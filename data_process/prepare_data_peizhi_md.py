"""
将 data/txt_gen/*_配置.txt 每行与 data/txt/*_配置.json 中对应的 markdown 文档对齐，
生成嵌入模型微调数据集。

- query    : data/txt_gen/*_配置.txt 的一行（自然语言参数描述）
- positive : data/txt/*_配置.json 中语义对应的 document 字段（完整 markdown 表格行）
- negative : 同参数名、不同车型的 markdown 文档（hard negative）

对应关系：txt_gen 文件中的每一行是从 JSON document 的表格某列提取后转写而来的，
因此 N 个版本的同一参数行 → 映射回同一个 JSON document（该参数的多版本对比行）。

匹配逻辑：
  1. 从 txt_gen 行解析出参数关键词（param / feature / value 等）
  2. 对 JSON document 的 param_name（车型版本列）做归一化（去单位括号、去分隔符）
  3. 用子串匹配找到对应 document

用法：
    python -m data_process.prepare_data_peizhi_md
    python -m data_process.prepare_data_peizhi_md --txt_dir data/txt_gen --json_dir data/txt
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
_TXT_DIR  = _ROOT / "data" / "txt_gen"
_JSON_DIR = _ROOT / "data" / "txt"
_OUT_DIR  = _ROOT / "data_process" / "data"

# ── 复用 prepare_data_peizhi 的解析逻辑 ──────────────────────────────────────
from data_process.prepare_data_peizhi import (
    parse_line  as _parse_txt_line,
    _normalize_line,
    _MODEL_FAMILIES,
)


# ── 参数名归一化 ────────────────────────────────────────────────────────────────

def _norm_param(s: str) -> str:
    """
    归一化参数名，用于跨数据源匹配：
      1. 去掉括号内的单位/说明（中英文括号均处理）
      2. 去掉 * × / - 等分隔符
      3. 压缩空白
    示例：
      "长*宽*高 (mm)" → "长宽高"
      "轴距 (mm)"     → "轴距"
      "全国零售价(元)" → "全国零售价"
      "满载离地间隙"  → "满载离地间隙"（不变）
    """
    s = re.sub(r"\s*[（(][^)）]*[)）]", "", s)  # 去括号及内容
    s = re.sub(r"[*×/\-]", "", s)               # 去分隔符
    s = re.sub(r"\s+", "", s)                   # 去空白
    return s.strip()


# ── JSON 文档解析 ────────────────────────────────────────────────────────────

def _parse_md_row(line: str) -> list[str]:
    """将 markdown 表格行拆成各列（去掉首尾 |，按 | 分割，保留空单元格）。"""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_json_doc(item: dict) -> Optional[dict]:
    """
    将 JSON 配置文档解析为结构化字典：
      model      : 车型系列（来自 metadata.category）
      param_name : 原始参数名（markdown 数据行的 col-1）
      param_norm : 归一化后的参数名（用于匹配）
      document   : 原始 markdown 文本（用作 positive）
      id         : 文档 ID
    """
    document = item.get("document", "")
    lines = [l for l in document.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return None

    data_row = _parse_md_row(lines[-1])   # 数据行（最后一行）
    if len(data_row) < 2:
        return None

    param_name = data_row[1]
    return {
        "id":         item.get("id", ""),
        "model":      item.get("metadata", {}).get("category", ""),
        "param_name": param_name,
        "param_norm": _norm_param(param_name),
        "document":   document,
    }


def load_json_docs(json_dir: Path) -> list[dict]:
    """加载所有 *_配置.json 文件，返回解析后的文档列表。"""
    docs: list[dict] = []
    for path in sorted(json_dir.glob("*_配置.json")):
        items = json.loads(path.read_text(encoding="utf-8"))
        parsed_count = 0
        for item in items:
            parsed = parse_json_doc(item)
            if parsed:
                docs.append(parsed)
                parsed_count += 1
        print(f"  {path.name}: {parsed_count} 文档")
    return docs


# ── 匹配索引 ─────────────────────────────────────────────────────────────────

def build_match_index(
    docs: list[dict],
) -> dict[str, dict[str, list[dict]]]:
    """
    构建 model × param_norm → [doc] 的二级索引，用于快速匹配。
    同一 model + param_norm 可能有多个文档（极少），取第一个。
    """
    idx: dict[str, dict[str, list[dict]]] = {}
    for doc in docs:
        idx.setdefault(doc["model"], {}).setdefault(doc["param_norm"], []).append(doc)
    return idx


def _find_doc(
    family: str,
    query_param: str,
    idx: dict[str, dict[str, list[dict]]],
) -> Optional[dict]:
    """
    在 idx[family] 中找与 query_param 最匹配的文档。
    匹配优先级：
      1. 归一化后精确匹配
      2. query_norm 是某 param_norm 的子串（或反向）→ 取最长公共子串的那个
    """
    if family not in idx:
        return None

    model_idx = idx[family]
    qnorm = _norm_param(query_param)

    # 1. 精确匹配
    if qnorm in model_idx:
        return model_idx[qnorm][0]

    # 2. 子串匹配 — 找最佳（common length 最长）
    best: Optional[dict] = None
    best_score = 0
    for param_norm, doc_list in model_idx.items():
        if qnorm in param_norm:
            score = len(qnorm)
        elif param_norm in qnorm:
            score = len(param_norm)
        else:
            continue
        if score > best_score:
            best_score = score
            best = doc_list[0]

    return best


def find_matching_doc(
    parsed: dict,
    idx: dict[str, dict[str, list[dict]]],
) -> Optional[dict]:
    """给定一条 txt_gen 解析结果，找对应的 JSON 文档。"""
    family = parsed["family"]
    t = parsed["type"]

    if t == "value":
        qp = parsed.get("param", "")
    elif t == "feat":
        qp = parsed.get("feature", "")
    elif t in ("plain",):
        qp = parsed.get("param", "")
    elif t == "node":
        qp = parsed.get("value", "")
    else:
        qp = parsed.get("content", "")

    if not qp:
        return None
    return _find_doc(family, qp, idx)


# ── Negative mining ──────────────────────────────────────────────────────────

def _build_cross_index(
    docs: list[dict],
) -> dict[str, dict[str, list[dict]]]:
    """
    cross_index[param_norm][model] = [doc]
    用于跨车型的 hard negative：同参数名、不同车型。
    """
    cidx: dict[str, dict[str, list[dict]]] = {}
    for doc in docs:
        cidx.setdefault(doc["param_norm"], {}).setdefault(doc["model"], []).append(doc)
    return cidx


def _sample_negative(
    matched_doc: dict,
    cross_idx: dict[str, dict[str, list[dict]]],
    all_docs: list[dict],
    rng: random.Random,
) -> Optional[str]:
    family     = matched_doc["model"]
    param_norm = matched_doc["param_norm"]

    # 1. 同参数、不同车型（hard negative）
    same_param = cross_idx.get(param_norm, {})
    other_families = [f for f in same_param if f != family]
    if other_families:
        neg = rng.choice(same_param[rng.choice(other_families)])
        return neg["document"]

    # 2. 不同车型、任意参数（fallback）
    others = [d for d in all_docs if d["model"] != family]
    if others:
        return rng.choice(others)["document"]
    return None


# ── 数据集构建 ────────────────────────────────────────────────────────────────

def load_txt_lines(txt_dir: Path) -> list[tuple[str, dict]]:
    """
    加载所有 *_配置.txt 行并解析，返回 (raw_line, parsed) 列表。
    倒装句先归一化。
    """
    result: list[tuple[str, dict]] = []
    for path in sorted(txt_dir.glob("*_配置.txt")):
        raw_count = parsed_count = 0
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            raw_count += 1
            normalized = _normalize_line(raw)
            p = _parse_txt_line(normalized)
            if p:
                result.append((raw, p))
                parsed_count += 1
        print(f"  {path.name}: {raw_count} 行，{parsed_count} 条解析成功")
    return result


def build_pairs(
    txt_lines: list[tuple[str, dict]],
    json_docs: list[dict],
    idx: dict,
    cross_idx: dict,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)

    matched: list[dict] = []
    unmatched = 0

    for raw_line, parsed in txt_lines:
        doc = find_matching_doc(parsed, idx)
        if doc is None:
            unmatched += 1
            continue

        neg = _sample_negative(doc, cross_idx, json_docs, rng)
        pair: dict = {"query": raw_line, "positive": doc["document"]}
        if neg:
            pair["negative"] = neg
        matched.append(pair)

    # 打乱后分割
    rng.shuffle(matched)
    split = max(1, int(len(matched) * (1 - val_ratio)))
    return matched[:split], matched[split:]


# ── I/O ──────────────────────────────────────────────────────────────────────

def save_jsonl(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Saved {len(pairs):>5} pairs → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="配置 txt_gen → JSON document 对齐")
    parser.add_argument("--txt_dir",    default=str(_TXT_DIR))
    parser.add_argument("--json_dir",   default=str(_JSON_DIR))
    parser.add_argument("--output_dir", default=str(_OUT_DIR))
    parser.add_argument("--val_ratio",  type=float, default=0.1)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    txt_dir    = Path(args.txt_dir)
    json_dir   = Path(args.json_dir)
    output_dir = Path(args.output_dir)

    # ── 加载 JSON 文档 ──────────────────────────────────────────────────────
    print(f"Loading *_配置.json from {json_dir} …")
    json_docs = load_json_docs(json_dir)
    print(f"  Total JSON docs: {len(json_docs)}")

    idx       = build_match_index(json_docs)
    cross_idx = _build_cross_index(json_docs)

    # ── 加载 txt_gen 行 ─────────────────────────────────────────────────────
    print(f"\nLoading *_配置.txt from {txt_dir} …")
    txt_lines = load_txt_lines(txt_dir)
    print(f"  Total txt lines parsed: {len(txt_lines)}")

    # ── 构建训练对 ──────────────────────────────────────────────────────────
    print("\nMatching and building pairs …")
    train_pairs, val_pairs = build_pairs(
        txt_lines, json_docs, idx, cross_idx,
        val_ratio=args.val_ratio, seed=args.seed,
    )

    unmatched = len(txt_lines) - len(train_pairs) - len(val_pairs)
    has_neg   = sum(1 for p in train_pairs if "negative" in p)
    print(f"  Matched:   {len(train_pairs) + len(val_pairs)}")
    print(f"  Unmatched: {unmatched} ({unmatched / len(txt_lines) * 100:.1f}%)")
    print(f"  Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")
    print(f"  Pairs with hard negatives: {has_neg}/{len(train_pairs)}")

    # ── 保存 ────────────────────────────────────────────────────────────────
    print("\nSaving …")
    save_jsonl(train_pairs, output_dir / "train_peizhi_md.jsonl")
    save_jsonl(val_pairs,   output_dir / "val_peizhi_md.jsonl")
    print("Done.")


if __name__ == "__main__":
    main()
