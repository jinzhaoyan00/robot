"""
从 data/txt_gen/*_配置.txt 生成嵌入模型微调训练数据。

每行是形如 "{车型版本}{谓词}{参数/特性}" 的自然语言句子。
脚本将其转化为 (query, positive, negative) 三元组：
  - anchor (query)  : 模拟用户的自然语言提问
  - positive        : 能回答该问题的原始句子
  - negative (hard) : 同类参数但不同车型的句子

输出格式（每行一条）：
  {"query": "...", "positive": "...", "negative": "..."}

用法：
    python -m data_process.prepare_data_peizhi
    python -m data_process.prepare_data_peizhi --data_dir data/txt_gen --val_ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data" / "txt_gen"
_OUTPUT_DIR = _ROOT / "data_process" / "data"

# ── Car model family detection ────────────────────────────────────────────────

_MODEL_FAMILIES = ("EH7", "天工08", "天工06", "天工05")  # 长名在前，防止 "天工0" 提前截断

# 版本后缀（从最长到最短，避免贪婪截断）
_VERSION_SUFFIXES = (
    "四驱智选版车型", "四驱版车型", "智选版车型",
    "四驱先锋版", "四驱智选版", "四驱版",
    "先锋版", "智选版", "版本", "车型",
)
_SUFFIX_PAT = "|".join(re.escape(s) for s in _VERSION_SUFFIXES)

_RE_MV = re.compile(
    rf"^((?:{'|'.join(re.escape(f) for f in _MODEL_FAMILIES)})"
    rf"[\s\w\d]+?(?:{_SUFFIX_PAT}))\s*(.+?)。?\s*$"
)


def _extract_mv(line: str) -> tuple[str, str]:
    """返回 (model_version, remainder)；remainder 已去除句末 '。' 和首尾空白。"""
    m = _RE_MV.match(line)
    if m:
        mv = m.group(1).strip()
        rem = m.group(2).strip().rstrip("。").strip()
        return mv, rem
    return "", ""


def _family(mv: str) -> str:
    for f in _MODEL_FAMILIES:
        if mv.startswith(f):
            return f
    return "unknown"


# ── Inverted-sentence pre-processor ──────────────────────────────────────────
# 天工05 有些句子结构是 "{feature}在{mv}上为{value}。"，需要改写为正常顺序。
_RE_INVERTED = re.compile(
    rf"^(.+?)在((?:{'|'.join(re.escape(f) for f in _MODEL_FAMILIES)})[\s\w\d]+?(?:{_SUFFIX_PAT}))上(?:为|是)(.+?)。?\s*$"
)


def _normalize_line(line: str) -> str:
    """将倒装句改写为 '{mv}的{feature}为{value}。' 的正常形式。"""
    m = _RE_INVERTED.match(line.strip())
    if m:
        feature, mv, value = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        return f"{mv}的{feature}为{value}。"
    return line


# ── Sentence pattern classifiers ──────────────────────────────────────────────

# 有"的"：标准参数-值句  "的{param}[verb]{value}"
_RE_A = re.compile(r"^的(.+?)(?:为|是|具备|支持|采用|包含|包括|达|配备|提供)(.+)$")

# 功能性谓词句（无"的"）：{pred}{feature}
_FEAT_PREDS = (
    "不可选装", "可选装",                              # 长前缀放前
    "未配备", "不支持", "不具备",
    "搭载", "标配", "配有", "配备",
    "选装", "具备", "拥有", "带有", "使用",
    "采用", "支持", "无", "有",
)
_RE_B = re.compile(r"^(" + "|".join(re.escape(p) for p in _FEAT_PREDS) + r")(.+)$")

# 车型级陈述句（无"的"，直接以为/是开头）
_RE_C = re.compile(r"^(?:为|是)(.+)$")

# 无分隔符：{param}{numeric_value}  尝试从末尾数字分割
# 覆盖所有 ASCII 字母（kW/km/L/N·m/...）和 CJK 字符（升/秒/座/毫米/...）
_RE_D_TAIL = re.compile(
    r"^(.+?)(\d[\d.,/*×\-a-zA-Z·°()（）%³²¹㎞\u4e00-\u9fff]*)$"
)

# 嵌入式 为/是/达/采用："{desc}[为|是|达|采用]{value}"（无"的"，无已知谓词开头）
_RE_D2 = re.compile(r"^(.+?)(?:为|是|达|采用)(.+)$")


# ── Query templates ───────────────────────────────────────────────────────────

def _query_value(mv: str, param: str, rng: random.Random) -> str:
    param = param.rstrip("为是")
    templates = [
        f"{mv}的{param}是多少？",
        f"{mv}的{param}是什么？",
        f"请问{mv}的{param}？",
        f"{mv}，{param}怎么样？",
    ]
    return rng.choice(templates)


_PRED_QUERY: dict[str, list[str]] = {
    "标配":  ["{mv}是否标配{f}？", "{mv}有没有{f}？", "{mv}配备了{f}吗？"],
    "未配备": ["{mv}是否配备{f}？", "{mv}有没有{f}？", "{mv}配置了{f}吗？"],
    "配备":  ["{mv}有没有{f}？", "{mv}是否配备了{f}？"],
    "配有":  ["{mv}有没有{f}？", "{mv}配置了{f}吗？"],
    "搭载":  ["{mv}有没有{f}？", "{mv}搭载了{f}吗？"],
    "选装":  ["{mv}的{f}是标配还是选装？", "{mv}能否加装{f}？"],
    "采用":  ["{mv}有没有{f}？", "{mv}用的是什么材质/系统？"],
    "使用":  ["{mv}使用的是什么{f}？", "{mv}有没有{f}？"],
    "带有":  ["{mv}有没有{f}？", "{mv}带了{f}吗？"],
    "拥有":  ["{mv}有{f}吗？", "{mv}是否具备{f}？"],
    "支持":  ["{mv}是否支持{f}？", "{mv}有{f}功能吗？"],
    "不支持": ["{mv}是否支持{f}？", "{mv}支持{f}吗？"],
    "具备":  ["{mv}有{f}吗？", "{mv}是否具备{f}？"],
    "不具备":  ["{mv}有{f}吗？", "{mv}是否具备{f}？"],
    "无":     ["{mv}有没有{f}？", "{mv}是否配备{f}？"],
    "有":     ["{mv}有{f}吗？", "{mv}是否有{f}？"],
    "可选装":  ["{mv}的{f}是否可以选装？", "{mv}能加装{f}吗？"],
    "不可选装": ["{mv}的{f}是否可以选装？", "{mv}能加装{f}吗？"],
}


def _query_feat(mv: str, pred: str, feature: str, rng: random.Random) -> str:
    f = feature
    tmpl_list = _PRED_QUERY.get(pred, ["{mv}的{f}配置是什么？"])
    tmpl = rng.choice(tmpl_list)
    try:
        return tmpl.format(mv=mv, f=f)
    except Exception:
        return f"{mv}是否配备{f}？"


def _query_nodē(mv: str, value: str, rng: random.Random) -> str:
    templates = [
        f"{mv}是什么类型的车？",
        f"{mv}的驱动或版本类型是什么？",
        f"请介绍{mv}的基本配置。",
    ]
    return rng.choice(templates)


def _query_plain(mv: str, param: str, rng: random.Random) -> str:
    param = param.rstrip("为是").strip()
    templates = [
        f"{mv}的{param}是多少？",
        f"{mv}的{param}是什么？",
        f"请问{mv}的{param}参数？",
    ]
    return rng.choice(templates)


def _query_other(mv: str, content: str, rng: random.Random) -> str:
    snippet = content[:8].rstrip("为是，。")
    templates = [
        f"{mv}的{snippet}相关配置是什么？",
        f"{mv}有关{snippet}的参数是多少？",
    ]
    return rng.choice(templates)


# ── Parser ───────────────────────────────────────────────────────────────────

def parse_line(line: str) -> Optional[dict]:
    """
    将一行配置文本解析为结构化 dict：
      type   : 'value' | 'feat' | 'node' | 'plain' | 'other'
      mv     : 车型版本字符串
      family : 车型系列（EH7 / 天工05 / 天工06 / 天工08）
      topic  : 参数主题关键词（用于负样本分组）
      text   : 原始句子
      ... type-specific fields ...
    """
    line = line.strip()
    if not line:
        return None

    # 倒装句改写："{feature}在{mv}上为{value}。" → "{mv}的{feature}为{value}。"
    line = _normalize_line(line)

    # 确保句末有句号（部分句子以 ³ 等上标结尾）
    raw = line
    if not line.endswith("。"):
        line = line.rstrip(" \t") + "。"

    mv, rem = _extract_mv(line)
    if not mv:
        return None

    family = _family(mv)

    # ── A. 有"的"：标准参数-值句 ─────────────────────────────────────────
    if rem.startswith("的"):
        m = _RE_A.match(rem)
        if m:
            param, value = m.group(1).strip(), m.group(2).strip()
            return {
                "type": "value", "mv": mv, "family": family,
                "param": param, "value": value,
                "topic": param,
                "text": raw,
            }

    # ── B. 功能性谓词句 ──────────────────────────────────────────────────
    m = _RE_B.match(rem)
    if m:
        pred, feature = m.group(1), m.group(2).strip()
        return {
            "type": "feat", "mv": mv, "family": family,
            "pred": pred, "feature": feature,
            "topic": feature[:10],
            "text": raw,
        }

    # ── C. 车型级陈述：以 为/是 开头 ────────────────────────────────────
    m = _RE_C.match(rem)
    if m:
        value = m.group(1).strip()
        return {
            "type": "node", "mv": mv, "family": family,
            "value": value,
            "topic": value[:8],
            "text": raw,
        }

    # ── D. 无分隔符：尝试末尾数值分割 ───────────────────────────────────
    m = _RE_D_TAIL.match(rem)
    if m:
        param = m.group(1).strip().rstrip("为是").strip()
        value = m.group(2).strip()
        if param:
            return {
                "type": "plain", "mv": mv, "family": family,
                "param": param, "value": value,
                "topic": param,
                "text": raw,
            }

    # ── D2. 嵌入式谓词：{desc}[为|是|达|采用]{value}（无"的"） ──────────
    m = _RE_D2.match(rem)
    if m:
        param = m.group(1).strip().rstrip("为是达").strip()
        value = m.group(2).strip()
        if param and value:
            return {
                "type": "plain", "mv": mv, "family": family,
                "param": param, "value": value,
                "topic": param,
                "text": raw,
            }

    # ── E. Fallback ──────────────────────────────────────────────────────
    if rem:
        return {
            "type": "other", "mv": mv, "family": family,
            "content": rem,
            "topic": rem[:8],
            "text": raw,
        }

    return None


# ── Query builder ─────────────────────────────────────────────────────────────

def make_query(rec: dict, rng: random.Random) -> str:
    t = rec["type"]
    mv = rec["mv"]
    if t == "value":
        return _query_value(mv, rec["param"], rng)
    elif t == "feat":
        return _query_feat(mv, rec["pred"], rec["feature"], rng)
    elif t == "node":
        return _query_nodē(mv, rec["value"], rng)
    elif t == "plain":
        return _query_plain(mv, rec["param"], rng)
    else:
        return _query_other(mv, rec.get("content", ""), rng)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_config_files(data_dir: Path) -> list[dict]:
    records: list[dict] = []
    pattern = "*_配置.txt"
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"在 {data_dir} 下没有找到 {pattern} 文件，请检查路径。"
        )
    for path in files:
        skipped = 0
        parsed_count = 0
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            rec = parse_line(raw_line)
            if rec is None:
                skipped += 1
                continue
            rec["source"] = path.stem
            records.append(rec)
            parsed_count += 1
        print(f"  {path.name}: {parsed_count} 条已解析，{skipped} 条跳过")
    return records


# ── Negative mining ───────────────────────────────────────────────────────────

def _build_topic_index(
    records: list[dict],
) -> dict[str, dict[str, list[int]]]:
    """
    返回 topic_index[topic_key][family] = [record_indices]
    用于快速检索同 topic、不同车型的 hard negative。
    """
    idx: dict[str, dict[str, list[int]]] = {}
    for i, rec in enumerate(records):
        tk = rec["topic"]
        fam = rec["family"]
        idx.setdefault(tk, {}).setdefault(fam, []).append(i)
    return idx


def _sample_negative(
    rec: dict,
    rec_idx: int,
    records: list[dict],
    topic_index: dict[str, dict[str, list[int]]],
    rng: random.Random,
) -> Optional[str]:
    fam = rec["family"]
    tk = rec["topic"]

    # 1. 同 topic，不同车型（hard negative）
    same_topic = topic_index.get(tk, {})
    other_fams = [f for f in same_topic if f != fam]
    if other_fams:
        chosen_fam = rng.choice(other_fams)
        neg_idx = rng.choice(same_topic[chosen_fam])
        return records[neg_idx]["text"]

    # 2. 同车型，不同 topic（medium negative）
    all_topics = list(topic_index.keys())
    rng.shuffle(all_topics)
    for other_tk in all_topics:
        if other_tk == tk:
            continue
        same_fam_list = topic_index[other_tk].get(fam, [])
        if same_fam_list:
            neg_idx = rng.choice(same_fam_list)
            if neg_idx != rec_idx:
                return records[neg_idx]["text"]

    # 3. 任意其他记录（last resort）
    candidates = [i for i in range(len(records)) if i != rec_idx]
    if candidates:
        return records[rng.choice(candidates)]["text"]
    return None


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_pairs(
    records: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    topic_index = _build_topic_index(records)

    # 打乱顺序，使 train/val 分布均匀
    order = list(range(len(records)))
    rng.shuffle(order)

    pairs: list[dict] = []
    for rec_idx in order:
        rec = records[rec_idx]
        query = make_query(rec, rng)
        positive = rec["text"]
        negative = _sample_negative(rec, rec_idx, records, topic_index, rng)

        pair: dict = {"query": query, "positive": positive}
        if negative:
            pair["negative"] = negative
        pairs.append(pair)

    split = max(1, int(len(pairs) * (1 - val_ratio)))
    return pairs[:split], pairs[split:]


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_jsonl(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"  Saved {len(pairs):>5} pairs → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="从 *_配置.txt 生成嵌入模型微调数据")
    parser.add_argument("--data_dir",  default=str(_DATA_DIR), help="txt_gen 目录路径")
    parser.add_argument("--output_dir", default=str(_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--val_ratio",  type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"Loading *_配置.txt from {data_dir} …")
    records = load_config_files(data_dir)
    print(f"  Total parsed: {len(records)}")

    type_counts = {}
    for r in records:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
    print("  Sentence types:", {k: v for k, v in sorted(type_counts.items())})

    print("\nBuilding training pairs …")
    train_pairs, val_pairs = build_pairs(records, val_ratio=args.val_ratio, seed=args.seed)
    print(f"  Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    has_neg = sum(1 for p in train_pairs if "negative" in p)
    print(f"  Pairs with hard negatives: {has_neg}/{len(train_pairs)}")

    print("\nSaving …")
    save_jsonl(train_pairs, output_dir / "train_peizhi.jsonl")
    save_jsonl(val_pairs,   output_dir / "val_peizhi.jsonl")
    print("Done.")


if __name__ == "__main__":
    main()
