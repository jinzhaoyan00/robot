"""
从 data/txt_gen/*_答疑.txt 生成嵌入模型微调训练数据。

每个文件由多个"问题/答案"块组成，脚本将其转化为 (query, positive, negative) 三元组：
  - anchor (query)  : 原始问题（已是自然语言提问，直接使用）
  - positive        : 对应答案（多行合并）
  - negative (hard) : 同类话题但不同车型的答案

此外，对于问题中含"这款车/这车"的隐式指代，会额外生成一条将指代替换为车型名称
的训练对，扩充数据多样性。

输出格式（每行一条）：
  {"query": "...", "positive": "...", "negative": "..."}

用法：
    python -m data_process.prepare_data_qa
    python -m data_process.prepare_data_qa --data_dir data/txt_gen --val_ratio 0.1
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

# ── Topic classification ───────────────────────────────────────────────────────

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "价格":   ["价格", "售价", "多少钱", "优惠", "折扣", "分期", "首付", "贷款", "置换", "补贴", "报价", "指导价"],
    "续航":   ["续航", "里程", "CLTC", "电量", "电池容量", "电耗"],
    "充电":   ["充电", "快充", "补能", "充电桩", "超充", "慢充", "V2L"],
    "动力":   ["驱动", "四驱", "后驱", "前驱", "动力", "扭矩", "功率", "加速", "电机", "马力", "车速"],
    "安全":   ["安全", "气囊", "防撞", "制动", "碰撞", "车身", "质保", "保修", "三包"],
    "维保":   ["维修", "保养", "售后", "4S", "年检", "保险"],
    "智驾":   ["智能驾驶", "辅助驾驶", "自动驾驶", "泊车", "领航", "辅驾", "ADAS", "视觉"],
    "外观":   ["外观", "颜色", "车身", "设计", "造型", "外形", "尺寸", "轴距"],
    "内饰":   ["内饰", "座椅", "空调", "音响", "屏幕", "中控", "方向盘", "氛围灯", "香氛"],
    "购车":   ["预订", "订购", "提车", "交付", "退订", "下订", "订车", "购买", "交定", "定金"],
}


def _topic_of(text: str) -> str:
    for topic, keywords in _TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return topic
    return "other"


# ── Parser ─────────────────────────────────────────────────────────────────────

def _parse_qa_file(path: Path) -> list[dict]:
    """
    解析一个 *_答疑.txt 文件，返回 Q&A 记录列表。
    每条记录包含：question, answer, model, topic。
    """
    car_model = path.stem.replace("_答疑", "")
    records: list[dict] = []

    current_q: Optional[str] = None
    answer_lines: list[str] = []
    in_answer = False

    def _flush():
        if current_q and answer_lines:
            answer = "\n".join(answer_lines)
            topic = _topic_of(current_q + answer)
            records.append({
                "question": current_q,
                "answer": answer,
                "model": car_model,
                "topic": topic,
            })

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        if line.startswith("问题："):
            _flush()
            current_q = line[3:].strip()
            answer_lines = []
            in_answer = False

        elif line.startswith("答案："):
            in_answer = True
            first = line[3:].strip()
            answer_lines = [first] if first else []

        elif in_answer and line:
            answer_lines.append(line)

        elif not line:
            # blank line — keep accumulating; flush happens on next "问题："
            pass

    _flush()  # last block
    return records


def load_qa_files(data_dir: Path) -> list[dict]:
    pattern = "*_答疑.txt"
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"在 {data_dir} 下没有找到 {pattern} 文件，请检查路径。")

    all_records: list[dict] = []
    for path in files:
        recs = _parse_qa_file(path)
        print(f"  {path.name}: {len(recs)} 条 Q&A")
        all_records.extend(recs)
    return all_records


# ── Negative mining ────────────────────────────────────────────────────────────

def _build_index(
    records: list[dict],
) -> dict[str, dict[str, list[int]]]:
    """返回 topic_index[topic][model] = [record_indices]。"""
    idx: dict[str, dict[str, list[int]]] = {}
    for i, rec in enumerate(records):
        idx.setdefault(rec["topic"], {}).setdefault(rec["model"], []).append(i)
    return idx


def _sample_negative(
    rec: dict,
    rec_idx: int,
    records: list[dict],
    index: dict[str, dict[str, list[int]]],
    rng: random.Random,
) -> Optional[str]:
    topic = rec["topic"]
    model = rec["model"]

    # 1. 同话题、不同车型（hard negative）
    same_topic = index.get(topic, {})
    other_models = [m for m in same_topic if m != model]
    if other_models:
        chosen = rng.choice(other_models)
        neg_idx = rng.choice(same_topic[chosen])
        return records[neg_idx]["answer"]

    # 2. 不同车型、任意话题（fallback）
    all_models = {r["model"] for r in records}
    other_models = [m for m in all_models if m != model]
    if other_models:
        chosen_model = rng.choice(list(other_models))
        candidates = [i for i, r in enumerate(records) if r["model"] == chosen_model]
        return records[rng.choice(candidates)]["answer"]

    return None


# ── Dataset builder ────────────────────────────────────────────────────────────

_RE_THIS_CAR = re.compile(r"这款车|这车(?!型)")  # 匹配"这款车"和"这车"（非"这车型"）


def _expand_pairs(rec: dict) -> list[tuple[str, str]]:
    """
    返回该记录生成的 (query, positive) 元组列表。
    对于含隐式指代（"这款车"/"这车"）的问题，额外生成
    一条将指代替换为车型名称的版本。
    """
    q, ans, model = rec["question"], rec["answer"], rec["model"]
    pairs = [(q, ans)]

    if _RE_THIS_CAR.search(q):
        q_explicit = _RE_THIS_CAR.sub(model, q)
        if q_explicit != q:
            pairs.append((q_explicit, ans))

    return pairs


def build_pairs(
    records: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    index = _build_index(records)

    # 打乱顺序保证 train/val 分布均衡
    order = list(range(len(records)))
    rng.shuffle(order)

    pairs: list[dict] = []
    for rec_idx in order:
        rec = records[rec_idx]
        neg = _sample_negative(rec, rec_idx, records, index, rng)

        for query, positive in _expand_pairs(rec):
            pair: dict = {"query": query, "positive": positive}
            if neg:
                pair["negative"] = neg
            pairs.append(pair)

    split = max(1, int(len(pairs) * (1 - val_ratio)))
    return pairs[:split], pairs[split:]


# ── I/O ────────────────────────────────────────────────────────────────────────

def save_jsonl(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"  Saved {len(pairs):>4} pairs → {path}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="从 *_答疑.txt 生成嵌入模型微调数据")
    parser.add_argument("--data_dir",   default=str(_DATA_DIR),   help="txt_gen 目录路径")
    parser.add_argument("--output_dir", default=str(_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--val_ratio",  type=float, default=0.1,  help="验证集比例")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"Loading *_答疑.txt from {data_dir} …")
    records = load_qa_files(data_dir)
    print(f"  Total Q&A pairs: {len(records)}")

    topic_dist = {}
    for r in records:
        topic_dist[r["topic"]] = topic_dist.get(r["topic"], 0) + 1
    print("  Topic distribution:", {k: v for k, v in sorted(topic_dist.items())})

    print("\nBuilding training pairs …")
    train_pairs, val_pairs = build_pairs(records, val_ratio=args.val_ratio, seed=args.seed)

    implicit_refs = sum(1 for r in records if _RE_THIS_CAR.search(r["question"]))
    has_neg = sum(1 for p in train_pairs if "negative" in p)
    print(f"  Base Q&A: {len(records)}  |  含隐式指代扩展: {implicit_refs}")
    print(f"  Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")
    print(f"  Pairs with hard negatives: {has_neg}/{len(train_pairs)}")

    print("\nSaving …")
    save_jsonl(train_pairs, output_dir / "train_qa.jsonl")
    save_jsonl(val_pairs,   output_dir / "val_qa.jsonl")
    print("Done.")


if __name__ == "__main__":
    main()
