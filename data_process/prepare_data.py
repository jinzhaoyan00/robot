"""
从 data/txt/*.json 生成微调用的训练对，输出到 finetune/data/train.jsonl 和 val.jsonl。

每行格式：{"query": "...", "positive": "...", "negative": "..."}

数据策略：
  - 答疑 文件：提取 "问题" 列作为 query，完整文档作为 positive
  - 配置 文件：由 "参数" 和 "车型版本" 列构造自然语言 query
  - 电池保修 文件：使用模板 query
  - 负样本：从不同车型的文档中随机采样（hard negative）

用法：
    python -m data_process.prepare_data
    python -m data_process.prepare_data --data_dir data/txt --output_dir finetune/data --val_ratio 0.1
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data" / "txt"
_OUTPUT_DIR = _ROOT / "data_process" / "data"


# ── Markdown table parser ────────────────────────────────────────────────────

def _split_md_row(line: str) -> list[str]:
    """Split a markdown table row into cells, stripping leading/trailing pipes."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(c.replace("-", "").replace(" ", "")) == set() for c in cells)


def parse_md_table(document: str) -> dict:
    """
    Parse a single-data-row markdown table into a dict.

    Handles two layouts found in the knowledge base:

    Layout A – 答疑 / generic (header + sep + data):
        | 类别 | 问题 | 话术 |
        | --- | --- | --- |
        | 车型信息 | EH7是什么车？ | 回答... |

    Layout B – 配置 (header + sep + data, first two cols are param categories):
        | 参数 | 车型版本 | 版本A | 版本B |
        | --- | --- | --- | --- |
        | 价格 | 全国零售价(元) | 208800 | 218800 |
    """
    lines = [l for l in document.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return {}

    headers = _split_md_row(lines[0])
    # Skip separator rows to find the first data row
    data_line_idx = next(
        (i for i in range(1, len(lines)) if not _is_separator_row(_split_md_row(lines[i]))),
        None,
    )
    if data_line_idx is None:
        return {}

    values = _split_md_row(lines[data_line_idx])
    return dict(zip(headers, values))


# ── Per-tag extraction logic ─────────────────────────────────────────────────

def _extract_qa(doc: dict) -> Optional[tuple[str, str]]:
    """答疑: 问题列 → (query, document)."""
    row = parse_md_table(doc["document"])
    question = row.get("问题", "").strip()
    if len(question) < 5:
        return None
    return question, doc["document"]


def _extract_config(doc: dict) -> Optional[tuple[str, str]]:
    """配置: 由参数类别+参数名拼成自然语言 query."""
    row = parse_md_table(doc["document"])
    category = doc["metadata"].get("category", "")

    # Layout: 参数=price_category, 车型版本=specific_param
    param_cat = row.get("参数", "").strip()
    param_name = row.get("车型版本", "").strip()

    if not param_cat and not param_name:
        return None

    if param_cat and param_name:
        query = f"{category}的{param_cat}中{param_name}是多少？"
    elif param_cat:
        query = f"{category}的{param_cat}是多少？"
    else:
        query = f"{category}的{param_name}是多少？"

    return query, doc["document"]


# Template queries for 电池保修 documents
_BATTERY_TEMPLATES = [
    "{cat}的电池保修政策是什么？",
    "{cat}电池质量保证期限是多久？",
    "{cat}动力电池容量衰减多少可以保修？",
    "{cat}的电池保修条件有哪些？",
    "{cat}电池出现问题怎么办？",
]


def _extract_battery(doc: dict, rng: random.Random) -> Optional[tuple[str, str]]:
    """电池保修: 从模板中随机抽取 query."""
    cat = doc["metadata"].get("category", "红旗")
    document = doc["document"].strip()
    if not document:
        return None
    template = rng.choice(_battery_templates_for(cat))
    return template.format(cat=cat), document


def _battery_templates_for(cat: str) -> list[str]:
    return [t.format(cat=cat) for t in _BATTERY_TEMPLATES]


# ── Hard negative mining ─────────────────────────────────────────────────────

def _sample_negative(
    doc: dict,
    model_docs: dict[str, list[dict]],
    rng: random.Random,
) -> Optional[str]:
    """Sample a document from a different car model as a hard negative."""
    own_cat = doc["metadata"].get("category", "")
    others = [c for c in model_docs if c != own_cat and model_docs[c]]
    if not others:
        return None
    neg_doc = rng.choice(model_docs[rng.choice(others)])
    return neg_doc["document"]


# ── Main pipeline ────────────────────────────────────────────────────────────

def load_documents(data_dir: Path) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(data_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            item.setdefault("metadata", {})
            item["metadata"]["_source"] = path.stem
        docs.extend(items)
    return docs


def build_pairs(
    docs: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)

    # Index documents by car model for negative mining
    model_docs: dict[str, list[dict]] = {}
    for doc in docs:
        cat = doc["metadata"].get("category", "unknown")
        model_docs.setdefault(cat, []).append(doc)

    pairs: list[dict] = []

    for doc in docs:
        tags = doc["metadata"].get("tags", [])

        if "答疑" in tags:
            result = _extract_qa(doc)
        elif "配置" in tags:
            result = _extract_config(doc)
        elif "电池保修" in tags:
            result = _extract_battery(doc, rng)
        else:
            continue

        if result is None:
            continue

        query, positive = result
        negative = _sample_negative(doc, model_docs, rng)

        pair: dict = {"query": query, "positive": positive}
        if negative:
            pair["negative"] = negative
        pairs.append(pair)

    rng.shuffle(pairs)
    split = max(1, int(len(pairs) * (1 - val_ratio)))
    return pairs[:split], pairs[split:]


def save_jsonl(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"  Saved {len(pairs):>5} pairs → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 bge-large-zh 微调训练数据")
    parser.add_argument("--data_dir", default=str(_DATA_DIR), help="data/txt 目录路径")
    parser.add_argument("--output_dir", default=str(_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"Loading documents from {data_dir} …")
    docs = load_documents(data_dir)
    print(f"  Loaded {len(docs)} documents from {len(list(data_dir.glob('*.json')))} files")

    print("Building training pairs …")
    train_pairs, val_pairs = build_pairs(docs, val_ratio=args.val_ratio, seed=args.seed)
    print(f"  Total: {len(train_pairs) + len(val_pairs)}  |  train: {len(train_pairs)}  |  val: {len(val_pairs)}")

    print("Saving …")
    save_jsonl(train_pairs, output_dir / "train.jsonl")
    save_jsonl(val_pairs, output_dir / "val.jsonl")
    print("Done.")


if __name__ == "__main__":
    main()
