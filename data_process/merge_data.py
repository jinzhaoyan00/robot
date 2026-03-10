"""
将 data_process/data/ 下所有 train_*.jsonl 和 val_*.jsonl 分别合并为：
  - data_process/data/train_all.jsonl
  - data_process/data/val_all.jsonl

用于后续微调训练（finetune/train.py 默认读取这两个文件）。

各子数据集说明：
  train.jsonl / val.jsonl             : 基础 QA + 配置 + 电池保修（prepare_data.py）
  train_peizhi.jsonl / val_peizhi.jsonl: 配置参数句对（prepare_data_peizhi.py）
  train_qa.jsonl / val_qa.jsonl       : 答疑问答对（prepare_data_qa.py）
  train_peizhi_md.jsonl / val_peizhi_md.jsonl: 配置句 → markdown 文档对齐（prepare_data_peizhi_md.py）

用法：
    python -m data_process.merge_data
    python -m data_process.merge_data --data_dir data_process/data --shuffle --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_ROOT     = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data_process" / "data"


def merge_splits(
    data_dir: Path,
    prefix: str,
    output_file: Path,
    shuffle: bool = True,
    seed: int = 42,
) -> int:
    """
    合并 data_dir 下所有匹配 f"{prefix}_*.jsonl" 的文件（不含输出文件自身），
    写入 output_file，返回合并后的总行数。

    Args:
        data_dir    : 数据目录
        prefix      : "train" 或 "val"
        output_file : 输出路径
        shuffle     : 是否打乱顺序（建议训练集打乱）
        seed        : 随机种子
    """
    # 匹配 train.jsonl、train_*.jsonl（val 同理），排除输出文件自身
    sources = sorted(
        p for p in data_dir.glob("*.jsonl")
        if (p.stem == prefix or p.stem.startswith(f"{prefix}_"))
        and p.resolve() != output_file.resolve()
    )

    if not sources:
        print(f"  [{prefix}] 未找到 {prefix}*.jsonl 文件，跳过。")
        return 0

    records: list[dict] = []
    for path in sources:
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  [{prefix}] {path.name}: {len(lines)} 条")
        records.extend(lines)

    if shuffle:
        random.seed(seed)
        random.shuffle(records)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  [{prefix}] 合并完成 → {output_file}  (共 {len(records)} 条)")
    return len(records)


def merge_all(
    data_dir: Path = _DATA_DIR,
    train_output: Path | None = None,
    val_output: Path | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[int, int]:
    """
    合并训练集和验证集，返回 (train_count, val_count)。

    Args:
        data_dir     : 包含各子数据集的目录（默认 data_process/data）
        train_output : 合并后训练集路径（默认 data_dir/train_all.jsonl）
        val_output   : 合并后验证集路径（默认 data_dir/val_all.jsonl）
        shuffle      : 是否打乱训练集顺序
        seed         : 随机种子
    """
    if train_output is None:
        train_output = data_dir / "train_all.jsonl"
    if val_output is None:
        val_output = data_dir / "val_all.jsonl"

    print(f"数据目录: {data_dir}")
    print()

    n_train = merge_splits(data_dir, "train", train_output, shuffle=shuffle, seed=seed)
    print()
    n_val   = merge_splits(data_dir, "val",   val_output,   shuffle=False,   seed=seed)

    return n_train, n_val


def main() -> None:
    parser = argparse.ArgumentParser(description="合并子数据集为 train_all/val_all")
    parser.add_argument("--data_dir",     default=str(_DATA_DIR), help="数据目录")
    parser.add_argument("--train_output", default=None,           help="训练集输出路径")
    parser.add_argument("--val_output",   default=None,           help="验证集输出路径")
    parser.add_argument("--no_shuffle",   action="store_true",    help="不打乱训练集顺序")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_output = Path(args.train_output) if args.train_output else None
    val_output   = Path(args.val_output)   if args.val_output   else None

    n_train, n_val = merge_all(
        data_dir=data_dir,
        train_output=train_output,
        val_output=val_output,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
    print()
    print(f"汇总: train_all={n_train} 条  |  val_all={n_val} 条")


if __name__ == "__main__":
    main()
