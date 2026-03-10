"""
微调 BAAI/bge-large-zh 嵌入模型。

使用 sentence-transformers v3 + MultipleNegativesRankingLoss（InfoNCE）。
模型使用 CLS token 作为句子表示，与 index_builder/embedder.py 保持一致。

用法：
    # 使用默认配置（先运行 prepare_data.py 生成数据）
    python -m finetune.train

    # 自定义路径和超参
    python -m finetune.train \\
        --base_model_dir model_cache/BAAI/bge-large-zh \\
        --output_dir     model_cache/trained/BAAI/bge-large-zh \\
        --num_epochs 5   --batch_size 32 --learning_rate 1e-5
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for pkg in ("sentence_transformers", "datasets", "accelerate"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            "缺少依赖，请先安装：\n"
            f"  pip install {' '.join(missing)}\n"
            "或：  pip install -r finetune/requirements_finetune.txt",
            file=sys.stderr,
        )
        sys.exit(1)


def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(
            f"找不到数据文件：{path}\n"
            "请先运行：python -m data_process.prepare_data",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_hf_dataset(pairs: list[dict]):
    """Convert list[{query, positive, [negative]}] → HuggingFace Dataset."""
    from datasets import Dataset

    has_neg = any("negative" in p for p in pairs)
    data = {
        "anchor":   [p["query"]    for p in pairs],
        "positive": [p["positive"] for p in pairs],
    }
    if has_neg:
        # Fall back to positive when a pair has no explicit negative
        data["negative"] = [p.get("negative", p["positive"]) for p in pairs]
    return Dataset.from_dict(data)


def build_model(model_dir: str, max_seq_length: int):
    """
    Load bge-large-zh with CLS-token pooling.

    bge models were trained with CLS pooling (not mean pooling), so we
    explicitly configure the Pooling layer to match that setup.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer

    transformer = Transformer(model_dir, max_seq_length=max_seq_length)
    pooling = Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
    )
    return SentenceTransformer(modules=[transformer, pooling])


def build_ir_evaluator(val_pairs: list[dict], name: str = "val"):
    """
    Build an InformationRetrievalEvaluator for NDCG@10 / MRR@10 monitoring.
    Each val pair is treated as a single-relevant-document IR query.
    """
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    queries, corpus, relevant = {}, {}, {}
    for i, pair in enumerate(val_pairs):
        qid, did = f"q{i}", f"d{i}"
        queries[qid]  = pair["query"]
        corpus[did]   = pair["positive"]
        relevant[qid] = {did}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        name=name,
        show_progress_bar=False,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    from finetune.config import FinetuneConfig

    cfg = FinetuneConfig()
    p = argparse.ArgumentParser(description="微调 BAAI/bge-large-zh")
    p.add_argument("--base_model_dir", default=cfg.base_model_dir)
    p.add_argument("--output_dir",     default=cfg.output_dir)
    p.add_argument("--train_file",     default=cfg.train_file)
    p.add_argument("--val_file",       default=cfg.val_file)
    p.add_argument("--num_epochs",     type=int,   default=cfg.num_epochs)
    p.add_argument("--batch_size",     type=int,   default=cfg.batch_size)
    p.add_argument("--learning_rate",  type=float, default=cfg.learning_rate)
    p.add_argument("--warmup_ratio",   type=float, default=cfg.warmup_ratio)
    p.add_argument("--max_seq_length", type=int,   default=cfg.max_seq_length)
    p.add_argument("--eval_steps",     type=int,   default=cfg.eval_steps)
    p.add_argument("--save_steps",     type=int,   default=cfg.save_steps)
    p.add_argument("--save_total_limit", type=int, default=cfg.save_total_limit)
    p.add_argument("--logging_steps",  type=int,   default=cfg.logging_steps)
    p.add_argument("--seed",           type=int,   default=cfg.seed)
    return p.parse_args()


def main() -> None:
    _check_deps()

    from sentence_transformers import SentenceTransformerTrainer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import (
        BatchSamplers,
        SentenceTransformerTrainingArguments,
    )

    args = parse_args()

    use_cuda = torch.cuda.is_available()
    print(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if use_cuda else 'CPU'}")
    print(f"Base model : {args.base_model_dir}")
    print(f"Output dir : {args.output_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading data …")
    train_pairs = load_jsonl(args.train_file)
    val_pairs   = load_jsonl(args.val_file)
    print(f"  train: {len(train_pairs)}  |  val: {len(val_pairs)}")

    train_dataset = make_hf_dataset(train_pairs)
    val_dataset   = make_hf_dataset(val_pairs)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nLoading model …")
    model = build_model(args.base_model_dir, args.max_seq_length)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # MultipleNegativesRankingLoss (InfoNCE with in-batch + explicit negatives)
    loss = MultipleNegativesRankingLoss(model)

    # ── Evaluator ─────────────────────────────────────────────────────────────
    evaluator = build_ir_evaluator(val_pairs, name="val")

    # ── Training arguments ────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_pairs) // args.batch_size)
    warmup_steps = int(steps_per_epoch * args.num_epochs * args.warmup_ratio)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        # Mixed precision: fp16 on CUDA, plain fp32 on CPU
        fp16=use_cuda,
        bf16=False,
        # Evaluation & checkpointing
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        # NO_DUPLICATES prevents false negatives in MNRL batches
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=args.seed,
        run_name="bge-large-zh-domain",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    print("\nStarting training …\n")
    trainer.train()

    # Save final model (HuggingFace-compatible format)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    print(f"\nFinal model saved → {output_path}")

    # Also export the tokenizer alongside the model weights so it can be
    # loaded directly with AutoTokenizer / GTEEmbedder.
    model[0].tokenizer.save_pretrained(str(output_path))
    print("Tokenizer saved.")


if __name__ == "__main__":
    main()
