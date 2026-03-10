"""Fine-tuning configuration for BAAI/bge-large-zh."""

from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FinetuneConfig:
    # ── Paths ────────────────────────────────────────────────────────────────
    base_model_dir: str = str(_ROOT / "vllm" / "models" / "BAAI" / "bge-large-zh")
    output_dir: str = str(_ROOT / "vllm" / "trained" / "BAAI" / "bge-large-zh")
    train_file: str = str(_ROOT / "data_process" / "data" / "train_all.jsonl")
    val_file: str = str(_ROOT / "data_process" / "data" / "val_all.jsonl")

    # ── Training hyperparameters ─────────────────────────────────────────────
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 512

    # ── Data split ───────────────────────────────────────────────────────────
    val_ratio: float = 0.1
    seed: int = 42

    # ── Checkpointing & logging ──────────────────────────────────────────────
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 10
    logging_steps: int = 10
