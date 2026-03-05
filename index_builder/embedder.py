"""
嵌入模型封装（默认：BAAI/bge-large-zh）

使用 modelscope.hub.snapshot_download 下载模型，
通过 HuggingFace transformers 进行推理，输出 L2 归一化向量。

模型 ID 从 .env 中的 EMBED_MODEL_ID 读取，默认为 BAAI/bge-large-zh。
"""

import os
import threading
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import List

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

_MODEL_LOAD_LOCK = threading.Lock()

MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-large-zh")
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "model_cache")


def _download_model() -> str:
    """从 ModelScope 下载模型，返回本地路径。"""
    from modelscope.hub.snapshot_download import snapshot_download
    os.makedirs(_CACHE_DIR, exist_ok=True)
    print(f"  Downloading model {MODEL_ID} …")
    local_path = snapshot_download(MODEL_ID, cache_dir=_CACHE_DIR)
    print(f"  Model ready: {local_path}")
    return local_path


class GTEEmbedder:
    """
    嵌入模型封装，支持批量推理。
    默认使用 BAAI/bge-large-zh，可通过 .env 的 EMBED_MODEL_ID 切换。
    输出的向量已做 L2 归一化，可直接用于余弦相似度计算。
    """

    MAX_LENGTH = 512
    BATCH_SIZE = 32

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        with _MODEL_LOAD_LOCK:
            # 双重检查，防止多线程重复加载
            if self._model is not None:
                return
            model_dir = _download_model()
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self._model = AutoModel.from_pretrained(model_dir)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()
            print(f"  Embed model [{MODEL_ID}] loaded on {self._device}")

    @torch.no_grad()
    def _infer_batch(self, texts: List[str]) -> np.ndarray:
        self._load()
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        outputs = self._model(**encoded)
        # CLS token 作为句子表示
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)

    def embed(self, texts: List[str]) -> np.ndarray:
        """将文本列表转为归一化向量矩阵 (N, D)。"""
        all_vecs: List[np.ndarray] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start: start + self.BATCH_SIZE]
            all_vecs.append(self._infer_batch(batch))
        return np.vstack(all_vecs)

    def embed_one(self, text: str) -> np.ndarray:
        """嵌入单条文本，返回 shape (D,) 的向量。"""
        return self.embed([text])[0]

    def dim(self) -> int:
        """返回嵌入维度。"""
        self._load()
        # 用一条空文本探测维度
        return self._infer_batch([""]).shape[1]
