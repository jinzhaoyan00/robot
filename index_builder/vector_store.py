"""
FAISS 向量存储模块
使用 ModelScope GTE-base 中文嵌入模型生成向量，存储在 FAISS 索引中。
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any

from .embedder import GTEEmbedder


class VectorStore:
    """基于 FAISS 的向量存储，使用 ModelScope BAAI/bge-large-zh 模型生成嵌入向量。"""

    BATCH_SIZE = 32

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.index_path = os.path.join(store_path, "faiss.index")
        self.metadata_path = os.path.join(store_path, "metadata.json")
        os.makedirs(store_path, exist_ok=True)
        self._embedder = GTEEmbedder()

    def build(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        """从文本列表构建并持久化 FAISS 索引。"""
        if len(texts) != len(metadata):
            raise ValueError("texts 和 metadata 长度必须一致")

        print(f"Building FAISS index: {len(texts)} texts …")
        all_vecs: List[np.ndarray] = []

        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start: start + self.BATCH_SIZE]
            vecs = self._embedder.embed(batch)
            all_vecs.append(vecs)
            done = min(start + self.BATCH_SIZE, len(texts))
            print(f"  Embedded {done}/{len(texts)}")

        matrix = np.vstack(all_vecs)
        dim = matrix.shape[1]

        # 内积索引（向量已 L2 归一化，内积 == 余弦相似度）
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"FAISS index saved: {index.ntotal} vectors, dim={dim}")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """向量相似度检索，返回 top-k 结果（含 score 字段）。"""
        index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        query_vec = self._embedder.embed([query])
        scores, indices = index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                entry = metadata[idx].copy()
                entry["score"] = float(score)
                results.append(entry)
        return results
