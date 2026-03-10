"""
ChromaDB 向量存储模块
使用 ModelScope BAAI/bge-large-zh 嵌入模型生成向量，存储在 ChromaDB 中。
"""

import os
import json
import chromadb
from typing import List, Dict, Any

from .embedder import GTEEmbedder


class VectorStore:
    """基于 ChromaDB 的向量存储，使用 ModelScope BAAI/bge-large-zh 模型生成嵌入向量。"""

    BATCH_SIZE = 8
    COLLECTION_NAME = "car"

    def __init__(self, store_path: str):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        self._embedder = GTEEmbedder()
        self._client = chromadb.PersistentClient(path=store_path)

    def build(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        """从文本列表构建并持久化 ChromaDB 索引。"""
        if len(texts) != len(metadata):
            raise ValueError("texts 和 metadata 长度必须一致")

        try:
            self._client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        collection = self._client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"Building ChromaDB index: {len(texts)} texts …")

        for start in range(0, len(texts), self.BATCH_SIZE):
            batch_texts = texts[start : start + self.BATCH_SIZE]
            batch_meta = metadata[start : start + self.BATCH_SIZE]
            vecs = self._embedder.embed(batch_texts)

            collection.add(
                ids=[str(start + i) for i in range(len(batch_texts))],
                embeddings=vecs.tolist(),
                documents=batch_texts,
                metadatas=[_flatten_metadata(m) for m in batch_meta],
            )
            done = min(start + self.BATCH_SIZE, len(texts))
            print(f"  Embedded {done}/{len(texts)}")

        print(f"ChromaDB collection saved: {collection.count()} vectors")

    def search(
        self,
        query: str,
        k: int = 5,
        tags: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        向量相似度检索，返回 top-k 结果（含 score 和 text 字段）。

        Args:
            tags: 可选标签列表。不为空时只检索 metadata 中包含任意一个标签的文档。
                  标签对应 metadata 中的 tag_<name> 布尔字段（由 build() 写入）。
        """
        collection = self._client.get_collection(name=self.COLLECTION_NAME)
        query_vec = self._embedder.embed([query])

        where: Dict[str, Any] | None = None
        if tags:
            if len(tags) == 1:
                where = {f"tag_{tags[0]}": {"$eq": True}}
            else:
                where = {"$or": [{f"tag_{t}": {"$eq": True}} for t in tags]}

        results = collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=k,
            include=["metadatas", "distances", "documents"],
            where=where,
        )

        output = []
        for meta, dist, doc in zip(
            results["metadatas"][0],
            results["distances"][0],
            results["documents"][0],
        ):
            entry = _unflatten_metadata(meta)
            entry["score"] = float(1.0 - dist)
            entry["text"] = doc
            output.append(entry)
        return output


def _flatten_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    将复杂类型的 metadata 值序列化为字符串（ChromaDB 只支持基础标量类型）。
    对于列表类型额外展开为 tag_<item>=True 的布尔字段，用于 where 过滤。
    """
    flat: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        elif isinstance(v, list):
            flat[k] = json.dumps(v, ensure_ascii=False)
            for item in v:
                if isinstance(item, str):
                    flat[f"tag_{item}"] = True
        else:
            flat[k] = json.dumps(v, ensure_ascii=False)
    return flat


def _unflatten_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """将字符串形式的 JSON 值还原为原始类型（仅处理以 { 或 [ 开头的字符串）。"""
    result: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, str) and v and v[0] in ("{", "["):
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                result[k] = v
        else:
            result[k] = v
    return result
