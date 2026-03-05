"""
BM25 稀疏关键词索引模块
使用 jieba 对中文文本分词，基于 rank-bm25 构建 Okapi BM25 索引。
"""

import os
import json
import pickle
import warnings
from typing import List, Dict, Any

import jieba
from rank_bm25 import BM25Okapi

# 屏蔽 jieba pkg_resources 废弃警告
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")


def _tokenize(text: str) -> List[str]:
    """使用 jieba 对中文文本分词，过滤空白 token。"""
    return [tok for tok in jieba.cut(text) if tok.strip()]


class BM25Index:
    """基于 jieba + rank-bm25 的中文关键词检索索引。"""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.index_path = os.path.join(store_path, "bm25.pkl")
        self.metadata_path = os.path.join(store_path, "bm25_metadata.json")
        os.makedirs(store_path, exist_ok=True)

    def build(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        """构建并持久化 BM25 索引。"""
        if len(texts) != len(metadata):
            raise ValueError("texts 和 metadata 长度必须一致")

        print(f"Building BM25 index: {len(texts)} texts …")
        tokenized = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)

        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": bm25, "texts": texts, "tokenized": tokenized}, f)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"BM25 index saved: {len(texts)} documents")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """BM25 关键词检索，返回 top-k 结果（含 score 和 text 字段）。"""
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        query_tokens = _tokenize(query)
        scores = data["bm25"].get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            entry = metadata[idx].copy()
            entry["text"] = data["texts"][idx]
            entry["score"] = float(scores[idx])
            results.append(entry)
        return results
