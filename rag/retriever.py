"""
RAG 检索模块：分别从向量数据库、BM25 索引和 KuZu 知识图谱中检索相关文档。

每路检索各返回 k 条结果（默认 20），结果统一附加 `source` 字段标识来源。
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# 项目根目录：确保 index_builder 包可导入，并定位 .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env", override=False)

def _resolve(key: str, default: str) -> str:
    val = os.getenv(key, default)
    p = Path(val)
    return str(p if p.is_absolute() else _PROJECT_ROOT / p)

FAISS_DIR  = _resolve("FAISS_DIR",  "vector_db/faiss")
BM25_DIR   = _resolve("BM25_DIR",   "vector_db/bm25")
KUZU_DIR   = _resolve("KUZU_DIR",   "vector_db/kuzu_db")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "20"))

from index_builder.vector_store import VectorStore  # noqa: E402
from index_builder.bm25_index import BM25Index      # noqa: E402


# ──────────────────────────────────────────────────────────────
# 向量检索
# ──────────────────────────────────────────────────────────────

class VectorRetriever:
    """基于 FAISS 向量索引的语义相似度检索。"""

    def __init__(self, store_path: str = FAISS_DIR):
        self._store = VectorStore(store_path)

    def search(self, query: str, k: int = RETRIEVE_K) -> List[Dict[str, Any]]:
        """
        检索与 query 语义最相近的 k 条文档。

        Returns:
            list of dict: 每条包含 text, file, line, score, source 字段。
        """
        results = self._store.search(query, k=k)
        for r in results:
            r["source"] = "vector"
        return results


# ──────────────────────────────────────────────────────────────
# BM25 关键词检索
# ──────────────────────────────────────────────────────────────

class BM25Retriever:
    """基于 jieba + BM25Okapi 的关键词匹配检索。"""

    def __init__(self, store_path: str = BM25_DIR):
        self._index = BM25Index(store_path)

    def search(self, query: str, k: int = RETRIEVE_K) -> List[Dict[str, Any]]:
        """
        检索与 query 关键词最相关的 k 条文档。

        Returns:
            list of dict: 每条包含 text, file, line, score, source 字段。
        """
        results = self._index.search(query, k=k)
        for r in results:
            r["source"] = "bm25"
        return results


# ──────────────────────────────────────────────────────────────
# 知识图谱检索
# ──────────────────────────────────────────────────────────────

class GraphRetriever:
    """
    基于 KuZu 全文搜索（FTS）的知识图谱检索。

    同时查询 Episodic（片段记忆）、Entity（实体摘要）和边（事实）三张表，
    合并后按得分排序，返回 top-k 条。
    """

    def __init__(self, db_path: str = KUZU_DIR):
        self.db_path = db_path

    def search(self, query: str, k: int = RETRIEVE_K) -> List[Dict[str, Any]]:
        """
        在 KuZu FTS 索引中检索相关片段、实体和事实。

        Returns:
            list of dict: 每条包含 text, score, source 字段；
                          图谱不可用时返回空列表并打印警告。
        """
        if not os.path.exists(self.db_path):
            print(f"[WARN] KuZu 数据库不存在: {self.db_path}，跳过图谱检索。")
            return []

        try:
            import kuzu
            db   = kuzu.Database(self.db_path, read_only=True)
            conn = kuzu.Connection(db)
            results: List[Dict[str, Any]] = []

            # 对 FTS 查询字符串中的单引号进行转义
            safe_q = query.replace("'", "''")

            # 1. Episodic 片段记忆（最完整的原始文本）
            results.extend(
                self._query_fts(
                    conn,
                    table="Episodic",
                    index="episode_content",
                    query=safe_q,
                    return_expr="node.content AS text",
                    source_tag="graph_episode",
                    limit=k // 2,
                )
            )

            # 2. Entity 实体摘要（name + summary）
            results.extend(
                self._query_fts(
                    conn,
                    table="Entity",
                    index="node_name_and_summary",
                    query=safe_q,
                    return_expr="node.name + CASE WHEN node.summary IS NOT NULL "
                                "AND node.summary <> '' THEN '：' + node.summary ELSE '' END AS text",
                    source_tag="graph_entity",
                    limit=k // 4,
                )
            )

            # 3. RelatesToNode_ 关系事实（边上的 fact 字段）
            results.extend(
                self._query_fts(
                    conn,
                    table="RelatesToNode_",
                    index="edge_name_and_fact",
                    query=safe_q,
                    return_expr="CASE WHEN e.fact IS NOT NULL AND e.fact <> '' "
                                "THEN e.fact ELSE e.name END AS text",
                    source_tag="graph_fact",
                    limit=k // 4,
                    node_alias="e",
                )
            )

            conn.close()

            # 按得分降序，去掉空文本，截取 top-k
            results = [r for r in results if r.get("text", "").strip()]
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]

        except Exception as exc:
            print(f"[WARN] 知识图谱检索失败: {exc}")
            return []

    # ── 内部辅助 ─────────────────────────────────────────────

    @staticmethod
    def _query_fts(
        conn,
        table: str,
        index: str,
        query: str,
        return_expr: str,
        source_tag: str,
        limit: int,
        node_alias: str = "node",
    ) -> List[Dict[str, Any]]:
        """
        执行单条 KuZu FTS 查询，返回 {text, score, source} 列表。
        失败时静默返回空列表。
        """
        cypher = (
            f"CALL QUERY_FTS_INDEX('{table}', '{index}', '{query}') "
            f"YIELD {node_alias}, score "
            f"RETURN {return_expr}, score "
            f"ORDER BY score DESC LIMIT {max(1, limit)}"
        )
        try:
            result = conn.execute(cypher)
            rows: List[Dict[str, Any]] = []
            while result.has_next():
                row = result.get_next()
                text  = str(row[0]).strip() if row[0] is not None else ""
                score = float(row[1]) if row[1] is not None else 0.0
                rows.append({"text": text, "score": score, "source": source_tag})
            return rows
        except Exception as exc:
            print(f"  [WARN] FTS({table}): {exc}")
            return []
