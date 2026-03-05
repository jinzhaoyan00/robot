"""
RAG 主入口：混合三路检索 + RRF 重排序 + 阿里云大模型生成回答。

流程：
  1. 向量检索    → top-20（FAISS 余弦相似度）
  2. BM25 检索   → top-20（jieba 分词 + BM25Okapi）
  3. 图谱检索    → top-20（KuZu FTS 全文检索）
  4. RRF 融合    → 60 条 → top-10（Reciprocal Rank Fusion）
  5. LLM 生成    → 调用阿里云 qwen-plus 生成回答

用法：
  python main_rag.py "EH7 的续航里程是多少？"
  python main_rag.py "EH7 的价格" --quiet   # 不打印中间步骤
"""

import asyncio
import argparse
import sys

from rag.retriever import VectorRetriever, BM25Retriever, GraphRetriever
from rag.reranker  import reciprocal_rank_fusion
from rag.generator import generate_answer
from rag.config    import RETRIEVE_K, TOP_N


async def rag_query(query: str, verbose: bool = True) -> str:
    """
    执行完整 RAG 流程，返回 LLM 生成的回答。

    Args:
        query:   用户查询字符串。
        verbose: 是否打印每步进度和中间结果。

    Returns:
        模型生成的回答。
    """
    sep = "=" * 60

    if verbose:
        print(f"\n{sep}")
        print(f"  查询：{query}")
        print(sep)

    # ── Step 1-3：三路并行检索各 20 条 ─────────────────────────
    vector_retriever = VectorRetriever()
    bm25_retriever   = BM25Retriever()
    graph_retriever  = GraphRetriever()

    if verbose:
        print(f"\n[1/4] 向量检索（FAISS）…")
    vector_results = vector_retriever.search(query, k=RETRIEVE_K)
    if verbose:
        print(f"      ✓ {len(vector_results)} 条")

    if verbose:
        print(f"[2/4] BM25 关键词检索…")
    bm25_results = bm25_retriever.search(query, k=RETRIEVE_K)
    if verbose:
        print(f"      ✓ {len(bm25_results)} 条")

    if verbose:
        print(f"[3/4] 知识图谱检索（KuZu FTS）…")
    graph_results = graph_retriever.search(query, k=RETRIEVE_K)
    if verbose:
        print(f"      ✓ {len(graph_results)} 条")

    total = len(vector_results) + len(bm25_results) + len(graph_results)
    if verbose:
        print(f"\n      合计候选：{total} 条")

    # ── Step 4：RRF 融合 → top-10 ───────────────────────────────
    if verbose:
        print(f"\n[4/4] RRF 融合重排序 → top-{TOP_N}…")

    top_results = reciprocal_rank_fusion(
        [vector_results, bm25_results, graph_results],
        top_n=TOP_N,
    )

    if verbose:
        print(f"      ✓ 选出 {len(top_results)} 条最优片段\n")
        print("  ┌─ Top-10 参考片段 " + "─" * 42)
        for i, r in enumerate(top_results, 1):
            srcs    = "/".join(r.get("sources") or [r.get("source", "?")])
            rrf     = r.get("rrf_score", 0.0)
            preview = (r.get("text") or "").replace("\n", " ")[:70]
            print(f"  │ [{i:2d}] [{srcs}] rrf={rrf:.4f}  {preview}…")
        print("  └" + "─" * 59)

    # ── Step 5：LLM 生成回答 ────────────────────────────────────
    if verbose:
        print(f"\n  调用大模型生成回答…")

    answer = await generate_answer(query, top_results)

    if verbose:
        print(f"\n{sep}")
        print("  回答")
        print(sep)
        print(answer)
        print(sep)

    return answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="混合 RAG 系统：向量 + BM25 + 知识图谱 + 阿里云大模型"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="红旗 EH7 的续航里程和价格是多少？",
        help="用户查询（默认：红旗 EH7 的续航里程和价格是多少？）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="只输出最终回答，不打印中间步骤",
    )
    args = parser.parse_args()

    answer = asyncio.run(rag_query(args.query, verbose=not args.quiet))

    if args.quiet:
        print(answer)


if __name__ == "__main__":
    main()
