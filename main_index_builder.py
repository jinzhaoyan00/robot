"""
主入口：读取 data/ 目录文本，构建 FAISS 向量索引、BM25 关键词索引和知识图谱。
所有索引与数据库持久化至 vector_db/ 目录。
"""

import os
import re
import asyncio
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

from index_builder.vector_store import VectorStore
from index_builder.bm25_index import BM25Index
from index_builder.knowledge_graph import KnowledgeGraph

# ── 路径配置 ──────────────────────────────────────
DATA_DIR = "data"
VECTOR_DB_DIR = "vector_db"
FAISS_DIR = os.path.join(VECTOR_DB_DIR, "faiss")
BM25_DIR = os.path.join(VECTOR_DB_DIR, "bm25")
KUZU_DIR = os.path.join(VECTOR_DB_DIR, "kuzu_db")

# 每个知识图谱 episode 包含的文本块数量
# 每个 episode 对应一次 LLM 提取调用；块数越多调用次数越少，但每次处理复杂度更高
KG_EPISODE_LINES = 1

# 分块分隔符（匹配独占一行的 "FAW-Robotics"，允许前后有空白字符）
_CHUNK_SEPARATOR = re.compile(r"(?m)^\s*FAW-Robotics\s*$")


# ── 数据读取 ──────────────────────────────────────

def read_data(data_dir: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    读取 data/ 下所有 .txt 文件，以独占一行的 "FAW-Robotics" 为分隔符
    切分成文本块，每个非空块作为一条独立的检索单元。
    """
    texts: List[str] = []
    metadata: List[Dict[str, Any]] = []

    for txt_file in sorted(Path(data_dir).glob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8")
        chunks = _CHUNK_SEPARATOR.split(raw)
        chunk_idx = 0
        for raw_chunk in chunks:
            text = raw_chunk.strip()
            if text:
                chunk_idx += 1
                texts.append(text)
                metadata.append({
                    "text": text,
                    "file": txt_file.name,
                    "chunk": chunk_idx,
                })

    print(f"Loaded {len(texts)} chunks from {data_dir}/")
    return texts, metadata


def build_episodes(
    texts: List[str],
    metadata: List[Dict[str, Any]],
    batch_size: int = KG_EPISODE_LINES,
) -> List[Dict[str, str]]:
    """将文本块分组为知识图谱 episode 列表。"""
    episodes: List[Dict[str, str]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        batch_meta  = metadata[start: start + batch_size]
        src_file    = batch_meta[0]["file"]
        chunk_start = batch_meta[0]["chunk"]
        chunk_end   = batch_meta[-1]["chunk"]
        episodes.append({
            "name":   f"{src_file}_C{chunk_start}-{chunk_end}",
            "body":   "\n\n---\n\n".join(batch_texts),
            "source": f"文件: {src_file}，块 {chunk_start}–{chunk_end}",
        })
    return episodes


# ── 构建流程 ──────────────────────────────────────

async def build_all(build_kg: bool = True) -> None:
    texts, metadata = read_data(DATA_DIR)
    if not texts:
        print("data/ 目录下无有效文本，退出。")
        return

    # 1. FAISS 向量索引
    print("\n" + "=" * 50)
    print("Step 1/3  FAISS 向量索引")
    print("=" * 50)
    vs = VectorStore(FAISS_DIR)
    vs.build(texts, metadata)

    # 2. BM25 关键词索引
    print("\n" + "=" * 50)
    print("Step 2/3  BM25 关键词索引")
    print("=" * 50)
    bm25 = BM25Index(BM25_DIR)
    bm25.build(texts, metadata)

    # 3. 知识图谱（需要 LLM API 调用，可选关闭）
    if build_kg:
        print("\n" + "=" * 50)
        print("Step 3/3  知识图谱（Graphiti + KuZu）")
        print("=" * 50)
        episodes = build_episodes(texts, metadata)
        kg = KnowledgeGraph(KUZU_DIR)
        await kg.build(episodes)

    print("\n" + "=" * 50)
    print("所有索引构建完成！")
    print(f"  FAISS  → {FAISS_DIR}")
    print(f"  BM25   → {BM25_DIR}")
    if build_kg:
        print(f"  KuZu   → {KUZU_DIR}")
    print("=" * 50)

    # ── 简单检索测试 ──────────────────────────────
    _run_search_test(vs, bm25)


def _run_search_test(vs: VectorStore, bm25: BM25Index) -> None:
    test_queries = ["EH7价格", "轴距", "电池保修"]
    print("\n" + "=" * 50)
    print("检索功能测试")
    print("=" * 50)
    for q in test_queries:
        print(f"\n查询: 「{q}」")
        print("  [FAISS] Top-3:")
        for r in vs.search(q, k=3):
            preview = r['text'].replace('\n', ' ')[:80]
            print(f"    score={r['score']:.4f}  {preview}")
        print("  [BM25]  Top-3:")
        for r in bm25.search(q, k=3):
            preview = r['text'].replace('\n', ' ')[:80]
            print(f"    score={r['score']:.4f}  {preview}")


# ── 入口 ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建向量索引与知识图谱")
    parser.add_argument(
        "--skip-kg",
        action="store_true",
        help="跳过知识图谱构建步骤（不调用 LLM API）",
    )
    args = parser.parse_args()

    asyncio.run(build_all(build_kg=not args.skip_kg))
