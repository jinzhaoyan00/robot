"""
主入口：读取 data/txt/ 目录文本，构建 ChromaDB 向量索引、BM25 关键词索引和知识图谱。
所有索引与数据库持久化至 vector_db/ 目录。

断点续建：进度保存在 tmp/build_state.json，中断后重新运行即可从断点继续。
使用 --reset 参数可清除进度、从头重新构建。
"""

import os
import re
import asyncio
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

from index_builder.vector_store import VectorStore
from index_builder.bm25_index import BM25Index
from index_builder.knowledge_graph import KnowledgeGraph

# ── 路径配置 ──────────────────────────────────────
DATA_DIR = "data/txt"
VECTOR_DB_DIR = "vector_db"
CHROMA_DIR = os.path.join(VECTOR_DB_DIR, "chromadb")
BM25_DIR = os.path.join(VECTOR_DB_DIR, "bm25")
KUZU_DIR = os.path.join(VECTOR_DB_DIR, "kuzu_db")

# ── 断点续建状态 ───────────────────────────────────
TMP_DIR = "tmp"
_STATE_FILE = os.path.join(TMP_DIR, "build_state.json")

_DEFAULT_STATE: Dict[str, Any] = {
    "vector_done": False,
    "bm25_done": False,
    "kg_done": False,
    "kg_completed_count": 0,   # 已成功写入 KG 的 episode 数量
}


def _load_state() -> Dict[str, Any]:
    """读取断点状态；文件不存在时返回默认（全部未完成）。"""
    if os.path.exists(_STATE_FILE):
        with open(_STATE_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        state = dict(_DEFAULT_STATE)
        state.update(saved)
        return state
    return dict(_DEFAULT_STATE)


def _save_state(state: Dict[str, Any]) -> None:
    """将当前状态持久化到 tmp/build_state.json。"""
    os.makedirs(TMP_DIR, exist_ok=True)
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _reset_state() -> None:
    """删除断点文件，下次运行将从头开始。"""
    if os.path.exists(_STATE_FILE):
        os.remove(_STATE_FILE)
        print(f"已重置进度文件：{_STATE_FILE}")

# 每个知识图谱 episode 包含的文本块数量
# 每个 episode 对应一次 LLM 提取调用；块数越多调用次数越少，但每次处理复杂度更高

# ── 数据读取 ──────────────────────────────────────

def read_data(data_dir: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    读取 data/txt/ 下所有 .json 文件，每个 JSON 条目作为一条独立的检索单元。
    顶层 id 字段（如存在）会合并进 metadata，方便下游使用。
    """
    texts: List[str] = []
    metadata: List[Dict[str, Any]] = []

    for json_file in sorted(Path(data_dir).glob("*.json")):
        items = json.loads(json_file.read_text(encoding="utf-8"))
        for item in items:
            texts.append(item["document"])
            meta = dict(item["metadata"])
            if "id" in item:
                meta["id"] = item["id"]
            metadata.append(meta)

    pattern = r'问题：(.*?)\n答案：([\s\S]*?)(?=\n问题：|$)'    
    for txt_file in sorted(Path(data_dir).glob("*答疑_gen.txt")):
        tmp = txt_file.name.split('/')[-1].split('_')[0:2]
        id = 1
        with open(txt_file, 'r', encoding="utf-8") as f:
            data = f.read()
            for q, a in re.findall(pattern, data):
                texts.append(f"问题: {q}\n答案: {a}" )
                meta = {
                    "category": tmp[0],
                    "tags": [
                        tmp[0],
                        tmp[1]
                    ],
                    'id': f'{tmp[0]}_{tmp[1]}_gen_{id}'
                }
                id = id + 1
                metadata.append(meta)

    for txt_file in sorted(Path(data_dir).glob("*配置_gen.txt")):
        tmp = txt_file.name.split('/')[-1].split('_')[0:2]
        id = 1
        with open(txt_file, 'r', encoding="utf-8") as f:
            data = f.readlines()
            for line in data:
                texts.append(line)
                meta = {
                    "category": tmp[0],
                    "tags": [
                        tmp[0],
                        tmp[1]
                    ],
                    'id': f'{tmp[0]}_{tmp[1]}_gen_{id}'
                }
                id = id + 1
                metadata.append(meta)

    print(f"Loaded {len(texts)} chunks from {data_dir}/")
    return texts, metadata


def build_episodes(data_dir: str) -> List[Dict[str, str]]:
    """将文本块分组为知识图谱 episode 列表。"""
    episodes: List[Dict[str, str]] = []
    for json_file in sorted(Path(data_dir).glob("*.json")):
        items = json.loads(json_file.read_text(encoding="utf-8"))
        episodes.extend(items)
    
    return episodes


# ── 构建流程 ──────────────────────────────────────

async def build_all(build_kg: bool = False) -> None:
    state = _load_state()

    # 显示当前进度
    if any([state["vector_done"], state["bm25_done"], state["kg_completed_count"] > 0]):
        print("检测到未完成的构建进度，从断点继续：")
        print(f"  向量索引: {'已完成' if state['vector_done'] else '未完成'}")
        print(f"  BM25 索引: {'已完成' if state['bm25_done'] else '未完成'}")
        print(f"  知识图谱: {state['kg_completed_count']} episodes 已完成"
              + ("（全部完成）" if state["kg_done"] else ""))

    texts, metadata = read_data(DATA_DIR)
    if not texts:
        print("data/ 目录下无有效文本，退出。")
        return

    # 1. ChromaDB 向量索引
    print("\n" + "=" * 50)
    if state["vector_done"]:
        print("Step 1/3  ChromaDB 向量索引  [已完成，跳过]")
        vs = VectorStore(CHROMA_DIR)
    else:
        print("Step 1/3  ChromaDB 向量索引")
        print("=" * 50)
        vs = VectorStore(CHROMA_DIR)
        vs.build(texts, metadata)
        state["vector_done"] = True
        _save_state(state)

    # 2. BM25 关键词索引
    print("\n" + "=" * 50)
    if state["bm25_done"]:
        print("Step 2/3  BM25 关键词索引  [已完成，跳过]")
        bm25 = BM25Index(BM25_DIR)
    else:
        print("Step 2/3  BM25 关键词索引")
        print("=" * 50)
        bm25 = BM25Index(BM25_DIR)
        bm25.build(texts, metadata)
        state["bm25_done"] = True
        _save_state(state)

    # 3. 知识图谱（需要 LLM API 调用，可选关闭）
    if build_kg:
        print("\n" + "=" * 50)
        if state["kg_done"]:
            print("Step 3/3  知识图谱（Graphiti + KuZu）  [已完成，跳过]")
        else:
            print("Step 3/3  知识图谱（Graphiti + KuZu）")
            print("=" * 50)
            episodes = build_episodes(DATA_DIR)

            def _on_episode_done(completed_count: int) -> None:
                state["kg_completed_count"] = completed_count
                _save_state(state)

            kg = KnowledgeGraph(KUZU_DIR)
            await kg.build(
                episodes,
                start_from=state["kg_completed_count"],
                on_episode_done=_on_episode_done,
            )
            state["kg_done"] = True
            _save_state(state)

    print("\n" + "=" * 50)
    print("所有索引构建完成！")
    print(f"  ChromaDB → {CHROMA_DIR}")
    print(f"  BM25   → {BM25_DIR}")
    if build_kg:
        print(f"  KuZu   → {KUZU_DIR}")
    print("=" * 50)


# ── 入口 ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建向量索引与知识图谱")
    parser.add_argument(
        "--build-kg",
        action="store_true",
        help="构建知识图谱步骤",
        default=False,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清除断点进度，从头重新构建所有索引",
    )
    args = parser.parse_args()

    if args.reset:
        _reset_state()

    asyncio.run(build_all(build_kg=args.build_kg))
