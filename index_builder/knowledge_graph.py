"""
知识图谱构建模块
使用 graphiti-core + KuZu 图数据库提取和存储知识图谱。
LLM 采用阿里云 DashScope（OpenAI 兼容接口），嵌入模型采用 ModelScope BAAI/bge-large-zh。
"""

import os
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, List, Dict, Any, Optional

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.cross_encoder.client import CrossEncoderClient

from graphiti_core.llm_client.openai_base_client import DEFAULT_MAX_TOKENS
from graphiti_core.nodes import EpisodeType


from .embedder import GTEEmbedder as _GTEEmbedder
from .dashscope_client import DashScopeClient

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

ALIBABA_API_KEY  = os.getenv("ALIBABA_API_KEY",  "")
ALIBABA_BASE_URL = os.getenv("ALIBABA_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
ALIBABA_MODEL    = os.getenv("ALIBABA_MODEL",    "qwen-plus")
KG_MAX_TOKENS    = int(os.getenv("KG_MAX_TOKENS", "4096"))


class _GraphitiGTEEmbedder(EmbedderClient):
    """
    将 GTEEmbedder 包装为 graphiti EmbedderClient 异步接口。
    同步推理通过 run_in_executor 在线程池中执行，避免阻塞事件循环。
    """

    def __init__(self):
        self._inner = _GTEEmbedder()

    async def create(self, input_data) -> List[float]:
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, (list, tuple)):
            text = " ".join(str(t) for t in input_data)
        else:
            text = str(input_data)

        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(None, self._inner.embed, [text])
        return vecs[0].tolist()

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(None, self._inner.embed, input_data_list)
        return [v.tolist() for v in vecs]


class _PassthroughReranker(CrossEncoderClient):
    """
    简单的 pass-through reranker，按原顺序返回 passages。
    用于不支持专用重排序 API 的场景。
    """

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(p, float(len(passages) - i)) for i, p in enumerate(passages)]


class KnowledgeGraph:
    """
    知识图谱构建器：
    - 使用阿里云 Qwen LLM 从文本中提取实体和关系
    - 使用 KuZu 嵌入式图数据库持久化存储
    - 使用 Graphiti 管理时序知识图谱
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        # 只创建父目录；KuZu 会自行初始化数据库目录，不能预先创建
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    def _build_graphiti(self) -> Graphiti:
        llm_config = LLMConfig(
            api_key=ALIBABA_API_KEY,
            base_url=ALIBABA_BASE_URL,
            model=ALIBABA_MODEL,
            small_model=ALIBABA_MODEL,
            # max_tokens=KG_MAX_TOKENS,
        )
        llm_client = DashScopeClient(config=llm_config)
        embedder = _GraphitiGTEEmbedder()
        kuzu_driver = KuzuDriver(db=self.db_path)

        return Graphiti(
            llm_client=llm_client,
            embedder=embedder,
            graph_driver=kuzu_driver,
            cross_encoder=_PassthroughReranker(),
            max_coroutines=1,
        )

    def _create_fts_indices(self, graphiti: "Graphiti") -> None:
        """
        在 KuZu 数据库中手动创建全文搜索索引。
        graphiti 的 build_indices_and_constraints 对 KuZu 是 no-op，需要手动创建。
        通过 graphiti.driver.db（同一 kuzu.Database 实例）执行，确保在同一数据库会话中生效。
        """
        import kuzu as _kuzu
        kuzu_db = graphiti.driver.db
        conn = _kuzu.Connection(kuzu_db)
        fts_queries = [
            "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
            "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
            "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);",
            "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
        ]
        for q in fts_queries:
            try:
                conn.execute(q)
            except Exception as exc:
                # 索引已存在时可能报错，忽略
                if "already exist" not in str(exc).lower():
                    print(f"  [FTS] {exc}")
        conn.close()
        print("  FTS indices created.")


    async def build(
        self,
        episodes: List[Dict[str, Any]],
        start_from: int = 0,
        on_episode_done: Optional[Callable[[int], None]] = None,
    ) -> None:
        """
        从 episodes 列表构建知识图谱并写入 KuZu 数据库。

        参数：
            episodes       : episode 字典列表，每项包含 document / metadata / id
            start_from     : 跳过前 N 个 episode（断点续建时传入已完成数量）
            on_episode_done: 每成功写入一个 episode 后调用，参数为累计完成总数；
                             可用于在外部保存断点进度
        """
        total = len(episodes)
        print(f"Building knowledge graph: {total} episodes ...")
        if start_from > 0:
            print(f"  断点续建：跳过前 {start_from} 个已完成的 episodes，"
                  f"从第 {start_from + 1} 个开始")

        graphiti = self._build_graphiti()
        await graphiti.build_indices_and_constraints()
        # KuZu 驱动的 build_indices_and_constraints 是 no-op，需手动创建 FTS 索引
        self._create_fts_indices(graphiti)

        completed = start_from  # 累计完成数（含本次运行前已写入的）
        for i, ep in enumerate(episodes):
            if i < start_from:
                continue

            ep_id = ep.get("id", str(i))
            print(f"  Episode {i + 1}/{total}  ({ep_id})")
            try:
                await graphiti.add_episode(
                    name=ep["metadata"]["category"],
                    episode_body=ep["document"],
                    source=EpisodeType.text,
                    source_description="\t".join(ep["metadata"]["tags"]),
                    reference_time=datetime.now(timezone.utc),
                )
                completed += 1
                if on_episode_done is not None:
                    on_episode_done(completed)
            except Exception as exc:
                print(f"  [WARN] Episode {ep_id} skipped: {exc}")

        await graphiti.close()
        print(f"Knowledge graph saved to: {self.db_path}")


    async def build_error(
        self,
        episodes: List[Dict[str, Any]],
        start_from: int = 0,
        on_episode_done: Optional[Callable[[int], None]] = None,
    ) -> None:
        """
        从 episodes 列表构建知识图谱并写入 KuZu 数据库。
        .将错误的数据重新加入到图数据中，

        参数：
            episodes       : episode 字典列表，每项包含 document / metadata / id
            start_from     : 跳过前 N 个 episode（断点续建时传入已完成数量）
            on_episode_done: 每成功写入一个 episode 后调用，参数为累计完成总数；
                             可用于在外部保存断点进度
        """
        total = len(episodes)
        print(f"Building knowledge graph: {total} episodes ...")
        if start_from > 0:
            print(f"  断点续建：跳过前 {start_from} 个已完成的 episodes，"
                  f"从第 {start_from + 1} 个开始")

        graphiti = self._build_graphiti()
        await graphiti.build_indices_and_constraints()
        # KuZu 驱动的 build_indices_and_constraints 是 no-op，需手动创建 FTS 索引
        self._create_fts_indices(graphiti)

        completed = start_from  # 累计完成数（含本次运行前已写入的）
        for i, ep in enumerate(episodes):
            if i < start_from:
                continue

            ep_id = ep.get("id", str(i))
            print(f"  Episode {i + 1}/{total}  ({ep_id})")
            try:
                error_list = ['209']
                error_list = [int(item) for item in error_list]
                if i+1 not in error_list:
                    continue
                print(ep["document"])
                await graphiti.add_episode(
                    name=ep["metadata"]["category"],
                    episode_body=ep["document"],
                    source=EpisodeType.text,
                    source_description="\t".join(ep["metadata"]["tags"]),
                    reference_time=datetime.now(timezone.utc),
                )
                completed += 1
                if on_episode_done is not None:
                    on_episode_done(completed)
            except Exception as exc:
                print(f"  [WARN] Episode {ep_id} skipped: {exc}")

        await graphiti.close()
        print(f"Knowledge graph saved to: {self.db_path}")
