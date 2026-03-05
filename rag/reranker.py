"""
结果融合与重排序模块。

使用 RRF（Reciprocal Rank Fusion）算法融合三路检索结果：
  - 每路结果的排名 r 贡献得分 1 / (k + r)
  - 相同文本在多路中的得分累加（跨源去重）
  - 按 RRF 得分降序，取 top-n 条最优结果

RRF 不依赖各路原始分数的尺度，天然适合融合余弦相似度（向量）、
BM25 关键词分和 FTS 全文分等异质分数。
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

RRF_K = int(os.getenv("RRF_K", "60"))
TOP_N = int(os.getenv("TOP_N", "10"))


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    rrf_k: int = RRF_K,
    top_n: int = TOP_N,
) -> List[Dict[str, Any]]:
    """
    对多路检索结果进行 RRF 融合并去重，返回最优 top-n 条。

    Args:
        result_lists: 多路检索结果；每路已按相关性降序排列。
        rrf_k:        RRF 平滑系数（默认 60）。
        top_n:        返回条数（默认 10）。

    Returns:
        融合后的 top-n 结果列表，每条新增 `rrf_score` 和 `sources` 字段。
    """
    # text -> 融合条目
    merged: Dict[str, Dict[str, Any]] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            text = (item.get("text") or "").strip()
            if not text:
                continue

            rrf_score = 1.0 / (rrf_k + rank)
            src       = item.get("source", "unknown")

            if text in merged:
                merged[text]["rrf_score"] += rrf_score
                # 记录该条目来自哪些检索源
                if src not in merged[text]["sources"]:
                    merged[text]["sources"].append(src)
            else:
                entry = {**item, "rrf_score": rrf_score, "sources": [src]}
                merged[text] = entry

    sorted_items = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_items[:top_n]
