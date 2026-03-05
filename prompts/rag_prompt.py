"""
RAG（知识检索）路径的提示词。

包含：
  RAG_SYSTEM             — 回答生成时的系统提示词
  build_rag_user_message — 将参考资料 + 用户问题拼成 user 消息的工厂函数
"""

from typing import List, Dict, Any

RAG_SYSTEM = """\
你是一个专业的汽车知识助手，专注于红旗 EH7 电动汽车的相关信息。
请严格依据下方【参考资料】回答用户的问题，做到准确、简洁、有条理。
- 如果参考资料中包含数字、规格或价格，请原文引用，不要自行推断。
- 如果参考资料中没有足够信息回答问题，请如实告知用户，不要编造内容。
"""


def _build_context(top_results: List[Dict[str, Any]]) -> str:
    """将 top-N 检索结果格式化为带编号的参考资料段落。"""
    lines: List[str] = []
    for i, item in enumerate(top_results, start=1):
        text    = (item.get("text") or "").strip()
        sources = item.get("sources") or [item.get("source", "?")]
        src_str = "/".join(sources)
        lines.append(f"[{i}] [{src_str}] {text}")
    return "\n".join(lines)


def build_rag_user_message(query: str, top_results: List[Dict[str, Any]]) -> str:
    """
    将检索结果和用户问题组合成发给 LLM 的 user 消息。

    Args:
        query:       用户原始问题。
        top_results: 经 RRF 融合后的 top-N 检索片段列表。

    Returns:
        格式化后的 user 消息字符串。
    """
    context = _build_context(top_results)
    return (
        f"【参考资料】\n{context}\n\n"
        f"【用户问题】\n{query}"
    )
