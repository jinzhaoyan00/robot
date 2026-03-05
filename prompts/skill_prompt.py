"""
本地技能调用（skill）路径的提示词。

包含：
  build_skill_extract_system — 动态构建参数提取系统提示词（含技能列表）
  SKILL_ANSWER_SYSTEM        — 根据技能执行结果生成自然语言回答
"""


def build_skill_extract_system(extract_hint: str) -> str:
    """
    构建技能参数提取系统提示词。

    Args:
        extract_hint: 由 skills.get_extract_hint() 返回的参数示例片段。

    Returns:
        完整的系统提示词字符串。
    """
    return (
        "从用户请求中识别要调用的本地技能及所需参数，输出 JSON。\n\n"
        "可用技能及 JSON 格式：\n"
        f"{extract_hint}\n\n"
        "只输出 JSON，不输出任何其他内容。"
    )


SKILL_ANSWER_SYSTEM = """\
你是一个智能助手，已知查询结果，请用自然语言简洁友好地回答用户。
"""
