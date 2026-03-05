"""
意图分类提示词。

支持六种意图：
  chat     — 日常对话 / 常识问答
  rag      — EH7 知识库查询
  tool     — 数学计算（MCP 远程调用）
  skill    — 本地技能调用
  ethics   — 涉及不当内容（政治/恐怖/黄赌毒/歧视等）
  fallback — 超出系统能力范围的问题
"""

# 六种意图的描述模板（{skill_hint} 将被运行时的技能列表替换）
_CATEGORIES_TEMPLATE = """\
  chat     —— 日常对话、问候、闲聊，或与红旗 EH7 汽车无关的一般性常识问题。
  rag      —— 需要查询红旗 EH7 汽车知识库的问题，例如：价格、续航、配置、
              尺寸、电池、保修、充电、安全配置等规格和产品信息。
  tool     —— 需要数学计算的请求：加法、减法、乘法、除法。
  skill    —— 本地技能调用，包括：
{skill_hint}
  ethics   —— 涉及政治敏感、恐怖主义、黄赌毒、社会歧视或其他不当言论的内容。
  fallback —— 问题超出系统能力范围：既不属于 EH7 汽车知识，也不是日常对话常识，\
且不属于数学计算、技能调用或不当内容的其他问题。\
"""

_SUFFIX = (
    '只输出 JSON，格式：'
    '{"intent": "chat"|"rag"|"tool"|"skill"|"ethics"|"fallback", "reason": "简要说明"}\n'
    '不要输出其他任何内容。'
)

# 合法意图集合（供调用方校验）
VALID_INTENTS = frozenset({"chat", "rag", "tool", "skill", "ethics", "fallback"})


def build_intent_system(skill_hint: str) -> str:
    """
    构建意图分类系统提示词。

    Args:
        skill_hint: 由 skills.get_intent_hint() 返回的技能列表描述。

    Returns:
        完整的系统提示词字符串。
    """
    categories = _CATEGORIES_TEMPLATE.format(skill_hint=skill_hint)
    return (
        "你是一个意图分类器，负责判断用户输入属于以下六类之一：\n\n"
        + categories
        + "\n\n"
        + _SUFFIX
    )
