"""
意图分类提示词。

支持七种意图：
  chat       — 日常对话 / 常识问答
  rag        — 红旗 天工05、天工06、天工08、EH7 汽车知识问答
  tool       — 数学计算（MCP 远程调用）
  skill      — 本地技能调用
  self_intro — 自我介绍（询问助手身份/名字/能力等）
  ethics     — 涉及不当内容（政治/恐怖/黄赌毒/歧视等）
  fallback   — 超出系统能力范围的问题

当意图为 rag 时，同时输出知识库标签（可多选）：
  电池保修 — 用户询问电池保修/维修/质保相关信息
  配置     — 用户询问汽车参数/配置/性能/价格等规格信息
  答疑     — 用户询问汽车常识性知识问答，红旗 天工05、天工06、天工08、EH7 汽车知识问答
"""

# 七种意图的描述模板（{skill_hint} 将被运行时的技能列表替换）
_CATEGORIES_TEMPLATE = """\
  chat       —— 日常对话、问候、闲聊，或与红旗 天工05、天工06、天工08、EH7 汽车无关的一般性常识问题。
  rag        —— 需要查询红旗 天工05、天工06、天工08、EH7 汽车知识库的问题，例如：价格、续航、配置、
                尺寸、电池、保修、充电、安全配置等规格和产品信息。
  tool       —— 需要数学计算的请求：加法、减法、乘法、除法。
  skill      —— 本地技能调用，包括：
{skill_hint}
  self_intro —— 用户询问助手的身份、名字、能力或要求助手自我介绍，例如：你叫什么名字、你是谁、\
你能做什么、介绍一下你自己等。
  ethics     —— 涉及政治敏感、恐怖主义、黄赌毒、社会歧视或其他不当言论的内容。
  fallback   —— 问题超出系统能力范围：既不属于 红旗 天工05、天工06、天工08、EH7 汽车知识，也不是日常对话常识，\
且不属于数学计算、技能调用或不当内容的其他问题。\
"""

_RAG_TAGS_RULE = """\

当 intent 为 "rag" 时，还须在 tags 字段中给出本次查询涉及的知识库分类（当不容易区分时，同时返回多个分类）：
  "电池保修" —— 用户询问电池保修、质保期、电池维修等相关信息。
  "配置"     —— 用户询问汽车参数、配置、性能、价格、尺寸、续航等规格信息，问题较具体。
  "答疑"     —— 用户询问汽车日常使用、功能操作等常识性知识，问题较抽象。
tags 只能从以上三个值中选取，不得添加其他值。若 intent 不是 "rag"，tags 输出空数组。\
"""

_SUFFIX = (
    '只输出 JSON，格式：'
    '{"intent": "chat"|"rag"|"tool"|"skill"|"self_intro"|"ethics"|"fallback", '
    '"tags": ["电池保修"|"配置"|"答疑", ...], "reason": "简要说明"}\n'
    '不要输出其他任何内容。'
)

# 合法意图集合（供调用方校验）
VALID_INTENTS = frozenset({"chat", "rag", "tool", "skill", "self_intro", "ethics", "fallback"})

# rag 意图下合法的标签集合（供调用方校验）
VALID_RAG_TAGS = frozenset({"电池保修", "配置", "答疑"})


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
        "你是一个意图分类器，负责判断用户输入属于以下七类之一：\n\n"
        + categories
        + _RAG_TAGS_RULE
        + "\n\n"
        + _SUFFIX
    )
