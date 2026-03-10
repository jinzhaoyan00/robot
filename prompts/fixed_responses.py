"""
固定回复语模板。

包含：
  ETHICS_RESPONSE   — 涉及道德伦理（政治/恐怖/黄赌毒/歧视等）时的拒绝回复
  FALLBACK_RESPONSE — 问题超出系统能力范围时的兜底回复
"""

# 道德伦理拦截回复
ETHICS_RESPONSE = "对不起，我无法回答这个问题。请您换个问题。"

# 兜底话术（超出系统能力范围）
FALLBACK_RESPONSE = "抱歉，暂时没有找到答案。您可以换个问题吗？"

# 自我介绍回复
SELF_INTRO_RESPONSE = "我是小莫，一汽集团的智能机器人，有什么可以帮您呢？"
