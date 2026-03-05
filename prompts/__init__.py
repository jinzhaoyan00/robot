"""
提示词统一管理包。

所有提示词常量和构建函数统一从此包导出，业务代码只需：

    from prompts import (
        build_intent_system, VALID_INTENTS,
        CHAT_SYSTEM,
        RAG_SYSTEM, build_rag_user_message,
        TOOL_EXTRACT_SYSTEM, TOOL_ANSWER_SYSTEM,
        build_skill_extract_system, SKILL_ANSWER_SYSTEM,
        ETHICS_RESPONSE, FALLBACK_RESPONSE,
    )

各子模块分工：
  intent_prompt   — 意图分类（6 类：chat/rag/tool/skill/ethics/fallback）
  chat_prompt     — 日常对话系统提示词
  rag_prompt      — RAG 知识检索系统提示词 + 用户消息构建
  tool_prompt     — MCP 数学工具提取 + 回答提示词
  skill_prompt    — 本地技能参数提取 + 回答提示词
  fixed_responses — 道德伦理拒绝回复 + 兜底话术
"""

from prompts.intent_prompt import build_intent_system, VALID_INTENTS
from prompts.chat_prompt   import CHAT_SYSTEM
from prompts.rag_prompt    import RAG_SYSTEM, build_rag_user_message
from prompts.tool_prompt   import TOOL_EXTRACT_SYSTEM, TOOL_ANSWER_SYSTEM
from prompts.skill_prompt  import build_skill_extract_system, SKILL_ANSWER_SYSTEM
from prompts.fixed_responses import ETHICS_RESPONSE, FALLBACK_RESPONSE

__all__ = [
    "build_intent_system",
    "VALID_INTENTS",
    "CHAT_SYSTEM",
    "RAG_SYSTEM",
    "build_rag_user_message",
    "TOOL_EXTRACT_SYSTEM",
    "TOOL_ANSWER_SYSTEM",
    "build_skill_extract_system",
    "SKILL_ANSWER_SYSTEM",
    "ETHICS_RESPONSE",
    "FALLBACK_RESPONSE",
]
