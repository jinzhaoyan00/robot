"""
阿里云 DashScope LLM 客户端
继承 graphiti BaseOpenAIClient，完全使用 chat.completions API，
兼容 DashScope（不支持 OpenAI Responses API）。
"""

import json
import logging
import typing
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graphiti_core.llm_client.openai_base_client import (
    BaseOpenAIClient,
    DEFAULT_REASONING,
    DEFAULT_VERBOSITY,
)

from pathlib import Path
import os
from dotenv import load_dotenv

# 最大生成token数，与输入token数加在一起，构成模型支持的最大上下文长度
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
DEFAULT_MAX_TOKENS = os.getenv("DEFAULT_MAX_TOKENS", 6384)

from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message

import openai as _openai

logger = logging.getLogger(__name__)


class DashScopeClient(BaseOpenAIClient):
    """
    针对阿里云 DashScope OpenAI 兼容接口的 LLM 客户端。
    统一使用 chat.completions + JSON mode 代替 OpenAI Responses API。
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str = DEFAULT_REASONING,
        verbosity: str = DEFAULT_VERBOSITY,
    ):
        super().__init__(config, cache, max_tokens, reasoning, verbosity)
        if config is None:
            config = LLMConfig()
        self.client = client or AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )


    # ── 必须实现的抽象方法（内部不会被调用，用 _generate_response 绕过）──

    async def _create_structured_completion(self, *args, **kwargs) -> Any:  # type: ignore[override]
        raise NotImplementedError("Use _generate_response instead")

    async def _create_completion(  # type: ignore[override]
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        **kwargs,
    ) -> Any:
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    # ── 核心重写：统一使用 chat.completions + JSON 解析 ──────────────────

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 0,   # 0 → 回退到 self.max_tokens（由 config.max_tokens 决定）
        model_size: ModelSize = ModelSize.medium,
    ) -> tuple[dict[str, Any], int, int]:
        openai_messages = self._convert_messages_to_openai_format(messages)
        model = self._get_model_for_size(model_size)

        # 若需要结构化输出，在消息末尾追加 JSON schema 说明
        if response_model is not None:
            schema = response_model.model_json_schema()
            schema_note = (
                "\n\nRespond ONLY with a valid JSON object matching this schema "
                f"(no extra text):\n{json.dumps(schema, ensure_ascii=False)}"
            )
            msgs = list(openai_messages)
            if msgs and msgs[-1].get("role") == "user":
                last = dict(msgs[-1])
                last["content"] = str(last.get("content", "")) + schema_note
                msgs[-1] = last
            else:
                msgs.append({"role": "user", "content": schema_note})
            openai_messages = msgs

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content or "{}"
            input_tokens = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
            output_tokens = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0

            return json.loads(raw), input_tokens, output_tokens

        except _openai.RateLimitError:
            raise RateLimitError
        except Exception as exc:
            logger.error(f"Error in generating LLM response: {exc}")
            raise
