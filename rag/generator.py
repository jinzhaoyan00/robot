"""
RAG 答案生成模块。

将 top-N 检索片段和用户问题拼装为 prompt，调用阿里云 DashScope（Qwen）
大模型生成最终回答。提供普通版（返回字符串）和流式版（异步生成器）两个接口。
"""

import os
from pathlib import Path
from typing import AsyncGenerator, List, Dict, Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

ALIBABA_API_KEY  = os.getenv("ALIBABA_API_KEY",  "")
ALIBABA_BASE_URL = os.getenv("ALIBABA_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
ALIBABA_MODEL    = os.getenv("ALIBABA_MODEL",    "qwen-plus")

from prompts import RAG_SYSTEM, build_rag_user_message  # noqa: E402


async def generate_answer(
    query: str,
    top_results: List[Dict[str, Any]],
    model: str = ALIBABA_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> str:
    """
    调用 DashScope 大模型，根据检索结果生成问题回答。

    Args:
        query:       用户原始问题。
        top_results: 经 RRF 融合后的 top-N 检索片段。
        model:       模型名称。
        max_tokens:  最大输出 token 数。
        temperature: 采样温度（越低越确定）。

    Returns:
        模型生成的回答字符串。
    """
    client = AsyncOpenAI(api_key=ALIBABA_API_KEY, base_url=ALIBABA_BASE_URL)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user",   "content": build_rag_user_message(query, top_results)},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return (response.choices[0].message.content or "").strip()


async def generate_answer_stream(
    query: str,
    top_results: List[Dict[str, Any]],
    model: str = ALIBABA_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> AsyncGenerator[str, None]:
    """
    流式版本：逐 token 产出模型回答，供调用方边收边打印。

    Args:
        query:       用户原始问题。
        top_results: 经 RRF 融合后的 top-N 检索片段。
        model:       模型名称。
        max_tokens:  最大输出 token 数。
        temperature: 采样温度。

    Yields:
        每次模型输出的文本片段（delta）。
    """
    client = AsyncOpenAI(api_key=ALIBABA_API_KEY, base_url=ALIBABA_BASE_URL)

    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user",   "content": build_rag_user_message(query, top_results)},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta
