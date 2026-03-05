"""
对话主程序：循环接收用户输入，识别意图并路由至对应处理器。

意图分类（由 LLM 判断）：
  chat     → 阿里云大模型直接生成回答（保持多轮对话历史）
  rag      → RAG 混合检索 + 大模型生成（红旗 EH7 知识库）
  tool     → MCP 远程调用（数学计算）
  skill    → 本地技能调用（日期时间查询、单位换算等）
  ethics   → 道德伦理拦截（政治/恐怖/黄赌毒/歧视等不当内容）
  fallback → 兜底话术（超出系统能力范围的问题）

运行：
  python main_dialog.py
  python main_dialog.py --mcp-url http://localhost:8000   # 指定 MCP 服务器地址
"""

import re
import asyncio
import json
import argparse
import sys
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

ALIBABA_API_KEY  = os.getenv("ALIBABA_API_KEY",  "")
ALIBABA_BASE_URL = os.getenv("ALIBABA_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
ALIBABA_MODEL    = os.getenv("ALIBABA_MODEL",    "qwen-plus")
RETRIEVE_K       = int(os.getenv("RETRIEVE_K", "20"))
TOP_N            = int(os.getenv("TOP_N",      "10"))

from rag.retriever import VectorRetriever, BM25Retriever, GraphRetriever  # noqa: E402
from rag.reranker  import reciprocal_rank_fusion                           # noqa: E402
from rag.generator import generate_answer_stream                           # noqa: E402
import skills as _skills                                                   # noqa: E402
from prompts import (                                                      # noqa: E402
    build_intent_system, VALID_INTENTS,
    CHAT_SYSTEM,
    TOOL_EXTRACT_SYSTEM, TOOL_ANSWER_SYSTEM,
    build_skill_extract_system, SKILL_ANSWER_SYSTEM,
    ETHICS_RESPONSE, FALLBACK_RESPONSE,
)


# ══════════════════════════════════════════════════════════════════
# MCP 工具执行器
# ══════════════════════════════════════════════════════════════════

class MCPExecutor:
    """
    MCP 工具执行器。

    优先尝试通过 MCP SSE 协议连接远程服务器（需要 mcp 包已安装）；
    如果服务器不可达或 mcp 包未安装，则退回到本地直接执行（与
    mcp/server.py 的实现逻辑完全一致）。
    """

    def __init__(self, server_url: str = "http://localhost:8000"):
        self._server_url = server_url.rstrip("/")
        self._sse_url    = f"{self._server_url}/sse"

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用 MCP 工具，返回执行结果。"""
        result = await self._try_remote(tool_name, arguments)
        if result is not None:
            return result
        return self._local_exec(tool_name, arguments)

    async def _try_remote(self, tool_name: str, arguments: dict) -> Any | None:
        """尝试通过 SSE 连接远程 MCP 服务器。失败时返回 None。"""
        project_root  = os.path.dirname(os.path.abspath(__file__))
        original_path = sys.path.copy()
        sys.path = [p for p in sys.path if p != project_root and p != ""]
        try:
            from mcp import ClientSession              # type: ignore
            from mcp.client.sse import sse_client      # type: ignore

            async with sse_client(self._sse_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    res = await session.call_tool(tool_name, arguments)
                    if res.content:
                        return res.content[0].text
            return None
        except Exception:
            return None
        finally:
            sys.path = original_path

    @staticmethod
    def _local_exec(tool_name: str, arguments: dict) -> Any:
        """本地直接执行（与 mcp/server.py 逻辑一致）。"""
        a = float(arguments.get("a", 0))
        b = float(arguments.get("b", 0))
        if tool_name == "add":
            return a + b
        if tool_name == "subtract":
            return a - b
        if tool_name == "multiply":
            return a * b
        if tool_name == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        raise ValueError(f"未知工具: {tool_name}")


# ══════════════════════════════════════════════════════════════════
# 对话系统
# ══════════════════════════════════════════════════════════════════

class DialogSystem:
    """
    多轮对话系统，集成意图识别、RAG 检索和 MCP 远程调用。
    """

    def __init__(self, mcp_server_url: str = "http://localhost:8000"):
        self._client   = AsyncOpenAI(api_key=ALIBABA_API_KEY, base_url=ALIBABA_BASE_URL)
        self._executor = MCPExecutor(server_url=mcp_server_url)
        self._chat_history: list[dict] = []

    # ── 1. 意图分类 ──────────────────────────────────────────────

    async def classify_intent(self, query: str) -> tuple[str, str]:
        """
        调用 LLM 判断用户意图。

        Returns:
            (intent, reason)  intent ∈ VALID_INTENTS
        """
        system_prompt = build_intent_system(_skills.get_intent_hint())
        response = await self._client.chat.completions.create(
            model=ALIBABA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            max_tokens=128,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "{}").strip()
        try:
            parsed = json.loads(raw)
            intent = parsed.get("intent", "chat").lower()
            reason = parsed.get("reason", "")
            if intent not in VALID_INTENTS:
                intent = "chat"
        except json.JSONDecodeError:
            intent, reason = "chat", "JSON 解析失败，默认 chat"
        return intent, reason

    # ── 2. 简单对话（多轮，流式）─────────────────────────────────

    async def handle_chat(self, query: str) -> str:
        """流式调用大模型，边生成边打印，维持多轮对话历史。"""
        self._chat_history.append({"role": "user", "content": query})
        messages = [
            {"role": "system", "content": CHAT_SYSTEM},
            *self._chat_history,
        ]

        stream = await self._client.chat.completions.create(
            model=ALIBABA_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=True,
        )

        chunks: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                print(delta, end="", flush=True)
                chunks.append(delta)

        answer = "".join(chunks)
        self._chat_history.append({"role": "assistant", "content": answer})
        if len(self._chat_history) > 20:
            self._chat_history = self._chat_history[-20:]
        return answer

    # ── 3. 知识检索（RAG，流式）──────────────────────────────────

    async def handle_rag(self, query: str) -> str:
        """三路检索 + RRF 融合后，流式生成 EH7 相关问题的回答。"""
        vector_results = VectorRetriever().search(query, k=RETRIEVE_K)
        bm25_results   = BM25Retriever().search(query, k=RETRIEVE_K)
        graph_results  = GraphRetriever().search(query, k=RETRIEVE_K)
        top_results    = reciprocal_rank_fusion(
            [vector_results, bm25_results, graph_results],
            top_n=TOP_N,
        )

        chunks: list[str] = []
        async for delta in generate_answer_stream(query, top_results):
            print(delta, end="", flush=True)
            chunks.append(delta)

        return "".join(chunks)

    # ── 4. 远程调用（MCP，流式）──────────────────────────────────

    async def handle_tool(self, query: str) -> str:
        """
        第一轮：让 LLM 从自然语言中提取工具名和参数（JSON 输出）。
        第二轮：通过 MCP 执行工具，再流式生成自然语言回答。
        """
        extract_resp = await self._client.chat.completions.create(
            model=ALIBABA_MODEL,
            messages=[
                {"role": "system", "content": TOOL_EXTRACT_SYSTEM},
                {"role": "user",   "content": query},
            ],
            max_tokens=64,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = (extract_resp.choices[0].message.content or "{}").strip()

        try:
            parsed   = json.loads(raw)
            fn_name  = parsed.get("tool", "")
            fn_args  = {"a": float(parsed.get("a", 0)), "b": float(parsed.get("b", 0))}
        except (json.JSONDecodeError, ValueError):
            fn_name, fn_args = "", {}

        if fn_name:
            try:
                result_str = str(await self._executor.call(fn_name, fn_args))
            except Exception as exc:
                result_str = f"错误: {exc}"
        else:
            result_str = "无法识别计算操作"

        stream = await self._client.chat.completions.create(
            model=ALIBABA_MODEL,
            messages=[
                {"role": "system",    "content": TOOL_ANSWER_SYSTEM},
                {"role": "user",      "content": query},
                {"role": "assistant", "content": f"计算结果：{result_str}"},
                {"role": "user",      "content": "请用自然语言把计算结果告诉我。"},
            ],
            max_tokens=256,
            temperature=0.3,
            stream=True,
        )

        chunks: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                print(delta, end="", flush=True)
                chunks.append(delta)

        return "".join(chunks)

    # ── 5. 本地技能调用（Skills，流式）───────────────────────────

    async def handle_skill(self, query: str) -> str:
        """
        第一轮：让 LLM 从自然语言中提取技能名和参数（JSON 输出）。
        第二轮：本地执行技能，再流式生成自然语言回答。
        """
        extract_resp = await self._client.chat.completions.create(
            model=ALIBABA_MODEL,
            messages=[
                {"role": "system", "content": build_skill_extract_system(_skills.get_extract_hint())},
                {"role": "user",   "content": query},
            ],
            max_tokens=128,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = (extract_resp.choices[0].message.content or "{}").strip()

        try:
            parsed     = json.loads(raw)
            skill_name = parsed.pop("skill", "")
            params     = parsed
        except (json.JSONDecodeError, ValueError):
            skill_name, params = "", {}

        result_str = _skills.execute_skill(skill_name, params) if skill_name \
                     else "无法识别要调用的技能"

        stream = await self._client.chat.completions.create(
            model=ALIBABA_MODEL,
            messages=[
                {"role": "system",    "content": SKILL_ANSWER_SYSTEM},
                {"role": "user",      "content": query},
                {"role": "assistant", "content": f"查询结果：{result_str}"},
                {"role": "user",      "content": "请用自然语言告诉我结果。"},
            ],
            max_tokens=256,
            temperature=0.3,
            stream=True,
        )

        chunks: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                print(delta, end="", flush=True)
                chunks.append(delta)

        return "".join(chunks)

    # ── 6. 道德伦理拦截 ───────────────────────────────────────────

    async def handle_ethics(self, query: str) -> str:  # noqa: ARG002
        """涉及不当内容时直接返回固定拒绝回复。"""
        print(ETHICS_RESPONSE, end="", flush=True)
        return ETHICS_RESPONSE

    # ── 7. 兜底话术 ───────────────────────────────────────────────

    async def handle_fallback(self, query: str) -> str:  # noqa: ARG002
        """超出系统能力范围时直接返回固定兜底回复。"""
        print(FALLBACK_RESPONSE, end="", flush=True)
        return FALLBACK_RESPONSE

    # ── 8. 主处理入口 ─────────────────────────────────────────────

    async def process(self, query: str, show_intent: bool = True) -> str:
        """
        分类意图，打印意图标签和"助手: "前缀，然后路由到对应的流式处理器。
        处理器负责边生成边打印；本方法返回完整回答字符串（供程序化调用）。

        query 可能携带 /no_think 等模型控制前缀，分类时自动剥离以保证准确性；
        各 handler 接收原始 query（含前缀），用于抑制模型思考输出。
        """
        # 剥离 /no_think 等控制前缀，仅用于意图分类
        classify_query = re.sub(r"^(/no_think|/think)\s*\n?", "", query).strip()
        intent, reason = await self.classify_intent(classify_query)

        if show_intent:
            label = {
                "chat":     "💬 对话",
                "rag":      "📚 知识检索",
                "tool":     "🔧 远程调用",
                "skill":    "⚙️ 工具调用",
                "ethics":   "🚫 道德伦理",
                "fallback": "❓ 兜底话术",
            }.get(intent, intent)
            print(f"  [意图: {label}] {reason}")

        print("助手: ", end="", flush=True)

        if intent == "rag":
            return await self.handle_rag(query)
        if intent == "tool":
            return await self.handle_tool(query)
        if intent == "skill":
            return await self.handle_skill(query)
        if intent == "ethics":
            return await self.handle_ethics(query)
        if intent == "fallback":
            return await self.handle_fallback(query)
        return await self.handle_chat(query)

    # ── 9. 对话循环 ───────────────────────────────────────────────

    async def run(self, show_intent: bool = True) -> None:
        """启动交互式对话循环，输入 exit/quit 退出。"""
        print("=" * 60)
        print("  红旗 EH7 智能对话助手")
        print("  支持：日常对话 | EH7 知识查询 | 数学计算 | 工具调用")
        print("  输入 exit 或 quit 退出")
        print("=" * 60)

        while True:
            try:
                query = input("\n你: ").strip()
                query = "/no_think\n" + query
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit", "退出", "bye"):
                print("再见！")
                break

            print()
            try:
                await self.process(query, show_intent=show_intent)
                print()
            except Exception as exc:
                print(f"\n[错误] {exc}")


# ══════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="红旗 EH7 智能对话助手（意图识别 + RAG + MCP 远程调用）"
    )
    parser.add_argument(
        "--mcp-url",
        default="http://localhost:8888",
        help="MCP 服务器地址（默认: http://localhost:8888）",
    )
    parser.add_argument(
        "--no-intent-label",
        action="store_true",
        help="不打印意图分类标签",
    )
    args = parser.parse_args()

    system = DialogSystem(mcp_server_url=args.mcp_url)
    asyncio.run(system.run(show_intent=not args.no_intent_label))


if __name__ == "__main__":
    main()
