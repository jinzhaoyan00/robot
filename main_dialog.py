"""
对话主程序：循环接收用户输入，识别意图并路由至对应处理器。

意图分类（由 LLM 判断）：
  chat     → 阿里云大模型直接生成回答（保持多轮对话历史）
  rag      → RAG 混合检索 + 大模型生成（红旗 EH7/天工05/天工06/天工08 知识库）
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
from typing import Any, AsyncGenerator

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
    build_intent_system, VALID_INTENTS, VALID_RAG_TAGS,
    CHAT_SYSTEM,
    TOOL_EXTRACT_SYSTEM, TOOL_ANSWER_SYSTEM,
    build_skill_extract_system, SKILL_ANSWER_SYSTEM,
    ETHICS_RESPONSE, FALLBACK_RESPONSE, SELF_INTRO_RESPONSE,
)


# ══════════════════════════════════════════════════════════════════
# <think> 标签过滤器（流式状态机）
# ══════════════════════════════════════════════════════════════════

class ThinkStripper:
    """
    从流式 token 中实时过滤 <think>...</think> 块。

    使用状态机 + 前缀暂存缓冲区，可正确处理标签跨 token 分割的情况：
    - 正常状态：发现 <think> 之前的内容立即输出；若末尾是 <think> 的
      某个前缀则暂存，等待后续 token 确认。
    - think 状态：丢弃所有内容直到找到 </think>；结束标签之后的
      首行换行符一并跳过。
    """

    _OPEN  = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf      = ""
        self._in_think = False

    def feed(self, token: str) -> str:
        """输入一个流式 token，返回可立即安全输出的文本（可能为空字符串）。"""
        self._buf += token
        return self._process()

    def finalize(self) -> str:
        """流结束时清空缓冲区；think 块内未闭合的内容一律丢弃。"""
        if self._in_think:
            self._buf = ""
            return ""
        out, self._buf = self._buf, ""
        return out

    def _process(self) -> str:
        out: list[str] = []
        while True:
            if self._in_think:
                idx = self._buf.find(self._CLOSE)
                if idx != -1:
                    # 找到结束标签：丢弃到标签末尾，跳过紧跟的换行
                    self._buf = self._buf[idx + len(self._CLOSE):].lstrip("\n")
                    self._in_think = False
                    # 继续处理 </think> 之后的剩余内容
                else:
                    # 保留可能是 </think> 前缀的末尾部分，等待后续 token
                    keep = next(
                        (n for n in range(len(self._CLOSE) - 1, 0, -1)
                         if self._buf.endswith(self._CLOSE[:n])),
                        0,
                    )
                    self._buf = self._buf[-keep:] if keep else ""
                    break
            else:
                idx = self._buf.find(self._OPEN)
                if idx != -1:
                    # 输出 <think> 之前的内容，进入 think 状态
                    out.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self._OPEN):]
                    self._in_think = True
                    # 继续处理 <think> 之后的剩余内容
                else:
                    # 末尾可能是 <think> 的某个前缀，暂存等待
                    keep = next(
                        (n for n in range(len(self._OPEN) - 1, 0, -1)
                         if self._buf.endswith(self._OPEN[:n])),
                        0,
                    )
                    safe_end = len(self._buf) - keep
                    out.append(self._buf[:safe_end])
                    self._buf = self._buf[safe_end:]
                    break
        return "".join(out)


# ══════════════════════════════════════════════════════════════════
# MCP 工具执行器
# ══════════════════════════════════════════════════════════════════

class MCPExecutor:
    """
    MCP 工具执行器。

    通过 MCP SSE 协议连接远程服务器调用工具（需要 mcp 包已安装）。
    如果服务器不可达或 mcp 包未安装，call() 返回 None，
    调用方应向用户提示服务不可用。
    """

    def __init__(self, server_url: str = "http://localhost:8000"):
        self._server_url = server_url.rstrip("/")
        self._sse_url    = f"{self._server_url}/sse"

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用 MCP 工具，返回执行结果。"""
        result = await self._try_remote(tool_name, arguments)
        return result

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

        # 启动时预加载所有检索索引，避免每次 RAG 查询重复读取磁盘
        self._vector_retriever = VectorRetriever()
        self._bm25_retriever   = BM25Retriever()
        self._graph_retriever  = GraphRetriever()

    # ── 1. 意图分类 ──────────────────────────────────────────────

    async def classify_intent(self, query: str) -> tuple[str, list[str], str]:
        """
        调用 LLM 判断用户意图，并在意图为 rag 时提取知识库标签。

        Returns:
            (intent, tags, reason)
              intent ∈ VALID_INTENTS
              tags   — rag 意图下的知识库标签列表（其余意图为空列表）
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
            raw_tags = parsed.get("tags", []) if intent == "rag" else []
            tags = [t for t in raw_tags if t in VALID_RAG_TAGS]
        except json.JSONDecodeError:
            intent, tags, reason = "chat", [], "JSON 解析失败，默认 chat"
        return intent, tags, reason

    # ── 2. 简单对话（多轮，流式）─────────────────────────────────

    async def handle_chat(self, query: str) -> AsyncGenerator[str, None]:
        """流式调用大模型，逐 token yield；所有 chunk 产出后更新多轮历史。"""
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
                chunks.append(delta)
                yield delta

        answer = "".join(chunks)
        self._chat_history.append({"role": "assistant", "content": answer})
        if len(self._chat_history) > 20:
            self._chat_history = self._chat_history[-20:]

    # ── 3. 知识检索（RAG，流式）──────────────────────────────────

    async def handle_rag(
        self, query: str, tags: list[str] | None = None
    ) -> AsyncGenerator[str, None]:
        """
        三路检索 + RRF 融合后，流式生成 EH7 相关问题的回答。

        Args:
            tags: 由意图分类返回的知识库标签，向量检索时作为 metadata 过滤条件。
        """
        vector_results = self._vector_retriever.search(query, k=RETRIEVE_K, tags=tags)
        bm25_results   = self._bm25_retriever.search(query, k=RETRIEVE_K)
        graph_results  = self._graph_retriever.search(query, k=RETRIEVE_K)
        top_results    = reciprocal_rank_fusion(
            [vector_results, bm25_results, graph_results],
            top_n=TOP_N,
        )

        async for delta in generate_answer_stream(query, top_results):
            yield delta

    # ── 4. 远程调用（MCP，流式）──────────────────────────────────

    async def handle_tool(self, query: str) -> AsyncGenerator[str, None]:
        """
        第一轮：让 LLM 从自然语言中提取工具名和参数（JSON 输出，非流式）。
        第二轮：通过 MCP 执行工具，再流式生成自然语言回答并逐 token yield。
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
                result_str = None
        else:
            result_str = None

        if result_str is None or result_str == "None":
            yield '远程服务调用失败，请检测服务是否可用。'
        else:
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

            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta

    # ── 5. 本地技能调用（Skills，流式）───────────────────────────

    async def handle_skill(self, query: str) -> AsyncGenerator[str, None]:
        """
        第一轮：让 LLM 从自然语言中提取技能名和参数（JSON 输出，非流式）。
        第二轮：本地执行技能，再流式生成自然语言回答并逐 token yield。
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

        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    # ── 6. 自我介绍 ───────────────────────────────────────────────

    async def handle_self_intro(self, query: str) -> AsyncGenerator[str, None]:  # noqa: ARG002
        """用户询问助手身份时直接 yield 固定自我介绍回复。"""
        yield SELF_INTRO_RESPONSE

    # ── 7. 道德伦理拦截 ───────────────────────────────────────────

    async def handle_ethics(self, query: str) -> AsyncGenerator[str, None]:  # noqa: ARG002
        """涉及不当内容时直接 yield 固定拒绝回复。"""
        yield ETHICS_RESPONSE

    # ── 8. 兜底话术 ───────────────────────────────────────────────

    async def handle_fallback(self, query: str) -> AsyncGenerator[str, None]:  # noqa: ARG002
        """超出系统能力范围时直接 yield 固定兜底回复。"""
        yield FALLBACK_RESPONSE

    # ── 9. 主处理入口 ─────────────────────────────────────────────

    async def process(self, query: str, show_intent: bool = True) -> AsyncGenerator[str, None]:
        """
        分类意图，打印意图标签和"助手: "前缀，然后路由到对应的流式处理器，
        将 handler 产出的 token 逐一 yield 给调用方。

        query 可能携带 /no_think 等模型控制前缀，分类时自动剥离以保证准确性；
        各 handler 接收原始 query（含前缀），用于抑制模型思考输出。
        """
        # 剥离 /no_think 等控制前缀，仅用于意图分类
        classify_query = re.sub(r"^(/no_think|/think)\s*\n?", "", query).strip()
        intent, tags, reason = await self.classify_intent(classify_query)

        if show_intent:
            label = {
                "chat":       "💬 简单对话",
                "rag":        "📚 知识检索",
                "tool":       "🔧 远程调用",
                "skill":      "⚙️ 工具调用",
                "self_intro": "🤖 自我介绍",
                "ethics":     "🚫 道德伦理",
                "fallback":   "❓ 兜底话术",
            }.get(intent, intent)
            tag_hint = f" [{', '.join(tags)}]" if tags else ""
            print(f"[意图: {label}{tag_hint}] {reason}")

        print("助手: ", end="", flush=True)

        if intent == "rag":
            handler = self.handle_rag(query, tags=tags)
        elif intent == "tool":
            handler = self.handle_tool(query)
        elif intent == "skill":
            handler = self.handle_skill(query)
        elif intent == "self_intro":
            handler = self.handle_self_intro(query)
        elif intent == "ethics":
            handler = self.handle_ethics(query)
        elif intent == "fallback":
            handler = self.handle_fallback(query)
        else:
            handler = self.handle_chat(query)

        stripper = ThinkStripper()
        async for token in handler:
            out = stripper.feed(token)
            if out:
                yield out
        tail = stripper.finalize()
        if tail:
            yield tail

    # ── 10. 对话循环 ──────────────────────────────────────────────

    async def run(self, show_intent: bool = True) -> None:
        """启动交互式对话循环，输入 exit/quit 退出。"""
        print("=" * 60)
        print("  红旗汽车智能对话助手（EH7 / 天工05 / 天工06 / 天工08）")
        print("  支持：日常对话 | 红旗汽车知识查询 | 本地技能 | 远程工具调用")
        print("  输入 exit 或 quit 退出")
        print("=" * 60)

        while True:
            try:
                query = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit", "退出", "bye"):
                print("再见！")
                break

            try:
                query = "/no_think\n" + query
                async for token in self.process(query, show_intent=show_intent):
                    print(token, end="", flush=True)
            except Exception as exc:
                print(f"\n[错误] {exc}")


# ══════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="红旗汽车智能对话助手（意图识别 + RAG + MCP 远程调用）"
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
