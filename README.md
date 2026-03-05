# 红旗 EH7 智能对话助手

基于阿里云 Qwen 大模型，集成 **意图识别 → 多路 RAG 检索 → MCP 远程工具调用 → 本地技能调用** 的多轮对话系统，专为红旗 EH7 车型知识问答场景设计。

---

## 功能概览

| 意图类型 | 触发条件 | 处理方式 |
|---------|---------|---------|
| `chat` | 日常闲聊、通用问题 | Qwen 多轮对话（保留最近 20 条历史） |
| `rag` | 红旗 EH7 相关知识查询 | 三路混合检索 + RRF 融合 + Qwen 生成 |
| `tool` | 数学计算 | LLM 提取参数 → MCP 远程调用 |
| `skill` | 日期时间查询、单位换算等 | LLM 提取参数 → 本地技能执行 |
| `ethics` | 涉及政治/暴恐/违法内容 | 固定拒绝话术 |
| `fallback` | 超出系统能力范围 | 固定兜底话术 |

所有回答均以**流式输出**呈现，并自动过滤模型的 `<think>…</think>` 推理过程。

---

## 项目结构

```
robot/
├── data/                        # 红旗 EH7 原始文本（以 "FAW-Robotics" 行分块）
│   ├── eh7_1.txt
│   ├── eh7_2.txt
│   ├── eh7_3.txt
│   └── eh7_4.txt
│
├── index_builder/               # 离线索引构建模块
│   ├── embedder.py              # ModelScope BAAI/bge-large-zh 嵌入模型
│   ├── vector_store.py          # FAISS 向量索引（IndexFlatIP，余弦相似度）
│   ├── bm25_index.py            # BM25 关键词索引（jieba + rank-bm25）
│   ├── knowledge_graph.py       # 知识图谱构建（graphiti-core + KuZu）
│   └── dashscope_client.py      # graphiti 专用 DashScope LLM 客户端
│
├── rag/                         # 在线检索与生成模块
│   ├── retriever.py             # VectorRetriever / BM25Retriever / GraphRetriever
│   ├── reranker.py              # Reciprocal Rank Fusion (RRF) 融合排序
│   └── generator.py             # 流式 / 非流式 LLM 答案生成
│
├── prompts/                     # 提示词统一管理
│   ├── intent_prompt.py         # 意图分类（6 类）
│   ├── chat_prompt.py           # 日常对话系统提示词
│   ├── rag_prompt.py            # RAG 知识检索提示词
│   ├── tool_prompt.py           # MCP 工具参数提取 + 回答提示词
│   ├── skill_prompt.py          # 本地技能参数提取 + 回答提示词
│   └── fixed_responses.py       # 道德伦理拒绝 + 兜底话术
│
├── skills/                      # 本地技能（插件式目录结构）
│   ├── datetime/                # 日期时间查询技能
│   │   ├── SKILL.md             # 技能元数据（YAML frontmatter）
│   │   └── scripts/execute.py   # 执行逻辑
│   └── unit_converter/          # 单位换算技能
│       ├── SKILL.md
│       └── scripts/execute.py
│
├── mcp/                         # MCP 远程工具服务
│   ├── server.py                # FastMCP 服务端（加减乘除，SSE 协议）
│   └── client.py                # MCP 客户端（测试用）
│
├── vector_db/                   # 索引持久化目录（构建后自动生成，不纳入版本控制）
│   ├── faiss/                   # FAISS 向量索引 + metadata.json
│   ├── bm25/                    # BM25 索引 + bm25_metadata.json
│   └── kuzu_db/                 # KuZu 图数据库（知识图谱）
│
├── model_cache/                 # 嵌入模型缓存（不纳入版本控制）
│
├── main_index_builder.py        # 离线入口：构建全部索引
├── main_dialog.py               # 在线入口：启动对话系统
├── .env.example                 # 环境变量模板
├── requirements.txt
└── README.md
```

---

## 技术栈

| 组件 | 选型 |
|-----|-----|
| 对话大模型 | 阿里云 DashScope Qwen（OpenAI 兼容接口） |
| 嵌入模型 | [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)（ModelScope 缓存） |
| 向量检索 | [FAISS](https://faiss.ai/) IndexFlatIP（余弦相似度） |
| 关键词检索 | [rank-bm25](https://github.com/dorianbrown/rank_bm25) Okapi BM25 + jieba 分词 |
| 图谱检索 | [KuZu](https://kuzudb.com/) 全文搜索（FTS） |
| 知识图谱构建 | [graphiti-core](https://github.com/getzep/graphiti) + KuZu 后端 |
| 远程工具协议 | [MCP](https://modelcontextprotocol.io/)（SSE 传输） |
| 融合排序 | Reciprocal Rank Fusion (RRF) |

---

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/jinzhaoyan00/robot.git
cd robot
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 首次运行时会从 ModelScope 自动下载嵌入模型（BAAI/bge-large-zh，约 1.3 GB），请确保网络畅通。

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入必要配置：

```ini
# 阿里云 DashScope API（必填）
ALIBABA_API_KEY=your-api-key-here
ALIBABA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ALIBABA_MODEL=qwen-plus

# MCP 服务器（可选，不启动时自动降级为本地执行）
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8000

# RAG 检索参数（可选，以下为默认值）
RETRIEVE_K=20    # 每路检索条数
TOP_N=10         # RRF 融合后送入 LLM 的条数
RRF_K=60         # RRF 平滑系数

# 索引路径（可选）
FAISS_DIR=vector_db/faiss
BM25_DIR=vector_db/bm25
KUZU_DIR=vector_db/kuzu_db
EMBED_MODEL_ID=BAAI/bge-large-zh
```

---

## 使用步骤

### Step 1：构建索引（离线，只需运行一次）

```bash
# 构建全部索引（FAISS + BM25 + 知识图谱）
python main_index_builder.py

# 仅构建向量和 BM25 索引（不调用 LLM API，速度快）
python main_index_builder.py --skip-kg
```

构建流程：
1. 读取 `data/` 下所有 `.txt` 文件，以独占一行的 `FAW-Robotics` 为分隔符切块
2. **FAISS**：嵌入所有文本块，构建内积向量索引
3. **BM25**：jieba 分词后构建 Okapi BM25 索引
4. **知识图谱**（可选）：调用 Qwen 提取实体和关系，写入 KuZu 图数据库

构建完成后，索引文件保存至 `vector_db/`。

### Step 2：启动对话（可选：先启动 MCP 服务器）

```bash
# 启动 MCP 数学工具服务器（可选）
python mcp/server.py

# 启动对话系统（默认连接 http://localhost:8888 的 MCP 服务器）
python main_dialog.py

# 指定 MCP 服务器地址
python main_dialog.py --mcp-url http://localhost:8000

# 不显示意图分类标签
python main_dialog.py --no-intent-label
```

---

## 对话系统架构

```
用户输入
  │
  ▼
意图分类（Qwen，JSON 输出）
  │
  ├─ chat     ─→ 多轮对话（保留最近 20 条历史）
  │
  ├─ rag      ─→ 向量检索（FAISS）
  │              BM25 检索
  │              图谱检索（KuZu FTS）
  │              RRF 融合排序（top-N 片段）
  │              Qwen 流式生成答案
  │
  ├─ tool     ─→ Qwen 提取工具名 + 参数（非流式）
  │              MCP 远程调用（SSE）或本地降级执行
  │              Qwen 流式生成自然语言回答
  │
  ├─ skill    ─→ Qwen 提取技能名 + 参数（非流式）
  │              本地技能执行（datetime / unit_converter / …）
  │              Qwen 流式生成自然语言回答
  │
  ├─ ethics   ─→ 固定拒绝话术（不调用 LLM）
  │
  └─ fallback ─→ 固定兜底话术（不调用 LLM）
  │
  ▼
ThinkStripper（过滤 <think>…</think>）
  │
  ▼
逐 token 流式打印至终端
```

### 索引预加载

`DialogSystem` 启动时一次性将所有索引加载进内存，后续每次 RAG 查询均复用缓存，避免重复读取磁盘：

- `VectorRetriever`：FAISS index + metadata 常驻内存
- `BM25Retriever`：BM25 对象 + metadata 常驻内存
- `GraphRetriever`：KuZu 连接持久保持

---

## 扩展技能

在 `skills/` 下创建子目录，遵循以下结构即可被自动发现和加载：

```
skills/
└── my_skill/
    ├── SKILL.md             # 元数据（YAML frontmatter）
    └── scripts/
        └── execute.py       # 执行逻辑
```

`SKILL.md` frontmatter 示例：

```yaml
---
name: my_skill
description: 这个技能的功能描述（供意图分类 LLM 参考）
params_hint: '{"param1": "值1", "param2": "值2"}'
---
```

`execute.py` 须导出 `execute(**kwargs) -> str` 函数。

---

## MCP 工具服务

`mcp/server.py` 提供四个数学工具，通过 SSE 协议对外暴露：

| 工具 | 说明 |
|-----|-----|
| `add(a, b)` | 加法 |
| `subtract(a, b)` | 减法 |
| `multiply(a, b)` | 乘法 |
| `divide(a, b)` | 除法（b=0 时抛出异常） |

对话系统优先通过 MCP SSE 协议连接服务器；服务器不可达时自动降级为本地执行，无需手动干预。

---

## RAG 检索说明

三路检索各取 `RETRIEVE_K`（默认 20）条结果，经 RRF 融合后取前 `TOP_N`（默认 10）条送入 LLM：

| 检索路 | 实现 | 优势 |
|-------|-----|-----|
| 向量检索 | FAISS + bge-large-zh | 捕捉语义相似度 |
| 关键词检索 | BM25Okapi + jieba | 精确关键词匹配 |
| 图谱检索 | KuZu FTS | 结构化知识（实体、关系、原文片段） |

图谱检索同时查询三类节点：
- `Episodic`：原始文本片段（完整上下文）
- `Entity`：实体名称 + 摘要
- `RelatesToNode_`：实体间关系事实

---

## 注意事项

1. **首次运行**：嵌入模型从 ModelScope 自动下载（约 1.3 GB），需要访问 `modelscope.cn`。
2. **知识图谱构建**：依赖 DashScope API，数据量大时受速率限制影响，耗时较长（每 episode 一次 LLM 调用）。
3. **KuZu 数据库**：`vector_db/kuzu_db` 由 KuZu 自动创建，**不要**预先创建该目录。
4. **GPU 加速**：bge-large-zh 嵌入模型自动检测并使用 CUDA GPU（如有）。
5. **`.env` 安全**：`.env` 文件已加入 `.gitignore`，切勿提交至版本控制。
