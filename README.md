# 红旗汽车智能对话助手

基于阿里云 Qwen 大模型，集成 **意图识别 → 多路 RAG 检索 → MCP 远程工具 → 本地技能** 的流式多轮对话系统，专为红旗 EH7、天工05、天工06、天工08 车型知识问答场景设计。

---

## 功能概览

| 意图类型 | 触发条件 | 处理方式 |
|---------|---------|---------|
| `chat` | 日常闲聊、通用问题 | Qwen 多轮对话（保留最近 20 条历史） |
| `rag` | EH7 / 天工05 / 天工06 / 天工08 知识查询 | 三路混合检索 + RRF 融合 + Qwen 流式生成 |
| `tool` | 数学计算（加减乘除） | LLM 提取参数 → MCP SSE 远程调用 → Qwen 生成回答 |
| `skill` | 日期时间查询、单位换算等 | LLM 提取参数 → 本地技能执行 → Qwen 生成回答 |
| `self_intro` | "你叫什么名字"、"你是谁"、"你能做什么" | 固定自我介绍话术（不调用 LLM） |
| `ethics` | 涉及政治/暴恐/违法等不当内容 | 固定拒绝话术（不调用 LLM） |
| `fallback` | 超出系统能力范围 | 固定兜底话术（不调用 LLM） |

所有回答均以**流式输出**逐 token 打印，并自动过滤推理模型的 `<think>…</think>` 块。

---

## 项目结构

```
robot/
│
├── main_dialog.py              # 在线入口：启动多轮对话系统
├── main_index_builder.py       # 离线入口：构建向量 / BM25 / 知识图谱索引
│
├── prompts/                    # 提示词统一管理（所有 LLM 指令集中在此）
│   ├── __init__.py             # 统一导出入口
│   ├── intent_prompt.py        # 意图分类（7 类，含 RAG 标签规则）
│   ├── chat_prompt.py          # 日常对话系统提示词
│   ├── rag_prompt.py           # RAG 检索系统提示词 + user 消息构建
│   ├── tool_prompt.py          # MCP 工具参数提取 + 自然语言回答提示词
│   ├── skill_prompt.py         # 本地技能参数提取 + 自然语言回答提示词
│   └── fixed_responses.py      # 自我介绍 / 道德伦理拒绝 / 兜底话术常量
│
├── rag/                        # 在线检索与生成模块
│   ├── retriever.py            # VectorRetriever / BM25Retriever / GraphRetriever
│   ├── reranker.py             # Reciprocal Rank Fusion (RRF) 多路融合排序
│   └── generator.py            # 流式 / 非流式 LLM 答案生成
│
├── index_builder/              # 离线索引构建模块
│   ├── embedder.py             # GTEEmbedder：BAAI/bge-large-zh 嵌入推理
│   ├── vector_store.py         # ChromaDB 向量存储（余弦相似度，批量写入）
│   ├── bm25_index.py           # BM25 关键词索引（jieba 分词 + rank-bm25）
│   ├── knowledge_graph.py      # 知识图谱构建（graphiti-core + KuZu）
│   └── dashscope_client.py     # graphiti 专用 DashScope LLM 客户端
│
├── skills/                     # 本地技能（插件式目录，自动发现加载）
│   ├── __init__.py             # REGISTRY 注册表 + 对外调度接口
│   ├── datetime/               # 日期时间查询技能
│   │   ├── SKILL.md            # 技能元数据（YAML frontmatter）
│   │   └── scripts/execute.py  # execute(query_type) → str
│   └── unit_converter/         # 单位换算技能（长度 / 重量 / 温度）
│       ├── SKILL.md
│       └── scripts/execute.py  # execute(value, from_unit, to_unit) → str
│
├── mcp/                        # MCP 远程工具服务
│   ├── server.py               # FastMCP SSE 服务端（add/subtract/multiply/divide）
│   ├── client.py               # MCP 客户端（测试用）
│   └── mcp_config.json         # 服务器配置
│
├── data_process/               # 数据预处理脚本（原始数据 → 知识库 JSON）
│   ├── data_preprocess.py      # Excel（.xlsx）→ JSON，处理配置/答疑/电池保修表
│   ├── prepare_data.py         # 构建向量检索训练三元组（query / pos / neg）
│   ├── prepare_data_peizhi.py  # 从 *_配置.txt 构建训练对
│   ├── prepare_data_peizhi_md.py # 对齐配置文本与 JSON markdown 文档
│   ├── prepare_data_qa.py      # 从 *_答疑.txt 构建训练对
│   └── merge_data.py           # 合并所有 train/val JSONL 分片
│
├── finetune/                   # 嵌入模型微调（可选）
│   ├── train.py                # SentenceTransformer + MNRL 训练脚本
│   ├── config.py               # FinetuneConfig 数据类
│   └── requirements_finetune.txt # 微调专用依赖
│
├── data/
│   ├── txt/                    # 知识库 JSON 文件（供索引构建读取）
│   │   ├── EH7_配置.json       # EH7 车型参数规格表（markdown 格式）
│   │   ├── EH7_答疑.json       # EH7 常见问题解答
│   │   ├── EH7_电池保修.json   # EH7 电池保修政策
│   │   ├── 天工05_*.json       # 天工05 对应三类文档
│   │   ├── 天工06_*.json       # 天工06 对应三类文档
│   │   └── 天工08_*.json       # 天工08 对应三类文档
│   └── 知识库-v2/              # 原始 Excel 数据源（.xlsx）
│
├── vector_db/                  # 索引持久化目录（自动生成，不纳入版本控制）
│   ├── chromadb/               # ChromaDB 向量索引
│   ├── bm25/                   # BM25 索引（bm25.pkl + bm25_metadata.json）
│   └── kuzu_db/                # KuZu 图数据库（知识图谱，可选）
│
├── model_cache/                # 嵌入模型本地缓存（自动下载，不纳入版本控制）
├── tmp/                        # 索引构建断点状态（build_state.json）
│
├── .env.example                # 环境变量模板
├── requirements.txt            # 核心依赖
└── README.md
```

---

## 技术栈

| 组件 | 技术选型 |
|-----|---------|
| 对话大模型 | 阿里云 DashScope Qwen（OpenAI 兼容接口） |
| 嵌入模型 | [BAAI/bge-large-zh](https://modelscope.cn/models/BAAI/bge-large-zh)（ModelScope 缓存，CLS 池化 + L2 归一化） |
| 向量检索 | [ChromaDB](https://www.trychroma.com/)（余弦相似度，持久化） |
| 关键词检索 | [rank-bm25](https://github.com/dorianbrown/rank_bm25) Okapi BM25 + jieba 分词 |
| 图谱检索 | [KuZu](https://kuzudb.com/) 全文搜索（FTS，可选） |
| 知识图谱构建 | [graphiti-core](https://github.com/getzep/graphiti) + KuZu 驱动 |
| 融合排序 | Reciprocal Rank Fusion（RRF） |
| 远程工具协议 | [MCP](https://modelcontextprotocol.io/)（SSE 传输，FastMCP） |
| 流式过滤 | `ThinkStripper`：状态机实时过滤 `<think>…</think>` |

---

## 快速开始

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

编辑 `.env`，填入以下配置：

```ini
# 阿里云 DashScope API（必填）
ALIBABA_API_KEY=your-api-key-here
ALIBABA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ALIBABA_MODEL=qwen-plus

# MCP 服务器地址（可选）
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8000

# RAG 检索参数（可选，以下为默认值）
RETRIEVE_K=20        # 每路检索返回条数
TOP_N=10             # RRF 融合后送入 LLM 的条数
RRF_K=60             # RRF 平滑系数

# 索引存储路径（可选，以下为默认值）
CHROMA_DIR=vector_db/chromadb
BM25_DIR=vector_db/bm25
KUZU_DIR=vector_db/kuzu_db
EMBED_MODEL_ID=BAAI/bge-large-zh
```

### 4. 构建索引（离线，只需运行一次）

```bash
# 推荐：仅构建向量 + BM25 索引（无需调用 LLM API）
python main_index_builder.py --skip-kg

# 完整构建（包含知识图谱，需要 DashScope API，耗时较长）
python main_index_builder.py

# 从头重建（清除已有进度）
python main_index_builder.py --reset --skip-kg
```

构建完成后索引保存至 `vector_db/`，断点进度记录在 `tmp/build_state.json`。

### 5. 启动对话

```bash
# 可选：先启动 MCP 数学工具服务器
python mcp/server.py

# 启动对话系统
python main_dialog.py

# 指定 MCP 服务器地址
python main_dialog.py --mcp-url http://localhost:8000

# 不显示意图分类标签
python main_dialog.py --no-intent-label
```

---

## 系统架构

### 对话流程

```
用户输入
  │
  ▼
意图分类（Qwen JSON 输出，7 类）
  │
  ├─ chat       ──► Qwen 多轮流式对话（最近 20 条历史）
  │
  ├─ rag        ──► 向量检索（ChromaDB + bge-large-zh）
  │                  BM25 关键词检索（jieba）
  │                  知识图谱检索（KuZu FTS，可选）
  │                  RRF 融合排序 → top-N 片段
  │                  Qwen 流式生成答案
  │
  ├─ tool       ──► Qwen 提取 tool/a/b（JSON）
  │                  MCP SSE 远程调用（add/subtract/multiply/divide）
  │                  Qwen 流式生成自然语言回答
  │
  ├─ skill      ──► Qwen 提取 skill/params（JSON）
  │                  本地技能执行（datetime / unit_converter）
  │                  Qwen 流式生成自然语言回答
  │
  ├─ self_intro ──► 固定话术："我是小莫，一汽集团的智能机器人…"
  │
  ├─ ethics     ──► 固定拒绝话术
  │
  └─ fallback   ──► 固定兜底话术
       │
       ▼
  ThinkStripper（流式过滤 <think>…</think>）
       │
       ▼
  逐 token 打印至终端
```

### 索引构建流程

```
data/txt/*.json
  │
  ├─ Step 1/3 ──► ChromaDB 向量索引
  │                bge-large-zh 批量嵌入（BATCH_SIZE=8）
  │                余弦相似度，HNSW 索引
  │
  ├─ Step 2/3 ──► BM25 关键词索引
  │                jieba 分词 → BM25Okapi
  │                持久化为 bm25.pkl + bm25_metadata.json
  │
  └─ Step 3/3 ──► 知识图谱（可选，--skip-kg 跳过）
                   Qwen 提取实体 + 关系
                   写入 KuZu 图数据库
                   手动创建 FTS 索引（Episodic / Entity / RelatesToNode_）
```

---

## RAG 检索说明

三路检索各取 `RETRIEVE_K`（默认 20）条，经 RRF 融合后保留 `TOP_N`（默认 10）条送入 LLM：

| 检索路 | 实现 | 优势 |
|-------|-----|------|
| 向量检索 | ChromaDB + BAAI/bge-large-zh | 语义相似度，泛化能力强 |
| 关键词检索 | BM25Okapi + jieba | 精确关键词匹配，型号/规格检索准确 |
| 图谱检索 | KuZu FTS（可选） | 结构化知识，实体关系推理 |

图谱检索同时查询三类节点（各占 k/2、k/4、k/4）：

| 节点类型 | FTS 索引 | 返回内容 |
|---------|---------|---------|
| `Episodic` | `episode_content` | 原始文本片段（完整上下文） |
| `Entity` | `node_name_and_summary` | 实体名称 + 摘要 |
| `RelatesToNode_` | `edge_name_and_fact` | 实体间关系事实 |

意图分类器为 `rag` 意图同时提取知识库标签（`配置` / `答疑` / `电池保修`），向量检索阶段据此过滤 metadata，提升精度。

---

## 本地技能扩展

在 `skills/` 下新建子目录，放入以下两个文件即可被自动发现并注册：

```
skills/
└── my_skill/
    ├── SKILL.md             # 技能元数据（YAML frontmatter）
    └── scripts/
        └── execute.py       # 须导出 execute(**kwargs) -> str
```

**`SKILL.md` 示例：**

```yaml
---
name: my_skill
description: 技能功能描述（供意图分类 LLM 判断使用）
params_hint: '{"skill": "my_skill", "param1": "示例值"}'
---
```

**内置技能：**

| 技能名 | 功能 | 参数 |
|--------|------|------|
| `datetime` | 查询当前日期 / 时间 / 星期 | `query_type`: full \| date \| time \| weekday |
| `unit_converter` | 长度 / 重量 / 温度换算 | `value`, `from_unit`, `to_unit` |

---

## MCP 工具服务

`mcp/server.py` 通过 SSE 协议对外暴露四个数学工具：

| 工具 | 说明 |
|-----|------|
| `add(a, b)` | 加法 |
| `subtract(a, b)` | 减法 |
| `multiply(a, b)` | 乘法 |
| `divide(a, b)` | 除法（b=0 时抛出异常） |

```bash
# 启动服务器（读取 .env 中的 MCP_SERVER_HOST/PORT）
python mcp/server.py
# SSE 端点：http://localhost:8000/sse
```

对话系统通过 `--mcp-url` 指定服务器地址（默认 `http://localhost:8888`）；服务器不可达时向用户返回提示信息。

---

## 数据处理与微调（可选）

### 数据预处理

`data_process/` 将原始 Excel 数据转换为知识库 JSON：

```bash
python data_process/data_preprocess.py   # Excel → JSON
python data_process/prepare_data.py      # 构建检索训练三元组
python data_process/merge_data.py        # 合并所有训练/验证集
```

### 嵌入模型微调

```bash
pip install -r finetune/requirements_finetune.txt
python finetune/train.py
```

基于 `SentenceTransformer + MultipleNegativesRankingLoss`，在车型知识库数据上进一步优化 `bge-large-zh` 的检索效果。

---

## 配置参考

| 变量 | 默认值 | 说明 |
|-----|--------|------|
| `ALIBABA_API_KEY` | — | DashScope API 密钥（必填） |
| `ALIBABA_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | API 端点 |
| `ALIBABA_MODEL` | `qwen-plus` | 使用的模型名称 |
| `RETRIEVE_K` | `20` | 每路检索返回条数 |
| `TOP_N` | `10` | RRF 融合后送入 LLM 的条数 |
| `RRF_K` | `60` | RRF 平滑系数 |
| `CHROMA_DIR` | `vector_db/chromadb` | ChromaDB 索引路径 |
| `BM25_DIR` | `vector_db/bm25` | BM25 索引路径 |
| `KUZU_DIR` | `vector_db/kuzu_db` | KuZu 图数据库路径 |
| `EMBED_MODEL_ID` | `BAAI/bge-large-zh` | ModelScope 嵌入模型 ID |
| `MCP_SERVER_HOST` | `0.0.0.0` | MCP 服务器监听地址 |
| `MCP_SERVER_PORT` | `8000` | MCP 服务器端口 |
| `KG_MAX_TOKENS` | `4096` | 知识图谱构建时 LLM 最大 token 数 |

---

## 注意事项

1. **首次运行**：嵌入模型从 ModelScope 自动下载（约 1.3 GB），需能访问 `modelscope.cn`。
2. **GPU 加速**：嵌入模型自动检测 CUDA，有 GPU 时显著提速。
3. **知识图谱**：依赖 DashScope API，每个 episode 消耗一次 LLM 调用，数据量大时耗时较长，建议先用 `--skip-kg` 验证基本功能。
4. **KuZu 目录**：`vector_db/kuzu_db` 由 KuZu 自行初始化，**不要**预先创建该目录。
5. **断点续建**：索引构建中断后直接重新运行即可从 `tmp/build_state.json` 恢复；`--reset` 清除进度重头构建。
6. **`.env` 安全**：`.env` 已加入 `.gitignore`，切勿提交至版本控制。
