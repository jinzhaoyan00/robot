# Robot — 多模态知识索引系统

从结构化文本中自动构建 **FAISS 向量索引**、**BM25 关键词索引** 和 **Graphiti 知识图谱**，存储于 KuZu 图数据库，支持语义检索与知识图谱查询。

---

## 项目结构

```
robot/
├── data/                        # 原始文本数据（每行一条记录）
│   ├── eh7_1.txt
│   ├── eh7_2.txt
│   ├── eh7_3.txt
│   └── eh7_4.txt
│
├── index_builder/               # 索引构建模块
│   ├── __init__.py
│   ├── embedder.py              # ModelScope GTE-base 中文嵌入模型
│   ├── vector_store.py          # FAISS 向量索引构建与检索
│   ├── bm25_index.py            # BM25 关键词索引构建与检索
│   ├── knowledge_graph.py       # Graphiti + KuZu 知识图谱构建
│   └── dashscope_client.py      # 阿里云 DashScope LLM 客户端（OpenAI 兼容）
│
├── vector_db/                   # 持久化存储目录（构建后自动生成）
│   ├── faiss/
│   │   ├── faiss.index          # FAISS 向量索引文件
│   │   └── metadata.json        # 文档元数据（文件名、行号、原文）
│   ├── bm25/
│   │   ├── bm25.pkl             # BM25 索引对象
│   │   └── bm25_metadata.json   # 文档元数据
│   ├── kuzu_db/                 # KuZu 图数据库（知识图谱）
│   └── .model_cache/            # GTE 模型缓存（自动下载）
│
├── main.py                      # 主入口：构建全部索引
├── requirements.txt             # Python 依赖
└── README.md
```

---

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 嵌入模型 | [ModelScope GTE-base 中文版](https://www.modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-base)（768 维） |
| 向量数据库 | [FAISS](https://faiss.ai/)（IndexFlatIP，余弦相似度） |
| 关键词索引 | [rank-bm25](https://github.com/dorianbrown/rank_bm25)（Okapi BM25） + jieba 分词 |
| 知识图谱框架 | [graphiti-core](https://github.com/getzep/graphiti) |
| 图数据库 | [KuZu](https://kuzudb.com/)（嵌入式图数据库） |
| LLM | 阿里云 DashScope Qwen（OpenAI 兼容接口） |

---

## 安装

```bash
pip install -r requirements.txt
```

**requirements.txt 核心依赖：**
```
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
jieba>=0.42.1
modelscope>=1.9.0
graphiti-core[kuzu]>=0.28.0
openai>=1.0.0
numpy>=1.24.0
transformers
torch
datasets
addict
oss2
```

> **注意**：首次运行时会自动从 ModelScope 下载 GTE-base 模型（约 400 MB），请确保网络畅通。

---

## 快速开始

### 构建全部索引（含知识图谱）

```bash
python main.py
```

构建流程：
1. **Step 1** — 读取 `data/` 下所有 `.txt` 文件，逐行嵌入并构建 FAISS 索引
2. **Step 2** — jieba 分词后构建 BM25 索引
3. **Step 3** — 调用阿里云 Qwen LLM 提取实体和关系，写入 KuZu 图数据库
4. **检索测试** — 运行预设查询，展示 FAISS 和 BM25 检索结果

### 跳过知识图谱（仅构建向量和 BM25 索引）

```bash
python main.py --skip-kg
```

> 适用于无需 LLM API 的离线环境，或仅需向量/关键词检索的场景。

---

## 代码说明

### `index_builder/embedder.py` — GTE 嵌入模型

使用 `modelscope.hub.snapshot_download` 下载模型，通过 HuggingFace `transformers` 进行推理：

```python
from index_builder.embedder import GTEEmbedder

emb = GTEEmbedder()
vec = emb.embed_one("红旗EH7价格是多少？")   # shape (768,)
mat = emb.embed(["文本1", "文本2"])           # shape (2, 768)
```

### `index_builder/vector_store.py` — FAISS 向量检索

```python
from index_builder.vector_store import VectorStore

vs = VectorStore("vector_db/faiss")
results = vs.search("EH7轴距", k=5)
# [{'text': '...', 'file': 'eh7_1.txt', 'line': 11, 'score': 0.593}, ...]
```

### `index_builder/bm25_index.py` — BM25 关键词检索

```python
from index_builder.bm25_index import BM25Index

bm25 = BM25Index("vector_db/bm25")
results = bm25.search("电池保修政策", k=5)
# [{'text': '...', 'file': '...', 'line': ..., 'score': 7.27}, ...]
```

### `index_builder/knowledge_graph.py` — 知识图谱

```python
import asyncio
from index_builder.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph("vector_db/kuzu_db")
asyncio.run(kg.build([{
    "name": "ep_001",
    "body": "EH7 600 智选版售价 208800 元，轴距 3000mm。",
    "source": "配置表"
}]))
```

### `index_builder/dashscope_client.py` — 阿里云 LLM 客户端

针对 DashScope OpenAI 兼容接口的自定义 graphiti LLM 客户端：
- 重写 `_generate_response`，统一使用 `chat.completions` + JSON mode
- 绕过 graphiti 默认调用的 OpenAI Responses API（DashScope 不支持）

---

## 知识图谱 Schema（KuZu）

| 节点类型 | 说明 |
|---------|------|
| `Entity` | 提取的实体（产品、特性、政策等） |
| `Episodic` | 原始文本片段（每 10 行一个 episode） |
| `Community` | 实体社区（批量构建时生成） |

| 关系类型 | 说明 |
|---------|------|
| `RELATES_TO` | 实体之间的语义关系 |
| `MENTIONS` | Episode 提及 Entity |

---

## 配置说明

主要配置项均在各模块顶部常量中定义：

| 配置项 | 位置 | 默认值 |
|-------|------|-------|
| 嵌入模型 ID | `index_builder/embedder.py` | `iic/nlp_gte_sentence-embedding_chinese-base` |
| 阿里云 API Key | `index_builder/knowledge_graph.py` | `sk-b4349f588d9f46dfbcb749bde4566019` |
| 阿里云模型 | `index_builder/knowledge_graph.py` | `qwen-plus` |
| Episode 行数 | `main.py` | `10` |

---

## 性能参考

测试环境：Linux，NVIDIA GPU，数据集 604 行

| 步骤 | 耗时 |
|------|------|
| FAISS 索引构建（604 条） | ~45 秒 |
| BM25 索引构建（604 条） | < 1 秒 |
| 知识图谱（61 个 episodes） | ~38 分钟（受 LLM API 速率限制） |

**知识图谱结果：**
- 实体节点：178 个
- MENTIONS 边：486 条
- RELATES_TO 关系：966 条

---

## 注意事项

1. **首次运行**：GTE 模型会从 ModelScope 自动下载（约 400 MB），需要网络访问 `modelscope.cn`。
2. **API 限制**：知识图谱构建依赖阿里云 DashScope API，受速率限制影响，大量数据时耗时较长。
3. **KuZu 数据库**：`vector_db/kuzu_db` 由 KuZu 自动创建，**不要**预先创建该目录。
4. **GPU 加速**：GTE 嵌入模型会自动检测并使用 CUDA GPU（如有）。
