# 混合检索 RAG —— 上市公司年报中文知识库

面向 A 股上市公司年报的中文混合检索 RAG 系统：**dense + sparse 双路召回 → RRF 融合 → 云端重排 → LLM 生成**的端到端可检索问答链路，并将检索能力封装为 MCP 服务供其他 Agent 调用。

> **为什么是「混合检索」**：纯向量检索（dense）对财务术语、精确代码、数字提取这类硬匹配容易漏召回，纯关键词检索（sparse）又不懂语义改写。本项目用 BGE-M3 同时产出稠密向量与学习型稀疏向量，再叠加 jieba + BM25 的词法路，三条路经 RRF 融合互补，实测改写类查询召回显著优于单路。

## ✨ 特性

- **三命名向量**：`dense`（BGE-M3，1024 维）+ `bm25`（jieba + BM25，服务端 IDF）+ `bge_sparse`（learned sparse，白给）
- **双路召回 + RRF 融合**：dense / sparse 各召回 top50，RRF 按名次融合，保留分路排名明细
- **云端重排**：前 25 条送硅基流动 `bge-reranker-v2-m3` 重排取 top5
- **增量入库**：服务端 IDF 口径下新增/替换报告零全局重算，`--codes` 指定公司增量
- **可溯源引用**：每个命中块带公司/代码/年度/章节/页码/块类型 11 字段元数据，答案可溯源
- **MCP 服务**：检索侧封装为 streamable-http MCP，供 Claude Desktop / Cursor / 任意 Agent 调用（`search` / `list_companies` / `get_chunk_detail` / `health`）
- **离线评测**：自写五检索指标（Recall@5 / Precision@5 / HitRate@5 / MRR / NDCG@5）+ LLM 裁判

## 📊 评测结果（M4，50 题）

| 指标 | 得分 |
|---|---|
| Recall@5 | **0.58** |
| MRR | **0.433** |
| NDCG@5 | **0.470** |
| 裁判均分（5 分制） | **3.72** |

分题型召回：数字提取 0.857 > 术语精确 0.722 ≫ 语义改写 0.222（改写题是当前最大短板，见架构文档 §9 归因）。

## 🏗️ 技术栈

| 层 | 技术 |
|---|---|
| 嵌入 | BGE-M3（本地权重，FlagEmbedding 1.4.2） |
| 向量库 | Qdrant（Docker 容器，三命名向量 + 服务端 IDF） |
| 切块 | LlamaIndex `SentenceSplitter`（768/80） |
| 分词 | jieba（`lcut_for_search`） |
| 重排 | 硅基流动 `bge-reranker-v2-m3` |
| 生成 | 智谱 GLM `glm-5.3-flash` |
| 评测 | 自写五指标 + DeepSeek 裁判 |
| MCP | `mcp` SDK（streamable-http） |

## 📁 目录结构

```
├── src/                     # 核心代码（每模块单文件）
│   ├── download_pdf.py      # 年报 PDF 下载（巨潮接口）
│   ├── parse_pdf.py         # PyMuPDF 解析（title/text/table 三分类）
│   ├── chunk_documents.py   # 切块（768/80 + 元数据注入）
│   ├── index_documents.py   # BGE-M3 编码 + 三命名向量入库
│   ├── retrieve.py          # 检索（双路召回 + RRF + 重排）
│   ├── llm_client.py        # Prompt 组装 + LLM 生成
│   ├── evaluate.py          # 评测（出题/跑批/打分）
│   ├── mcp_server.py        # MCP 服务（检索侧对外暴露）
│   ├── utils.py             # 日志等工具
│   └── check_env.py         # 环境自检
├── scripts/                 # 服务启动脚本
│   ├── start_ui.py          # Gradio 交互界面启动器（逻辑自原 src/ui.py 迁入）
│   └── serve_mcp.py         # MCP 服务按需启动器
├── config/
│   ├── config.yaml          # 主配置（qdrant/embedding/retrieval/llm 等八段）
│   ├── companies.json       # 语料公司清单
│   └── .env.example         # 密钥模板（复制为 .env）
├── data/
│   ├── raw/                 # 年报 PDF 原件（需自行下载，见 usage.md）
│   ├── chunks/              # 切块产物（已入库，供演示/重建索引）
│   └── eval/                # 评测产物（评测集 + 报告）
├── docs/                    # 架构 / MCP / 使用文档
├── init_collection.py       # 独立建表脚本（三命名向量 schema）
├── docker-compose.yml       # Qdrant 容器一键启动
├── test_mcp_client.py       # MCP 客户端测试脚本
└── requirements.txt         # 依赖清单（含安装步骤）
```

## 🚀 快速开始

完整的环境搭建、数据下载、索引构建、检索与 MCP 接入步骤见 **[usage.md](usage.md)**。这里给出最小路径：

```bash
# 1. 创建环境并装依赖（详见 usage.md 或 requirements.txt 注释）
conda create -n py_rag python=3.12 -y && conda activate py_rag
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 启动 Qdrant 容器
docker compose up -d

# 3. 建表 + 入库（chunks 已入库，可直接跳过；重建用 index_documents.py）
python init_collection.py

# 4. 启动交互界面，前端直接提问测试
python scripts/start_ui.py
# 浏览器打开 http://127.0.0.1:7860，输入问题即可检索问答
```

> **其他 Agent 接入**：如需将检索能力暴露给 Claude Desktop / Cursor 等 Agent，可另启 MCP 服务（`python scripts/serve_mcp.py`，接入地址 `http://127.0.0.1:8100/mcp`），详见 [docs/mcp.md](docs/mcp.md)。日常交互测试推荐用上面的 `start_ui.py`。

## 📚 文档

| 文档 | 说明 |
|---|---|
| [usage.md](usage.md) | 详细使用手册（环境/数据/索引/检索/MCP/评测/FAQ） |
| [docs/architecture.md](docs/architecture.md) | 架构文档（唯一活跃架构文档，含设计决策与风险） |
| [docs/mcp.md](docs/mcp.md) | MCP 服务接入文档（四个工具 + 客户端配置） |

## ⚠️ 环境注意

- **导入顺序硬约束**（仅旧环境）：`FlagEmbedding` 必须最先 import（torch 与 OpenSSL/MKL 的 DLL 冲突）。新环境 `py_rag`（torch 2.14 + numpy 2.x）下此约束已消失，但代码已固化安全顺序，无需改动。
- **GPU**：GTX 1650（TU117 无 Tensor Core）下 fp16 反而慢 4.7 倍，配置固定 fp32。
- **transformers 必须锁 4.x**：5.9.0 实测 import 即段错误，`requirements.txt` 已锁 `<5.0`。

## 📄 License

待定（开源前请补充）。
