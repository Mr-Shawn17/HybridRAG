# 使用手册（usage.md）

本文档是混合检索 RAG 项目的详细使用指南，覆盖从零搭建环境到检索、评测、MCP 接入的完整流程。

> 快速概览见 [README.md](README.md)，架构设计见 [docs/architecture.md](docs/architecture.md)，MCP 接入见 [docs/mcp.md](docs/mcp.md)。

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [启动 Qdrant](#3-启动-qdrant)
4. [构建索引](#4-构建索引)
5. [检索与问答](#5-检索与问答)
6. [MCP 服务](#6-mcp-服务)
7. [评测](#7-评测)
8. [常见问题（FAQ）](#8-常见问题faq)

---

## 1. 环境准备

### 1.1 前置依赖

- **Python 3.12**（推荐 conda 管理）
- **Docker Desktop**（Qdrant 容器）
- **NVIDIA GPU**（可选，BGE-M3 编码可跑 CPU，但慢）

### 1.2 创建环境

```bash
conda create -n py_rag python=3.12 -y
conda activate py_rag
```

### 1.3 安装依赖

**完整步骤见 `requirements.txt` 末尾注释**，关键顺序（避免版本冲突）：

```bash
# ① torch（CUDA 版，按驱动选 cu130；CPU 版换 /whl/cpu）
pip install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130

# ② 先单独锁 transformers 4.x（5.9.0 实测 import 即段错误，必须锁 <5.0）
pip install "transformers>=4.44.2,<5.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

# ③ 装其余依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **版本说明**：torch 2.14.1+cu130 / transformers 4.57.6 / numpy 2.5.4（pyarrow 26 要求 ≥2.0）/ FlagEmbedding 1.4.2 / qdrant-client 1.20.0 / mcp 2.3.0。

### 1.4 验证环境

```bash
python -c "import torch, transformers, FlagEmbedding; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

输出版本号且 `cuda=True` 即就绪。

### 1.5 下载模型权重

BGE-M3 权重（约 2.3G）**不入库**，需自行下载到 `model/BGE-M3/`：

```bash
# 方式一：从 Hugging Face 下载（国内可配 HF_ENDPOINT 镜像）
huggingface-cli download BAAI/bge-m3 --local-dir model/BGE-M3

# 方式二：从 ModelScope 下载
# pip install modelscope && python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='model/BGE-M3')"
```

### 1.6 配置密钥（可选）

```bash
cp config/.env.example config/.env
# 编辑 config/.env 填入 key
```

| Key | 用途 | 是否必需 |
|---|---|---|
| `SILICONFLOW_API_KEY` | 云端重排（E3 实验组） | 用 E3 重排才需要 |
| `GLM_API_KEY` | LLM 生成（start_ui.py 问答） | 用问答总结才需要 |
| `DEEPSEEK_API_KEY` | 评测裁判 | 跑评测才需要 |

> **纯检索链路（无重排、无生成）完全本地运行，无需任何 key**。检索 + BM25 + RRF 融合 + BGE-M3 编码都不依赖外部 API。

## 2. 数据准备

### 2.1 年报 PDF

60 份年报 PDF（`data/raw/`）**不入库**，通过巨潮资讯网公开接口下载：

```bash
python src/download_pdf.py
# 下载产物：data/raw/{2023,2024,2025}/ + manifest.json
```

下载的公司清单见 `config/companies.json`（20 家，覆盖金融/家电/白酒/医药/汽车/能源/科技等行业）。

### 2.2 数据产物说明

| 目录 | 内容 | 是否入库 | 生成方式 |
|---|---|---|---|
| `data/raw/` | 年报 PDF 原件 | ❌ | `download_pdf.py` |
| `data/parsed/` | 解析中间产物 JSON | ❌ | `parse_pdf.py` |
| `data/chunks/` | 切块产物 JSON | ✅（已入库） | `chunk_documents.py` |
| `data/embeddings/` | 编码缓存 npz | ❌ | `index_documents.py` 阶段 A |
| `data/eval/` | 评测集 + 报告 | ✅ | `evaluate.py` |

> `data/chunks/`（56M）已随仓库入库，供演示或重建索引使用；其余均可按上述命令再生。

## 3. 启动 Qdrant

```bash
docker compose up -d          # 后台启动 qdrant 容器（映射 6333/6334 端口）
docker compose down           # 停止（数据卷保留）
docker compose down -v        # 停止并清空数据（慎用）
```

启动后验证：

```bash
curl http://127.0.0.1:6333/collections
```

### 建表（collection + 三命名向量 schema）

```bash
python init_collection.py              # 创建 collection（已存在则跳过）
python init_collection.py --recreate   # 删除重建（清空数据）
```

collection `annual_reports` 含三命名向量：
- `dense`（1024 维，COSINE）
- `bm25`（稀疏，`Modifier.IDF` 服务端维护 df/idf）
- `bge_sparse`（稀疏，原始 relu 权重，无 modifier）

> `init_collection.py` 只依赖 `qdrant-client` + `pyyaml`，**不加载模型**，秒级完成。索引入库时 `index_documents.py` 也会自动建表，二选一即可。

## 4. 构建索引

```bash
# 全量两阶段（编码 + 入库；首次编码约 20~40min，重跑命中缓存免 GPU）
python src/index_documents.py

# 指定公司（增量 / 小样本）
python src/index_documents.py --codes 600519

# 只处理前 N 份（试跑）
python src/index_documents.py --limit 2

# 删旧 collection 重建
python src/index_documents.py --recreate

# 整份替换（入库前先删所选报告旧点）
python src/index_documents.py --codes 000001 --replace

# 跳过编码（要求 npz 缓存齐全）
python src/index_documents.py --skip-encode
```

**日常新增一份年报**：

```bash
# 切块产物放 data/chunks/ 后：
python src/index_documents.py --codes <code> --limit 1
```

## 5. 检索与问答

> **日常交互测试推荐用 Gradio UI（5.1）**：启动后在前端界面直接输入问题即可，无需写代码。需要编程集成时再用 Python API（5.2）。

### 5.1 Gradio UI

```bash
python scripts/start_ui.py
# 浏览器打开 http://127.0.0.1:7860
```

启动即预热（加载 BGE-M3 + 连接 Qdrant），首问不卡。界面支持：

- **提问**：输入自然语言问题，点「检索并回答」或回车提交
- **实验组切换**：E0~E4 五组路由（E3 为完整链路，含云端重排）
- **元数据过滤**：按股票代码 / 年份区间限定召回范围
- **引用溯源**：每个命中块带公司 / 代码 / 年度 / 章节 / 页码
- **进度反馈**：查询期间显示 spinner + 扫动进度条 + 实时计时，完成显示「检索 x.xs + 生成 x.xs」

### 5.2 Python API

```python
import sys
sys.path.insert(0, "src")
from retrieve import Retriever, build_filter

r = Retriever()  # 惰性加载 BGE-M3

# 完整混合链路（E3，含云端重排）
hits = r.retrieve("平安银行2023年末不良贷款率是多少", exp="E3")

# 纯本地检索（E2，零 API 成本）
hits = r.retrieve("...", exp="E2")

# 带过滤（限定股票代码 + 年份区间）
flt = build_filter(codes=["000001"], year_from=2023, year_to=2024)
hits = r.retrieve("...", exp="E3", flt=flt)
```

**实验组路由**（`exp` 参数）：

| 实验组 | 召回路 | 重排 |
|---|---|---|
| E0 | 纯 dense 向量 | 否 |
| E1 | 纯 BM25 词法 | 否 |
| E2 | dense + bm25 混合 | 否 |
| E3 | dense + bm25 混合 | ✅（默认） |
| E4 | dense + bge_sparse | 否 |

## 6. MCP 服务

将检索侧封装为 MCP 服务，**供其他 Agent（Claude Desktop / Cursor 等）调用**。日常人工交互测试请直接用上面的 `scripts/start_ui.py`（5.1），无需启动本服务。接入详见 [docs/mcp.md](docs/mcp.md)。

```bash
# 启动（按需）
python scripts/serve_mcp.py                    # host/port 读 config（默认 127.0.0.1:8100）
python scripts/serve_mcp.py --port 9000        # 换端口（覆盖 config）
python scripts/serve_mcp.py --log-path D:/logs/mcp.log   # 自定义日志路径

# 测试客户端（纯标准库，零依赖）
python test_mcp_client.py --list
python test_mcp_client.py health
python test_mcp_client.py list_companies
python test_mcp_client.py search --query "平安银行不良贷款率" --exp E3
python test_mcp_client.py get_chunk_detail --chunk_id "000001_2023_00002"
```

四个工具：`search`（混合检索）/ `list_companies`（语料范围）/ `get_chunk_detail`（单条完整元数据）/ `health`（健康检查）。

## 7. 评测

```bash
# 生成评测集（50 题，分层抽样 seed=42）
python src/evaluate.py gen

# 跑批（断点续跑）
python src/evaluate.py run

# 打分（五指标 + 裁判报告）
python src/evaluate.py score
```

产出到 `data/eval/`：`eval_set.json`（评测集）/ `run_results.json`（结果）/ `report.md`（报告）。

**五指标**：Recall@5 / Precision@5 / HitRate@5 / MRR / NDCG@5，金标准 = 出题源 chunk 唯一命中。

## 8. 常见问题（FAQ）

### Q1：`import transformers` 段错误？

transforms 5.x 在本机实测 import 即段错误，必须锁 4.x（`pip install "transformers>=4.44.2,<5.0"`）。见 requirements.txt 注释。

### Q2：`import FlagEmbedding` 崩溃？

旧环境（torch 老版本）存在「FlagEmbedding 必须最先 import」的 DLL 顺序约束。新环境 `py_rag`（torch 2.14 + numpy 2.x）下冲突已消失，但代码已固化安全顺序，直接 `import index_documents` 即可，勿在之前抢先 import numpy/hashlib。

### Q3：`search` 报错但 `health` 正常？

先查 `SILICONFLOW_API_KEY` 是否配置（E3 含云端重排，缺 key 会失败）。零成本方案用 `exp=E2` 或 `rerank=false` 绕过重排。

### Q4：Qdrant 连接失败？

确认容器已启动：`docker compose up -d`，再 `curl http://127.0.0.1:6333/collections` 验证。连接参数在 `config/config.yaml` 的 `qdrant` 段。

### Q5：模型加载慢 / 显存不足？

BGE-M3 首次加载约 20s；GTX 1650（4G 显存）下 `batch_size` 已调 16，仍 OOM 可再调小（config.yaml `embedding.batch_size`）。fp16 已关（TU117 无 Tensor Core，fp16 反而慢 4.7 倍）。

### Q6：数据 / 权重为什么不在仓库里？

年报 PDF（326M）、模型权重（4.3G）体积过大且可公开下载，已 .gitignore。仓库只含源码 + 配置 + 文档 + `data/chunks`（56M 切块产物）。
