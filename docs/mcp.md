# MCP 服务接入文档（v1.9）

把「混合检索 RAG」的检索侧封装成 MCP（Model Context Protocol）服务，供其他 Agent（Claude Desktop / Cursor / WorkBuddy 自定义 MCP / 任何 MCP 客户端）调用。

> **定位**：本服务用于**其他 Agent 集成**。日常人工交互测试请直接用 `scripts/start_ui.py`（前端提问），无需启动本服务。

## 1. 概述

- **代码**：`src/mcp_server.py`（服务逻辑）+ `scripts/serve_mcp.py`（按需启动器：拉起 + 探活）
- **传输**：streamable-http（常驻服务，多客户端共享同一 BGE-M3 模型实例）
- **默认地址**：`http://127.0.0.1:8100/mcp`（host/port 见 `config/config.yaml` 的 `mcp` 段，或命令行 `--host`/`--port` 覆盖）
- **运行环境**：`py_rag`（Python 3.12，见 `requirements.txt`）

## 2. 启动

```bash
conda activate py_rag
cd D:\VSCodeProject\混合检索RAG
python scripts/serve_mcp.py                # host/port 读 config（默认 127.0.0.1:8100）
python scripts/serve_mcp.py --port 9000    # 换端口（覆盖 config）
python scripts/serve_mcp.py --host 0.0.0.0 # 局域网（自担风险，无鉴权）
```

> 也可直接跑底层实现 `python src/mcp_server.py`（等价，但无探活/复用逻辑，不推荐）。`scripts/serve_mcp.py` 会先探活复用、再拉起并轮询就绪。

**前置依赖**：
- Qdrant 容器在跑（`docker start qdrant`），collection `annual_reports` 已入库（40,192 点）
- 云端重排 key：`SILICONFLOW_API_KEY`（E3 完整链路用；缺失时 E3 会失败，可用 `exp=E2` 或 `rerank=false` 绕过）

## 3. 四个工具

### 3.1 `search` — 混合检索（核心）

| 参数 | 类型 | 说明 |
|---|---|---|
| `query` | str（必填） | 查询语句，如「平安银行2023年末不良贷款率是多少」 |
| `exp` | str | 检索口径：`E3`（默认，混合+重排）/ `E2`（混合不重排，零 API 成本）/ `E0`（纯向量）/ `E1`（纯 BM25）/ `E4`（向量+learned sparse） |
| `codes` | list[str] | 限定股票代码，如 `["000001","600036"]`，None=全库 |
| `year_from` / `year_to` | int | 年份闭区间 |
| `rerank` | bool | 覆盖默认重排行为（True 强开/False 强关） |

**返回**：`hits` 命中列表（每条含 `chunk_id`/公司/代码/年度/页码/正文/各路分数 + `rrf_score`/`rerank_score`），`timings_ms` 各阶段耗时。首次调用含模型加载约 5s（GTX 1650 fp32）。

### 3.2 `list_companies` — 语料覆盖范围

无参数。返回 20 家公司（`stock_code` + chunk 数 + 覆盖年度）、全库年度分布、总点数。**建议 search 前先调用**，拿到准确股票代码再过滤。

### 3.3 `get_chunk_detail` — 单条完整元数据

参数：`chunk_id`（search 命中项的 `chunk_id` 字段）。返回该 chunk 的完整 11 字段 payload（含 `source_file`/`section`/`page_num`/`is_ocr`/完整 `text`）。

### 3.4 `health` — 健康检查

无参数。返回 Qdrant 连通性、collection 点数、模型加载态、rerank 配置、检索统计。**search 报错时先调它定位**（区分 Qdrant 未启动 / API key 缺失 / 模型未加载）。

## 4. 客户端接入

### 4.1 Claude Desktop

`claude_desktop_config.json` 里加：

```json
{
  "mcpServers": {
    "annual-report-rag": {
      "type": "http",
      "url": "http://127.0.0.1:8100/mcp"
    }
  }
}
```

### 4.2 Cursor

Cursor Settings → MCP → Add new MCP server → 选 HTTP，填 `http://127.0.0.1:8100/mcp`。

### 4.3 WorkBuddy 自定义 MCP

连接器管理 → 添加自定义 MCP → 选 streamable-http 类型，URL 填 `http://127.0.0.1:8100/mcp`。

### 4.4 程序化调用（官方 SDK）

```python
import asyncio
from mcp import Client

async def main():
    async with Client("http://127.0.0.1:8100/mcp") as client:
        r = await client.call_tool("search", {"query": "平安银行不良贷款率"})
        print(r.structured_content)

asyncio.run(main())
```

> **注意**：mcp SDK 2.3.0 的 `Client.call_tool()` 在 Windows + httpx2 下存在 URL 编码 bug（请求路径被编码为 `http%3A//...` 导致 404）。用 curl 直连、或 Claude Desktop / Cursor 等成熟客户端均正常。若需程序化调用且遇到此问题，可改用低层 `streamable_http_client` 手动发 JSON-RPC，或降级 `mcp<2`。

## 5. 设计要点

- **连接复用**：进程内单例。轻客户端（仅 Qdrant，服务 list_companies/get_chunk_detail/health）与完整 Retriever（含 BGE-M3，服务 search）均惰性初始化。
- **并发保护**：全局异步锁串行化 Qdrant/GPU 访问（4GB 显存跑不下并发 encode）；阻塞操作经 `asyncio.to_thread` 下放线程池。
- **错误约定**：可预期失败（Qdrant 未启动/重排 API 失败/参数非法）抛 `ToolError`（客户端收到可读消息）；意外异常由 SDK 记 traceback。
- **导入顺序**：沿用项目硬约束（`FlagEmbedding` 最先加载）。注意 py_rag 新环境（torch 2.14 + numpy 2.x）下 DLL 冲突已消失，此约束理论上可放宽，但代码已固化安全顺序，无需改动。

## 6. 环境说明（2026-10-25 新建）

- **环境**：`py_rag`（Python 3.12.13），替代原 `py312`（其 transformers 5.9.0 import 即段错误，已废弃检索用途）
- **关键版本**：torch 2.14.1+cu130 / transformers 4.57.6（锁 4.x，避开损坏的 5.9.0）/ numpy 2.5.4（pyarrow 26 要求 ≥2.0）/ mcp 2.3.0
- **安装**：见 `requirements.txt` 末尾「py_rag 环境完整安装步骤」

## 7. 遗留问题 / 已知限制

- mcp SDK 2.3.0 客户端 `call_tool()` 的 URL 编码 bug（见 §4.4 注意），属上游 SDK 问题，非本服务端缺陷。
- 语料实际覆盖 **2023–2025** 三个年度（`list_companies` 返回为准），非早期文档里的 2021–2023。
