# -*- coding: utf-8 -*-
"""MCP 服务（Model Context Protocol）：把检索侧暴露给其他 Agent 调用。

架构 v1.9 新增：streamable-http 传输（默认 127.0.0.1:8100/mcp），
四个工具——

  search            混合检索（BGE-M3 dense + BM25 双路召回 → RRF 融合
                    → 云端重排，默认 E3 完整链路；即 retrieve.Retriever）
  list_companies    语料覆盖范围（Qdrant facet 按 stock_code/fiscal_year
                    聚合，毫秒级，不加载模型）
  health            健康检查（Qdrant 连通/点数/模型加载态/rerank 配置）
  get_chunk_detail  按 chunk_id 取单条完整 payload（point id =
                    uuid5(NAMESPACE_URL, chunk_id)，见 index_documents）

设计口径：
  - 连接复用：进程内单例——轻客户端（仅 Qdrant，list_companies/
    get_chunk_detail/health 用）与完整 Retriever（含 BGE-M3 模型，
    search 用）都惰性初始化；首次 search 含模型加载（约 20s，GTX 1650
    fp32），之后毫秒级。
  - 并发保护：全局异步锁串行化所有 Qdrant/GPU 访问（4GB 显存跑不下
    并发 encode）；阻塞操作经 asyncio.to_thread 下放线程池，不卡事件
    循环。
  - 错误约定：可预期失败（Qdrant 未启动/重排 API 失败/参数非法）抛
    mcp ToolError——客户端收到可读消息；意外异常由 SDK 记 traceback。
  - 部署形态：常驻服务，多客户端共享同一模型实例（对比 stdio 子进程
    模式每次冷启动都要重载 20s 模型，这是选 streamable-http 的主因）。

导入顺序硬约束：与 retrieve.py 同款——本模块最先 import index_documents
（其内部保证 FlagEmbedding 先于 hashlib/numpy/jieba 加载，否则 Windows
import 阶段即 segfault，2026-09-15 实测）。mcp SDK（pydantic/httpx 等）
全部排在其后。

用法（项目根目录，py312 环境；需 Qdrant 容器在跑）：
  python src/mcp_server.py                       # 127.0.0.1:8100
  python src/mcp_server.py --port 9000           # 换端口
  python src/mcp_server.py --host 0.0.0.0        # 监听全部网卡（局域网）

客户端接入（Claude Desktop / Cursor / WorkBuddy 自定义 MCP 等）：
  url = http://<host>:<port>/mcp，详见 docs/mcp.md
"""

from __future__ import annotations

import os

# 离线加载本地权重（model/BGE-M3），与 index_documents 一致
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 必须最先！勿把后面的 import 挪到它之前（见模块 docstring 导入顺序硬约束）
import index_documents as idx  # noqa: E402
import retrieve  # noqa: E402

import argparse  # noqa: E402
import asyncio  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

# 日志路径：默认 <项目根>/log/mcp_server.log；可用环境变量 MCP_LOG_PATH
# 覆盖（serve_mcp.py 会把可配置的日志路径经此传入），路径不存在会自动创建。
#
# 注意：不能用 utils.Log.init_logger——它内部有 root_logger.handlers 守卫，
# index_documents import 时已初始化 root logger（写到 index_documents.log），
# 后续调用会被守卫拦下，导致 MCP 日志混进 index_documents.log。故此处自建
# 独立的命名 logger + 专属文件 handler，与 root logger 解耦。
_default_log = idx.LOG_DIR / "mcp_server.log"
_log_path = Path(os.environ.get("MCP_LOG_PATH", str(_default_log)))

_fmt = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
_log_dir = _log_path.parent
_log_dir.mkdir(parents=True, exist_ok=True)

# MCP 自身业务日志：独立命名 logger，文件 handler 写到 _log_path。
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.INFO)
logger.propagate = False  # 不向 root logger 传播，避免写入 index_documents.log
if not any(getattr(h, "_mcp_file", False) for h in logger.handlers):
    _fh = logging.FileHandler(str(_log_path), encoding="utf-8")
    _fh._mcp_file = True
    _fh.setFormatter(_fmt)
    _fh.setLevel(logging.INFO)
    logger.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    _sh.setLevel(logging.INFO)
    logger.addHandler(_sh)


def _route_to_mcp_log(name: str) -> None:
    """把某个 logger 命名空间的日志路由到 mcp_server.log（同时仍输出控制台）。

    用于 mcp.server.*（SDK 内部）与 uvicorn 的日志：它们默认 propagate 到
    root logger（→ index_documents.log），这里显式挂同一个文件 handler 并
    阻断传播，使 MCP 服务相关日志全部归到 mcp_server.log。
    """
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    if not any(getattr(h, "_mcp_file", False) for h in lg.handlers):
        lg.addHandler(_fh)
        if not any(isinstance(h, logging.StreamHandler) for h in lg.handlers):
            lg.addHandler(_sh)


# SDK 内部日志 + uvicorn 一并路由到 mcp_server.log
_route_to_mcp_log("mcp.server")
_route_to_mcp_log("uvicorn")
_route_to_mcp_log("uvicorn.access")
_route_to_mcp_log("uvicorn.error")

# ── 单例状态（惰性初始化，全局锁串行化）──────────────────────────


class _State:
    """进程级单例：轻客户端（Qdrant only）与完整 Retriever（含模型）。"""

    client: idx.QdrantClient | None = None  # type: ignore[name-defined]
    retriever: retrieve.Retriever | None = None
    init_error: str | None = None  # Retriever 最近一次初始化失败原因


_init_lock = asyncio.Lock()  # 串行化惰性初始化
_exec_lock = asyncio.Lock()  # 串行化 Qdrant/GPU 访问（4GB 显存防并发 OOM）

_search_count = 0
_search_total_ms = 0.0


def _init_client() -> idx.QdrantClient:  # type: ignore[name-defined]
    """轻客户端：仅连 Qdrant，不加载 BGE-M3（facet/直查用）。"""
    cfg = idx.load_config()
    client = idx.get_qdrant_client(cfg)
    if not client.collection_exists(idx.COLLECTION_NAME):
        raise RuntimeError(
            f"collection {idx.COLLECTION_NAME} 不存在（请先完成 M2b 入库，"
            "并确认 Qdrant 容器在运行：docker start qdrant）"
        )
    return client


async def _get_client() -> idx.QdrantClient:  # type: ignore[name-defined]
    async with _init_lock:
        if _State.client is None:
            try:
                _State.client = await asyncio.to_thread(_init_client)
            except Exception as exc:  # noqa: BLE001
                raise ToolError(
                    f"Qdrant 连接失败：{exc}（确认容器在跑：docker start qdrant）"
                ) from exc
        return _State.client


async def _get_retriever() -> retrieve.Retriever:
    """完整 Retriever：Qdrant + BGE-M3 模型（首次约 20s 模型加载）。"""
    async with _init_lock:
        if _State.retriever is None:
            try:
                retrieve._load_dotenv()  # .env/注册表兜底（幂等 setdefault）
                _State.retriever = await asyncio.to_thread(retrieve.Retriever)
                _State.init_error = None
            except Exception as exc:  # noqa: BLE001
                _State.init_error = f"{type(exc).__name__}: {exc}"
                raise ToolError(f"Retriever 初始化失败：{exc}") from exc
        return _State.retriever


# ── 工具实现 ────────────────────────────────────────────────────


def _hit_dict(rank: int, h: dict) -> dict:
    """retrieve() 命中项 → 对外精简 dict（保留定位与溯源所需全字段）。"""
    p = h.get("payload") or {}
    out = {
        "rank": rank,
        "chunk_id": p.get("chunk_id"),
        "company_name": p.get("company_name"),
        "stock_code": p.get("stock_code"),
        "fiscal_year": p.get("fiscal_year"),
        "section": p.get("section"),
        "page_num": p.get("page_num"),
        "chunk_type": p.get("chunk_type"),
        "text": h.get("text", ""),
        "rrf_score": round(float(h.get("rrf_score", 0.0)), 6),
        "score_dense": h.get("score_dense"),
        "rank_dense": h.get("rank_dense"),
        "score_sparse": h.get("score_sparse"),
        "rank_sparse": h.get("rank_sparse"),
    }
    if h.get("rerank_score") is not None:
        out["rerank_score"] = round(float(h["rerank_score"]), 6)
        out["rerank_rank"] = h.get("rerank_rank")
    return out


server = MCPServer(
    name="annual-report-rag",
    instructions=(
        "上市公司年报混合检索知识库（60 份年报 / 20 家公司 × 2021-2023）。"
        "典型用法：先 list_companies 确认语料覆盖的公司与股票代码，"
        "再 search 按问题检索（可按股票代码/年份过滤），"
        "需要单条完整元数据时用 get_chunk_detail(chunk_id)。"
    ),
)


@server.tool()
async def search(
    query: str,
    exp: str = "E3",
    codes: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    rerank: bool | None = None,
) -> dict:
    """在年报语料库中混合检索，返回最相关的 chunk（默认 5 条）。

    Args:
        query: 查询语句（自然语言，如"平安银行2023年末不良贷款率是多少"）。
        exp: 实验组/检索口径。E3=混合检索+云端重排（默认，效果最好）；
            E2=混合检索不重排（零 API 成本）；E0=纯向量；E1=纯 BM25；
            E4=向量+learned sparse 对照组。
        codes: 限定股票代码列表（如 ["000001", "600036"]），None=全库。
        year_from: 年份区间起（含，如 2022）。
        year_to: 年份区间止（含，如 2023）。
        rerank: 覆盖默认重排行为（True 强制开/False 强制关；默认按 exp）。

    Returns:
        dict: hits 命中列表（每条含 chunk_id/公司/代码/年度/页码/正文/
        各路分数），timings_ms 各阶段耗时。首次调用含模型加载约 20s。
    """
    global _search_count, _search_total_ms
    if exp not in retrieve.EXP_ROUTES:
        raise ToolError(
            f"未知实验组 {exp}，可选：{'/'.join(retrieve.EXP_ROUTES)}"
        )
    if not query or not query.strip():
        raise ToolError("query 不能为空")
    r = await _get_retriever()
    flt = retrieve.build_filter(
        codes=codes, year_from=year_from, year_to=year_to
    )
    t0 = time.perf_counter()
    async with _exec_lock:  # GPU/Qdrant 串行
        try:
            hits = await asyncio.to_thread(
                r.retrieve, query.strip(), exp, flt, rerank
            )
        except (RuntimeError, ValueError) as exc:
            # 可预期失败：重排 API 挂、Qdrant 断连、参数非法——透传给调用方
            raise ToolError(str(exc)) from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _search_count += 1
    _search_total_ms += elapsed_ms
    return {
        "query": query.strip(),
        "exp": exp,
        "filtered": flt is not None,
        "n_hits": len(hits),
        "hits": [_hit_dict(i, h) for i, h in enumerate(hits, 1)],
        "timings_ms": r.last_timings_ms,
    }


@server.tool()
async def list_companies() -> dict:
    """列出语料库覆盖的公司（股票代码+chunk 数）与年度分布。

    调用 search 前建议先执行本工具：拿到准确股票代码再过滤，
    避免公司名/代码猜错导致空结果。

    Returns:
        dict: companies 公司列表（stock_code/n_company_chunks/years 覆盖
        年度），years 全库年度分布，n_points 总点数。
    """
    client = await _get_client()

    def _do() -> dict:
        comp_resp = client.facet(
            idx.COLLECTION_NAME, key="stock_code", limit=100, exact=True
        )
        year_resp = client.facet(
            idx.COLLECTION_NAME, key="fiscal_year", limit=20, exact=True
        )
        # 每家公司再聚合一次年度覆盖（20 家 × 毫秒级 facet）
        from qdrant_client import models as qm

        comp_years: dict[str, list] = {}
        for hit in comp_resp.hits:
            code = str(hit.value)
            yr = client.facet(
                idx.COLLECTION_NAME,
                key="fiscal_year",
                facet_filter=qm.Filter(
                    must=[qm.FieldCondition(key="stock_code",
                                            match=qm.MatchValue(value=code))]
                ),
                limit=10,
                exact=True,
            )
            comp_years[code] = sorted(int(h.value) for h in yr.hits)
        return {
            "companies": [
                {"stock_code": str(h.value), "n_chunks": h.count}
                for h in comp_resp.hits
            ],
            "years": {str(h.value): h.count for h in year_resp.hits},
            "company_years": comp_years,
            "n_points": client.count(
                idx.COLLECTION_NAME, exact=True
            ).count,
        }

    async with _exec_lock:
        try:
            return await asyncio.to_thread(_do)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"facet 聚合失败：{exc}") from exc


@server.tool()
async def get_chunk_detail(chunk_id: str) -> dict:
    """按 chunk_id 取单条 chunk 的完整元数据与正文。

    search 返回的每条命中都带 chunk_id；当正文需要完整溯源（源文件、
    章节、页码、OCR 标记等 11 字段 payload）时用本工具二次拉取。

    Args:
        chunk_id: chunk 唯一标识（search 命中项的 chunk_id 字段）。

    Returns:
        dict: point_id（uuid5 点 id）、payload 全部 11 字段（含完整 text）。
    """
    if not chunk_id or not chunk_id.strip():
        raise ToolError("chunk_id 不能为空")
    client = await _get_client()
    # point id = uuid5(NAMESPACE_URL, chunk_id)，与入库侧同口径（§2.1）
    pid = str(uuid.uuid5(idx.CHUNK_NS, chunk_id.strip()))

    def _do() -> list:
        return client.retrieve(
            idx.COLLECTION_NAME, ids=[pid], with_payload=True
        )

    async with _exec_lock:
        try:
            pts = await asyncio.to_thread(_do)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Qdrant 直查失败：{exc}") from exc
    if not pts:
        raise ToolError(f"chunk_id 未命中：{chunk_id}（确认来自 search 返回）")
    p = pts[0]
    return {"point_id": str(p.id), "payload": p.payload or {}}


@server.tool()
async def health() -> dict:
    """健康检查：Qdrant 连通性、collection 点数、模型加载态、rerank 配置。

    search 报错时先调本工具定位：区分 Qdrant 未启动、API key 缺失、
    模型未加载等问题。

    Returns:
        dict: ok 总体状态 + qdrant/model/rerank/stats 各分项。
    """
    out: dict[str, Any] = {"ok": True}
    # ① Qdrant
    try:
        client = await _get_client()
        n = await asyncio.to_thread(
            client.count, idx.COLLECTION_NAME, exact=True
        )
        out["qdrant"] = {"ok": True, "n_points": n.count,
                         "collection": idx.COLLECTION_NAME}
    except ToolError as exc:
        out["ok"] = False
        out["qdrant"] = {"ok": False, "error": str(exc)}
        return out
    # ② 模型（惰性加载：未触发过 search 时如实报告）
    if _State.retriever is not None:
        out["model"] = {"loaded": True, "device": _State.retriever.cfg
                        ["embedding"].get("device", "?")}
    else:
        out["model"] = {
            "loaded": False,
            "note": "惰性加载：首次 search 时加载（约 20s）",
            **({"last_init_error": _State.init_error}
               if _State.init_error else {}),
        }
    # ③ rerank 配置（E3 依赖 SILICONFLOW_API_KEY）
    rr = (idx.load_config().get("rerank") or {})
    enabled = bool(rr) and bool(rr.get("enabled", True))
    key_env = str(rr.get("api_key_env", "SILICONFLOW_API_KEY"))
    out["rerank"] = {
        "configured": enabled,
        "model": rr.get("model"),
        "api_key_set": bool(os.environ.get(key_env, "").strip()),
        "api_key_env": key_env,
        "note": "E3 默认走云端重排；key 缺失时 E3 会失败，可用 exp=E2 或 "
                "rerank=false 绕过",
    }
    # ④ 服务统计
    out["stats"] = {
        "n_searches": _search_count,
        "avg_search_ms": round(_search_total_ms / _search_count, 1)
        if _search_count else None,
    }
    out["exp_routes"] = list(retrieve.EXP_ROUTES)
    return out


# ── CLI ─────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="年报混合检索 MCP 服务（streamable-http）"
    )
    ap.add_argument("--host", default=None, help="监听地址（默认读 config mcp 段，再退 127.0.0.1）")
    ap.add_argument("--port", type=int, default=None, help="监听端口（默认读 config mcp 段，再退 8100）")
    args = ap.parse_args()

    cfg = idx.load_config()
    mcp_cfg = cfg.get("mcp") or {}
    host = args.host or mcp_cfg.get("host", "127.0.0.1")
    port = int(args.port or mcp_cfg.get("port", 8100))

    retrieve._load_dotenv()  # .env/注册表兜底（rerank key 等）

    logger.info("MCP 服务启动：%s:%s/mcp（streamable-http）", host, port)
    logger.info(
        "工具：search / list_companies / get_chunk_detail / health；"
        "语料 collection=%s", idx.COLLECTION_NAME
    )
    server.run("streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
