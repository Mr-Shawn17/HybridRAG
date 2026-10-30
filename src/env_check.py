# -*- coding: utf-8 -*-
"""服务启动前的环境检查（v1.11：serve_mcp.py / start_ui.py 启动前统一调用）。

目的：把"启动后才发现环境没准备好"前移到启动阶段，逐项打印 ✓/!/✗ 并在
任一**硬阻塞项**失败时打印 [ERROR] 并中止启动，便于问题排查。

检查项按服务区分（用户 2026-10-30 定版）：
  - 通用硬项（两服务都查，失败即中止）：
      ① Qdrant 容器在线 + collection `annual_reports` 存在
      ② BGE-M3 本地权重 model/BGE-M3 就位（search 编码必需）
  - serve_mcp 专属软项：SILICONFLOW_API_KEY（E3 云端重排；缺则仅 WARN）
  - start_ui 专属软项：SILICONFLOW_API_KEY + GLM_API_KEY（E3 重排 / LLM 生成）

设计口径：
  - 复用 index_documents 的常量（COLLECTION_NAME/ROOT_PATH）保证单一真相源，
    不重复定义连接参数——Qdrant 探活与 get_qdrant_client 同款读 config。
  - 检查前先 retrieve._load_dotenv()（config/.env + Machine 注册表兜底），
    这样 key 检查才与运行时真实可见的环境变量一致（setdefault 幂等）。
  - 本模块**不加载 BGE-M3 模型**（那要 20s，且启动器稍后会自己加载）——
    只判断权重目录/关键文件是否存在，快速、零副作用。
  - 每个检查函数返回 (ok: bool, msg: str)，异常会被框架捕获标为 FAIL。

导入顺序：本模块 import index_documents（其内部保证 FlagEmbedding 最先），
勿在之前 import numpy/hashlib（见 index_documents docstring 硬约束）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

# 必须最先！复用 index_documents 的常量与安全导入顺序（见模块 docstring）
import index_documents as idx  # noqa: E402

PASS, WARN, FAIL = "✓", "!", "✗"

# BGE-M3 本地权重关键文件（判断就位与否，不加载模型）
# 注意：本项目权重为 pytorch_model.bin（非 safetensors），以实机文件为准
_MODEL_MARKERS = ("config.json", "sentence_bert_config.json", "pytorch_model.bin")


# ── 检查函数（每项返回 (ok, msg)）───────────────────────────────────


def _check_qdrant() -> tuple[bool, str]:
    """Qdrant 服务端在线 + collection 存在（硬阻塞项）。"""
    import json
    import urllib.request

    cfg = idx.load_config()
    q = cfg["qdrant"]
    url = f"http://{q['host']}:{int(q.get('port', 6333))}/collections"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        # 常见场景翻译成人话，便于排查；其余透传原始异常
        hint = (
            "容器未启动 → 运行 `docker start qdrant` 或 `docker compose up -d`"
            if _is_conn_refused(exc)
            else f"{type(exc).__name__}: {exc}"
        )
        return False, f"无法连接 Qdrant（{url}）：{hint}"
    names = [c["name"] for c in data.get("result", {}).get("collections", [])]
    if idx.COLLECTION_NAME not in names:
        return False, (
            f"Qdrant 在线但缺 collection {idx.COLLECTION_NAME}"
            f"（现有 {names or '无'}）→ 先跑 index_documents 入库"
        )
    return True, f"{url} 在线，collection={idx.COLLECTION_NAME} 存在"


def _is_conn_refused(exc: Exception) -> bool:
    """判断是否为连接拒绝（容器/端口未监听）——含 111 errno 或 URLError 包裹。"""
    errno = getattr(exc, "errno", None)
    if errno is not None:
        return errno in (111, 10061)  # Linux ECONNREFUSED / Windows WSAECONNREFUSED
    reason = getattr(exc, "reason", None)
    if isinstance(reason, Exception):
        return _is_conn_refused(reason)
    return False


def _check_model() -> tuple[bool, str]:
    """BGE-M3 本地权重就位（硬阻塞项）。"""
    model_dir = idx.ROOT_PATH / "model" / "BGE-M3"
    if not model_dir.is_dir():
        return False, f"模型目录缺失：{model_dir}"
    missing = [m for m in _MODEL_MARKERS if not (model_dir / m).exists()]
    if missing:
        return False, f"权重不完整，缺 {missing}（{model_dir}）"
    return True, f"本地权重就位：{model_dir}"


def _check_api_key(env_name: str, purpose: str) -> tuple[bool, str]:
    """API key 是否已配置（软项：缺 key 不阻塞，仅 WARN）。"""
    key = os.environ.get(env_name, "").strip()
    if key:
        return True, f"{env_name} 已配置（{purpose}）"
    return False, f"{env_name} 未配置（{purpose}；缺 key 时对应能力会失败）"


# ── 检查清单（按服务区分）─────────────────────────────────────────────


def _iter_checks(service: str):
    """按服务产出 (name, is_hard, check_fn) 序列。

    service: "mcp" | "ui"
    is_hard=True 的项失败会中止启动（FAIL）；False 的项失败仅 WARN。
    """
    yield "Qdrant 服务端", True, _check_qdrant
    yield "BGE-M3 模型权重", True, _check_model

    # 重排 key：E3 云端重排依赖（软项，两服务都用 E3 默认链路）
    rr = (idx.load_config().get("rerank") or {})
    rerank_env = str(rr.get("api_key_env", "SILICONFLOW_API_KEY"))
    yield (
        f"重排 API key（{rerank_env}）",
        False,
        lambda: _check_api_key(rerank_env, "E3 云端重排"),
    )

    # LLM key：仅 UI 问答生成需要（MCP 不调 LLM，不查）
    if service == "ui":
        llm = (idx.load_config().get("llm") or {})
        llm_env = str(llm.get("api_key_env", "GLM_API_KEY"))
        yield (
            f"LLM API key（{llm_env}）",
            False,
            lambda: _check_api_key(llm_env, "问答生成"),
        )


def run_checks(service: str, out=None) -> bool:
    """执行环境检查，逐项打印，返回「是否全部硬项通过」。

    service: "mcp" | "ui"。
    out: 可选输出回调（print 风格，默认 print）；便于调用方统一走 logger。
    返回 True=可继续启动；False=存在硬阻塞项，调用方应中止（sys.exit 非零）。

    失败/警告时同步写 error/warning 日志（logger.error / logger.warning）。
    """
    import logging

    logger = logging.getLogger("env_check")
    emit = out or print

    emit("=" * 64)
    emit(f"环境检查（{service}）")
    emit("=" * 64)

    all_hard_ok = True
    for name, is_hard, fn in _iter_checks(service):
        try:
            ok, msg = fn()
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        if ok:
            emit(f"{PASS} {name}: {msg}")
        elif is_hard:
            all_hard_ok = False
            emit(f"{FAIL} {name}: {msg}")
            logger.error("[环境检查失败] %s: %s", name, msg)
        else:
            emit(f"{WARN} {name}: {msg}")
            logger.warning("[环境检查警告] %s: %s", name, msg)

    emit("=" * 64)
    if all_hard_ok:
        emit("环境检查：硬项全部通过 ✓")
    else:
        emit("环境检查：存在未就绪的必需项 ✗（请按上方提示修复后重试）")
    emit("=" * 64)
    return all_hard_ok


if __name__ == "__main__":
    # 独立运行：默认按 ui 全量检查（开发自检）
    import retrieve  # noqa: F401  触发 _load_dotenv 兜底

    retrieve._load_dotenv()
    ok = run_checks("ui")
    raise SystemExit(0 if ok else 1)
