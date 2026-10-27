# -*- coding: utf-8 -*-
"""切块产物向量化入库（M2，架构 v1.3 §2.1 离线索引链路 / ED6 / ED9）。

单文件两阶段（顺序执行，各自可断点续跑）：

  A encode  BGE-M3 一次前向同时产出 dense(1024) 与 learned sparse，
            按报告落盘缓存 data/embeddings/<stem>.npz；重跑时缓存命中
            即跳过 GPU（全量 ~4 万 chunk 编码约 20~40min，重跑代价高）。
  B upload  组装 PointStruct（3 命名向量 + 11 payload 字段，§3.1）；
            bm25 向量客户端只算 tf 饱和+长度归一值，**无任何全库统计
            环节**（v1.6 口径，见下），阶段 B 按 --codes 只处理所选报告
            → 增量入库天然成立；uuid5(NAMESPACE_URL, chunk_id) 作 point
            id → 幂等重入，批量 upsert 到 Qdrant 服务端（Docker 容器，
            config.yaml qdrant 段）。

关键口径（勿改，改了检索端必须全量重建）：
  - BM25 分工（v1.6 修订，取代 v1.3 的"权重全在客户端算完"）：df/idf 这类
    随语料变化的全局统计交服务端——collection 建法为 bm25 命名稀疏向量挂
    Modifier.IDF（Qdrant ≥1.10 特性，容器 1.19.1），打分时服务端对查询向量
    逐维乘 idf = ln(1+(N-df+0.5)/(df+0.5))，与旧客户端 bm25_idf() 同式；
    N/df 由 Qdrant 随点增删自动维护 → **新增/替换文档零全局重算**。
    客户端入库只存 tf 饱和+长度归一部分 tf·(k1+1)/(tf+k1·(1-b+b·dl/avgdl))，
    avgdl 取冻结常量 sparse.avgdl_frozen（慢变量，出处与漂移对策见
    config.yaml 注释）；查询向量每命中词仍填 1.0 → 服务端得分即 Σ BM25(t,d)。
    已知滞后：删点/覆盖 upsert 后相关词 df 偏计，等 segment optimizer 合并
    自愈（qdrant/qdrant#6690、#6735）；千级/4万的小规模替换影响可忽略
    （RRF 只看名次），大规模重灌后介意就跑一次 --recreate --skip-encode。
    （v1.3 旧注"服务端没有 IDF 参数"系对 1.10 前行为的判断，已按实机版本修订。）
  - 词 → 维度：md5(term)[:4] 大端转 u32。bm25 与 bge_sparse 共用此映射，
    每份报告的 npz 可独立生成、无全局词表依赖（词表 ~60 万级，
    2^32 空间期望碰撞 ~40 对，对打分影响可忽略，v1.3 定版接受；v1.6 起
    服务端按维度计 df，碰撞词共享 posting list，量级同上不构成新风险）。
  - bge_sparse 不挂 Modifier.IDF（存原始 relu 权重，纯点积）——E4 换
    稀疏路时行为可预期；如需 IDF 归一须连同查询端一起改，另行实验。
    **v1.6 注**：bm25 路挂 IDF 后两稀疏路不再同构，恢复 E4 对照时需把
    "idf 加权有无"计入变量差异（见 architecture.md ED9 v1.6 注）。

导入顺序硬约束：`FlagEmbedding` 必须紧随标准库 os 之后**最先**导入，
hashlib / numpy / jieba / yaml / qdrant 全部排在它之后，否则 Windows import
阶段即 access-violation segfault（0xC0000005，脚本进不了 main、日志全空）。
真因是 torch 原生库与 OpenSSL(_hashlib)、MKL(numpy) 的 DLL 冲突——旧记录只写
"先于 jieba"，实测先 import hashlib 或 numpy 同样崩（2026-09-15 faulthandler
二分定位）。本文件已固化安全顺序；retrieve.py 等下游凡要用这三者，一律
`import index_documents`（导入本模块即完成安全排序），切勿在其之前抢先
import hashlib / numpy。

用法（项目根目录，py312 环境）：
  python src/index_documents.py                  # 全量两阶段
  python src/index_documents.py --codes 600519   # 指定公司（小样本/增量入库）
  python src/index_documents.py --limit 2        # 只处理前 N 份
  python src/index_documents.py --recreate       # 删旧 collection 重建
  python src/index_documents.py --replace        # 整份替换：入库前先删所选报告旧点
  python src/index_documents.py --skip-encode    # 跳过阶段 A（要求 npz 缓存齐全）
  # 新增一份年报的日常操作：切块产物放 data/chunks/ 后
  #   python src/index_documents.py --codes <code> --limit 1
  # 某份报告重新切块要替换旧的：上面命令加 --replace
输出：data/embeddings/*.npz、Qdrant 服务端 collection: annual_reports
（v1.6 起不再产出 bm25_stats.json；data/bm25_stats/ 下旧表为历史产物，
  代码不再读写）
"""

from __future__ import annotations

import os

# 离线加载本地权重（model/BGE-M3），阻止 FlagEmbedding 向 HF 补拉文件
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── 导入顺序硬约束（2026-09-15 faulthandler 二分实测）────────────────
# torch 加载的原生库与 OpenSSL(libcrypto)、MKL 存在 DLL 冲突：**任何预加载
# 了 hashlib（_hashlib→OpenSSL）或 numpy（MKL）的代码先于 FlagEmbedding 导入，
# Windows 下 import 阶段即 access-violation segfault（0xC0000005，连 main 都
# 进不去、日志全空）**。实测矩阵：
#   hashlib→FlagEmbedding ✗   numpy→FlagEmbedding ✗   jieba→FlagEmbedding ✗
#   FlagEmbedding→hashlib ✓   FlagEmbedding→numpy→yaml→jieba→qdrant ✓
# 唯一安全写法：紧随标准库 os（不碰原生 DLL）之后**最先**导入 FlagEmbedding，
# 其余第三方与用到 hashlib/numpy 的标准库全部排在它后面。retrieve.py 等下游
# 复用本模块 tokenize/term_dim 时 `import index_documents` 即自动完成安全排序。
from FlagEmbedding import BGEM3FlagModel  # noqa: E402  isort: skip  # 必须最先！

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402  isort: skip
import yaml  # noqa: E402  isort: skip
import jieba  # noqa: E402  isort: skip
from qdrant_client import QdrantClient, models  # noqa: E402  isort: skip
from utils import Log

# logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_PATH / "config" / "config.yaml"
LOG_DIR = ROOT_PATH / "log"

file_name = Path(__file__).stem
logger = Log.init_logger(
    log_path=LOG_DIR / f"{file_name}.log",
    log_level=logging.INFO,
    when="D",
    interval=1,
    backupCount=7,  # 所有轮转文件总数（包含同一天内的多个分片），不是天数
    max_mb=10,  # 单个文件容量上限，单位MB
)

COLLECTION_NAME = "annual_reports"
DENSE_DIM = 1024
VECTOR_DENSE = "dense"
VECTOR_BM25 = "bm25"
VECTOR_BGE_SPARSE = "bge_sparse"
UPLOAD_BATCH = 256

# uuid5 命名空间：point id = uuid5(NAMESPACE_URL, chunk_id)，全局幂等
CHUNK_NS = uuid.NAMESPACE_URL


# ── 通用口径函数（tokenize/term_dim 被 retrieve.py 复用，勿改签名与实现）──


def tokenize(text: str) -> list[str]:
    """jieba search 分词 → 小写、去空白的 token 列表（G2 定版 search）。

    search 模式对财务长词（应收账款周转率）额外产出子词，部分匹配召回更稳；
    检索查询端必须用同一函数，否则词→维度映射对不上。
    """
    return [w.strip().lower() for w in jieba.lcut_for_search(text) if w.strip()]


def term_dim(term: str) -> int:
    """词 → u32 稀疏维度：md5(term) 前 4 字节大端整数（见模块 docstring）。"""
    return int.from_bytes(hashlib.md5(term.encode("utf-8")).digest()[:4], "big")


def bm25_weight(tf: int, dl: int, avgdl: float, k1: float, b: float) -> float:
    """单个 (词, 文档) 的 BM25 tf 饱和 + 长度归一部分，入库时算好（v1.6）。

    不含 idf——那半截由 Qdrant 服务端 Modifier.IDF 在打分时乘上，公式
    ln(1+(N-df+0.5)/(df+0.5)) 与旧客户端实现同式（旧函数已随阶段 B 删除）。
    avgdl 传冻结常量 cfg["sparse"]["avgdl_frozen"]，不再用全库实时统计。
    """
    return (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))


# ── 配置与数据装载 ──────────────────────────────────────────────


def load_config() -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    emb = cfg["embedding"]
    if emb.get("provider", "local") != "local":
        raise NotImplementedError(
            "provider=siliconflow 的云端编码尚未实现（架构 §3 降级开关预留位）"
        )
    return cfg


def get_qdrant_client(cfg: dict) -> QdrantClient:
    """构造 Qdrant 服务端客户端（v1.4：Docker 容器，取代本地模式 path=）。

    全项目唯一入口——retrieve.py / ui.py 一律经此建连，连接参数只从
    config.yaml 的 qdrant 段读取。服务端模式无本地锁问题，close() 保留
    （关 HTTP 连接池，无害）。容器未启动时构造不报错、首个请求才失败，
    故调用方紧跟 collection_exists/count 探活即可。
    """
    q = cfg["qdrant"]
    return QdrantClient(
        host=str(q["host"]),
        port=int(q.get("port", 6333)),
        api_key=q.get("api_key") or None,
        check_compatibility=False,  # 容器 1.19.1 vs 客户端 1.19.0，关版本比对告警
    )


def iter_chunk_files(
    codes: list[str] | None, limit: int | None, chunks_dir: Path
) -> list[Path]:
    """按 --codes/--limit 过滤报告级 chunk 文件（文件名序稳定）。"""
    files = sorted(chunks_dir.glob("*_chunk.json"))
    if codes:
        keep = set(codes)
        files = [f for f in files if f.name.split("_")[0] in keep]
    if limit:
        files = files[:limit]
    return files


def load_chunks(path: Path) -> tuple[list[dict], dict]:
    """读一份 chunk json → (chunks, 报告级 meta)。

    meta：company_name 从文件名推导（<code>_<year>_<公司名>_chunk.json），
    其余取自 chunk metadata（同一份内恒定字段）。
    """
    chunks = json.loads(path.read_text(encoding="utf-8"))
    parts = path.name[: -len("_chunk.json")].split("_")
    if not chunks:
        raise ValueError(f"空 chunk 文件：{path}")
    m = chunks[0]["metadata"]
    meta = {
        "company_name": parts[2] if len(parts) >= 3 else "unknown",
        "stock_code": m["stock_code"],
        "fiscal_year": int(m["year"]),
        "source_file": m["source_file"],
    }
    return chunks, meta


# ── 阶段 A：BGE-M3 编码（dense + learned sparse），npz 缓存 ──────


def get_m3_model(emb_cfg: dict) -> BGEM3FlagModel:
    device = emb_cfg.get("device", "cuda")
    model_path = str(ROOT_PATH / emb_cfg["model"])
    if not Path(model_path).exists():  # 本地目录缺失时退化为 HF id（镜像下载）
        model_path = emb_cfg.get("hf_model_id", "BAAI/bge-m3")
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    logger.info(
        "加载 BGE-M3：%s (device=%s, fp16=%s)",
        model_path,
        device,
        emb_cfg.get("use_fp16", False),
    )
    return BGEM3FlagModel(
        model_path, device=device, use_fp16=bool(emb_cfg.get("use_fp16", False))
    )


def encode_file(
    model: BGEM3FlagModel, chunks: list[dict], npz_path: Path, batch_size: int
) -> None:
    """
    对单份年报下全部chunk执行BGE-M3编码，输出dense向量 + BGE-M3稀疏向量，并压缩保存到npz缓存
    npz内部存储结构（数组顺序必须和chunks顺序严格对齐，用于后续读取校验）：
        dense      [N, 1024] float32：BGE-M3稠密向量，输出已归一化，可直接cosine计算
        sp_idx     flat int64数组：全部chunk的bge_sparse稀疏维度id，一维平铺拼接
        sp_val     flat float32数组：与sp_idx一一对应的稀疏权重值
        sp_offsets [N+1] int64数组：偏移数组，用来切分sp_idx/sp_val，取出单个chunk对应的稀疏向量段
    :param model: 已加载好的BGE-M3模型实例
    :param chunks: 当前年报的文本块列表
    :param npz_path: 向量缓存npz文件目标路径
    :param batch_size: 模型推理batch大小
    """
    # 提取所有chunk的正文文本，按顺序组成文本列表送入模型
    texts = [c["content"] for c in chunks]
    t0 = time.time()
    # BGE-M3一次性前向推理，同时返回dense向量与lexical稀疏权重，不启用colbert向量
    out = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    # 稠密向量转float32 numpy数组
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)

    idx_chunks: list[np.ndarray] = []  # 保存每个chunk稀疏维度数组，后续拼接
    val_chunks: list[np.ndarray] = []  # 保存每个chunk稀疏权重数组，后续拼接
    for lw in out["lexical_weights"]:
        # lexical_weights: dict{token:relu后的权重}, 过滤掉权重≤0的无效词
        items = [(t, w) for t, w in lw.items() if w > 0.0]
        # 对词项排序，保证每次运行维度顺序稳定
        items.sort(key=lambda x: x[0])
        # 将term映射为稀疏向量维度，生成numpy维度数组
        idx_chunks.append(
            np.fromiter(
                (term_dim(t) for t, _ in items), dtype=np.int64, count=len(items)
            )
        )
        # 提取对应的权重值生成numpy权重数组
        val_chunks.append(
            np.fromiter(
                (float(w) for _, w in items), dtype=np.float32, count=len(items)
            )
        )
    # 构造偏移数组：offsets[j] 代表第j个chunk稀疏向量在sp_idx的起始下标
    offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
    # 累加每个chunk稀疏长度，填充offsets[1:]
    np.cumsum([len(a) for a in idx_chunks], out=offsets[1:])

    # 确保缓存目录存在
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    # 先写入临时文件tmp.npz，写入成功后原子替换，防止程序中断生成损坏的npz文件
    tmp = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        dense=dense,
        sp_idx=np.concatenate(idx_chunks) if idx_chunks else np.empty(0, np.int64),
        sp_val=np.concatenate(val_chunks) if val_chunks else np.empty(0, np.float32),
        sp_offsets=offsets,
        n=len(chunks),
    )
    # 原子重命名，覆盖目标npz
    tmp.replace(npz_path)
    logger.info(
        "  编码 %d chunks（dense %s + sparse 均长 %.0f），%.1fs",
        len(chunks),
        dense.shape,
        offsets[-1] / max(len(chunks), 1),
        time.time() - t0,
    )


def stage_encode(files: list[Path], cfg: dict, *, skip: bool) -> None:
    """
    BGE-M3向量编码，生成dense稠密向量 + BGE-M3 learned稀疏向量，落地npz缓存文件。
    支持断点续跑：已存在且chunk数量匹配的缓存可直接跳过GPU编码，节省算力

    :param files: 需要处理的年报chunk文件列表
    :param cfg: 全局配置字典，包含embedding配置、缓存路径
    :param skip: 是否跳过编码阶段（对应命令行--skip-encode），开启后缓存缺失直接抛异常
    """
    # 读取embedding相关配置（模型路径、batch、device等）
    emb_cfg = cfg["embedding"]
    # 向量缓存目录
    cache_dir = ROOT_PATH / cfg["paths"]["embeddings"]
    # BGE-M3模型实例，懒加载：第一次需要编码时才加载，只加载一次
    model = None
    # 记录阶段A起始时间，用于统计总耗时
    t0 = time.time()

    # 遍历每一份年报chunk文件，i为当前序号，从1开始计数
    for i, f in enumerate(files, 1):
        # 读取当前年报的所有文本块chunks + 年报元信息（公司名、股票代码、年份等）
        chunks, meta = load_chunks(f)
        # 定义当前这份年报对应的npz缓存文件路径（文件名与chunk json同名，后缀npz）
        npz_path = cache_dir / f"{f.stem}.npz"
        # 如果缓存文件存在，则加载；否则置为None
        cached = np.load(npz_path) if npz_path.exists() else None

        # 缓存存在，并且缓存内保存的chunk数量和当前文件chunk数量完全一致 → 缓存有效，跳过编码
        if cached is not None and int(cached["n"]) == len(chunks):
            logger.info("(%d/%d) %s 缓存命中，跳过", i, len(files), f.name)
            continue

        # 开启--skip-encode模式，但缓存不存在/数量不匹配，无法继续，抛出异常终止
        if skip:
            raise FileNotFoundError(f"--skip-encode 但缓存缺失/行数不符：{npz_path}")

        # 缓存文件存在但是chunk数量对不上（旧缓存、chunk重新切分过），告警并重新编码
        if cached is not None:
            logger.warning(
                "  %s 缓存行数 %d ≠ chunk 数 %d，重编码",
                f.name,
                int(cached["n"]),
                len(chunks),
            )

        # 懒加载模型：仅第一次需要编码时加载BGE-M3，后续复用同一个模型，避免重复加载
        if model is None:
            model = get_m3_model(emb_cfg)

        # 打印日志：当前正在处理哪一份年报，以及chunk数量
        logger.info(
            "(%d/%d) %s 编码中（%s %s 年报，%d chunks）",
            i,
            len(files),
            f.name,
            meta["company_name"],
            meta["fiscal_year"],
            len(chunks),
        )
        # 调用encode_file：对当前年报所有chunk执行BGE-M3编码，并写入npz缓存
        encode_file(model, chunks, npz_path, int(emb_cfg.get("batch_size", 16)))

    # 全部文件处理完毕，输出阶段A总耗时
    logger.info("阶段A 完成，用时 %.1fs", time.time() - t0)


# ── 阶段 B：组装 points 入库（upload；v1.6 起无全库统计阶段）────────


def replace_report_points(
    client: QdrantClient, meta: dict, log_prefix: str = ""
) -> None:
    """--replace：删掉该报告（stock_code + fiscal_year）的全部旧点。

    整份替换（重新切块后 chunk_id 集合变了）入库前调用，残点不误留。
    注意删点后服务端相关词 df 递减有滞后（见模块 docstring 已知滞后），
    等 segment 优化合并自愈，小规模可忽略。
    """
    res = client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="stock_code",
                        match=models.MatchValue(value=str(meta["stock_code"])),
                    ),
                    models.FieldCondition(
                        key="fiscal_year",
                        match=models.MatchValue(value=int(meta["fiscal_year"])),
                    ),
                ]
            )
        ),
        wait=True,
    )
    logger.info(
        "%s--replace 删旧点 %s/%s → %s",
        log_prefix,
        meta["stock_code"],
        meta["fiscal_year"],
        res,
    )


def chunk_payload(c: dict, meta: dict) -> dict:
    """chunk → Qdrant payload（架构 §3.1 的 11 字段 schema）。"""
    m = c["metadata"]
    return {
        "company_name": meta["company_name"],
        "stock_code": m["stock_code"],
        "fiscal_year": int(m["year"]),
        "section": m["section"],
        "page_num": int(m["page_num"]),
        "chunk_idx": int(c["chunk_id"].rsplit("_", 1)[1]),
        "chunk_type": (m["block_types"] or ["unknown"])[0],
        "chunk_id": c["chunk_id"],
        "source_file": m["source_file"],
        "is_ocr": bool(m["is_ocr"]),
        "text": c["content"],  # M3 组装 Prompt 依赖，正文必须入库
    }


def bm25_sparse_vector(tokens: Counter[str], sparse_cfg: dict) -> models.SparseVector:
    """一个 chunk 的 token 频表 → bm25 文档向量（v1.6：只含 tf 饱和+长度归一值）。

    不再乘 idf——服务端 Modifier.IDF 打分时逐维乘当前 idf，df/N 随点增删
    自动维护，本函数无任何全库统计依赖（只读 config 的 k1/b/avgdl_frozen）。
    """
    k1, b = float(sparse_cfg["k1"]), float(sparse_cfg["b"])
    avgdl = float(sparse_cfg["avgdl_frozen"])
    dl = sum(tokens.values())
    idx, val = [], []
    for t, tf in tokens.items():
        idx.append(term_dim(t))
        val.append(bm25_weight(tf, dl, avgdl, k1, b))
    # hash 碰撞合并（同维相加；服务端按维度计 df，碰撞词共享 posting）
    merged: dict[int, float] = {}
    for i, v in zip(idx, val):
        merged[i] = merged.get(i, 0.0) + v
    return models.SparseVector(
        indices=list(merged.keys()), values=list(merged.values())
    )


def ensure_collection(client: QdrantClient, recreate: bool) -> None:
    """建 collection：1 dense(cosine) + 2 稀疏(dot)，payload 过滤索引。"""
    if recreate and client.collection_exists(COLLECTION_NAME):
        logger.info("--recreate：删除旧 collection %s", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        # 单 segment：服务端 IDF 统计按 segment 计算，多 segment 时 df/N 各算
        # 各的（官方文档明示）；4 万点规模单 segment 无性能代价，换来 idf 全库
        # 口径统一 + 删点后统计修正更及时（旧 collection 是 8 segments，重建一并解决）
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=1),
        vectors_config={
            VECTOR_DENSE: models.VectorParams(
                size=DENSE_DIM, distance=models.Distance.COSINE
            ),
        },
        sparse_vectors_config={
            # v1.6：bm25 挂服务端 IDF——客户端只存 tf 饱和值，df/N 统计由
            # Qdrant 维护、打分时逐维乘 idf（qdrant ≥1.10，容器 1.19.1；
            # 仅建 collection 时生效，改口径须 --recreate 重建）
            # （注：qdrant-client 1.19.0 里 modifier 是 SparseVectorParams 顶层
            #   字段，SparseIndexParams 不接受 modifier，勿嵌套写）
            VECTOR_BM25: models.SparseVectorParams(modifier=models.Modifier.IDF),
            # bge_sparse 存原始 relu 权重，不挂 modifier（E4 口径见模块 docstring）
            VECTOR_BGE_SPARSE: models.SparseVectorParams(),
        },
    )
    # 过滤检索（ED7）用到的 payload 索引
    for field, schema in [
        ("stock_code", models.KeywordIndexParams(type="keyword")),
        ("company_name", models.KeywordIndexParams(type="keyword")),
        ("chunk_type", models.KeywordIndexParams(type="keyword")),
        ("fiscal_year", models.IntegerIndexParams(type="integer")),
        ("page_num", models.IntegerIndexParams(type="integer")),
    ]:
        client.create_payload_index(
            COLLECTION_NAME, field_name=field, field_schema=schema
        )
    logger.info(
        "collection 已创建：%s（%s/%s/%s）",
        COLLECTION_NAME,
        VECTOR_DENSE,
        VECTOR_BM25,
        VECTOR_BGE_SPARSE,
    )


def stage_upload(
    files: list[Path], cfg: dict, *, recreate: bool, replace: bool
) -> None:
    """阶段 B：逐报告组装 points 并 upsert。

    v1.6 起无 stats 参数：bm25 值只依赖 chunk 自身 + config 常量，
    增量入库成立。replace=True 时入库前先删该报告旧点（整份替换）。
    """
    cache_dir = ROOT_PATH / cfg["paths"]["embeddings"]

    client = get_qdrant_client(cfg)
    q = cfg["qdrant"]
    logger.info("连接 Qdrant 服务端 %s:%s ...", q["host"], q.get("port", 6333))
    try:
        ensure_collection(client, recreate)
        total = 0
        t0 = time.time()
        batch: list[models.PointStruct] = []

        def flush() -> None:
            nonlocal batch
            if not batch:
                return
            client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
            batch = []

        for i, f in enumerate(files, 1):
            chunks, meta = load_chunks(f)
            if replace:
                replace_report_points(
                    client, meta, log_prefix=f"({i}/{len(files)}) {f.name} "
                )
            with np.load(cache_dir / f"{f.stem}.npz") as z:
                assert int(z["n"]) == len(chunks), f"缓存与 chunk 错位：{f}"
                dense = z["dense"]
                sp_idx, sp_val = z["sp_idx"], z["sp_val"]
                offsets = z["sp_offsets"]

            for j, c in enumerate(chunks):
                pid = str(uuid.uuid5(CHUNK_NS, c["chunk_id"]))
                lo, hi = int(offsets[j]), int(offsets[j + 1])
                bge_sparse = models.SparseVector(
                    indices=sp_idx[lo:hi].tolist(),
                    values=sp_val[lo:hi].astype(float).tolist(),
                )
                tokens = Counter(tokenize(c["content"]))
                batch.append(
                    models.PointStruct(
                        id=pid,
                        vector={
                            VECTOR_DENSE: dense[j].astype(float).tolist(),
                            VECTOR_BM25: bm25_sparse_vector(tokens, cfg["sparse"]),
                            VECTOR_BGE_SPARSE: bge_sparse,
                        },
                        payload=chunk_payload(c, meta),
                    )
                )
                if len(batch) >= UPLOAD_BATCH:
                    flush()
            total += len(chunks)
            have = client.count(COLLECTION_NAME, exact=True).count
            logger.info(
                "(%d/%d) %s 入库完成（累计 %d chunks / collection %d 点）",
                i,
                len(files),
                f.name,
                total,
                have,
            )
        flush()
        logger.info("阶段B（upload）完成：%d 点，用时 %.1fs", total, time.time() - t0)
    finally:
        client.close()  # 释放 HTTP 连接池（服务端模式无本地锁问题，习惯保留）


# ── 入口 ───────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="切块产物向量化入库（M2）")
    ap.add_argument("--codes", nargs="*", default=None, help="只处理指定股票代码")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 份（小样本）")
    ap.add_argument("--recreate", action="store_true", help="删除并重建 collection")
    ap.add_argument(
        "--replace",
        action="store_true",
        help="整份替换：每个所选报告入库前按 stock_code+fiscal_year 先删旧点"
        "（重新切块后 chunk 集合变化时用；纯新增报告不需要）",
    )
    ap.add_argument(
        "--skip-encode",
        action="store_true",
        help="跳过阶段A（要求缓存齐全，缺失即报错）",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # jieba 词典预热，避免并发首次加载告警（单进程顺序跑，纯降噪）
    jieba.setLogLevel(logging.WARNING)

    # 加载配置文件
    cfg = load_config()

    # 按照过滤条件筛选要处理的文本块，如果过滤条件为空，则返回所有
    files = iter_chunk_files(args.codes, args.limit, ROOT_PATH / cfg["paths"]["chunks"])

    if not files:
        logger.error("没有可入库的 chunk 文件（检查 --codes/--limit 与 data/chunks/）")
        return

    # 计算文件和文本块数量
    total_chunks = 0
    for f in files:
        chunks, _ = load_chunks(f)
        total_chunks += len(chunks)
    logger.info(
        "目标 %d 份 / %d chunks | batch_size=%s device=%s",
        len(files),
        total_chunks,
        cfg["embedding"].get("batch_size"),
        cfg["embedding"].get("device"),
    )

    stage_encode(files, cfg, skip=args.skip_encode)
    # v1.6：原阶段 B（全库 BM25 统计）已删——idf 由 Qdrant 服务端 Modifier.IDF 维护
    stage_upload(files, cfg, recreate=args.recreate, replace=args.replace)
    logger.info("M2 索引全部完成 ✅")


if __name__ == "__main__":
    main()
