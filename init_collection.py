# -*- coding: utf-8 -*-
"""独立建表脚本：在已启动的 Qdrant 中创建 collection 及三命名向量 schema。

与 index_documents.py 内嵌的 ensure_collection() 逻辑完全一致，抽出来是为了
「只建表、不灌数据」可独立复现——便于开源用户先验证环境（Qdrant 容器 + 客户端
连通 + schema 正确），再跑真正的索引入库。

关键口径（与 index_documents.py 保持同步，勿单方面改动）：
  - collection 名：annual_reports
  - 三命名向量：
      dense      → 1024 维稠密，COSINE（BGE-M3）
      bm25       → 稀疏，Modifier.IDF（服务端维护 df/idf，Qdrant ≥1.10）
      bge_sparse → 稀疏，无 modifier（存原始 relu 权重，纯点积）
  - 单 segment（default_segment_number=1）：服务端 IDF 按 segment 统计，
    单 segment 保证 df/N 全库口径统一（4 万点规模无性能代价）。
  - payload 过滤索引：stock_code/company_name/chunk_type（keyword）、
    fiscal_year/page_num（integer），供检索过滤（ED7）用。

用法（项目根目录）：
  python init_collection.py            # 若已存在则跳过（幂等）
  python init_collection.py --recreate # 删除重建（清空数据，慎用）

注意：本脚本只依赖 qdrant-client + pyyaml，**不 import FlagEmbedding/torch**，
因此不受「FlagEmbedding 必须最先 import」的 DLL 顺序约束，可独立快速运行。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

COLLECTION_NAME = "annual_reports"
DENSE_DIM = 1024
VECTOR_DENSE = "dense"
VECTOR_BM25 = "bm25"
VECTOR_BGE_SPARSE = "bge_sparse"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def get_client(cfg: dict) -> QdrantClient:
    q = cfg["qdrant"]
    return QdrantClient(
        host=str(q["host"]),
        port=int(q.get("port", 6333)),
        api_key=q.get("api_key") or None,
        check_compatibility=False,
    )


def create_collection(client: QdrantClient, recreate: bool) -> None:
    if recreate and client.collection_exists(COLLECTION_NAME):
        print(f"--recreate：删除旧 collection {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
    if client.collection_exists(COLLECTION_NAME):
        print(f"collection 已存在，跳过：{COLLECTION_NAME}")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=1),
        vectors_config={
            VECTOR_DENSE: models.VectorParams(
                size=DENSE_DIM, distance=models.Distance.COSINE
            ),
        },
        sparse_vectors_config={
            VECTOR_BM25: models.SparseVectorParams(modifier=models.Modifier.IDF),
            VECTOR_BGE_SPARSE: models.SparseVectorParams(),
        },
    )

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

    print(
        f"collection 已创建：{COLLECTION_NAME} "
        f"({VECTOR_DENSE}/{VECTOR_BM25}/{VECTOR_BGE_SPARSE})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="在 Qdrant 中创建 collection（建表）")
    ap.add_argument("--recreate", action="store_true", help="删除并重建 collection")
    args = ap.parse_args()

    cfg = load_config()
    client = get_client(cfg)
    try:
        # 探活：容器未启动时首个请求才失败，这里主动 ping
        if not client.collection_exists(COLLECTION_NAME) and not args.recreate:
            pass  # 不存在的预期路径，继续建表
        create_collection(client, recreate=args.recreate)
        info = client.get_collection(COLLECTION_NAME)
        print(
            f"确认：points={info.points_count}，向量={list(info.config.params.vectors or {}) + list(info.config.params.sparse_vectors or {})}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 建表失败：{exc}", file=sys.stderr)
        print(
            "       请确认 Qdrant 容器已启动：docker compose up -d",
            file=sys.stderr,
        )
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
