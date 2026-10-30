# -*- coding: utf-8 -*-
"""混合检索：输入问题 → 返回检索结果，就这一件事。

链路（本地召回 + 云端重排，不含 LLM 生成）：
  1. BGE-M3 encode 出 dense 向量；jieba 分词填 BM25 查询向量
     （命中词权重一律 1.0——文档侧存 tf 饱和值，服务端 Modifier.IDF 打分时
     逐词乘当前 idf，得分即 Σ BM25(t,d)；v1.6 口径，见 index_documents 定版注释。
     查询端逻辑与 v1.3 完全相同，勿在查询侧自行乘 idf 造成双重加权）
  2. Qdrant 命名向量按实验组召回（各 top_k_per_route）：
       E0 纯 dense / E1 纯 bm25 / E2、E3 双路 / E4 稀疏路换 bge_sparse
     （E4 注意：v1.6 起 bm25 挂服务端 IDF、bge_sparse 不挂，两稀疏路不再
      同构，对照解读需计入 idf 加权差异，见 architecture.md ED9 v1.6 注）
     payload 过滤（股票代码/年份区间）经 query_filter 同时下推到每一路，
     与 RRF 融合正交可叠加（ED7）。
  3. RRF 融合（Σ 1/(rrf_k + rank)，多路命中贡献相加）排序
  4. 重排（默认仅 E3；其余组 rerank=None 不重排，显式 rerank=True 可强制
     开——单路组除外）：融合序前 rerank_top_n 走 OpenAI 协议调云端重排
     （config.rerank，默认硅基流动 bge-reranker-v2-m3：OpenAI SDK
     client.post("/rerank")，与 llm 段同款 base_url/api_key_env 客户端管理），
     按 relevance_score 降序取 final_top_n。重排失败直接抛 RuntimeError
     （用户定版：不重试、不做 RRF 降级，防止带病结果）；config.rerank.enabled
     =false 或缺省段可整体跳过重排（维持 RRF 序）；单路实验组（E0/E1/E4）
     不重排——对照组的变量是召回路本身。

用法（项目根目录，py312 环境；需先完成 M2b 全量入库）：
  python src/retrieve.py --query "平安银行2023年末不良贷款率是多少？"
  python src/retrieve.py --query "..." --exp E0            # 纯 dense 对照
  python src/retrieve.py --query "..." --no-rerank         # E3 去 rerank
  python src/retrieve.py --query "..." --codes 000001 --year-from 2023
  python src/retrieve.py --interactive                     # 循环提问

代码调用：
  import retrieve
  with retrieve.Retriever() as r:
      hits = r.retrieve("问题")                # E3 默认：混合 + 重排
      flt = retrieve.build_filter(codes=["000001"], year_from=2023)
      hits = r.retrieve("问题", exp="E2", flt=flt, rerank=False)
      r.last_timings_ms                        # 最近一次各阶段毫秒耗时
      r.n_points                               # collection 点数

.env 兜底：config/.env 的 key 经 _load_dotenv() 注入（系统环境变量优先，
setdefault 语义）；本文件 main() 与 scripts/start_ui.py / llm_client.py 启动时均已调用。

导入顺序硬约束：本模块最先导入 index_documents（其内部保证 FlagEmbedding 先
于 hashlib/numpy/jieba 加载，否则 Windows segfault，2026-09-15 实测）。
凡下游要用 numpy/jieba/FlagEmbedding，一律先 import 本模块。
"""

from __future__ import annotations

import os

# 离线加载本地权重（model/BGE-M3），与 index_documents 一致
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 必须最先！勿把后面的 import 挪到它之前（见模块 docstring 导入顺序硬约束）
import index_documents as idx  # noqa: E402

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

from qdrant_client import models as m  # noqa: E402 （idx 链已安全加载）

logger = logging.getLogger(__name__)

# 实验组 → 参与召回/融合的命名向量（E3 = E2 + 重排，召回路相同）
EXP_ROUTES: dict[str, tuple[str, ...]] = {
    "E0": (idx.VECTOR_DENSE,),
    "E1": (idx.VECTOR_BM25,),
    "E2": (idx.VECTOR_DENSE, idx.VECTOR_BM25),
    "E3": (idx.VECTOR_DENSE, idx.VECTOR_BM25),
    "E4": (idx.VECTOR_DENSE, idx.VECTOR_BGE_SPARSE),
}

_SOURCE_KEYS = (
    "company_name",
    "stock_code",
    "fiscal_year",
    "page_num",
    "chunk_type",
)


def _load_dotenv() -> None:
    """config/.env 与 Machine 级注册表环境变量兜底（进程环境优先，setdefault）。

    WorkBuddy/IDE 会话常不继承系统环境变量（如 SILICONFLOW_API_KEY/
    GLM_API_KEY 写在 Machine 段），开发/评测脚本统一经本函数兜底：
    ① config/.env（项目内，开源时用户自填）；② Windows Machine 注册表
    （HKLM ...\\Session Manager\\Environment）。均只 setdefault，不覆盖。
    """
    for k, v in _iter_env_sources():
        if v:
            os.environ.setdefault(k, v)


def _iter_env_sources():
    """产出 (key, value)：先 .env 文件，后 Machine 注册表（仅缺失项有意义）。"""
    env_path = idx.ROOT_PATH / "config" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            yield k.strip(), v.strip()
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ) as key:
                for i in range(winreg.QueryInfoKey(key)[1]):
                    try:
                        name, val, _ = winreg.EnumValue(key, i)
                        yield name, str(val)
                    except OSError:
                        break
        except OSError:
            pass


def build_filter(
    codes: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> m.Filter | None:
    """payload 过滤条件 → Qdrant Filter（ED7：与 RRF 正交，各路同时下推）。

    codes：股票代码列表（MatchAny 多值 OR）；year_from/year_to：年份闭区间。
    全空返回 None（不过滤）。
    """
    must: list[m.FieldCondition] = []
    if codes:
        must.append(
            m.FieldCondition(
                key="stock_code",
                match=m.MatchAny(any=[str(c) for c in codes]),
            )
        )
    if year_from is not None or year_to is not None:
        must.append(
            m.FieldCondition(
                key="fiscal_year",
                range=m.Range(
                    gte=int(year_from) if year_from is not None else None,
                    lte=int(year_to) if year_to is not None else None,
                ),
            )
        )
    if not must:
        return None
    return m.Filter(must=must)


class Retriever:
    """持 Qdrant 服务端连接与 BGE-M3 模型，retrieve(query) 出最终命中。

    生命周期：with Retriever() as r: r.retrieve(q)
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or idx.load_config()
        rt = self.cfg["retrieval"]
        self.top_k_per_route = int(rt.get("top_k_per_route", 50))
        self.rrf_k = int(rt.get("rrf_k", 60))
        self.rerank_top_n = int(rt.get("rerank_top_n", 25))
        self.final_top_n = int(rt.get("final_top_n", 5))

        # 重排配置（缺省段或 enabled=false 时跳过重排走 RRF 序）
        rr = self.cfg.get("rerank") or {}
        self._rerank_cfg = rr
        self._rerank_enabled = bool(rr) and bool(rr.get("enabled", True))
        self._client = None  # OpenAI 客户端惰性创建（与 llm_client 同款管理）

        # 最近一次 retrieve 的各阶段毫秒耗时（encode/recall/rerank）
        self.last_timings_ms: dict = {}

        self.client = idx.get_qdrant_client(self.cfg)
        if not self.client.collection_exists(idx.COLLECTION_NAME):
            raise RuntimeError(
                f"collection {idx.COLLECTION_NAME} 不存在，请先完成 M2b 入库"
                "（python src/index_documents.py，并确认 Qdrant 容器在运行）"
            )
        self.n_points = self.client.count(
            idx.COLLECTION_NAME, exact=True
        ).count
        self.model = idx.get_m3_model(self.cfg["embedding"])

    # ── 查询编码：dense + bm25 + bge_sparse 三个查询向量 ────────

    def _encode(self, query: str) -> dict:
        """BGE-M3 一次 encode 出 dense + learned sparse；jieba 出 bm25。

        bm25 查询向量命中词权重一律 1.0（idf 服务端乘）；bge_sparse 查询
        向量取模型 lexical_weights 的正值项（E4 用）。
        """
        out = self.model.encode(
            [query], return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        dense = out["dense_vecs"][0].astype(float).tolist()  # type: ignore
        toks = set(idx.tokenize(query))
        bm25 = m.SparseVector(
            indices=[idx.term_dim(t) for t in toks], values=[1.0] * len(toks)
        )
        lw = out["lexical_weights"][0]
        items = sorted(
            (t, float(w)) for t, w in lw.items() if float(w) > 0.0
        )  # type: ignore
        bge_sp = m.SparseVector(
            indices=[idx.term_dim(t) for t, _ in items],
            values=[w for _, w in items],
        )
        return {
            idx.VECTOR_DENSE: dense,
            idx.VECTOR_BM25: bm25,
            idx.VECTOR_BGE_SPARSE: bge_sp,
        }

    # ── 单路召回（filter 与该路下推，ED7）───────────────────────

    def _route(
        self, name: str, qvector, flt: m.Filter | None = None
    ) -> list[tuple[str, float, dict]]:
        """一路 top_k，返回 (point_id, score, payload)。"""
        points = self.client.query_points(
            idx.COLLECTION_NAME,
            using=name,
            query=qvector,
            limit=self.top_k_per_route,
            query_filter=flt,
            with_payload=True,
        ).points
        return [(str(p.id), p.score, p.payload or {}) for p in points]

    # ── 云端重排：OpenAI 协议 POST /rerank（硅基流动 bge-reranker-v2-m3）──

    def _rerank(self, query: str, cands: list[dict]) -> list[dict]:
        """经 OpenAI SDK 调 rerank：client.post("{base_url}/rerank")。

        与 llm_client 生成侧同款客户端管理（base_url + api_key_env，
        key 不落盘）。返回按 relevance_score 降序的新列表，每项附
        "rerank_score"。失败直接抛 RuntimeError，不重试、不做 RRF 降级
        （用户定版）。
        """
        rr = self._rerank_cfg
        key_env = str(rr.get("api_key_env", "SILICONFLOW_API_KEY"))
        api_key = os.environ.get(key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"环境变量 {key_env} 未设置，无法调用重排 API")

        if self._client is None:  # 惰性创建，生命周期同 Retriever
            from openai import OpenAI

            self._client = OpenAI(
                base_url=rr["base_url"],
                api_key=api_key,
                timeout=60.0,
                max_retries=0,  # 失败立即抛错（用户定版：不重试不降级）
            )
        payload = {
            "model": rr["model"],
            "query": query,
            "documents": [c["text"] for c in cands],
            "top_n": len(cands),  # 全量回传打分，截断留给 retrieve()
            "return_documents": False,
        }
        try:
            # rerank 非 OpenAI 原生端点，用 SDK 的通用 HTTP 方法透传。
            # cast_to 必须写 dict[str, Any]：openai SDK ≥3.x 的 construct_type
            # 对裸 dict 走 get_args 解包会抛 ValueError（2026-10-17 实测 3.3.0），
            # 参数化泛型新旧版本均兼容
            resp = self._client.post(
                "/rerank", body=payload, cast_to=dict[str, Any]
            )
            results = resp["results"]
        except Exception as exc:  # OpenAIError/网络/响应结构异常统一带上下文抛出
            raise RuntimeError(
                f"重排 API 调用失败（{rr['base_url']}/rerank）：{exc}"
            ) from exc
        if not results:
            raise RuntimeError(f"重排 API 返回空 results：{str(resp)[:200]}")
        out: list[dict] = []
        for item in results:  # API 保证按 relevance_score 降序
            c = dict(cands[int(item["index"])])
            c["rerank_score"] = float(item["relevance_score"])
            out.append(c)
        return out

    # ── 主入口：按实验组召回 + RRF 融合 +（E3）云端重排 ────────

    def retrieve(
        self,
        query: str,
        exp: str = "E3",
        flt: m.Filter | None = None,
        rerank: bool | None = None,
    ) -> list[dict]:
        """混合检索一步到位，返回 final_top_n 个命中。

        参数：
          query  ：查询语句
          exp    ：实验组 E0/E1/E2/E3/E4（见 EXP_ROUTES；E3=E2+重排，其余组默认不重排）
          flt    ：payload 过滤（build_filter() 产物，None=不过滤）
          rerank ：None=按实验组默认（E3 开、其余关，且要求 config.rerank.enabled）；
                   True 强制开（单路组忽略）；False 强制关（对照组用）

        返回 list[dict]，每项：{"point_id", "payload"（11 字段原样）, "text",
        "source"（公司/代码/年度/页码/块类型）, "rrf_score", "score_dense",
        "score_sparse", "rank_dense", "rank_sparse", …}；重排开启时另附
        "rerank_score", "rerank_rank" 且按 rerank_score 降序，否则按 rrf_score
        降序。各阶段耗时（ms）写入 self.last_timings_ms（encode/recall/rerank）。
        """
        t_total = time.perf_counter()
        if exp not in EXP_ROUTES:
            raise ValueError(f"未知实验组 {exp}，可选：{'/'.join(EXP_ROUTES)}")
        routes = EXP_ROUTES[exp]
        # 重排默认按实验组定（架构 §7/§10 口径）：仅 E3 完整链路走重排，
        # E0/E1/E2/E4 是对照组、默认不重排；rerank 参数可显式覆盖
        # （True 强制开——单路组除外；False 强制关）
        do_rerank = self._rerank_enabled and exp == "E3"
        if rerank is not None:
            do_rerank = bool(rerank)
        if len(routes) < 2:
            do_rerank = False  # 单路对照组不重排（变量是召回路本身）

        t0 = time.perf_counter()
        qvs = self._encode(query)
        t_encode = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        merged: dict[str, dict] = {}
        # 逐路独立计分：同一 chunk 多路命中时，各自的 rank 各贡献一次 RRF 分
        for name in routes:
            for rank, (pid, score, payload) in enumerate(
                self._route(name, qvs[name], flt), 1
            ):
                cur = merged.get(pid)
                if cur is None:
                    cur = merged[pid] = {
                        "point_id": pid,
                        "payload": payload,
                        "text": payload.get("text", ""),
                        "source": {k: payload.get(k) for k in _SOURCE_KEYS},
                        "rrf_score": 0.0,
                        "score_dense": None,
                        "score_sparse": None,
                        "rank_dense": None,
                        "rank_sparse": None,
                    }
                if name == idx.VECTOR_DENSE:
                    cur["score_dense"], cur["rank_dense"] = score, rank
                else:  # bm25 / bge_sparse 都记入稀疏通用位（对照见 EXP_ROUTES）
                    cur["score_sparse"], cur["rank_sparse"] = score, rank
                cur["rrf_score"] += 1.0 / (self.rrf_k + rank)
        ranked = sorted(merged.values(), key=lambda c: -c["rrf_score"])
        t_recall = (time.perf_counter() - t0) * 1000

        if not do_rerank:
            self.last_timings_ms = {
                "encode": round(t_encode, 1),
                "recall": round(t_recall, 1),  # 含两路召回 + RRF 融合排序
                "rerank": "off",
            }
            return ranked[: self.final_top_n]

        t0 = time.perf_counter()
        reranked = self._rerank(query, ranked[: self.rerank_top_n])
        for rank, c in enumerate(reranked, 1):
            c["rerank_rank"] = rank
        t_rerank = (time.perf_counter() - t0) * 1000
        self.last_timings_ms = {
            "encode": round(t_encode, 1),
            "recall": round(t_recall, 1),
            "rerank": round(t_rerank, 1),
            "total": round((time.perf_counter() - t_total) * 1000, 1),
        }
        return reranked[: self.final_top_n]

    def close(self) -> None:
        self.client.close()  # 释放 HTTP 连接池（服务端模式，同 index_documents）
        if self._client is not None:
            self._client.close()  # 释放 OpenAI SDK 底层 httpx 连接池

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ── CLI ─────────────────────────────────────────────────────────


def _print_hits(query: str, hits: list[dict], elapsed: float, exp: str) -> None:
    reranked = bool(hits) and hits[0].get("rerank_score") is not None
    print(
        f"\n查询：{query}    命中 {len(hits)} 条（{elapsed:.2f}s，路线 {exp}，"
        f"{'rerank' if reranked else 'RRF（未重排）'}序）"
    )
    for i, h in enumerate(hits, 1):
        s = h["source"]
        rr = (
            f"  rerank={h['rerank_score']:.4f} (rank {h['rerank_rank']})"
            if h.get("rerank_score") is not None
            else ""
        )
        print(
            f"\n[{i}] rrf={h['rrf_score']:.5f}{rr}  "
            f"dense={h['score_dense']} (rank {h['rank_dense']})  "
            f"sparse={h['score_sparse']} (rank {h['rank_sparse']})"
        )
        print(
            f"    {s['company_name']} {s['fiscal_year']}年报 "
            f"p{s['page_num']} [{s['chunk_type']}] code={s['stock_code']}"
        )
        text = h["text"].replace("\n", " ")
        print(f"    {text[:120]}{'…' if len(text) > 120 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="混合检索（BGE-M3 dense + BM25，RRF 融合 + 云端重排）"
    )
    ap.add_argument("--query", "-q", default=None, help="查询语句")
    ap.add_argument("--interactive", action="store_true", help="循环提问模式")
    ap.add_argument(
        "--exp",
        default="E3",
        choices=list(EXP_ROUTES),
        help="实验组：E0 纯dense / E1 纯bm25 / E2 混合无重排 / E3 完整 / E4 learned sparse（默认 E3）",
    )
    ap.add_argument(
        "--no-rerank", action="store_true", help="强制跳过重排（走 RRF 序）"
    )
    ap.add_argument("--codes", nargs="*", default=None, help="限定股票代码（可多个）")
    ap.add_argument("--year-from", type=int, default=None, help="年份区间起（含）")
    ap.add_argument("--year-to", type=int, default=None, help="年份区间止（含）")
    args = ap.parse_args()
    if not (args.query or args.interactive):
        ap.error("需要 --query '...' 或 --interactive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    import jieba  # 经 idx 链已安全加载，这里只是降噪

    jieba.setLogLevel(logging.WARNING)
    _load_dotenv()  # config/.env 兜底（系统环境变量优先）

    flt = build_filter(
        codes=args.codes,
        year_from=args.year_from,
        year_to=args.year_to,
    )
    with Retriever() as r:

        def run_once(q: str) -> None:
            t0 = time.time()
            hits = r.retrieve(
                q, exp=args.exp, flt=flt, rerank=False if args.no_rerank else None
            )
            _print_hits(q, hits, time.time() - t0, args.exp)
            logger.info("timings_ms=%s", r.last_timings_ms)

        if args.query:
            run_once(args.query)
        while args.interactive:
            try:
                q = input("\n查询> ").strip()
                if q:
                    run_once(q)
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()
