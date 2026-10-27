# -*- coding: utf-8 -*-
"""M4 评测：出题 → 跑批 → 打分，三个子命令一个文件（架构 §7，v1.8 口径）。

链路：
  gen    分层抽样 chunk（公司×年份×块类型，seed 可复现）→ GLM（llm 段）逐块
         出题 {question, reference_answer, source_chunk_id, qtype}，题型配额
         锁死 术语18/改写18/数字14 → data/eval/eval_set.json
  run    逐题走 E3 完整链路（检索+重排；重排失败硬抛中断，不许带病入表——
         rerank=None 按 v1.7 实验组默认，E3 含重排），生成走 GLM 失败不阻塞
         （答案为空仅影响答案侧参考，检索指标不受影响），存 contexts/answer/
         latency/answer_llm_ms → data/eval/run_results.json（断点续跑：已跑
         的 question 跳过）
  score  自写五指标（金标准 = 出题源 chunk，唯一命中口径）：
           Recall@5     top5 是否含源 chunk
           Precision@5  命中数/5（合成题金标仅 1 条 → 0 或 0.2）
           HitRate@5    同 Recall@5（单金标时数值等价，报告并列呈现）
           MRR          1/首次命中排名（未命中 0）
           NDCG@5       DCG=1/log2(rank+1) / IDCG=1（单金标折减）
         + 答案侧参考：裁判（judge 段 DeepSeek）对 answer×reference_answer
         5 分制评分（1=完全错 … 5=完全正确），仅作参考不进主表
         按题型分层 + 最差 10 题（按 MRR 升序）归因 → data/eval/report.md

诚实声明（写进报告）：合成题天然贴近源 chunk，指标度量"找回出题素材的能力"
而非"找回全部真证据的能力"；本评测在无元数据过滤的裸查询下测量。

用法（项目根目录，py312 环境；Qdrant 容器在跑）：
  python src/evaluate.py gen
  python src/evaluate.py run
  python src/evaluate.py score
  python src/evaluate.py gen --n 3    # 小样本调试

依赖：无 ragas/langchain——指标全部自写；LLM 走 OpenAI SDK（llm/judge 段）。
金标准口径（2026-10-25 用户定版）：源 chunk 作唯一金标，邻块/同章节不算命中。

导入顺序硬约束：同 retrieve.py，本模块最先导入 index_documents（Windows
DLL segfault 防护），勿在其之前 import 任何第三方库。
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 必须最先（内部保证 FlagEmbedding 先于 hashlib/numpy/jieba 加载）
import index_documents as idx  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import retrieve as R  # noqa: E402 （idx 之后，安全）

logger = logging.getLogger(__name__)

ROOT = idx.ROOT_PATH
EVAL_DIR = ROOT / "data" / "eval"
EVAL_SET = EVAL_DIR / "eval_set.json"
RUN_RESULTS = EVAL_DIR / "run_results.json"
REPORT = EVAL_DIR / "report.md"

QTYPES = ("term", "paraphrase", "numeric")  # 与 config.eval.qtype_quota 键对齐


# ── 通用：OpenAI 客户端（llm/judge 段同款管理，key 不落盘）────────


def _client_for(seg: dict):
    """按配置段（llm/judge）建 OpenAI 客户端，缺 key 抛 RuntimeError。"""
    from openai import OpenAI

    env = str(seg.get("api_key_env", ""))
    key = os.environ.get(env, "").strip()
    if not key:
        raise RuntimeError(f"环境变量 {env} 未设置（config 段 {seg.get('provider')}）")
    return OpenAI(base_url=seg["base_url"], api_key=key, timeout=120.0, max_retries=1)


def _chat(client, model: str, system: str, user: str, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


# ── gen：分层抽样 + GLM 出题 ─────────────────────────────────────


GEN_SYS = (
    "你是年报知识问答数据集的标注员。阅读给定的上市公司年报片段，"
    "严格按指定题型出一道中文问答题并给出参考答案。"
)

GEN_TPL = {
    "term": (
        "【题型】术语精确——问题必须使用片段中的原文措辞（如财务科目名、指标名），"
        "像用户直接查询行业术语。\n"
    ),
    "paraphrase": (
        "【题型】语义改写——问题必须换一种问法，禁止照抄片段措辞，像不懂财务术语"
        "的普通人在问（例：原文'不良贷款率'可问'银行放出去的钱有多少收不回来'）。\n"
    ),
    "numeric": (
        "【题型】数字提取——问题答案必须是片段中的具体数字/金额/比率（保留单位），"
        "优先出表格类数据。\n"
    ),
}

GEN_FMT = (
    "\n严格只输出一个 JSON 对象（无 markdown 代码块、无多余文字）：\n"
    '{{"question": "...", "reference_answer": "..."}}\n'
    "要求：问题独立可答（不出现'该片段/上文'字样）；参考答案精确且只依据片段内容；"
    "若片段实在无法出该题型（无术语/无数字/内容空洞），输出 "
    '{{"question": "", "reference_answer": ""}}。\n\n'
    "【年报片段】\n{text}\n"
)


def _parse_json_obj(text: str) -> dict | None:
    """LLM 输出 → JSON 对象；容忍 ```json 包裹与前后杂讯。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _sample_chunks(cfg: dict) -> list[dict]:
    """分层抽样候选源块：numeric 优先 table，其余两类 text/table 兼收。

    返回 list[dict]：{chunk_id, content, chunk_type, company_name, fiscal_year,
    stock_code}，已按 max_per_report 控制多样性，顺序随机（seed 可复现）。
    """
    ec = cfg["eval"]
    rng = random.Random(int(ec.get("seed", 42)))
    chunks_dir = ROOT / cfg["paths"]["chunks"]
    lo = int(ec.get("chunk_len_min", 80))
    hi = int(ec.get("chunk_len_max", 900))

    pool: list[dict] = []
    per_report: Counter = Counter()
    for f in sorted(chunks_dir.glob("*_chunk.json")):
        items = json.loads(f.read_text(encoding="utf-8"))
        m = items[0]["metadata"]
        report_key = f"{m['stock_code']}_{m['year']}"
        if per_report[report_key] >= int(ec.get("max_per_report", 2)):
            continue
        # 每份报告先随机抽少量合格块（多样性靠跨报告，不靠单报告深挖）
        cands = [
            c for c in items
            if lo <= len(c["content"]) <= hi
            and c["metadata"]["block_types"][0] in ("text", "table")
        ]
        rng.shuffle(cands)
        for c in cands[: int(ec.get("max_per_report", 2))]:
            pool.append({
                "chunk_id": c["chunk_id"],
                "content": c["content"][:hi],
                "chunk_type": c["metadata"]["block_types"][0],
                "company_name": f.name.split("_")[2] if f.name.count("_") >= 3 else "?",
                "fiscal_year": int(m["year"]),
                "stock_code": m["stock_code"],
            })
            per_report[report_key] += 1
    rng.shuffle(pool)
    return pool


def cmd_gen(cfg: dict, n_overrides: int | None = None) -> None:
    """出题：按配额逐题找源块 → GLM 生成 → 校验落盘。"""
    ec = cfg["eval"]
    quota = {k: int(ec["qtype_quota"][k]) for k in QTYPES}
    if n_overrides:  # 小样本调试：等比缩到 n 题
        total = sum(quota.values())
        scale = n_overrides / total
        quota = {k: max(1, round(v * scale)) for k, v in quota.items()}
    n_target = sum(quota.values())
    logger.info("出题目标 %d 题，配额 %s", n_target, quota)

    client = _client_for(cfg["llm"])
    model = cfg["llm"]["model"]

    # 源块供给：numeric 优先 table；term/paraphrase 全类型
    pool = _sample_chunks(cfg)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in pool:
        by_type["table" if c["chunk_type"] == "table" else "text"].append(c)
    rng = random.Random(int(ec.get("seed", 42)) + 1)
    supply = {
        "numeric": by_type["table"] + by_type["text"],  # 表格优先（列表头即表格）
        "term": by_type["text"] + by_type["table"],
        "paraphrase": by_type["text"] + by_type["table"],
    }
    for k in supply:
        rng.shuffle(supply[k])

    eval_set: list[dict] = []
    used_chunks: set[str] = set()
    fails = 0
    t0 = time.time()
    for qtype in QTYPES:
        need = quota[qtype]
        src = supply[qtype]
        i = 0
        while need > 0 and i < len(src):
            c = src[i]
            i += 1
            if c["chunk_id"] in used_chunks:
                continue
            prompt = GEN_TPL[qtype] + GEN_FMT.format(text=c["content"])
            try:
                raw = _chat(client, model, GEN_SYS, prompt, 0.4)
            except Exception as e:
                logger.warning("出题调用失败（跳过该块）：%s", str(e)[:120])
                fails += 1
                continue
            obj = _parse_json_obj(raw)
            q = (obj or {}).get("question", "").strip() if obj else ""
            a = (obj or {}).get("reference_answer", "").strip() if obj else ""
            if not q or not a:
                fails += 1
                continue
            used_chunks.add(c["chunk_id"])
            eval_set.append({
                "qid": f"{qtype}-{len([e for e in eval_set if e['qtype'] == qtype]) + 1:03d}",
                "qtype": qtype,
                "question": q,
                "reference_answer": a,
                "source_chunk_id": c["chunk_id"],
                "source_meta": {
                    "company_name": c["company_name"],
                    "fiscal_year": c["fiscal_year"],
                    "chunk_type": c["chunk_type"],
                    "stock_code": c["stock_code"],
                },
            })
            need -= 1
            logger.info("(%d/%d) [%s] %s → %.60s…",
                        len(eval_set), n_target, qtype,
                        c["company_name"], q)
        if need > 0:
            logger.warning("[%s] 源块耗尽，缺 %d 题（可提高 max_per_report 或放宽长度）",
                           qtype, need)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_SET.write_text(
        json.dumps(eval_set, ensure_ascii=False, indent=2), encoding="utf-8")
    dist = Counter(e["qtype"] for e in eval_set)
    logger.info("出题完成：%d 题（配额 %s，弃块 %d），用时 %.0fs → %s",
                len(eval_set), dict(dist), fails, time.time() - t0, EVAL_SET)


# ── run：E3 跑批 ────────────────────────────────────────────────


def cmd_run(cfg: dict) -> None:
    """逐题 E3 检索（+重排）+ GLM 生成，存跑批结果。断点续跑。"""
    if not EVAL_SET.exists():
        raise FileNotFoundError(f"先跑 gen：{EVAL_SET} 不存在")
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    results: list[dict] = []
    done_q: set[str] = set()
    if RUN_RESULTS.exists():
        results = json.loads(RUN_RESULTS.read_text(encoding="utf-8"))
        done_q = {r["qid"] for r in results}
        logger.info("断点续跑：已完成 %d/%d", len(done_q), len(eval_set))

    import llm_client

    t0 = time.time()
    with R.Retriever(cfg) as r:
        for i, q in enumerate(eval_set, 1):
            if q["qid"] in done_q:
                continue
            t1 = time.time()
            # rerank=None：E3 默认含重排；重排失败硬抛（定版），中断整批——
            # 语义即"不许带病入表"（architecture.md §9）
            hits = r.retrieve(q["question"], exp="E3")
            retrieve_ms = round((time.time() - t1) * 1000, 1)
            contexts = [
                {
                    "chunk_id": h["payload"].get("chunk_id"),
                    "rank": rk,
                    "company_name": h["payload"].get("company_name"),
                    "fiscal_year": h["payload"].get("fiscal_year"),
                    "page_num": h["payload"].get("page_num"),
                    "chunk_type": h["payload"].get("chunk_type"),
                    "rrf_score": h.get("rrf_score"),
                    "rerank_score": h.get("rerank_score"),
                }
                for rk, h in enumerate(hits, 1)
            ]
            # GLM 生成（答案侧参考）：失败不阻塞——检索指标不受影响
            answer, reason = llm_client.generate_answer(q["question"], hits, cfg)
            results.append({
                **{k: q[k] for k in ("qid", "qtype", "question",
                                     "reference_answer", "source_chunk_id",
                                     "source_meta")},
                "contexts": contexts,
                "answer": answer,
                "answer_degraded": reason,
                "retrieve_ms": retrieve_ms,
                "timings_ms": dict(r.last_timings_ms),
            })
            RUN_RESULTS.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            hit = q["source_chunk_id"] in {c["chunk_id"] for c in contexts}
            logger.info("(%d/%d) %s hit@5=%s %sms +llm%s",
                        i, len(eval_set), q["qid"], hit, retrieve_ms,
                        "" if answer else "(降级)")
    logger.info("跑批完成：%d 题，用时 %.0fs → %s",
                len(results), time.time() - t0, RUN_RESULTS)


# ── score：五指标 + 裁判评分 + 分层报告 ──────────────────────────


def _metrics_one(result: dict, k: int = 5) -> dict:
    """单题五指标（金标准=source_chunk_id 唯一命中，用户定版口径）。"""
    gold = result["source_chunk_id"]
    ctx = result["contexts"][:k]
    ranks = [j + 1 for j, c in enumerate(ctx) if c["chunk_id"] == gold]
    hit = bool(ranks)
    first = ranks[0] if hit else 0
    return {
        f"recall@{k}": 1.0 if hit else 0.0,
        f"precision@{k}": len(ranks) / k,
        f"hitrate@{k}": 1.0 if hit else 0.0,
        "mrr": (1.0 / first) if first else 0.0,
        f"ndcg@{k}": (1.0 / math.log2(first + 1)) if first else 0.0,  # IDCG=1（单金标）
    }


def _judge_answer(cfg: dict, results: list[dict]) -> dict[str, float]:
    """裁判（DeepSeek）5 分制评 answer vs reference_answer → {qid: score}。"""
    client = _client_for(cfg["judge"])
    model = cfg["judge"]["model"]
    sys_p = ("你是评测裁判。对【系统答案】与【参考答案】做 1~5 分整分评分："
             "5=完全正确（数字/单位/事实全对）；4=基本正确（细节小瑕疵）；"
             "3=部分正确；2=大部分错误；1=完全错误或答非所问。"
             "只输出一个整数。")
    scores: dict[str, float] = {}
    for r in results:
        if not r.get("answer"):
            continue  # 生成降级的题不评分
        user_p = (f"【问题】{r['question']}\n【参考答案】{r['reference_answer']}\n"
                  f"【系统答案】{r['answer'][:1200]}")
        try:
            raw = _chat(client, model, sys_p, user_p, 0.0)
            digits = [ch for ch in raw if ch.isdigit()]
            if digits:
                scores[r["qid"]] = min(5.0, max(1.0, float(digits[0])))
        except Exception as e:
            logger.warning("裁判调用失败（%s 跳过）：%s", r["qid"], str(e)[:120])
    return scores


def cmd_score(cfg: dict, skip_judge: bool = False) -> None:
    """算五指标（分题型分层）+ 裁判评分 + 最差 10 题归因 → report.md。"""
    if not RUN_RESULTS.exists():
        raise FileNotFoundError(f"先跑 run：{RUN_RESULTS} 不存在")
    results = json.loads(RUN_RESULTS.read_text(encoding="utf-8"))
    logger.info("打分 %d 题（金标准=源 chunk 唯一命中）", len(results))

    # 裁判评分（可选；失败不阻塞主表）
    judge_scores: dict[str, float] = {}
    if not skip_judge:
        try:
            judge_scores = _judge_answer(cfg, results)
            logger.info("裁判评分完成 %d 题", len(judge_scores))
        except RuntimeError as e:
            logger.warning("裁判不可用，跳过答案侧评分：%s", e)

    # 五指标：总体 + 分题型
    METRIC_KEYS = ["recall@5", "precision@5", "hitrate@5", "mrr", "ndcg@5"]
    overall: dict[str, float] = {}
    rows = []
    for r in results:
        m = _metrics_one(r)
        rows.append({**m, "qid": r["qid"], "qtype": r["qtype"],
                     "judge": judge_scores.get(r["qid"])})
        for k in METRIC_KEYS:
            overall[k] = overall.get(k, 0.0) + m[k]
    n = max(len(rows), 1)
    overall = {k: round(v / n, 4) for k, v in overall.items()}

    by_type: dict[str, dict[str, float]] = {}
    for qt in QTYPES:
        sub = [row for row in rows if row["qtype"] == qt]
        if not sub:
            continue
        by_type[qt] = {
            k: round(sum(row[k] for row in sub) / len(sub), 4) for k in METRIC_KEYS
        }
        js = [row["judge"] for row in sub if row["judge"] is not None]
        by_type[qt]["n"] = len(sub)
        by_type[qt]["judge_avg"] = round(sum(js) / len(js), 2) if js else None

    js_all = [row["judge"] for row in rows if row["judge"] is not None]
    judge_avg = round(sum(js_all) / len(js_all), 2) if js_all else None

    # 最差 10 题（MRR 升序，裁判分并列参考）
    worst = sorted(rows, key=lambda row: (row["mrr"], row.get("judge") or 6))[:10]

    # 延迟统计
    rts = [r["retrieve_ms"] for r in results]
    rts_sorted = sorted(rts)
    p50 = rts_sorted[len(rts) // 2] if rts else 0
    p95 = rts_sorted[min(len(rts) - 1, int(len(rts) * 0.95))] if rts else 0
    degraded = sum(1 for r in results if r.get("answer_degraded"))

    # ── 报告 ──
    L: list[str] = []
    L.append("# M4 评测报告（E3 完整链路）\n")
    L.append(f"- 题集：{len(results)} 题（"
             + "，".join(f"{qt} {by_type.get(qt, {}).get('n', 0)}" for qt in QTYPES)
             + "）；金标准 = 出题源 chunk（唯一命中口径）")
    L.append(f"- 检索延迟：P50 {p50:.0f}ms / P95 {p95:.0f}ms；"
             f"生成降级 {degraded}/{len(results)} 题")
    L.append("")
    L.append("## 主表（总体）\n")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    QT_CN = {"recall@5": "Recall@5（top5 含源块比例）",
             "precision@5": "Precision@5（命中数/5，单金标上限 0.2）",
             "hitrate@5": "HitRate@5（=Recall@5，单金标等价）",
             "mrr": "MRR（首次命中排名倒数）",
             "ndcg@5": "NDCG@5（位置折减，IDCG=1）"}
    for k in METRIC_KEYS:
        L.append(f"| {QT_CN[k]} | **{overall[k]:.4f}** |")
    L.append(f"| 裁判评分（答案侧参考，5 分制） | {judge_avg if judge_avg is not None else '—'} |")
    L.append("")
    L.append("## 分题型分层\n")
    L.append("| 题型 | n | Recall@5 | Precision@5 | HitRate@5 | MRR | NDCG@5 | 裁判均分 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for qt in QTYPES:
        d = by_type.get(qt)
        if not d:
            continue
        L.append(f"| {qt} | {d['n']} | {d['recall@5']:.4f} | {d['precision@5']:.4f} "
                 f"| {d['hitrate@5']:.4f} | {d['mrr']:.4f} | {d['ndcg@5']:.4f} "
                 f"| {d['judge_avg'] if d['judge_avg'] is not None else '—'} |")
    L.append("")
    L.append("## 最差 10 题（MRR 升序，归因素材）\n")
    L.append("| qid | 题型 | MRR | 裁判 | 问题 | 源块 | top1 |")
    L.append("|---|---|---|---|---|---|---|")
    rmap = {r["qid"]: r for r in results}
    for row in worst:
        r = rmap[row["qid"]]
        top1 = r["contexts"][0] if r["contexts"] else None
        t1 = (f"{top1['company_name']}{top1['fiscal_year']} p{top1['page_num']}"
              if top1 else "—")
        L.append(f"| {row['qid']} | {row['qtype']} | {row['mrr']:.2f} "
                 f"| {row['judge'] if row['judge'] is not None else '—'} "
                 f"| {r['question'][:40]}… "
                 f"| {r.get('source_meta', {}).get('company_name', '?')}"
                 f"{r.get('source_meta', {}).get('fiscal_year', '?')} "
                 f"[{r.get('source_meta', {}).get('chunk_type', '?')}] "
                 f"| {t1} |")
    L.append("")
    L.append("## 诚实声明\n")
    L.append("- 合成题天然贴近源 chunk：指标度量“找回出题素材的能力”，非"
             "“找回全部真正能支持答案的证据的能力”，绝对值偏乐观。")
    L.append("- 本评测在无元数据过滤的裸查询下测量；单金标口径下 Precision@5 "
             "上限 0.2、HitRate@5 与 Recall@5 数值等价，解读以 Recall@5/MRR/"
             "NDCG@5 为主。")
    L.append("- 裁判（DeepSeek）5 分制为答案侧参考指标，不依赖金标准，"
             "但含 LLM 主观性。")
    L.append("- 分题型差分（term vs paraphrase）反映对措辞变化的鲁棒性，"
             "比总平均更有信息量。")
    L.append("")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    logger.info("报告 → %s", REPORT)
    print("\n".join(L[:40]))


# ── CLI ─────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="M4 评测：gen 出题 / run 跑批 / score 打分")
    ap.add_argument("cmd", choices=["gen", "run", "score"],
                    help="gen=分层抽样出题；run=E3 跑批；score=五指标+报告")
    ap.add_argument("--n", type=int, default=None, help="gen 小样本调试题数")
    ap.add_argument("--skip-judge", action="store_true",
                    help="score 跳过裁判评分（只算检索五指标）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    import jieba
    jieba.setLogLevel(logging.WARNING)

    R._load_dotenv()  # .env + Machine 注册表兜底（GLM/DeepSeek/硅基流动 key）
    cfg = idx.load_config()

    if args.cmd == "gen":
        cmd_gen(cfg, n_overrides=args.n)
    elif args.cmd == "run":
        cmd_run(cfg)
    else:
        cmd_score(cfg, skip_judge=args.skip_judge)


if __name__ == "__main__":
    main()
