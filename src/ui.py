# -*- coding: utf-8 -*-
"""Gradio 检索界面（M3，架构 v1.3 §5：提问、结果列表、引用面板）。

与 retrieve.py 共享一个 Retriever 单例（Qdrant 已是服务端模式，无本地锁，
但仍复用单例省掉重复加载 BGE-M3 模型；gradio 同进程回调，天然满足）。

接口口径（v1.7，与 retrieve.py 对齐）：retrieve() 返回 list[dict]（每项含
payload/text/source/各路分数/rerank_score），阶段耗时在 r.last_timings_ms。
重排失败 retrieve 直接抛 RuntimeError（定版：不降级）——UI 层捕获后原样
呈现错误信息，不回退 RRF 序。

用法（项目根目录，py312 环境；需先完成 M2b 入库并安装 gradio）：
  python src/ui.py                 # http://127.0.0.1:7860
  python src/ui.py --share         # 临时外网链接（gradio tunnel，可选）
"""

from __future__ import annotations

import argparse
import logging
import time

# 导入顺序硬约束同 retrieve.py：index_documents 最先（内部保证 FlagEmbedding
# 先于 hashlib/numpy/jieba），gradio/retrieve 均在其后加载即安全
import index_documents as idx  # noqa: E402  isort: skip  # 必须最先！

import gradio as gr  # noqa: E402
import llm_client  # noqa: E402
import retrieve as R  # noqa: E402

logger = logging.getLogger(__name__)

_STATE: dict = {}  # {"retriever": Retriever}


def get_retriever() -> R.Retriever:
    if "retriever" not in _STATE:
        _STATE["retriever"] = R.Retriever(idx.load_config())
    return _STATE["retriever"]


def _fmt_citation(i: int, c: dict) -> str:
    p = c["payload"]
    lines = [f"**[{i}] {p.get('company_name','?')} {p.get('fiscal_year','?')} 年报** · "
             f"p.{p.get('page_num','?')} · {p.get('chunk_type','?')} · "
             f"章节：{p.get('section','')}"]
    score = (f"rerank={c['rerank_score']:.4f}" if c.get("rerank_score") is not None
             else f"rrf={c['rrf_score']:.5f}")
    d = ("—" if c.get("score_dense") is None else f"{c['score_dense']:.3f}")
    s = ("—" if c.get("score_sparse") is None else f"{c['score_sparse']:.3f}")
    lines.append(f"`{score}` · dense {d} · sparse {s}")
    text = (p.get("text") or "").strip()
    lines.append(text[:600] + ("…" if len(text) > 600 else ""))
    return "\n".join(lines)


def on_ask(query: str, exp: str, no_rerank: bool, no_llm: bool,
           code: str, year_from: str, year_to: str):
    if not query.strip():
        return "请输入问题", "", "—"
    r = get_retriever()
    codes = [c.strip() for c in code.replace("，", ",").split(",") if c.strip()]
    flt = R.build_filter(
        codes=codes or None,
        year_from=int(year_from) if year_from.strip() else None,
        year_to=int(year_to) if year_to.strip() else None)
    # 第一步：混合检索（v1.5 起 retrieve 不再内嵌生成）。
    # 重排失败按定版硬抛；UI 层捕获展示错误，不回退 RRF 序
    try:
        chunks = r.retrieve(query.strip(), exp=exp, flt=flt,
                            rerank=False if no_rerank else None)
    except RuntimeError as e:
        logger.warning("检索失败：%s", e)
        return f"检索失败：{e}", "", "—"

    # 第二步：LLM 生成（no_llm 开关照旧；失败降级只展示检索结果）
    answer_text, reason = None, None
    llm_ms = None
    if not no_llm:
        t0 = time.time()
        answer_text, reason = llm_client.generate_answer(
            query.strip(), chunks, r.cfg)
        llm_ms = round((time.time() - t0) * 1000, 1)
    degraded = [reason] if reason else []

    answer = answer_text or ("（未生成答案：" + "；".join(degraded or ["关闭"])
                             + "）\n\n以下为检索命中的参考资料，可人工判读。")
    citations = "\n\n---\n\n".join(
        _fmt_citation(i, c) for i, c in enumerate(chunks, 1))
    t = r.last_timings_ms

    def _ms(k):
        v = t.get(k, "—")
        return v if isinstance(v, str) else f"{v}ms"

    meta = (f"路线 **{exp}** · 库 {r.n_points:,} 点 · "
            f"encode {_ms('encode')} / 召回+RRF {_ms('recall')} / "
            f"rerank {_ms('rerank')}"
            + (f" / llm {llm_ms}ms" if llm_ms is not None else "")
            + (f" / 合计 {t['total']}ms" if isinstance(t.get("total"), (int, float)) else "")
            + (" · 生成降级: " + "、".join(degraded) if degraded else ""))
    return answer, citations, meta


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="年报混合检索 RAG") as demo:
        gr.Markdown("# 上市公司年报 · 混合检索 RAG\nBGE-M3 dense + jieba BM25 双路召回 → "
                    "RRF 融合 → 云重排 → 百炼生成，答案带页码溯源。")
        with gr.Row():
            with gr.Column(scale=5):
                query = gr.Textbox(label="问题", scale=2,
                                   placeholder="例：平安银行2023年末不良贷款率是多少？",
                                   lines=2)
                ask = gr.Button("检索并回答", variant="primary")
                answer = gr.Markdown(label="答案")
            with gr.Column(scale=2):
                exp = gr.Radio(["E3", "E0", "E1", "E2", "E4"], value="E3",
                               label="实验组（E3 完整链路；其余组默认不重排）")
                no_rerank = gr.Checkbox(False, label="强制跳过重排（E3 对照）")
                no_llm = gr.Checkbox(False, label="只检索不生成")
                code = gr.Textbox(label="限定股票代码（逗号分隔，可空）",
                                  placeholder="如 000001,600519")
                with gr.Row():
                    y_from = gr.Textbox(label="年份起", placeholder="2023")
                    y_to = gr.Textbox(label="年份止", placeholder="2025")
                meta = gr.Markdown("—")
        gr.Markdown("## 引用溯源")
        citations = gr.Markdown()
        ask.click(on_ask, [query, exp, no_rerank, no_llm, code, y_from, y_to],
                  [answer, citations, meta])
        query.submit(on_ask, [query, exp, no_rerank, no_llm, code, y_from, y_to],
                     [answer, citations, meta])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="年报混合检索 Gradio UI")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="gradio 临时外网链接")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    import jieba
    jieba.setLogLevel(logging.WARNING)

    R._load_dotenv()  # config/.env 兜底（setdefault，系统环境变量优先）
    get_retriever()  # 启动即预热（加载模型+开库），首问不卡
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
