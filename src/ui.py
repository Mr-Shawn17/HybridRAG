# -*- coding: utf-8 -*-
"""Gradio 检索界面（M3，架构 v1.3 §5：提问、结果列表、引用面板）。

与 retrieve.py 共享一个 Retriever 单例（Qdrant 已是服务端模式，无本地锁，
但仍复用单例省掉重复加载 BGE-M3 模型；gradio 同进程回调，天然满足）。

接口口径（v1.7，与 retrieve.py 对齐）：retrieve() 返回 list[dict]（每项含
payload/text/source/各路分数/rerank_score），阶段耗时在 r.last_timings_ms。
重排失败 retrieve 直接抛 RuntimeError（定版：不降级）——UI 层捕获后原样
呈现错误信息，不回退 RRF 序。

进度反馈（v1.9）：on_ask 改为生成器——流水线在后台线程跑，主线程每 0.2s
yield 一次状态条，前端据此刷出「spinner + 扫动进度条 + 实时计时」，查询期间
用户始终看得到后台在干活。阶段为两段式（检索中 → 生成中），因为 retrieve()
内部的 encode/召回/重排只在跑完后才通过 last_timings_ms 一次性给出；要更细
的粒度需给 retrieve.py 加 progress 回调，当前版本不做。

用法（项目根目录，py312 环境；需先完成 M2b 入库并安装 gradio）：
  python src/ui.py                 # http://127.0.0.1:7860
  python src/ui.py --share         # 临时外网链接（gradio tunnel，可选）
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

# 导入顺序硬约束同 retrieve.py：index_documents 最先（内部保证 FlagEmbedding
# 先于 hashlib/numpy/jieba），gradio/retrieve 均在其后加载即安全
import index_documents as idx  # noqa: E402  isort: skip  # 必须最先！

import gradio as gr  # noqa: E402
import llm_client  # noqa: E402
import retrieve as R  # noqa: E402

logger = logging.getLogger(__name__)

_STATE: dict = {}  # {"retriever": Retriever}

# 状态条样式：内联在 gr.HTML 里而不是走 Blocks(css=...)，避免 gradio 版本间
# css 参数位置变动；两条 CSS 动画（转圈 + 横向扫动）由浏览器侧执行，
# 服务端每 0.2s 只负责更新文案和秒数，动画本身不掉帧。
_STATUS_CSS = """<style>
.ragstat{display:flex;align-items:center;gap:10px;box-sizing:border-box;width:100%;
  margin:8px 0;padding:10px 14px;border:1px solid #e5e7eb;border-radius:10px;
  background:#fafafa;color:#4b5563;font-size:14px;line-height:1.4;
  font-family:ui-sans-serif,system-ui,"Microsoft YaHei",sans-serif}
.ragstat .spin{flex:0 0 auto;width:15px;height:15px;border-radius:50%;
  border:2.5px solid #e5e7eb;border-top-color:#f97316;
  animation:ragspin .75s linear infinite}
.ragstat .stage{flex:0 0 auto;white-space:nowrap}
.ragstat .bar{position:relative;flex:1 1 auto;min-width:60px;height:5px;
  border-radius:3px;background:#eef2f7;overflow:hidden}
.ragstat .bar::after{content:"";position:absolute;top:0;left:0;height:100%;width:45%;
  border-radius:3px;background:linear-gradient(90deg,rgba(249,115,22,0),#f97316,rgba(249,115,22,0));
  animation:ragslide 1.15s ease-in-out infinite}
.ragstat .t{flex:0 0 auto;min-width:52px;text-align:right;font-weight:600;
  font-variant-numeric:tabular-nums;color:#ea580c}
.ragstat .dot{flex:0 0 auto;width:16px;height:16px;border-radius:50%;display:flex;
  align-items:center;justify-content:center;font-size:10px;font-weight:700;color:#fff}
.ragstat.done{border-color:#bbf7d0;background:#f0fdf4;color:#166534}
.ragstat.done .dot{background:#22c55e}
.ragstat.done .t{color:#15803d}
.ragstat.err{border-color:#fecaca;background:#fef2f2;color:#991b1b}
.ragstat.err .dot{background:#ef4444}
.ragstat.err .t{color:#b91c1c}
.ragstat.idle{border-style:dashed;background:#fff;color:#9ca3af}
@keyframes ragspin{to{transform:rotate(360deg)}}
@keyframes ragslide{0%{transform:translateX(-110%)}100%{transform:translateX(240%)}}
</style>"""


def _status_html(stage: str, secs: float | None, state: str = "run") -> str:
    """渲染状态条。state: run（转圈+扫动+计时）/ done / err / idle。

    secs=None 时不渲染计时段（用于「请输入问题」这类还没开始跑的前置校验）。
    """
    if state == "idle":
        return _STATUS_CSS + (f'<div class="ragstat idle">{stage}</div>')
    timer = "" if secs is None else f'<span class="t">{secs:.1f}s</span>'
    if state == "done":
        return _STATUS_CSS + (
            f'<div class="ragstat done"><span class="dot">✓</span>'
            f'<span class="stage">{stage}</span>{timer}</div>'
        )
    if state == "err":
        return _STATUS_CSS + (
            f'<div class="ragstat err"><span class="dot">!</span>'
            f'<span class="stage">{stage}</span>{timer}</div>'
        )
    return _STATUS_CSS + (
        f'<div class="ragstat"><span class="spin"></span>'
        f'<span class="stage">{stage}</span><div class="bar"></div>'
        f"{timer}</div>"
    )


_IDLE_STATUS = _status_html("就绪 · 输入问题后点击「检索并回答」", 0.0, "idle")


def get_retriever() -> R.Retriever:
    if "retriever" not in _STATE:
        _STATE["retriever"] = R.Retriever(idx.load_config())
    return _STATE["retriever"]


def _fmt_citation(i: int, c: dict) -> str:
    p = c["payload"]
    lines = [
        f"**[{i}] {p.get('company_name','?')} {p.get('fiscal_year','?')} 年报** · "
        f"p.{p.get('page_num','?')} · {p.get('chunk_type','?')} · "
        f"章节：{p.get('section','')}"
    ]
    score = (
        f"rerank={c['rerank_score']:.4f}"
        if c.get("rerank_score") is not None
        else f"rrf={c['rrf_score']:.5f}"
    )
    d = "—" if c.get("score_dense") is None else f"{c['score_dense']:.3f}"
    s = "—" if c.get("score_sparse") is None else f"{c['score_sparse']:.3f}"
    lines.append(f"`{score}` · dense {d} · sparse {s}")
    text = (p.get("text") or "").strip()
    lines.append(text[:600] + ("…" if len(text) > 600 else ""))
    return "\n".join(lines)


def on_ask(
    query: str,
    exp: str,
    no_rerank: bool,
    no_llm: bool,
    code: str,
    year_from: str,
    year_to: str,
):
    """生成器：流水线丢到后台线程跑，主线程每 0.2s yield 一帧状态条。

    yield 顺序固定 (answer, citations, meta, status)。首帧先清空上一轮结果再
    亮出状态条，之后各帧用 gr.skip() 只更新状态条，出结果时一并落最终状态。
    """
    if not query.strip():
        yield ("请输入问题", "", "—", _status_html("请输入问题后再检索", None, "err"))
        return

    codes = [c.strip() for c in code.replace("，", ",").split(",") if c.strip()]
    try:
        flt = R.build_filter(
            codes=codes or None,
            year_from=int(year_from) if year_from.strip() else None,
            year_to=int(year_to) if year_to.strip() else None,
        )
    except ValueError:
        bad = year_from.strip() or year_to.strip()
        yield (
            f"筛选条件有误：年份需填整数，「{bad}」无法解析",
            "",
            "—",
            _status_html(
                f"筛选条件有误：年份需为整数（当前填的是「{bad}」）", None, "err"
            ),
        )
        return

    t0 = time.time()
    will_rerank = (not no_rerank) and exp == "E3"  # v1.7 口径：E3 默认开重排
    stage = {
        "text": "正在混合检索：encode → 双路召回 → RRF"
        + (" → rerank" if will_rerank else "")
    }
    box: dict = {}

    def _pipeline() -> None:
        """后台线程：检索 + 生成，边界处切换阶段文案。"""
        try:
            r = get_retriever()
            box["retriever"] = r
            box["chunks"] = r.retrieve(
                query.strip(), exp=exp, flt=flt, rerank=False if no_rerank else None
            )
            box["retrieve_s"] = time.time() - t0
            if no_llm:
                stage["text"] = "检索完成（已关闭生成），正在整理引用"
            else:
                stage["text"] = "检索完成，正在生成答案"
                t_llm = time.time()
                box["answer"], box["reason"] = llm_client.generate_answer(
                    query.strip(), box["chunks"], r.cfg
                )
                box["llm_ms"] = round((time.time() - t_llm) * 1000, 1)
        except BaseException as e:  # noqa: BLE001  线程内异常必须转交主线程
            box["err"] = e
        finally:
            box["done"] = True

    th = threading.Thread(target=_pipeline, name="rag-pipeline", daemon=True)
    th.start()

    # 首帧：清空上一轮内容，立刻亮出状态条（不等第一个 tick）
    yield "", "", "—", _status_html(stage["text"], 0.0)
    while not box.get("done"):
        time.sleep(0.2)
        yield gr.skip(), gr.skip(), gr.skip(), _status_html(
            stage["text"], time.time() - t0
        )
    th.join(1.0)
    elapsed = time.time() - t0

    if "err" in box:
        e = box["err"]
        logger.warning("查询失败：%s", e)
        msg = (
            f"检索失败：{e}"
            if isinstance(e, RuntimeError)
            else f"运行异常 {type(e).__name__}：{e}"
        )
        yield msg, "", "—", _status_html("查询失败", elapsed, "err")
        return

    r = box["retriever"]
    chunks = box.get("chunks") or []
    llm_ms = box.get("llm_ms")
    degraded = [box["reason"]] if box.get("reason") else []

    answer = box.get("answer") or (
        "（未生成答案："
        + "；".join(degraded or ["关闭"])
        + "）\n\n以下为检索命中的参考资料，可人工判读。"
    )
    citations = "\n\n---\n\n".join(_fmt_citation(i, c) for i, c in enumerate(chunks, 1))
    t = r.last_timings_ms

    def _ms(k):
        v = t.get(k, "—")
        return v if isinstance(v, str) else f"{v}ms"

    meta = (
        f"路线 **{exp}** · 库 {r.n_points:,} 点 · "
        f"encode {_ms('encode')} / 召回+RRF {_ms('recall')} / "
        f"rerank {_ms('rerank')}"
        + (f" / llm {llm_ms}ms" if llm_ms is not None else "")
        + (
            f" / 合计 {t['total']}ms"
            if isinstance(t.get("total"), (int, float))
            else ""
        )
        + f" · 本次查询 {elapsed:.1f}s"
        + (" · 生成降级: " + "、".join(degraded) if degraded else "")
    )

    done_stage = f"完成 · 检索 {box.get('retrieve_s', elapsed):.1f}s"
    if llm_ms is not None:
        done_stage += f" + 生成 {llm_ms / 1000:.1f}s"
    elif no_llm:
        done_stage += "（仅检索）"
    yield answer, citations, meta, _status_html(done_stage, elapsed, "done")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="年报混合检索 RAG") as demo:
        gr.Markdown(
            "# 上市公司年报 · 混合检索 RAG\nBGE-M3 dense + jieba BM25 双路召回 → "
            "RRF 融合 → 云重排 → 百炼生成，答案带页码溯源。"
        )
        with gr.Row():
            with gr.Column(scale=5):
                query = gr.Textbox(
                    label="问题",
                    scale=2,
                    placeholder="例：平安银行2023年末不良贷款率是多少？",
                    lines=2,
                )
                ask = gr.Button("检索并回答", variant="primary")
                status = gr.HTML(_IDLE_STATUS)
                answer = gr.Markdown(label="答案")
            with gr.Column(scale=2):
                exp = gr.Radio(
                    ["E3", "E0", "E1", "E2", "E4"],
                    value="E3",
                    label="实验组（E3 完整链路；其余组默认不重排）",
                )
                no_rerank = gr.Checkbox(False, label="强制跳过重排（E3 对照）")
                no_llm = gr.Checkbox(False, label="只检索不生成")
                code = gr.Textbox(
                    label="限定股票代码（逗号分隔，可空）",
                    placeholder="如 000001,600519",
                )
                with gr.Row():
                    y_from = gr.Textbox(label="年份起", placeholder="2023")
                    y_to = gr.Textbox(label="年份止", placeholder="2025")
                meta = gr.Markdown("—")
        gr.Markdown("## 引用溯源")
        citations = gr.Markdown()
        ask.click(
            on_ask,
            [query, exp, no_rerank, no_llm, code, y_from, y_to],
            [answer, citations, meta, status],
        )
        query.submit(
            on_ask,
            [query, exp, no_rerank, no_llm, code, y_from, y_to],
            [answer, citations, meta, status],
        )
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="年报混合检索 Gradio UI")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="gradio 临时外网链接")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    import jieba

    jieba.setLogLevel(logging.WARNING)

    R._load_dotenv()  # config/.env 兜底（setdefault，系统环境变量优先）
    get_retriever()  # 启动即预热（加载模型+开库），首问不卡
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
