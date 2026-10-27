# -*- coding: utf-8 -*-
"""LLM 生成薄封装（v1.5 自 retrieve.py 拆出）。

职责单一：基于用户 query + 混合检索结果（retrieve.Retriever.retrieve() 返回
dict 里的 chunks，纯 JSON 结构），调百炼 qwen（OpenAI 兼容层，token plan
个人版）生成带编号引用的中文答案。

刻意保持简单，作为参考实现：RAG 系统集成进 Agent 项目时本模块可整体替换，
只需守住 generate_answer(query, chunks) 的输入/输出约定。接入参数全部在
config.yaml 的 llm 段（base_url / model / temperature / api_key_env），API
key 从系统环境变量读取（默认 DASHSCOPE_API_KEY），代码与配置均不落盘 key。

降级约定：无 API key / 网络异常 → 返回 (None, 原因)，不抛异常、不断检索链。

代码调用：
  import retrieve, llm_client
  with retrieve.Retriever() as r:
      chunks = r.retrieve(query)                   # 1) 混合检索（list[dict]）
  answer, reason = llm_client.generate_answer(query, chunks)  # 2) 生成

单独运行（内部先检索再生成，端到端冒烟）：
  python src/llm_client.py --query "平安银行2023年末不良贷款率是多少？"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_PATH / "config" / "config.yaml"

CITATION_TMPL = "[{i}] {company}{year}年报 p.{page} {section}\n{text}"
SYSTEM_PROMPT = (
    "你是财报分析助手。只依据【参考资料】回答问题，引用处标注编号（如 [1][2]）；"
    "资料不足以回答时明确说「根据现有资料无法回答」，不要编造数字。"
    "回答用中文，简洁准确，涉及金额/比率时保留原文单位。"
)


def load_config() -> dict:
    """直接读 config.yaml。不 import index_documents，保持本模块轻量可搬移。"""
    import yaml
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _citation(i: int, chunk: dict) -> str:
    """单个 chunk dict → 带编号引用条目（字段来自入库 payload）。"""
    p = chunk.get("payload") or {}
    return CITATION_TMPL.format(
        i=i, company=p.get("company_name", "?"), year=p.get("fiscal_year", "?"),
        page=p.get("page_num", "?"), section=p.get("section", ""),
        text=(p.get("text") or "")[:1200],  # 防超长表格撑爆 prompt
    )


def build_prompt(query: str, chunks: list[dict]) -> str:
    """query + chunks → 用户消息（编号引用与 chunks 顺序一致）。"""
    refs = "\n\n".join(_citation(i, c) for i, c in enumerate(chunks, 1))
    return f"【参考资料】\n{refs}\n\n【问题】{query}\n"


def generate_answer(query: str, chunks: list[dict],
                    cfg: dict | None = None) -> tuple[str | None, str | None]:
    """调 LLM 生成答案。

    参数：query 为用户问题；chunks 为 retrieve() 返回的 list[dict]
    （每项含 payload.text/company_name/fiscal_year/page_num/section）。
    返回 (answer, None) 成功；(None, 降级原因) 失败——调用方决定是否只展示检索结果。
    """
    cfg = cfg or load_config()
    llm = cfg["llm"]
    env_name = llm.get("api_key_env", "DASHSCOPE_API_KEY")
    key = os.environ.get(env_name, "").strip()
    if not key:
        logger.warning("环境变量 %s 未设置，跳过 LLM 生成", env_name)
        return None, f"no_llm_key({env_name})"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=llm["base_url"], api_key=key)
        resp = client.chat.completions.create(
            model=llm["model"],
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": build_prompt(query, chunks)}],
            temperature=float(llm.get("temperature", 0.2)))
        return resp.choices[0].message.content, None
    except Exception as e:  # 网络/额度/模型异常统一降级，不断链
        logger.warning("LLM 生成失败：%s", e)
        return None, f"llm_error: {e}"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="LLM 生成（先检索再生成，端到端冒烟）")
    ap.add_argument("--query", "-q", required=True, help="查询语句")
    ap.add_argument("--exp", default="E3", help="检索实验组（见 retrieve.py）")
    ap.add_argument("--no-rerank", action="store_true", help="强制无重排")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    # 延迟导入：retrieve 内部保证 index_documents 最先加载（Windows DLL 顺序
    # 硬约束），本模块顶层只碰标准库，故此处导入即满足约束
    import retrieve as R
    R._load_dotenv()  # config/.env 兜底（系统环境变量优先）
    cfg = R.idx.load_config()
    with R.Retriever(cfg) as r:
        chunks = r.retrieve(args.query, exp=args.exp,
                            rerank=False if args.no_rerank else None)
        timings = r.last_timings_ms

    answer, reason = generate_answer(args.query, chunks, cfg)
    if answer:
        print("\n【答案】", answer)
    else:
        print("\n（未生成答案：", reason, "）")
    print("【引用】")
    for i, c in enumerate(chunks, 1):
        p = c["payload"]
        print(f"  [{i}] {p.get('company_name')}{p.get('fiscal_year')} "
              f"p{p.get('page_num')} {p.get('section', '')}")
    print("检索耗时:", timings,
          "| 生成降级:", reason or "无")


if __name__ == "__main__":
    main()
