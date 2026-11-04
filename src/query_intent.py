# -*- coding: utf-8 -*-
"""查询侧增强：第一级规则实体抽取 + 第二级 LLM 查询变换（2026-11-03 定版）。

第一级（纯规则，零成本）：
  公司名词典匹配（companies.json）+ 年份严格正则（年报邻近/公司名后紧跟），
  抽出 {codes, years} 喂给 retrieve.build_filter / MCP search 的过滤参数，
  掐掉「跨公司混淆」「跨年度错位」。

第二级（LLM 变换，全量触发 + 双路融合 + 失败回退，2026-11-03 用户定版）：
  每条查询先过 GLM 做术语归一改写（口语→年报标准措辞），输出
  {"intent", "queries"}；检索时原句+改写句各自召回、同一 RRF 池融合
  （共识块得分翻倍，改写跑偏时原句结果仍在池内兜底）；LLM 失败/超时/
  输出不合法时记 warning 回退原句（与 rerank 失败硬抛不同——改写失败
  只是增强没生效，回退即现状基线，非带病）。

  触发范围定版依据：词表覆盖度规则实测零区分度（16 万词语料词表下
  改写题覆盖率 0.7~0.93 全漏标），术语鸿沟是概念级信号、超出规则
  能力边界，故全量触发 LLM。

依赖：第一级仅标准库 + 读 config/companies.json；第二级另需 openai SDK
与 config.llm 段（GLM）。无模型加载、秒级返回，天然可测。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
COMPANIES_JSON = ROOT / "config" / "companies.json"

# 「X年年报 / X年年度报告 / X年的年报」：年报年度，最高置信
_RE_REPORT_YEAR = re.compile(r"(20[0-9]{2})\s*年\s*的?\s*(?:年度)?\s*报告")

# 「公司名后紧跟年份」：如「贵州茅台2024年」「平安银行 2023 年末」。
# 由 extract() 用已命中的公司简称拼出带公司名的正则，逐个捕获其后紧跟
# 的年份（允许公司名与年份间夹少量空格）。
_RE_TAIL_YEAR = re.compile(r"(20[0-9]{2})\s*年")


@dataclass
class QueryEntities:
    """规则抽取结果。codes/years 为空表示没抽到对应实体。"""
    raw_query: str = ""
    codes: list[str] = field(default_factory=list)   # 命中的股票代码
    years: list[int] = field(default_factory=list)   # 命中的年报年度
    companies: list[str] = field(default_factory=list)  # 命中的公司简称（供日志/调试）
    need_llm: bool = False                            # True=规则覆盖不了，交棒第二级

    @property
    def has_entity(self) -> bool:
        return bool(self.codes or self.years)

    def to_filter_args(self) -> dict:
        """→ retrieve.build_filter / MCP search 的参数形态。"""
        args: dict = {}
        if self.codes:
            args["codes"] = self.codes
        if self.years:
            # 多年度 → 闭区间覆盖（Qdrant 只支持 gte/lte 连续区间）
            args["year_from"] = min(self.years)
            args["year_to"] = max(self.years)
        return args


class _CompanyIndex:
    """公司简称→代码索引，读 companies.json，长词优先防子串误伤。"""

    def __init__(self, companies_json: Path = COMPANIES_JSON):
        data = json.loads(companies_json.read_text(encoding="utf-8"))
        self.default_years: list[int] = sorted(
            int(y) for y in data.get("default_years", [])
        )
        # (name, code) 按名称长度降序，保证「宁波银行」先于任何更短子串命中
        self._pairs = sorted(
            ((c["name"], str(c["code"])) for c in data["COMPANIES"]),
            key=lambda p: len(p[0]),
            reverse=True,
        )

    def match(self, query: str) -> tuple[list[str], list[str]]:
        """返回 (codes, companies)。一句可命中多家（罕见，但支持）。"""
        codes, names = [], []
        for name, code in self._pairs:
            if name in query:
                codes.append(code)
                names.append(name)
        return codes, names


def _extract_years(query: str, valid_years: list[int],
                   company_names: list[str]) -> list[int]:
    """严格年份抽取：只认「年报邻近」或「公司名后紧跟」的年报年度。

    裸年份（无年报上下文、非公司名后紧跟）一律不抽，避免把业务发生年
    误当年报年度。
    """
    found: set[int] = set()

    # 1) 年报邻近：2024年年报 / 2024年年度报告 / 2024年的年报
    for m in _RE_REPORT_YEAR.finditer(query):
        y = int(m.group(1))
        if y in valid_years:
            found.add(y)

    # 2) 公司名后紧跟年份：贵州茅台2024年 / 平安银行 2023 年末
    #    对每个命中的公司简称，在其出现位置之后窗口内捕获首个「XXXX年」
    for name in company_names:
        pos = query.find(name)
        while pos != -1:
            tail = query[pos + len(name): pos + len(name) + 12]  # 后 12 字符窗口
            m = _RE_TAIL_YEAR.search(tail)
            if m and int(m.group(1)) in valid_years:
                found.add(int(m.group(1)))
            pos = query.find(name, pos + 1)

    return sorted(found)


def extract(query: str, index: _CompanyIndex | None = None) -> QueryEntities:
    """主入口：规则抽取实体。

    Args:
        query: 原始查询文本。
        index: 公司索引（复用可避免重复读配置；默认新建）。

    Returns:
        QueryEntities；抽不到实体时 need_llm=True。
    """
    idx = index or _CompanyIndex()
    codes, names = idx.match(query)
    years = _extract_years(query, idx.default_years, names)

    ent = QueryEntities(
        raw_query=query,
        codes=codes,
        years=years,
        companies=names,
        need_llm=not (codes or years),
    )
    return ent


# ══ 第二级：LLM 查询变换（术语归一改写 + 拆分）════════════════


TRANSFORM_SYS = (
    "你是上市公司年报检索系统的查询改写器。把用户的口语化查询改写成贴合"
    "年报正式措辞的检索查询，用于混合检索（向量+BM25）。"
)

TRANSFORM_PROMPT = """改写规则：
1. 保留公司名、年份等实体信息，不改变问题意图，不自创数字与事实
2. 把口语说法替换为年报标准术语（会计科目/财务指标名），例如：
   "收不回来的钱" → "不良贷款"
   "还没拿到房产证的房子" → "未办妥产权证书的房屋建筑物"
   "帮忙查账的机构" → "会计师事务所"
   "拉新股东入伙" → "发行股票/吸收投资"
3. 若无把握对应到标准术语，保留原词，不要硬造
4. 原问题包含多个独立子问题时，拆成多条查询（每条一个子问题）
5. 查询已是标准术语措辞时，原样返回，不要为改而改

只输出一个 JSON 对象（无 markdown 代码块、无多余文字）：
{{"intent": "fact|comparison|concept|multi", "queries": ["改写后的查询1", "查询2（可选）"]}}

【用户查询】
{query}"""


@dataclass
class TransformResult:
    """第二级变换结果。"""
    ok: bool                       # False=LLM 失败/输出不合法，已回退
    intent: str = ""               # 观测用意图标签（fact/comparison/concept/multi）
    queries: list[str] = field(default_factory=list)  # 改写后查询（ok=True 时非空）
    fallback_reason: str = ""      # 回退原因（ok=False 时非空）


class Transformer:
    """LLM 查询变换器：GLM（config.llm 段），惰性建连，进程内复用。

    失败语义（用户定版）：超时/限速/输出不合法 → 记 warning 返回
    ok=False（回退原句），不抛异常——与 rerank 失败硬抛不同（rerank
    失败是结果带病，改写失败只是增强没生效，回退即现状基线）。
    """

    def __init__(self, llm_cfg: dict, timeout: float = 60.0):
        self._llm_cfg = llm_cfg
        self._timeout = timeout
        self._client = None

    def _ensure_client(self):
        import os

        if self._client is not None:
            return
        from openai import OpenAI

        env = str(self._llm_cfg.get("api_key_env", ""))
        key = os.environ.get(env, "").strip()
        if not key:
            raise RuntimeError(f"环境变量 {env} 未设置（config.llm 段）")
        self._client = OpenAI(
            base_url=self._llm_cfg["base_url"], api_key=key,
            timeout=self._timeout, max_retries=1,
        )

    def transform(self, query: str) -> TransformResult:
        """单次 LLM 改写。任何失败 → ok=False（调用方回退原句）。"""
        try:
            self._ensure_client()
            resp = self._client.chat.completions.create(
                model=self._llm_cfg["model"],
                messages=[
                    {"role": "system", "content": TRANSFORM_SYS},
                    {"role": "user", "content":
                     TRANSFORM_PROMPT.format(query=query)},
                ],
                temperature=0.0,  # 改写要确定性，不要发散
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("查询改写调用失败（回退原句）：%s", str(e)[:160])
            return TransformResult(ok=False, fallback_reason=f"llm_error:{str(e)[:80]}")

        obj = _parse_json_obj(raw)
        if not obj or not isinstance(obj.get("queries"), list):
            logger.warning("查询改写输出不合法（回退原句）：%.80s", raw)
            return TransformResult(ok=False, fallback_reason="bad_output")
        queries = [str(q).strip() for q in obj["queries"] if str(q).strip()]
        if not queries:
            return TransformResult(ok=False, fallback_reason="empty_queries")
        return TransformResult(
            ok=True,
            intent=str(obj.get("intent", ""))[:16],
            queries=queries,
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
        obj = json.loads(text[start: end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# ══ 编排：两级级联（规则实体 → LLM 改写 → 双路融合检索）═══════


def pipeline_query(
    query: str,
    retriever,
    transformer: Transformer | None,
    index: _CompanyIndex | None = None,
    exp: str = "E3",
    rerank: bool | None = None,
    flt_override=None,
) -> tuple[list[dict], dict]:
    """查询侧增强完整链路（两级级联），返回 (hits, 调试信息)。

    流程：
      1. 第一级：规则抽实体 → flt 过滤条件（抽不到则不过滤）
      2. 第二级：LLM 改写（transformer=None 时跳过；失败回退原句）
      3. 检索：[原句] + 改写 queries 去重合并 → retriever.retrieve
         多查询同池 RRF 融合（首元素=原句=重排基准句）

    flt_override：外部已确定过滤条件时传入（如 UI 手动条件与规则抽取合并后），
    跳过内部规则过滤直接使用——手动优先语义由调用方合并，本函数不做合并。

    调试信息：{"codes", "years", "rewrite_ok", "intent", "queries",
    "fallback_reason"}，供日志与评测记录。
    """
    ent = extract(query, index)

    queries = [query]
    dbg = {
        "codes": ent.codes,
        "years": ent.years,
        "rewrite_ok": False,
        "intent": "",
        "queries": [query],
        "fallback_reason": "",
    }
    if transformer is not None:
        tr = transformer.transform(query)
        dbg["rewrite_ok"] = tr.ok
        dbg["intent"] = tr.intent
        dbg["fallback_reason"] = tr.fallback_reason
        if tr.ok:
            for q in tr.queries:
                if q != query and q not in queries:
                    queries.append(q)
            dbg["queries"] = list(queries)

    if flt_override is not None:
        flt = flt_override
    elif ent.has_entity:
        import retrieve as R

        flt = R.build_filter(**ent.to_filter_args())
    else:
        flt = None

    hits = retriever.retrieve(queries, exp=exp, flt=flt, rerank=rerank)
    return hits, dbg


if __name__ == "__main__":
    # 自测 A：规则抽取
    print("═══ 第一级：规则实体抽取自测 ═══")
    cases = [
        "贵州茅台2024年的年报里，这家公司借出去的钱有多少？",
        "平安银行2023年末不良贷款率是多少？",
        "这家酒企平时会和老板旗下其他公司做生意，掏钱最多的是哪家？",
        "2024年公司新请来帮忙查账的是哪家机构？",
    ]
    idx = _CompanyIndex()
    for q in cases:
        e = extract(q, idx)
        print(f"Q: {q}\n  → codes={e.codes} years={e.years} need_llm={e.need_llm}\n")

    # 自测 B：LLM 改写（需 GLM_API_KEY；额度受限时可能回退，属预期）
    import os
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    if os.environ.get("GLM_API_KEY") or True:
        import retrieve as R  # noqa: F401（_load_dotenv 兜底注册表 key）

        R._load_dotenv()
        import index_documents as ix  # noqa: F401

        llm_cfg = ix.load_config()["llm"]
        print("═══ 第二级：LLM 改写自测 ═══")
        tr = Transformer(llm_cfg)
        for q in [
            "长城汽车2023年年报里，到年底公司账上已经算作资产、但房产证还没办好的房子，加起来值多少钱？",
            "迈瑞医疗2024年年报中，2024年公司新请来帮忙查账的是哪家机构？",
        ]:
            r = tr.transform(q)
            print(f"Q: {q[:50]}…\n  → ok={r.ok} intent={r.intent} "
                  f"queries={r.queries if r.ok else r.fallback_reason}\n")
