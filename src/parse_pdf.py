# -*- coding: utf-8 -*-
"""PyMuPDF 年报解析模块

与 v1（pymupdf_parser.py）的关系：
  - v1 输出 {meta, blocks:[{page_no, type, section, text/rows}], warnings}；
  - v2 按参考格式输出 {meta, source, blocks:[{block_type, content, page_num,
    section_path, is_ocr, raw_table}]}：章节由扁平 section 升级为
    section_path 标题栈（逐层嵌套，溯源/展示更直观），表格额外保留
    Markdown 管道序列化的 content 与原始二维数组 raw_table。
  - 底层能力沿用 v1 已验证的逻辑：正文字号众数、页眉页脚双条件剔除、
    表格 text 策略提取 + 空行/空列清理、扫描页检测、书签缺失时大字标题兜底。

与参考格式刻意保留的三处差异（均为本项目约定）：
  1. source 用项目相对路径（参考版是本机绝对路径；本项目要开源，
     见用户约定：所有路径一律项目相对，不暴露本机信息）；
  2. 表格区域只产出 table 块、文本流剔除表格区域。参考版会把表格内容
     再以 pipe-join 文本块输出一遍（table 块 + text 块双份冗余），v2
     不复刻该行为，避免切块阶段重复入库；
  3. 顶层多一个 warnings 键（扫描页等异常记录，供 M1 质检；参考版无此键，
     按 meta/source/blocks 三键读取的消费方不受影响）。

标题层级（section_path 深度）推断规则：
  - 首个识别到的标题 → level 0（文档根，永不被弹出）；
  - 目录行（点线引导+页码）或 "第X节/第X章" → level 1；
  - "一、"/"1、"/"1.1" 编号标题 → level 2；
  - "（一）" 括号编号 → level 3；
  - 其他大字标题（封面公司名/日期/报告全文等）→ 前一个块也是标题则
    level = 上一个标题 level + 1（封面连续标题逐层加深，与参考版一致），
    否则 level 1。
  入栈：先弹出所有 level >= 新标题 level 的旧标题再压入——同层替换、
  上层保留；文本/表格块取当前栈快照作 section_path，无标题时用 ["unknown"]。

表格 content 序列化（参考版同款）：
  "| 表头 |\n| --- | --- |\n| 行1 |"，raw_table 保留清洗后的二维数组。

meta 八字段来源：优先取下载 manifest（data/raw/manifest.json，含
title/adjunct_url/announcement_id），缺失时从文件名
<code>_<公司名>_<年份>_annual.pdf 推导；plate 按代码段位推导
（6 开头 sh、0/3 开头 sz、4/8 开头 bj）。

用法（项目根目录，py312 环境）：
  python src/parse/pymupdf_parser_v2.py                          # 演示：解析 data/raw 下第一份 PDF
  python src/parse/pymupdf_parser_v2.py data/raw/2023/xxx.pdf    # 指定一份
输出：data/parsed_v2/<code>_<year>_<name>.json（独立目录，不覆盖 v1 产物）
"""

from __future__ import annotations
import argparse
import time
import json
import re
import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parent.parent
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# ── 文本模式常量 ──────────────────────────────────────────────────

# 章节标题行："第一节/第一章"（允许空格变体）
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百\d]+[章节]")

# 数字编号标题："1.1 公司基本情况"（招商银行等版式，无"第X章"字样）
NUM_TITLE_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,2}\s+\S")

# 编号起始标题："一、xxx" / "1、xxx" / "1.xxx"（中文顿号或西文点必须紧随）
NUMBERED_RE = re.compile(r"^(?:[一二三四五六七八九十百]+|\d{1,3})\s*[、.．]\s*\S")

# 括号编号标题："（一）xxx" / "(一) xxx"
PAREN_NUM_RE = re.compile(r"^[（(]\s*[一二三四五六七八九十百]+\s*[）)]\s*\S")

# 目录行：点线引导符 + 结尾页码，如 "第一节 重要提示和释义 .......... 3"
# （点线可能是半角点/全角点/间隔号/省略号；页码最多 3 位）
TOC_LINE_RE = re.compile(r"[.．·•…]{2,}\s*\d{1,3}\s*$")

# 纯页码行："3 / 143"、"- 12 -"、"第 3 页 共 143 页"
PAGE_NO_RE = re.compile(r"^(?:[-—–]?\s*\d+\s*(?:[/|]\s*\d+|页)?|第\s*\d+\s*页.*)$")

# 同字号页眉行：与正文同字号（10pt）但固定贴顶的公司名/章节名，
# 按文本特征剔除——独立短行（<=20字）且不含数字和句读
HEADER_TEXT_RE = re.compile(r"^[^\d.,;:。；，%％/（(）)【】\[\]—…]{1,20}$")

# 表格区域内的页眉文本：年报页眉固定为「公司名 / 第X章 xxx / XXXX 年年度报告」
# 三段，被表格列对齐切碎后仍以短行形式混入表格首行（实测占 table chunk 9.8%）。
# 仅用于短行（<=30字、<=2 个非空单元格）判定，避免误伤同名的数据单元格。
TABLE_HF_TEXT_RE = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百\d]+[章节篇]\s*[\u4e00-\u9fff\s]{0,20}"  # 第五章 环境和社会责任
    r"|\d{4}\s*年(?:度)?(?:年度)?报告(?:全文)?"  # 2023 年年度报告
    r"|[\u4e00-\u9fffA-Za-z0-9（）()\s]{2,28}?(?:股份有限公司|有限责任公司|有限公司)"  # 公司名
    r")$"
)

# 表格页眉/页脚行的 y 带（与正文路径的贴顶页眉判定同口径）
TABLE_HF_MARGIN = 90.0
# 页眉行判定：拼接后字符数上限 / 非空单元格数上限
TABLE_HF_MAX_CHARS = 30
TABLE_HF_MAX_CELLS = 2

# 文件名推导：<code>_<公司名>_<年份>_annual.pdf
FNAME_META_RE = re.compile(r"^(\d{6})_(.+?)_(\d{4})_annual$")

# 扫描页判定阈值：整页可提取字符少于此数 → 视为扫描/图片页
SCANNED_PAGE_CHARS = 30

# 下载清单（manifest.json）中的公告直链前缀（静态服务器）
CNINFO_PDF_PREFIX = "http://static.cninfo.com.cn/"


# ── 基础工具 ──────────────────────────────────────────────────────


def to_rel_path(path: str | Path) -> str:
    """把路径压成项目相对路径（posix 风格）。

    项目内文件 → "data/raw/2023/xxx.pdf"（供 GitHub 开源，不暴露本机信息）；
    项目外文件 → 退化为文件名字符串，避免暴露目录结构。
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT_PATH).as_posix()
    except ValueError:
        return p.name


def _line_max_size(line: dict) -> float:
    """取一行内最大字号（标题行常由多个 span 拼成）。"""
    return max((s["size"] for s in line["spans"]), default=0.0)


def _is_bold(line: dict) -> bool:
    """行内任一 span 加粗即视为加粗（span flags bit 4 = bold）。"""
    return any(s["flags"] & 16 for s in line["spans"])


def _clean_cell(text: str) -> str:
    """表格单元格清洗：合并换行、去首尾空白。"""
    return re.sub(r"\s*\n\s*", "", text or "").strip()


def _row_joined(row: list[str]) -> tuple[str, int]:
    """表格行的非空单元格拼接文本 + 非空单元格数。"""
    cells = [c for c in row if c.strip()]
    return " ".join(cells).strip(), len(cells)


def _row_is_header_footer(row: list[str], y: float | None, page_height: float) -> bool:
    """表格行是否为混入的页眉/页脚（双条件：位置 + 文本特征）。

    年报页眉固定为「公司名 / 第X章 xxx / XXXX 年年度报告」三段，被表格列
    对齐切碎后仍以短行形式混入表格首尾行（实测占 table chunk 9.8%）。
    只剥离**边缘**行（首/尾），不动中间行——中间出现的同名文本是合法数据
    （如"释义"表里列出公司全称）。
    """
    text, n_cells = _row_joined(row)
    if not text or n_cells > TABLE_HF_MAX_CELLS or len(text) > TABLE_HF_MAX_CHARS:
        return False
    # 纯页码行（页脚）
    if PAGE_NO_RE.match(text):
        return True
    if not TABLE_HF_TEXT_RE.match(text):
        return False
    # 位置条件：能取到行 y 时要求落在页顶/页底边距带内（与正文页眉判定同口径）；
    # 取不到（PyMuPDF 内部结构变动）则退化为纯文本判定
    if y is None:
        return True
    margin = max(TABLE_HF_MARGIN, page_height * 0.10)
    return y < margin or y > page_height - margin


def _strip_table_header_footer(
    rows: list[list[str]],
    ys: list[float | None],
    page_height: float,
) -> list[list[str]]:
    """剥离表格首尾的页眉/页脚行，返回清洗后的 rows。

    ys 必须与 rows **等长且同序**（由调用方在过滤空行时同步维护）。
    """
    if len(ys) != len(rows):  # 防御：取不到对齐的行坐标时退化为纯文本判定
        ys = [None] * len(rows)

    start = 0
    while start < len(rows) and _row_is_header_footer(
        rows[start], ys[start], page_height
    ):
        start += 1
    end = len(rows)
    while end > start and _row_is_header_footer(
        rows[end - 1], ys[end - 1], page_height
    ):
        end -= 1
    return rows[start:end]


def _table_markdown(rows: list[list[str]]) -> str:
    """二维数组 → 参考版同款管道 Markdown（首行表头 + 分隔行）。

    "| 释义项 | 指 | 释义内容 |\\n| --- | --- | --- |\\n| ..."
    """
    if not rows:
        return ""
    ncols = len(rows[0])
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    sep = "| " + " | ".join(["---"] * ncols) + " |"
    return "\n".join([lines[0], sep] + lines[1:])


def _plate_of(code: str) -> str:
    """按股票代码段位推导交易所板块：6→sh，0/3→sz，4/8→bj，未知→空串。"""
    if code.startswith("6"):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    if code.startswith(("4", "8")):
        return "bj"
    return ""


# ── 正文字号统计 ──────────────────────────────────────────────────


def _body_font_size(doc: pymupdf.Document, sample_pages: int = 30) -> float:
    """采样前 N 页统计正文字号众数（年报正文统一 10~10.6pt 左右）。

    采样前 30 页足够稳定：覆盖目录+正文多种版式，且远快于全量扫描。
    """
    counter: dict[float, int] = {}
    for pno in range(min(sample_pages, doc.page_count)):
        for blk in doc[pno].get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                for span in line["spans"]:
                    text = span["text"].strip()
                    # 只统计含汉字的 span，滤掉页码/线条/数字串
                    if len(text) >= 4 and any("\u4e00" <= c <= "\u9fff" for c in text):
                        key = round(span["size"], 1)
                        counter[key] = counter.get(key, 0) + len(text)
    if not counter:
        return 10.5  # 兜底默认值
    return max(counter, key=counter.get)


# ── 页眉页脚 ──────────────────────────────────────────────────────


def _is_header_footer(line: dict, body_size: float, page_height: float) -> bool:
    """页眉/页脚判定：小字 + 贴近页顶/页底 双重条件。

    情形一（多数年报）：页眉页脚 9pt 小字，位于页面上下 ~70pt 边距内。
    情形二（平安银行等）：页眉与正文同字号（10pt）但固定贴顶，
    交给调用方按 HEADER_TEXT_RE 文本特征补充剔除（见 _parse_page_v2）。
    只用字号会误删正文小字注释，只用位置会误删大字封面元素，
    两者相与才安全。
    """
    bbox = line["bbox"]
    small = _line_max_size(line) < body_size * 0.95
    margin = max(70.0, page_height * 0.06)
    at_edge = bbox[3] < margin or bbox[1] > page_height - margin
    return small and at_edge


# ── 表格提取 ──────────────────────────────────────────────────────


def _page_tables(page: pymupdf.Page) -> tuple[list[list[list[str]]], list]:
    """提取本页表格（text 策略）并清洗：去空行/空列 + 剥离页眉页脚行。

    返回 (清洗后的 rows 列表, 原始 Table 对象列表)——两者**严格一一对应**
    （后者仅用于取 bbox 作排序键）。
    text 策略会把行距当分隔线，产出大量全空行；年报表格常有空白隔离行，
    逐行/逐列去掉全空项即可，无需识别哪些空行是"真分隔"。
    """
    clean: list[list[list[str]]] = []
    raws: list = []
    try:
        found = page.find_tables(strategy="text")
    except Exception as exc:  # 个别损坏页不阻塞整体
        logger.debug("find_tables 失败 p%s: %s", page.number, exc)
        return clean, raws

    page_height = page.rect.height
    for tab in found.tables:
        # 行 y 坐标：tab.rows[i].bbox 与 tab.extract()[i] 同序；
        # 取不到（内部结构变动）则置 None，退化为纯文本判定
        try:
            ys_all: list[float | None] = [r.bbox[1] for r in tab.rows]
        except Exception:  # noqa: BLE001
            ys_all = []

        extracted = tab.extract()
        if len(ys_all) != len(extracted):
            ys_all = [None] * len(extracted)

        # 去全空行（rows 与 ys 必须同步过滤，保持索引对齐）
        pairs = [
            ([_clean_cell(c) for c in row], y)
            for row, y in zip(extracted, ys_all)
            if any(c and _clean_cell(c) for c in row)
        ]
        if not pairs:
            continue
        rows = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]

        col_keep = [any(r[j] for r in rows) for j in range(len(rows[0]))]
        rows = [[c for j, c in enumerate(r) if col_keep[j]] for r in rows]  # 去全空列
        # 剥离混入的页眉/页脚行（v1.3 新增，见模块 docstring）
        rows = _strip_table_header_footer(rows, ys, page_height)
        # 至少 2 行 2 列才认为是有意义的表
        if len(rows) >= 2 and len(rows[0]) >= 2:
            clean.append(rows)
            raws.append(tab)  # 与 clean 同步 append，避免错位（v1.3 修复）
    return clean, raws


# ── 标题识别与层级 ────────────────────────────────────────────────


def _is_title_line(text: str, size: float, bold: bool, body_size: float) -> bool:
    """一行是否按"标题块"输出（其余行进入正文段落聚合）。

    三类命中：
      a. 大字行：字号 >= 正文 * 1.28（实测覆盖 14pt/15pt/18pt 标题，
         同时天然排除 9pt 页眉——阈值依据同 v1）；
      b. 目录行：点线引导 + 结尾页码（目录页条目通常与正文同字号）；
      c. 编号起始行：一、/1、/1.1/（一）且字号不低于正文 0.95 倍、
         全行 <= 50 字——正文里加粗编号段首（参考版即如此处理，
         如"五、公司经本次董事会审议通过的利润分配预案为..."整行 45 字
         成为标题块），限长避免把整段编号正文误判为标题。
    """
    if size >= body_size * 1.28:
        return True
    if TOC_LINE_RE.search(text):
        return True
    numbered = bool(
        NUMBERED_RE.match(text) or NUM_TITLE_RE.match(text) or PAREN_NUM_RE.match(text)
    )
    return numbered and size >= body_size * 0.95 and len(text) <= 50


def _title_level(
    text: str,
    size: float,
    body_size: float,
    has_root: bool,
    last_was_title: bool,
    last_title_level: int,
) -> int:
    """标题入栈层级（见模块 docstring 规则表）。"""
    if not has_root:
        return 0  # 首个标题 = 文档根
    if TOC_LINE_RE.search(text) or SECTION_RE.match(text):
        return 1  # 节级
    if PAREN_NUM_RE.match(text):
        return 3  # （一）级
    if NUMBERED_RE.match(text) or NUM_TITLE_RE.match(text):
        return 2  # 一、/1.1 级
    # 其他大字标题：封面连续标题逐层加深，否则挂在根下
    return last_title_level + 1 if last_was_title else 1


# ── 单页解析 ──────────────────────────────────────────────────────


def _parse_page_v2(
    page: pymupdf.Page, body_size: float
) -> tuple[list[dict], list[str]]:
    """解析单页 → 有序 block 列表（不含 section_path/page_num，后续统一填）+ warnings。

    返回的每个 block 额外带两个私有键（最终输出前剔除）：
      _y: 排序用纵坐标；_probe: (size, bold) 供标题层级判断。
    block 雏形含 block_type/content，表格块另含 raw_table。
    """
    warnings: list[str] = []
    raw_text = page.get_text("text")
    if len(raw_text.strip()) < SCANNED_PAGE_CHARS:
        warnings.append(
            f"p{page.number + 1}: 疑似扫描页/图片页（可提取字符 {len(raw_text.strip())}）"
        )

    tables, raw_tables = _page_tables(page)
    # 表格 bbox 的 y 范围：正文/标题行落在其中则跳过（防重复）
    table_spans = [(t.bbox[1], t.bbox[3]) for t in raw_tables]

    def in_table(y: float) -> bool:
        return any(y0 - 2 <= y <= y1 + 2 for y0, y1 in table_spans)

    page_height = page.rect.height
    items: list[dict] = []

    # ── 表格块 ──
    for rows, tab in zip(tables, raw_tables):
        items.append(
            {
                "_y": tab.bbox[1],
                "block_type": "table",
                "content": _table_markdown(rows),
                "raw_table": rows,
            }
        )

    # ── 标题块 + 正文段落（按 y 序走线，段落按行距聚合） ──
    lines: list[dict] = []  # {y, size, bold, text}
    for blk in page.get_text("dict")["blocks"]:
        for line in blk.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text or _is_header_footer(line, body_size, page_height):
                continue
            if PAGE_NO_RE.match(text) and _line_max_size(line) < body_size * 0.95:
                continue  # 独立页码行
            y = line["bbox"][1]
            margin = max(90.0, page_height * 0.10)
            # 同字号贴顶页眉（平安银行 10pt 公司名）：短行、无数字句读；
            # 大字章节标题（>= 正文 1.35 倍）是正文结构，保留
            if (
                y < margin
                and HEADER_TEXT_RE.match(text)
                and _line_max_size(line) < body_size * 1.35
            ):
                continue
            if in_table(y):
                continue  # 表格区域文字流剔除（防重复）
            lines.append(
                {
                    "y": y,
                    "size": _line_max_size(line),
                    "bold": _is_bold(line),
                    "text": text,
                }
            )
    lines.sort(key=lambda x: x["y"])

    buf: list[str] = []  # 段落缓冲（连续行拼接）
    buf_y = 0.0

    def flush():
        if buf:
            items.append(
                {
                    "_y": buf_y,
                    "block_type": "text",
                    "content": "".join(buf),
                    "raw_table": None,
                }
            )
            buf.clear()

    for ln in lines:
        is_title = _is_title_line(ln["text"], ln["size"], ln["bold"], body_size)
        if is_title:
            flush()
            items.append(
                {
                    "_y": ln["y"],
                    "block_type": "title",
                    "content": ln["text"],
                    "raw_table": None,
                    "_probe": (ln["size"], ln["bold"]),
                }
            )
            continue
        # 行间距超过 1.8 倍行高视为新段落起点（行高 ≈ 字号 * 1.5）
        if buf and ln["y"] - buf_y > ln["size"] * 1.8 * 1.5:
            flush()
        if not buf:
            buf_y = ln["y"]
        buf.append(ln["text"])
    flush()

    # 过短正文无信息量（孤立符号行），与 v1 口径一致
    items = [
        it for it in items if it["block_type"] != "text" or len(it["content"]) >= 2
    ]
    return items, warnings


# ── meta 组装 ─────────────────────────────────────────────────────


def _derive_meta(pdf_path: Path, meta_info: dict | None) -> dict:
    """八字段 meta：manifest 条目优先，文件名推导兜底。"""
    m = FNAME_META_RE.match(pdf_path.stem)
    code = m.group(1) if m else ""
    name = m.group(2) if m else ""
    year = m.group(3) if m else ""
    if meta_info:
        code = str(meta_info.get("code") or code)
        name = str(meta_info.get("name") or name)
        year = str(meta_info.get("year") or year)
    adjunct = (meta_info or {}).get("adjunct_url") or ""
    return {
        "stock_code": code,
        "plate": _plate_of(code),
        "company_name": name,
        "year": year,
        "title": str((meta_info or {}).get("title") or ""),
        "filename": pdf_path.name,
        "source_url": CNINFO_PDF_PREFIX + adjunct if adjunct else "",
        "announce_id": str((meta_info or {}).get("announcement_id") or ""),
    }


# ── 主入口 ────────────────────────────────────────────────────────


def parse_pdf(pdf_path: str | Path, meta_info: dict | None = None) -> dict:
    """解析一份年报 PDF → 参考格式结构化 dict。

    返回结构（对齐参考版，另附 warnings）：
      {
        "meta": {stock_code, plate, company_name, year, title,
                 filename, source_url, announce_id},
        "source": "data/raw/2023/xxx.pdf"（项目相对路径）,
        "blocks": [
          {"block_type": "title"|"text"|"table",
           "content": str,                # 表格为管道 Markdown
           "page_num": int,               # 1 起页码（与阅读器一致）
           "section_path": [str, ...],    # 标题栈快照（含标题块自身）
           "is_ocr": False,               # 数字原生 PDF，未做 OCR
           "raw_table": [[str]] | None},  # 仅表格块非空
          ...
        ],
        "warnings": [...]
      }
    """
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    body_size = _body_font_size(doc)

    blocks: list[dict] = []
    warnings: list[str] = []

    stack: list[tuple[int, str]] = []  # (level, 标题文本)
    has_root = False
    last_was_title = False
    last_title_level = 0

    for page in doc:
        pno = page.number + 1  # 1 起页码
        items, page_warns = _parse_page_v2(page, body_size)
        warnings.extend(page_warns)
        items.sort(key=lambda it: it["_y"])

        for it in items:
            it.pop("_y")  # 排序键，输出前剔除
            probe = it.pop("_probe", None)
            if it["block_type"] == "title":
                level = _title_level(
                    it["content"],
                    probe[0],
                    body_size,
                    has_root,
                    last_was_title,
                    last_title_level,
                )
                if level == 0:
                    stack.clear()  # 根标题：清栈重置
                else:
                    while stack and stack[-1][0] >= level:
                        stack.pop()  # 同层及更深旧标题出栈
                stack.append((level, it["content"]))
                has_root = True
                last_was_title, last_title_level = True, level
                it["section_path"] = [t for _, t in stack]  # 标题块含自身
            else:
                last_was_title = False
                it["section_path"] = [t for _, t in stack] if stack else ["unknown"]
            it["page_num"] = pno
            it["is_ocr"] = False  # 本项目 60 份均为数字原生 PDF，未走 OCR
            blocks.append(it)

    result = {
        "meta": _derive_meta(pdf_path, meta_info),
        "source": to_rel_path(pdf_path),
        "blocks": blocks,
        "warnings": warnings,
    }
    doc.close()
    return result


def save_parsed(result: dict, out_path: str | Path) -> None:
    """解析结果落盘（UTF-8 JSON，indent=2 与参考版排版一致）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# 解析单份pdf
def parse_pdf_single(pdf_path: str, meta_info: dict):
    """单份解析：自动匹配 manifest 填 meta，输出到 data/parsed/"""

    t0 = time.time()
    parsed_result = parse_pdf(pdf_path, meta_info=meta_info)

    # 输出名：<code>_<year>_<name>.json
    m = parsed_result["meta"]
    out_path = PARSED_DIR / f"{m['stock_code']}_{m['year']}_{m['company_name']}.json"
    save_parsed(parsed_result, out_path)

    print(f"out_path: {out_path}")

    # 摘要：块数分布 / 章节 source / 抽样标题栈
    from collections import Counter

    types = Counter(b["block_type"] for b in parsed_result["blocks"])
    depth = max((len(b["section_path"]) for b in parsed_result["blocks"]), default=0)
    print(f"输出: {out_path.relative_to(ROOT_PATH)}")
    print(f"meta: {m['stock_code']} {m['company_name']} {m['year']} plate={m['plate']}")
    print(
        f"块数: {len(parsed_result['blocks'])} {dict(types)} | section_path 最大深度 {depth} "
        f"| 警告 {len(parsed_result['warnings'])} | 用时 {time.time() - t0:.1f}s"
    )
    for b in [x for x in parsed_result["blocks"] if x["block_type"] == "title"][:5]:
        print("  title:", " > ".join(b["section_path"]))


# 批量解析所有PDF
def parse_pdf_all(codes: list[str] | None = None, limit: int | None = None):
    """批量解析：从 manifest.json 取 meta，逐份解析输出到 data/parsed/。

    :param codes: 只解析指定股票代码（None=全部）
    :param limit: 只解析前N份（小样本验证用）
    """
    manifest_path = RAW_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error("找不到 manifest：%s", manifest_path)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    items = list(manifest["items"].values())
    if codes:
        keep = set(codes)
        items = [it for it in items if str(it.get("code")) in keep]
    if limit:
        items = items[:limit]
    if not items:
        logger.error("过滤后清单为空，请检查 --codes / --limit")
        return

    logger.info("目标 %d 份年报", len(items))
    t_start = time.time()
    n_ok = 0
    for i, item in enumerate(items, 1):
        pdf_path = ROOT_PATH / item["file_path"]
        if not pdf_path.exists():
            logger.error("没有找到 pdf，路径：%s", pdf_path)
            continue
        t0 = time.time()
        parsed_result = parse_pdf(pdf_path, meta_info=item)
        m = parsed_result["meta"]
        out_path = (
            PARSED_DIR / f"{m['stock_code']}_{m['year']}_{m['company_name']}.json"
        )
        save_parsed(parsed_result, out_path)
        n_ok += 1
        logger.info(
            "(%d/%d) %s%s 年度报告解析成功 → %s，用时 %.1fs",
            i,
            len(items),
            item["name"],
            item["year"],
            out_path.name,
            time.time() - t0,
        )
    logger.info(
        "解析完成：%d/%d 份，总用时 %.1fs", n_ok, len(items), time.time() - t_start
    )


def main():
    ap = argparse.ArgumentParser(description="年报 PDF 解析（PyMuPDF）")
    ap.add_argument("--codes", nargs="*", default=None, help="只解析指定股票代码")
    ap.add_argument("--limit", type=int, default=None, help="只解析前 N 份（小样本）")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parse_pdf_all(codes=args.codes, limit=args.limit)


if __name__ == "__main__":
    main()
