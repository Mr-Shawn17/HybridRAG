# -*- coding: utf-8 -*-
"""基于 LlamaIndex 对解析产物（data/parsed/*.json）做文本分块。

v1.3 改造要点（相对初版的三处修复，依据见 docs/architecture.md §9 风险清单）：

1. **修复阅读顺序错乱**：初版 title/table 块遍历时立即 append，text 块先收集、
   批量切分后才在末尾 append，导致 chunk_idx 与文档位置脱钩（实测平安银行 2023：
   140 个 title + 304 个 table 占据 idx 0~443，1551 个 text 全在 444~1994，
   页码从 p2 跳到 p294）。现改为先按原始顺序构造"切分单元"（segment），
   text 单元批量切分后**按单元顺序回填**，chunk_idx 恢复文档位置语义，
   后续 small-to-big（命中碎片块后回捞前后块）才可行。

2. **修复 splitter 从未生效**：初版把每个 parsed text block 各自包成一个
   Document 送 splitter，而 parsed 侧 88640 个 text block 中 77751 个不足 100
   字符 → splitter 无内容可切，chunk_size/chunk_overlap 形同虚设（实测 text
   chunk 长度 min/p50/p90/max = 10/61/105/340 字符，≥1000 字符的为 0）。
   现改为把**连续的、同 section_path 的** text 块用 "\\n\\n" 聚合成一个单元再送
   splitter，让 chunk_size 真正生效；遇 table 块或章节切换即断开。
   聚合允许跨页（年报同一小节常跨页），page_num 记首块页、page_end 记末块页。

3. **title 块不再单独成 chunk**：纯标题短块无检索信息量，且会稀释召回结果
   （初版产出 1.95 万个 title chunk）。改为把标题信息作为**上下文前缀**拼进
   紧随其后的 text/table chunk 正文（默认取标题栈最末 2 层），完整层级路径仍
   存 metadata.section 供溯源与过滤。

附带：section_path 清洗。parse_pdf 的 _is_title_line 会把"1、本行董事会、
监事会及董事、监事、高级管理人员保证年度报告内容的真实、准确、完整，"这类
编号正文长句（<=50 字符）判为 title 并入栈，污染章节路径（实测抽样 13107 条
中 1501 条即 11.5% 的 section 含长句层级，最长 175 字符）。按长度阈值剔除。

用法（项目根目录，py312 环境）：
  python src/chunk_documents.py                 # 全量切块 60 份
  python src/chunk_documents.py --limit 5       # 小样本
  python src/chunk_documents.py --codes 600519  # 指定公司
  python src/chunk_documents.py --debug         # 额外输出 SentenceSplitter node 明细
输出：data/chunks/<code>_<year>_<公司名>_chunk.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from src.utils import Log

# logger = logging.getLogger(__name__)


ROOT_PATH = Path(__file__).resolve().parents[1]
PARSED_DIR = ROOT_PATH / "data" / "parsed"
CHUNK_DIR = ROOT_PATH / "data" / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DEBUG_DIR = ROOT_PATH / "data" / "chunk_debug"
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

# 标题栈清洗阈值：实测真标题 p50=11 字符，误判的编号正文长句 >=40 字符，取 25 兼顾
TITLE_LEVEL_MAX_CHARS = 25
# 拼进 chunk 正文的标题层级数（最末 N 层）：给碎片块一个主题锚点，又不冗长稀释向量
CONTEXT_LEVELS = 2
# 少于该字符数的 text 块视为碎文本丢弃（孤立符号行、表格残留）
MIN_TEXT_CHARS = 10

# 目录行特征：点线引导符 + 结尾页码（"重要提示 ......... 3"）。
# 年报目录页被 find_tables 识别成表格后，整表无检索价值却含大量章节名，
# 入库后会成为高干扰召回项（实测占 table chunk 0.7%），整表丢弃。
TOC_CELL_RE = re.compile(r"[.．·•…]{2,}\s*\d{1,3}\s*$")
# 一张表内命中目录特征的行数 >= 此值 → 判定为目录表
TOC_TABLE_MIN_ROWS = 3

# ── 表格页眉/页脚行清洗（切块层兜底）─────────────────────────────
# parse_pdf 已用 y 坐标剥离表格边缘的页眉行，但限制「<=2 个非空单元格」，
# 平安银行等年报的章节页眉被表格列对齐拆成 3~6 格（如 "第三"|"章 管理层
# 讨论与"|"分析"）而漏网（实测占 table chunk 1.7%）。切块层拿不到 y 坐标，
# 改用「文本特征 + 强标记」双重判定兜底：拼接整行去空白后匹配页眉模式，且
# 被剥的首/尾块必须含强标记（年度报告 / 第X章），避免误删以公司名等开头的真表格。
# 行拼接后字符数上限（页眉都是短行）
HEADER_ROW_MAX_CHARS = 30
# 页眉行文本特征：第X章 / XXXX年报告 / 公司全称 / 释义 / 重要提示 / 目录 / 纯页码
HEADER_ROW_RE = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百\d]+[章节篇][\u4e00-\u9fff]{0,24}"  # 第三章管理层讨论与分析
    r"|\d{4}年(?:度)?(?:年度)?报告(?:全文|摘要)?"  # 2023年年度报告
    r"|[\u4e00-\u9fffA-Za-z0-9（）()]{2,30}?(?:股份有限公司|有限责任公司|有限公司)"  # 公司名
    r"|释义|重要提示|目录|备查文件目录|公司简介"  # 常见固定页眉短词
    r"|\d{1,4}"  # 纯页码（页脚）
    r")$"
)
# 强标记：被剥的首/尾块至少含其一，才确认是页眉而非数据
HEADER_STRONG_RE = re.compile(
    r"\d{4}年(?:度)?(?:年度)?报告|第[一二三四五六七八九十百\d]+[章节篇]"
)


def _row_norm_text(row: list[str]) -> str:
    """拼接一行的非空单元格并去掉所有空白。

    跨页表格的页眉常被列对齐拆碎（"第三"|"章 管理层讨论与"|"分析"），
    直接 join 会留空格导致正则不匹配；去空白后还原成"第三章管理层讨论与分析"。
    """
    return re.sub(r"\s+", "", "".join(c for c in row if c.strip()))


def _is_header_row(row: list[str]) -> bool:
    """单行是否为页眉/页脚（纯文本特征判定，不含位置信息）。"""
    text = _row_norm_text(row)
    if not text or len(text) > HEADER_ROW_MAX_CHARS:
        return False
    return bool(HEADER_ROW_RE.match(text))


def _strip_header_footer_rows(rows: list[list[str]]) -> list[list[str]]:
    """剥离表格首尾的页眉/页脚行（切块层兜底，见上方常量注释）。

    只动**边缘**连续行；且首/尾被剥块必须含强标记（年度报告 / 第X章），
    否则整块保留——这样"以公司名开头的股东表"等真数据不会被误删。
    """
    n = len(rows)
    # 首部
    start = 0
    leading: list[str] = []
    while start < n and _is_header_row(rows[start]):
        leading.append(_row_norm_text(rows[start]))
        start += 1
    if leading and not any(HEADER_STRONG_RE.search(t) for t in leading):
        start = 0  # 无强标记 → 不是页眉，保留
    # 尾部
    end = n
    trailing: list[str] = []
    while end > start and _is_header_row(rows[end - 1]):
        trailing.append(_row_norm_text(rows[end - 1]))
        end -= 1
    if trailing and not any(HEADER_STRONG_RE.search(t) for t in trailing):
        end = n
    return rows[start:end]


def write_json(result, out_path: str | Path) -> None:
    """结果落盘（UTF-8 JSON，indent=2）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _clean_section(section_path: list[str]) -> list[str]:
    """标题栈清洗：剔除被误判为标题的正文长句层级。

    parse_pdf 把标题压栈后，后续块的 section_path 是栈快照；其中混入的长句
    层级对章节过滤与引用溯源无意义（反而让 section 长达 175 字符），按长度剔除。
    全部被剔除时兜底 ["unknown"]，不阻塞入库。
    """
    cleaned = [
        t.strip() for t in section_path if len(t.strip()) < TITLE_LEVEL_MAX_CHARS
    ]
    return cleaned or ["unknown"]


def _context_prefix(section: list[str], levels: int = CONTEXT_LEVELS) -> str:
    """取标题栈最末 N 层作为正文前缀（" > " 连接）。"""
    return " > ".join(section[-levels:]) if section else ""


def _table_to_text(rows: list[list[str]], max_chars: int) -> list[str]:
    """表格行 → 紧凑文本段。

    :param rows: raw_table 二维数组，表格原始结构化数据
    :param max_chars: 单段最大字符数（表格按字符截断，不是 token）
    :return: 切分后的表格文本片段列表
    规则：
    1. 提取表头，每一个分段都带上表头，保证单独一段表格可以看懂
    2. 逐行累加字符，超过阈值就新开一段
    3. 保证表格单行不会被拦腰切断
    """
    header = " | ".join(c for c in rows[0] if c)
    segments: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for row in rows[1:]:
        line = " | ".join(c for c in row if c)
        if not line:
            continue
        line_len = len(line)
        if buf and buf_len + line_len > max_chars:
            segments.append((header + "\n" if header else "") + "\n".join(buf))
            buf, buf_len = [], 0
        buf.append(line)
        buf_len += line_len
    if buf:
        segments.append((header + "\n" if header else "") + "\n".join(buf))
    return segments or ([header] if header else [])


def _is_toc_table(rows: list[list[str]]) -> bool:
    """判定是否为目录页被误识别成的表格。

    年报目录行形如「重要提示 .......... 3」，被 find_tables(text 策略) 识别为
    表格后，整表内容是"章节名 + 点线 + 页码"，无检索价值；且因含大量章节名，
    入库后会成为高干扰召回项（查询命中目录页而非正文）。

    判据：命中目录特征（点线引导 + 结尾页码）的行数 >= TOC_TABLE_MIN_ROWS，
    或占比过半。用双判据是因为部分目录表被表格列切碎后点线分散在单元格中。
    """
    if len(rows) < 2:
        return False
    hits = 0
    for row in rows:
        joined = " ".join(c for c in row if c.strip()).strip()
        if joined and TOC_CELL_RE.search(joined):
            hits += 1
        elif any(TOC_CELL_RE.search(c) for c in row if c.strip()):
            hits += 1
    return hits >= TOC_TABLE_MIN_ROWS or hits * 2 >= len(rows)


def _build_segments(parsed: dict, table_max_chars: int) -> list[dict]:
    """按解析块原始顺序构造"切分单元"（segment）列表。

    单元 = 一次切分与一个元数据归属的最小整体：
      - text 单元：连续同 section 的 text 块聚合体（待 splitter 切分）
      - table 单元：已按 table_max_chars 切好的单个表格片段（不再走 splitter）
    title 块不产出单元，其标题文本经 section_path 传递给后续单元作前缀。

    每项结构：
      {"kind": "text"|"table", "content": str, "section": [str],
       "page_num": int, "page_end": int, "is_ocr": bool, "pieces": [str]}
      pieces 在切分阶段回填，最终按单元顺序展开为 chunks。
    """
    segments: list[dict] = []
    buf_texts: list[str] = []
    buf_pages: list[int] = []
    buf_section: list[str] | None = None
    buf_ocr = False

    def flush():
        """把缓冲区聚合成一个 text 单元。"""
        nonlocal buf_texts, buf_pages, buf_section, buf_ocr
        if buf_texts:
            segments.append(
                {
                    "kind": "text",
                    "content": "\n\n".join(buf_texts),
                    "section": buf_section or ["unknown"],
                    "page_num": buf_pages[0],
                    "page_end": buf_pages[-1],
                    "is_ocr": buf_ocr,
                    "pieces": [],
                }
            )
        buf_texts, buf_pages, buf_section, buf_ocr = [], [], None, False

    for block in parsed["blocks"]:
        block_type = block["block_type"]
        page_num = block["page_num"]

        # title：不单独成 chunk，信息经 section_path 传给后续块（见模块 docstring 3）
        if block_type == "title":
            continue

        section = _clean_section(block["section_path"])

        if block_type == "text":
            text = block["content"].strip()
            if len(text) < MIN_TEXT_CHARS:
                continue
            # 章节切换 → 断开聚合（不同小节语义独立，不宜拼进同一 chunk）
            if buf_texts and section != buf_section:
                flush()
            buf_texts.append(text)
            buf_pages.append(page_num)
            buf_section = section
            buf_ocr = block["is_ocr"]

        elif block_type == "table":
            # 表格是独立检索单元：先断开 text 聚合，再逐段成单元
            flush()
            raw_table = block["raw_table"]
            if not raw_table:
                continue
            # 兜底剥离解析层漏网的页眉/页脚行（多单元格拆碎的章节名页眉等）
            rows = _strip_header_footer_rows(raw_table)
            if not rows:
                continue
            # 目录页被误识别成的表格：整表丢弃（无检索价值且高干扰，见 _is_toc_table）
            if _is_toc_table(rows):
                continue
            for seg in _table_to_text(rows, table_max_chars):
                segments.append(
                    {
                        "kind": "table",
                        "content": seg,
                        "section": section,
                        "page_num": page_num,
                        "page_end": page_num,
                        "is_ocr": block["is_ocr"],
                        "pieces": [seg],  # 表格已切好，不再过 splitter
                    }
                )

    flush()
    return segments


def chunk_parsed(
    parsed: dict,
    chunk_size: int = 768,  # 切分文本 token 上限
    chunk_overlap: int = 80,  # 相邻两个 chunk 之间重叠保留多少 token
    table_max_chars: int = 800,
    context_levels: int = CONTEXT_LEVELS,
    debug: bool = False,
) -> list[dict]:
    """对一份解析产物切块，返回 chunk 列表（不落盘）。

    流程：构造单元（聚合短 text 块）→ 批量切分所有 text 单元 → 按单元原始
    顺序展开为 chunks（保证 chunk_idx == 文档位置）→ 统一填 chunk_id。
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",  # 聚合时用它连接短块，故切分优先落在段落边界
        secondary_chunking_regex=r"[^,.;；。？！]+[,.;；。？！]?|[,.;；。？！]",
        include_prev_next_rel=True,
    )

    meta = parsed["meta"]
    stock_code = meta["stock_code"]
    company_name = meta["company_name"]
    year = meta["year"]
    source_filename = meta["filename"]

    segments = _build_segments(parsed, table_max_chars)

    # ── 批量切分所有 text 单元（一次调用，减少开销）──
    text_segs = [s for s in segments if s["kind"] == "text"]
    if text_segs:
        docs = [Document(text=s["content"]) for s in text_segs]
        all_nodes = splitter.get_nodes_from_documents(
            documents=docs, show_progress=debug
        )

        if debug:
            CHUNK_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            debug_path = (
                CHUNK_DEBUG_DIR / f"{stock_code}_{year}_{company_name}_debug.txt"
            )
            with open(debug_path, "w", encoding="utf-8") as f:
                for idx, node in enumerate(all_nodes):
                    f.write(f"===== Node {idx} =====\n")
                    f.write(json.dumps(node.model_dump(), ensure_ascii=False, indent=2))
                    f.write("\n")

        # 按 source_node 归组，回填到对应单元（一个单元可能切出多片）
        pieces_by_src: dict[str, list[str]] = {}
        for node in all_nodes:
            pieces_by_src.setdefault(node.source_node.node_id, []).append(
                node.get_content()
            )
        for seg, doc in zip(text_segs, docs):
            seg["pieces"] = pieces_by_src.get(doc.node_id) or [seg["content"]]

    # ── 按单元原始顺序展开为 chunks（顺序即文档阅读顺序）──
    chunks: list[dict] = []
    for seg in segments:
        prefix = _context_prefix(seg["section"], context_levels)
        section_str = " > ".join(seg["section"])
        for piece in seg["pieces"]:
            content = f"{prefix}\n{piece}" if prefix else piece
            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        "page_num": seg["page_num"],
                        "page_end": seg["page_end"],
                        "section": section_str,
                        "block_types": [seg["kind"]],
                        "is_ocr": seg["is_ocr"],
                        "stock_code": stock_code,
                        "year": str(year),
                        "source_file": source_filename,
                    },
                    "chunk_id": "",
                }
            )

    # ── 统一填全局 chunk_id：<股票代码>_<年份>_<5位补零序号> ──
    for idx, c in enumerate(chunks):
        c["chunk_id"] = f"{stock_code}_{year}_{idx:05d}"

    return chunks


def chunk_one(
    parsed_path: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    table_max_chars: int = 800,
    debug: bool = False,
    idx: int,
    file_num: int,
) -> Path:
    """切块单份解析产物并落盘，返回输出路径。"""
    t0 = time.time()
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    chunks = chunk_parsed(
        parsed,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        table_max_chars=table_max_chars,
        debug=debug,
    )
    m = parsed["meta"]
    out_path = (
        CHUNK_DIR / f"{m['stock_code']}_{m['year']}_{m['company_name']}_chunk.json"
    )
    write_json(chunks, out_path)
    logger.info(
        "(%d/%d) %s%s 年度报告切块成功：%d chunks → %s，用时 %.1fs",
        idx,
        file_num,
        m["company_name"],
        m["year"],
        len(chunks),
        out_path.name,
        time.time() - t0,
    )
    return out_path


def main():
    ap = argparse.ArgumentParser(description="年报解析产物切块")
    ap.add_argument(
        "--parsed", default="data/parsed", help="解析产物目录（相对项目根）"
    )
    ap.add_argument(
        "--chunk-size", type=int, default=768, help="SentenceSplitter token 上限"
    )
    ap.add_argument(
        "--chunk-overlap", type=int, default=80, help="相邻 chunk 重叠 token"
    )
    ap.add_argument(
        "--table-max-chars", type=int, default=800, help="表格单段最大字符数"
    )
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 份（小样本）")
    ap.add_argument("--codes", nargs="*", default=None, help="只处理指定股票代码")
    ap.add_argument(
        "--debug", action="store_true", help="额外输出 SentenceSplitter node 明细"
    )
    args = ap.parse_args()

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     datefmt="%H:%M:%S",
    # )

    parsed_dir = ROOT_PATH / args.parsed
    files = sorted(parsed_dir.glob("*.json"))
    if args.codes:
        keep = set(args.codes)
        files = [f for f in files if f.name.split("_")[0] in keep]
    if args.limit:
        files = files[: args.limit]
    if not files:
        logger.error("没有可切块的解析文件：%s", parsed_dir)
        return

    logger.info(
        "目标 %d 份 | chunk_size=%d overlap=%d table_max_chars=%d",
        len(files),
        args.chunk_size,
        args.chunk_overlap,
        args.table_max_chars,
    )
    t0 = time.time()
    total = 0
    for i, f in enumerate(files, 1):
        out = chunk_one(
            f,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            table_max_chars=args.table_max_chars,
            debug=args.debug,
            idx=i,
            file_num=len(files),
        )
        total += len(json.loads(out.read_text(encoding="utf-8")))

        # logger.info("(%d/%d) %s", i, len(files), f.name)
    logger.info(
        "切块完成：%d 份 / %d chunks，用时 %.1fs", len(files), total, time.time() - t0
    )


if __name__ == "__main__":
    main()
