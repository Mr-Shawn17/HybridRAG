# -*- coding: utf-8 -*-
"""巨潮资讯网年报下载模块（v1.3 扁平化：原 src/download/cninfo.py + cli.py 合并）

职责（architecture.md §5）：
  - 调用巨潮 hisAnnouncement/query 接口按 代码+年报类别+时间 检索公告
  - 标题甄别（区分年度报告 / 摘要 / 英文版 / 撤销说明）
  - 限频下载（间隔 + 指数退避重试，见 §9 风险清单"巨潮接口限频"）
  - manifest.json 记录（股票代码/公司/年份/公告信息/文件状态），供解析阶段填 meta

API 要点（静态端点，CN 版）：
  POST http://www.cninfo.com.cn/new/hisAnnouncement/query
    pageNum / pageSize / column=szse / tabName=fulltext
    seDate=YYYY-MM-DD~YYYY-MM-DD（闭区间，公开接口固定 ~ 格式）
    searchkey= ;category=category_ndbg_szsh ;isTitle=true ;stock=orgId,ajaxCode
  POST http://www.cninfo.com.cn/new/information/topSearch/query  → 按代码查 orgId

用法（项目根目录，py312 环境）：
  python src/download_pdf.py                                    # 全量：按 config/companies.json 跑
  python src/download_pdf.py --codes 600519 --years 2023        # 冒烟：指定单个代码 + 年份
  python src/download_pdf.py                                    # 补跑失败项（已有 ok 记录自动跳过）

清单文件默认 config/companies.json，可手动修改后直接重跑。
约定：路径一律项目相对路径（默认值即相对路径），开源友好。
"""

import re
import time
import json
import random
import logging
import argparse
from pathlib import Path
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

BASE = "http://www.cninfo.com.cn"
QUERY_URL = BASE + "/new/hisAnnouncement/query"
ORG_URL = BASE + "/new/information/topSearch/query"
# PDF 直链（静态服务器）：http://static.cninfo.com.cn/<adjunctUrl>
PDF_URL = "http://static.cninfo.com.cn/{}"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

CATEGORY_ANNUAL = "category_ndbg_szsh"   # 年报类别（沪深）
TITLE_RE = re.compile(r"年度报告")         # 命中"年度报告"才看候选
# 需要排除的变体：摘要/英文版/H股版/取消/更正说明（保留"更新后""更正后"这类实质性文件）
EXCLUDE_PATTERNS = ("摘要", "英文", "H股", "H 股", "已取消", "取消", "撤回", "更正公告")


def _post_json(session: requests.Session, url: str, data: dict,
               max_retry: int = 4, first_delay: float = 1.5) -> dict:
    """POST + 指数退避重试（网络错误 / 5xx / 空响应）。"""
    delay = first_delay
    for attempt in range(1, max_retry + 1):
        try:
            resp = session.post(url, data=data, headers=HEADERS, timeout=30)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retry:
                raise
            sleep_s = delay + random.uniform(0, 0.5)  # 抖动，防同步撞车
            logger.warning("请求失败(%s/4) %s → %.1fs 后重试", attempt, exc, sleep_s)
            time.sleep(sleep_s)
            delay *= 2
    return {}


def _try_get_org_id(session: requests.Session, code: str) -> dict | None:
    """topSearch 接口：按代码模糊查 orgId。"""
    data = {"keyWord": code, "maxNum": 10}
    payload = _post_json(session, ORG_URL, data)
    items = payload if isinstance(payload, list) else []
    for it in items:
        if it.get("code") == code:
            return {"orgId": it["orgId"], "name": it.get("zwjc", ""),
                    "code": code}
    return None


# ── 标题甄别 ──────────────────────────────────────────────────────

def pick_annual_report(announcements: list[dict]) -> dict | None:
    """从公告列表中挑出〈XXXX 年年度报告〉正文 PDF。

    巨潮按时间倒序返回同屏多条；同一报告年度可能有多条：
    正文 / 摘要 / 英文版 / 更新后。规则：排除摘要类 → 优先"更新后" → 首个。
    """
    candidates = [a for a in announcements
                  if TITLE_RE.search(a.get("announcementTitle", ""))
                  and not any(p in a.get("announcementTitle", "") for p in EXCLUDE_PATTERNS)]
    if not candidates:
        return None
    candidates.sort(key=lambda a: "更新后" not in a["announcementTitle"])  # 更新后靠前
    return candidates[0]


# ── 下载 ──────────────────────────────────────────────────────────

def download_pdf(session: requests.Session, adjunct_url: str, dest: Path,
                 max_retry: int = 3, first_delay: float = 2.0) -> bool:
    """下载 PDF 到本地，保留 .crash 后缀避免写入中断后留下坏文件。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = PDF_URL.format(adjunct_url)  # static 服务器直链，无需编码
    delay = first_delay
    for attempt in range(1, max_retry + 1):
        try:
            with session.get(url, headers=HEADERS, timeout=120, stream=True) as resp:
                resp.raise_for_status()
                with open(dest.with_suffix(dest.suffix + ".crash"), "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 15):
                        fh.write(chunk)
            final = dest.with_suffix(dest.suffix)  # .crash → .pdf
            if final.exists():
                final.unlink()
            dest.with_suffix(dest.suffix + ".crash").rename(final)
            return True
        except requests.RequestException as exc:
            logger.warning("下载失败(%s/%s) %s → 稍后重试", attempt, max_retry, exc)
            time.sleep(delay)
            delay *= 2
    return False


# ── 主入口 ────────────────────────────────────────────────────────

def _rel_posix(path: Path) -> str:
    """路径转项目相对 posix 字符串（开源友好，不暴露本机目录）。"""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fetch_annual_reports(companies: list[dict], raw_dir: Path, manifest_path: Path,
                         request_interval: float = 1.2, download_interval: float = 1.5) -> dict:
    """对清单中每个 (code, year) 下载对应年报 PDF 并写 manifest。"""
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _save_manifest():
        """增量落盘：中途崩溃也能保留已下载进度（断点续跑依据）。"""
        manifest["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    # 断点续跑：载入已有 manifest（计数字段按 items 重建，防历史不一致）
    manifest = {"generated_at": "", "total": 0, "ok": 0, "skipped": 0,
                "failed": 0, "items": {}}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.setdefault("items", {})
            manifest["total"] = len(manifest["items"])
            manifest["ok"] = sum(1 for v in manifest["items"].values()
                                 if v.get("status") == "ok")
            manifest["failed"] = sum(1 for v in manifest["items"].values()
                                     if v.get("status") == "failed")
            manifest["skipped"] = 0
            logger.info("载入已有 manifest：%d 项（ok=%d）", manifest["total"],
                        manifest["ok"])
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("manifest 读取失败（%s），重建", exc)
    manifest["skipped"] = 0

    session = requests.Session()

    # 预解析 orgId（缓存）
    org_ids: dict[str, str] = {}
    for comp in companies:
        code = comp["code"]
        if code in org_ids:
            continue
        info = _try_get_org_id(session, code)
        if info is None:
            logger.warning("[%s] topSearch 未命中 orgId，跳过", comp["name"])
            continue
        org_ids[code] = info["orgId"]
        logger.info("[org] %s %s → %s", code, info.get("name", ""), info["orgId"])
        time.sleep(request_interval)

    for comp in companies:
        code, name = comp["code"], comp["name"]
        org_id = org_ids.get(code)
        if not org_id:
            for year in comp["years"]:
                manifest["items"][f"{code}_{year}"] = {"code": code, "name": name,
                                                       "year": year, "status": "failed",
                                                       "reason": "orgId 未解析"}
                manifest["failed"] += 1
                manifest["total"] += 1
            continue

        for year in comp["years"]:
            key = f"{code}_{year}"
            existing = manifest["items"].get(key)
            if existing and existing.get("status") == "ok" and existing.get("file_path") \
                    and Path(existing["file_path"]).exists():
                manifest["skipped"] += 1
                manifest["total"] += 1
                logger.info("[skip] %s %s 年报已存在", name, year)
                continue

            year_path = raw_dir / str(year)
            year_path.mkdir(parents=True, exist_ok=True)
            # 检查是否有残留 .crash，删掉
            for crash in year_path.glob(f"{code}_*.pdf.crash"):
                crash.unlink(missing_ok=True)

            # seDate：该年度报告披露窗口（年报集中在次年 1~4 月，放宽到 12 月）
            se_date = f"{year + 1}-01-01~{year + 1}-12-31"
            data = {
                "pageNum": 1, "pageSize": 30, "column": "szse", "tabName": "fulltext",
                "plate": "", "stock": f"{code},{org_id}",  # 顺序必须是"代码,orgId"，两段非空，否则过滤失效
                "searchkey": "", "secid": "", "category": CATEGORY_ANNUAL,
                "trade": "", "seDate": se_date, "sortName": "", "sortType": "",
                "isHLtitle": "true",
            }
            try:
                payload = _post_json(session, QUERY_URL, data)
            except requests.RequestException as exc:
                logger.error("[%s %s] 接口重试耗尽：%s", name, year, exc)
                manifest["items"][key] = {"code": code, "name": name, "year": year,
                                          "status": "failed", "reason": str(exc)}
                manifest["failed"] += 1
                manifest["total"] += 1
                continue

            announcements = (payload or {}).get("announcements") or []
            target = pick_annual_report(announcements)

            if target is None:
                # 冬季窗口漏检兜底：往前扩 1 个月再试一次（部分公司 12 月底已挂）
                se_date2 = f"{year}-12-01~{year + 1}-04-30"
                data["seDate"] = se_date2
                payload = _post_json(session, QUERY_URL, data)
                announcements = (payload or {}).get("announcements") or []
                target = pick_annual_report(announcements)

            if target is None:
                logger.warning("[%s %s] 未找到年报公告（可能尚未披露或接口变更）", name, year)
                manifest["items"][key] = {"code": code, "name": name, "year": year,
                                          "status": "failed", "reason": "公告未命中"}
                manifest["failed"] += 1
                manifest["total"] += 1
                time.sleep(download_interval)
                continue

            title = target["announcementTitle"]
            adjunct_url = target["adjunctUrl"]           # e.g. finalpage/2025-03-28/xxxx.PDF
            adjunct_size = target.get("adjunctSize", 0)
            pdf_path = year_path / f"{code}_{name}_{year}_annual.pdf"

            logger.info("[down] %s %s | %s | %.1f KB", name, year, title, adjunct_size)
            if download_pdf(session, adjunct_url, pdf_path):
                manifest["items"][key] = {
                    "code": code, "name": name, "year": year,
                    "org_id": org_id, "title": title,
                    "adjunct_url": adjunct_url, "adjunct_size": adjunct_size,
                    "secid": target.get("secId", ""),
                    "announcement_id": target.get("announcementId", ""),
                    "announcement_time": target.get("announcementTime", 0),
                    "file_path": _rel_posix(pdf_path),
                    "status": "ok",
                }
                manifest["ok"] += 1
            else:
                manifest["items"][key] = {"code": code, "name": name, "year": year,
                                          "status": "failed", "reason": "下载失败",
                                          "title": title, "adjunct_url": adjunct_url}
                manifest["failed"] += 1
            manifest["total"] += 1
            _save_manifest()
            time.sleep(download_interval)

        # 每家公司间多歇一小会，降低风控触发概率
        time.sleep(request_interval)

    _save_manifest()
    logger.info("完成：total=%s ok=%s skipped=%s failed=%s → %s",
                manifest["total"], manifest["ok"], manifest["skipped"],
                manifest["failed"], manifest_path)
    return manifest


# ── CLI ───────────────────────────────────────────────────────────

def _resolve(p: str) -> Path:
    """相对路径按项目根解析，绝对路径原样使用。"""
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description="巨潮资讯网年报批量下载")
    parser.add_argument("--companies", default="config/companies.json",
                        help="采样清单 JSON 路径（默认 config/companies.json，相对项目根）")
    parser.add_argument("--raw", default="data/raw",
                        help="PDF 输出目录（默认 data/raw，相对项目根）")
    parser.add_argument("--manifest", default="data/raw/manifest.json",
                        help="manifest 输出路径（默认 data/raw/manifest.json）")
    parser.add_argument("--codes", nargs="*", default=None,
                        help="只下载指定股票代码（冒烟用），如 --codes 600519 000858")
    parser.add_argument("--years", nargs="*", type=int, default=None,
                        help="只下载指定年份，如 --years 2023")
    parser.add_argument("--interval", type=float, default=1.5,
                        help="下载间隔秒数（限频保护，默认 1.5）")
    args = parser.parse_args()

    companies = json.loads(_resolve(args.companies).read_text(encoding="utf-8"))["COMPANIES"]
    if args.codes:
        companies = [c for c in companies if c["code"] in set(args.codes)]
    if args.years:
        companies = [
            {**c, "years": [y for y in c["years"] if y in set(args.years)]}
            for c in companies
        ]
        companies = [c for c in companies if c["years"]]

    if not companies:
        logger.error("过滤后清单为空，请检查 --codes / --years")
        return

    logger.info("目标：%d 家公司 / %d 份年报 → %s",
                len(companies), sum(len(c["years"]) for c in companies), args.raw)
    fetch_annual_reports(
        companies=companies,
        raw_dir=_resolve(args.raw),
        manifest_path=_resolve(args.manifest),
        download_interval=args.interval,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
