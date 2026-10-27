# -*- coding: utf-8 -*-
"""MCP 服务客户端测试脚本（模拟外部 Agent 调用）。

直接走 streamable-http 协议（JSON-RPC over HTTP），用标准库 urllib 实现，
不依赖 mcp SDK 客户端——规避 mcp 2.3.0 在 Windows+httpx2 下的 call_tool
URL 编码 bug（详见 docs/mcp.md §4.4）。

协议流程（streamable-http）：
  initialize 握手 → 拿到 mcp-session-id → tools/call 调用指定工具。

用法（项目根目录，任意 Python 3 环境，无需 py_rag 依赖）：
  python test_mcp_client.py                       # 交互式菜单选方法
  python test_mcp_client.py health                # 调指定方法（无参数）
  python test_mcp_client.py list_companies
  python test_mcp_client.py search --query "平安银行2023年不良贷款率"
  python test_mcp_client.py search --query "..." --exp E2 --codes 000001 --year_from 2023 --year_to 2024
  python test_mcp_client.py get_chunk_detail --chunk_id "000001_2023_00002"

可选：--url 指定服务地址（默认 http://127.0.0.1:8100/mcp）。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8100/mcp"


class MCPError(Exception):
    """MCP 调用错误（含服务端返回的错误信息）。"""


class Client:
    """极简 streamable-http MCP 客户端（同步，标准库）。"""

    def __init__(self, url: str = DEFAULT_URL):
        self.url = url
        self.session_id: str | None = None
        self._rid = 0
        # 强制直连：忽略系统代理（HTTP_PROXY 等环境变量），否则 localhost
        # 请求会被转发到代理端口导致 502/拒绝连接。
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _post(self, method: str, params: dict | None = None) -> dict:
        """发一个 JSON-RPC 请求，返回 result；错误时抛 MCPError。"""
        self._rid += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._rid,
            "method": method,
            "params": params or {},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id

        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener.open(req, timeout=120) as resp:
                # 记录 session id（服务端在 initialize 响应头里返回）
                sid = resp.headers.get("mcp-session-id")
                if sid:
                    self.session_id = sid
                ctype = resp.headers.get("content-type", "")
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise MCPError(
                f"HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')}"
            ) from exc
        except urllib.error.URLError as exc:
            raise MCPError(
                f"连接失败：{exc.reason}（确认服务已启动：python serve_mcp.py）"
            ) from exc

        # 解析响应：可能是纯 JSON，也可能是 SSE（text/event-stream）。
        if "text/event-stream" in ctype:
            data = self._parse_sse(body)
        else:
            data = json.loads(body)
        if "error" in data:
            raise MCPError(f"服务端错误：{data['error']}")
        return data.get("result", {})

    @staticmethod
    def _parse_sse(body: str) -> dict:
        """解析 SSE 响应，返回最后一个 data: 行的 JSON 对象。

        streamable-http 的 SSE 形如：
            event: message
            data: {"jsonrpc":"2.0","id":1,"result":{...}}
        """
        last_data = None
        for line in body.splitlines():
            if line.startswith("data:"):
                last_data = line[len("data:") :].strip()
        if last_data is None:
            raise MCPError(f"无法解析 SSE 响应：{body[:200]}")
        return json.loads(last_data)

    def initialize(self) -> dict:
        """握手，返回服务端信息。"""
        return self._post(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        )

    def list_tools(self) -> list[dict]:
        """列出可用工具。"""
        return self._post("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """调用工具，返回结构化结果。"""
        r = self._post("tools/call", {"name": name, "arguments": arguments or {}})
        # streamable-http 下工具结果可能是 {"content": [...]} 或直接 dict
        content = r.get("content", [])
        if content:
            # MCP 标准：content 是 [{type:"text", text:"..."}]
            for item in content:
                if item.get("type") == "text":
                    try:
                        return json.loads(item["text"])
                    except (json.JSONDecodeError, TypeError):
                        return {"text": item["text"]}
        return r


# ── 各方法的参数解析 ────────────────────────────────────────────


def _args_search(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(prog="search")
    ap.add_argument("--query", required=True, help="查询语句")
    ap.add_argument("--exp", default="E3", help="检索口径 E0~E4（默认 E3）")
    ap.add_argument("--codes", nargs="*", default=None, help="股票代码列表")
    ap.add_argument("--year_from", type=int, default=None)
    ap.add_argument("--year_to", type=int, default=None)
    ap.add_argument(
        "--rerank", type=lambda s: s.lower() in ("1", "true", "yes"), default=None
    )
    a = ap.parse_args(argv)
    out = {"query": a.query, "exp": a.exp}
    if a.codes is not None:
        out["codes"] = a.codes
    if a.year_from is not None:
        out["year_from"] = a.year_from
    if a.year_to is not None:
        out["year_to"] = a.year_to
    if a.rerank is not None:
        out["rerank"] = a.rerank
    return out


def _args_get_chunk_detail(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(prog="get_chunk_detail")
    ap.add_argument("--chunk_id", required=True)
    a = ap.parse_args(argv)
    return {"chunk_id": a.chunk_id}


METHOD_PARSERS = {
    "search": _args_search,
    "get_chunk_detail": _args_get_chunk_detail,
}


# ── 输出格式化 ──────────────────────────────────────────────────


def _print_result(name: str, result: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"方法：{name}  返回结果")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def _print_search(result: dict) -> None:
    """search 结果友好打印（命中列表逐条展示）。"""
    print(f"\n查询：{result.get('query')}")
    print(f"实验组：{result.get('exp')}   命中数：{result.get('n_hits')}")
    timings = result.get("timings_ms") or {}
    print(f"耗时(ms)：{timings}")
    print("-" * 60)
    for h in result.get("hits", []):
        print(
            f"[#{h.get('rank')}] {h.get('company_name')}({h.get('stock_code')}) "
            f"{h.get('fiscal_year')}年 第{h.get('page_num')}页 {h.get('chunk_type')}"
        )
        print(f"    章节：{h.get('section')}   chunk_id：{h.get('chunk_id')}")
        print(f"    正文：{h.get('text', '')[:120]}")
        print(
            f"    rrf={h.get('rrf_score')}  dense={h.get('score_dense')}  "
            f"sparse={h.get('score_sparse')}"
            + (f"  rerank={h.get('rerank_score')}" if "rerank_score" in h else "")
        )
        print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="MCP 服务客户端测试脚本（模拟外部 Agent 调用）",
        add_help=True,
    )
    ap.add_argument("method", nargs="?", help="要调用的方法名（缺省进入交互菜单）")
    ap.add_argument(
        "method_args", nargs=argparse.REMAINDER, help="方法参数（--key value）"
    )
    ap.add_argument(
        "--url", default=DEFAULT_URL, help=f"服务地址（默认 {DEFAULT_URL}）"
    )
    ap.add_argument("--list", action="store_true", help="仅列出服务端可用工具")
    # 先把 method 之后的内容整体当作 method_args，避免 --url 被吞
    argv = sys.argv[1:]
    # 手动拆：找 --url / --list，其余归 method / method_args
    url = DEFAULT_URL
    list_only = False
    rest: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--url":
            url = argv[i + 1]
            i += 2
        elif argv[i] == "--list":
            list_only = True
            i += 1
        else:
            rest.append(argv[i])
            i += 1

    method = rest[0] if rest else None
    method_args = rest[1:] if len(rest) > 1 else []

    client = Client(url)

    # 握手
    try:
        info = client.initialize()
    except MCPError as exc:
        print(f"[错误] 握手失败：{exc}")
        return 1
    print(
        f"[已连接] {info.get('serverInfo', {}).get('name', '?')} "
        f"v{info.get('serverInfo', {}).get('version', '?')}"
    )

    # 列出工具
    if list_only or method is None:
        tools = client.list_tools()
        print(f"\n服务端可用工具（{len(tools)} 个）：")
        for t in tools:
            print(f"  - {t.get('name')}: {t.get('description', '')[:60]}")
        if list_only:
            return 0
        # 交互菜单
        print("\n请输入要调用的方法名，或直接回车退出：")
        method = input("方法名> ").strip()
        if not method:
            return 0

    # 构造参数
    if method in METHOD_PARSERS:
        arguments = METHOD_PARSERS[method](method_args)
    elif method_args:
        # 通用 --key value 解析
        arguments = {}
        it = iter(method_args)
        for k in it:
            key = k.lstrip("-")
            val = next(it, None)
            arguments[key] = val
    else:
        arguments = {}

    print(f"\n调用方法：{method}  参数：{arguments}")

    try:
        result = client.call_tool(method, arguments)
    except MCPError as exc:
        print(f"[错误] 调用失败：{exc}")
        return 1

    # 按方法定制输出
    if method == "search":
        _print_search(result)
    else:
        _print_result(method, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
