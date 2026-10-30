# -*- coding: utf-8 -*-
"""按需启动 MCP 服务的启动器（v1.4 起迁入 scripts/，逻辑在 src/mcp_server.py）。

职责单一：只做「拉起 + 探活」，真正的服务逻辑在 src/mcp_server.py。
  - 启动前探活：端口已有实例则直接复用，不重复拉起；
  - 启动后轮询 HTTP 端点直到就绪，打印接入地址。

用法（项目根目录，py_rag 环境）：
  python scripts/serve_mcp.py                  # host/port 读 config mcp 段（默认 127.0.0.1:8100）
  python scripts/serve_mcp.py --port 9000      # 换端口（覆盖 config）
  python scripts/serve_mcp.py --host 0.0.0.0   # 监听全部网卡（局域网，自担风险）
  python scripts/serve_mcp.py --log-path D:/logs/mcp.log   # 自定义日志路径

接入地址：http://<host>:<port>/mcp，详见 docs/mcp.md。

日志：默认写到 <项目根>/log/mcp_server.log（LOG_PATH 常量）。改这里的
LOG_PATH 或加 --log-path 参数即可自定义，路径经 MCP_LOG_PATH 环境变量
传给 src/mcp_server.py。

解释器：默认用「运行本脚本的那个 python」（用 py_rag 的 python 启动即可，
不再硬编码本机 conda 路径）；如需指定别的解释器，设 PYTHON_BIN 环境变量。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # scripts/ 的上一级 = 项目根
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
MCP_SCRIPT = ROOT / "src" / "mcp_server.py"

# 日志路径：默认 <项目根>/log/mcp_server.log。想改日志位置就改这一行，
# 或运行时用 --log-path 参数覆盖。
LOG_PATH = ROOT / "log" / "mcp_server.log"


def _read_mcp_defaults() -> tuple[str, int]:
    """读 config.yaml 的 mcp 段，返回 (host, port) 默认值。

    仅用于探活/打印文案，不传给子进程——真正的端口解析由 mcp_server.py
    自己读 config 决定（保持单一真相源）。config 缺失/解析失败时退
    127.0.0.1:8100。
    """
    host, port = "127.0.0.1", 8100
    try:
        import yaml

        cfg_path = ROOT / "config" / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                mcp_cfg = (yaml.safe_load(f) or {}).get("mcp") or {}
            host = str(mcp_cfg.get("host", host))
            port = int(mcp_cfg.get("port", port))
    except Exception:  # noqa: BLE001  配置读取失败不阻塞启动，退默认值
        pass
    return host, port


def _endpoint_alive(host: str, port: int, timeout: float = 1.5) -> bool:
    """探测端点是否在监听（不判断返回码，能连上 TCP 拿到响应即视为在跑）。

    streamable-http 端点对裸 GET 会返回 405/502 等，只要 TCP 建立、拿到
    任何 HTTP 响应即说明服务在监听。
    """
    url = f"http://{host}:{port}/mcp"
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # 有 HTTP 响应（哪怕 405/502）即视为在监听
    except Exception:
        return False


def _wait_ready(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _endpoint_alive(host, port):
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="按需启动年报混合检索 MCP 服务（streamable-http）"
    )
    ap.add_argument("--host", default=None, help="监听地址（默认读 config mcp 段，再退 127.0.0.1）")
    ap.add_argument("--port", type=int, default=None, help="监听端口（默认读 config mcp 段，再退 8100）")
    ap.add_argument(
        "--log-path",
        default=str(LOG_PATH),
        help=f"日志文件路径（默认 {LOG_PATH}）",
    )
    args = ap.parse_args()

    if not Path(PYTHON).exists():
        print(f"[错误] 找不到 python 解释器：{PYTHON}")
        print("       请用 py_rag 环境的 python 运行本脚本，或设 PYTHON_BIN 指定")
        return 1

    # 解析实际 host/port：用户显式传入优先，否则读 config（不传给子进程，
    # 让 mcp_server.py 自己读 config 保持单一真相源）
    cfg_host, cfg_port = _read_mcp_defaults()
    host = args.host or cfg_host
    port = args.port if args.port is not None else cfg_port

    url = f"http://{host}:{port}/mcp"

    # 1. 已有实例则复用
    if _endpoint_alive(host, port):
        print(f"[复用] 端口 {port} 已有 MCP 服务在跑，直接使用：{url}")
        return 0

    # 1.5 启动前环境检查（v1.11：硬项未就绪则中止，软项仅提示）
    #      key 检查前先加载 config/.env + Machine 注册表兜底，保证与运行时
    #      真实可见的环境变量一致（retrieve._load_dotenv 为幂等 setdefault）。
    #      sys.path 已含 src/（见下方 import），此处延迟导入避免重复加载模型。
    sys.path.insert(0, str(ROOT / "src"))
    import retrieve as _R  # noqa: E402  触发 _load_dotenv 兜底
    import env_check  # noqa: E402

    _R._load_dotenv()
    if not env_check.run_checks("mcp"):
        print()
        print("[错误] 环境检查未通过，MCP 服务未启动。请按上方 ✗ 项提示修复后重试。")
        return 1

    # 2. 拉起服务（前台运行，Ctrl+C 停止）
    print(f"[启动] 正在拉起 MCP 服务：{url}")
    print(f"       解释器：{PYTHON}")
    print(f"       日志：{args.log_path}")
    print("       （首次 search 触发 BGE-M3 模型加载，约 20s）")
    print("       （按 Ctrl+C 停止服务）")
    print()

    # 日志路径经 MCP_LOG_PATH 传给 mcp_server.py（继承当前环境变量）
    env = os.environ.copy()
    env["MCP_LOG_PATH"] = args.log_path

    # 用户显式传 --host/--port 才传给子进程；未传则让 mcp_server.py 自己读 config
    cmd = [PYTHON, str(MCP_SCRIPT)]
    if args.host is not None:
        cmd += ["--host", args.host]
    if args.port is not None:
        cmd += ["--port", str(args.port)]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)

    # 3. 轮询就绪
    try:
        if _wait_ready(host, port):
            print(f"[就绪] 服务已启动，接入地址：{url}")
        else:
            print(
                "[警告] 60s 内未探到服务就绪，可能模型加载较慢或启动失败，"
                f"请查看 {args.log_path}"
            )
        # 前台挂起，跟随服务进程（Ctrl+C 时一并退出）
        proc.wait()
    except KeyboardInterrupt:
        print("\n[停止] 正在停止服务...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
