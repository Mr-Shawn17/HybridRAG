# 环境自检脚本（M0 验收项）
# 执行：python src/check_env.py
# 检查：依赖可导入 / config 可读 / Qdrant 服务端在线 / GPU 可用性（不阻塞）/ 数据产物就位

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 项目根（src/ 的上一级）
sys.path.insert(0, str(ROOT / "src"))

PASS, WARN, FAIL = "✓", "!", "✗"
results = []


def check(name, fn):
    try:
        ok, msg = fn()
        level = PASS if ok else WARN
        results.append((level, name, msg))
    except Exception as exc:  # noqa: BLE001
        results.append((FAIL, name, str(exc)))


# 1. 依赖（导入顺序敏感：jieba 先于 FlagEmbedding 导入会触发 DLL segfault，
#    本项目统一约定 FlagEmbedding 最先导入，见 src/index_documents.py 注释）
def _deps():
    import importlib.metadata as md
    import FlagEmbedding  # noqa: F401  必须最先
    import llama_index.core, qdrant_client, jieba, pymupdf, yaml  # noqa
    ver = lambda pkg: md.version(pkg)  # noqa: E731
    return True, (f"llama-index {ver('llama-index-core')} / qdrant {ver('qdrant-client')} "
                  f"/ pymupdf {ver('pymupdf')} / jieba {ver('jieba')}")

check("核心依赖", _deps)


# 2. config
def _config():
    import yaml
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["embedding"]["model"] == "model/BGE-M3", cfg["embedding"]["model"]
    for k in ("chunks", "embeddings"):  # v1.6：paths.bm25_stats 已废弃（idf 交服务端）
        assert k in cfg["paths"], f"paths.{k} 缺失"
    assert "avgdl_frozen" in cfg["sparse"], "sparse.avgdl_frozen 缺失（v1.6 BM25 口径必需）"
    assert "host" in cfg["qdrant"], "qdrant.host 缺失（v1.4 起服务端模式）"
    return True, (f"chunk={cfg['chunking']['chunk_size']}/{cfg['chunking']['chunk_overlap']} "
                  f"tokenizer={cfg['sparse']['tokenizer']} "
                  f"bm25 k1={cfg['sparse']['k1']} b={cfg['sparse']['b']} "
                  f"avgdl_frozen={cfg['sparse']['avgdl_frozen']}")
check("config.yaml", _config)


# 3. Qdrant 服务端探活（v1.4：Docker 容器取代本地模式，必须先起容器）
def _qdrant():
    import json
    import urllib.request
    import yaml
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    q = cfg["qdrant"]
    url = f"http://{q['host']}:{int(q.get('port', 6333))}/collections"
    with urllib.request.urlopen(url, timeout=5) as r:
        data = json.loads(r.read())
    names = [c["name"] for c in data.get("result", {}).get("collections", [])]
    if "annual_reports" not in names:
        return False, f"服务在线但缺 collection annual_reports（现有 {names}）→ 跑 index_documents 入库"
    return True, f"{url} 在线，collection: {names}"

check("Qdrant 服务端", _qdrant)


# 4. GPU（不阻塞：异常时切 provider: siliconflow）
def _gpu():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return True, f"{name} {mem:.1f}GB"
    return False, "CUDA 不可用 → embedding 切 provider: siliconflow"

check("GPU（不阻塞）", _gpu)


# 4. 数据产物
def _data():
    raw = len(list((ROOT / "data" / "raw").glob("*/*.pdf")))
    parsed = len(list((ROOT / "data" / "parsed").glob("*.json")))
    chunks = len(list((ROOT / "data" / "chunks").glob("*_chunk.json")))
    ok = raw >= 50 and parsed >= 50 and chunks >= 50
    return ok, f"raw={raw} PDF / parsed={parsed} JSON / chunks={chunks} JSON"

check("数据产物（M1+M2a）", _data)


# 6. .env（M3 前需要，现在仅提醒）
def _env():
    if (ROOT / "config" / ".env").exists():
        return True, "已存在"
    return False, "未创建 → 复制 config/.env.example 为 config/.env 并填 key"

check(".env（M3 前需就位）", _env)


print("=" * 60)
for level, name, msg in results:
    print(f"{level} {name}: {msg}")
print("=" * 60)
fails = sum(1 for l, _, _ in results if l == FAIL)
print("结论:", "全部通过" if fails == 0 else f"{fails} 项失败")
