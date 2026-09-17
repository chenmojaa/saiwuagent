"""检查 MCP 服务是否真的可用（不是只看配置）。

用法：
    python scripts/check_mcp.py                 # 检查所有 enabled 的服务，只做握手 + tools/list
    python scripts/check_mcp.py playwright      # 按 preset_id 或 name 过滤
    python scripts/check_mcp.py playwright --browse   # 额外真的开一次浏览器联网

为什么需要它：UI 上的「测试」按钮只检查启动命令是否存在（见 api/mcp.py
test_server），不握手、不列工具，配置写错也能显示通过。这个脚本走完整的
stdio JSON-RPC 握手，能真正区分「配好了」和「只是填了字段」。

注意 list_tools() 失败时返回 [] 而不是抛异常，所以空列表即失败。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agent.tools.mcp_client import MCPServerSpec, MCPSession  # noqa: E402

REGISTRY = ROOT / "backend" / "data" / "mcp" / "servers.json"


def load_servers() -> list[dict]:
    if not REGISTRY.is_file():
        print(f"找不到注册表：{REGISTRY}")
        raise SystemExit(1)
    return json.loads(REGISTRY.read_text(encoding="utf-8")).get("servers") or []


def check(entry: dict, browse: bool) -> bool:
    spec = MCPServerSpec.from_registry_entry(entry)
    label = f"{entry.get('name')} ({entry.get('id')})"
    print(f"\n=== {label} ===")
    print(f"    {spec.command} {' '.join(spec.args)}")

    session = MCPSession(spec, cwd=str(ROOT), init_timeout=60.0, call_timeout=120.0)
    try:
        t0 = time.time()
        tools = session.list_tools()
        if not tools:
            print("    FAIL  握手或 tools/list 返回空")
            return False
        print(f"    OK    {len(tools)} 个工具（{time.time() - t0:.1f}s）")

        if not browse:
            return True

        # 真开一次浏览器，验证 Chromium 就绪 + 出网正常。
        # 这一步是 tools/list 覆盖不到的：列工具不需要浏览器，也不碰网络。
        t1 = time.time()
        out = session.call("browser_navigate", {"url": "https://www.bing.com/search?q=test"})
        title = session.call("browser_evaluate", {"function": "() => document.title"})
        ok = "Page Title" in out and "### Result" in title
        print(f"    {'OK  ' if ok else 'FAIL'}  浏览器联网（{time.time() - t1:.1f}s）")
        for line in out.splitlines():
            if line.startswith("- Page Title"):
                print(f"          {line.strip()}")
        return ok
    finally:
        session.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", help="按 preset_id 或 name 过滤（不区分大小写）")
    ap.add_argument("--browse", action="store_true", help="额外真的开浏览器联网验证")
    ap.add_argument("--all", action="store_true", help="包含已禁用的服务")
    args = ap.parse_args()

    servers = load_servers()
    if not args.all:
        servers = [s for s in servers if s.get("enabled")]
    if args.filter:
        needle = args.filter.lower()
        servers = [
            s for s in servers
            if needle in str(s.get("preset_id", "")).lower()
            or needle in str(s.get("name", "")).lower()
        ]
    if not servers:
        print("没有匹配的（已启用的）服务。用 --all 可包含已禁用的。")
        return 1

    results = [check(s, args.browse) for s in servers]
    passed = sum(results)
    print(f"\n结果：{passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
