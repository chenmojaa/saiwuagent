"""Regression tests for the tool-call permission gate in answer.py.

修的是这个线上 bug（2026-09-17 截图复现）：

    [Error] {"detail":"'StructuredTool' object has no attribute 'get'"}

`answer.py` 的权限门控把 `tool` 当成 dict 用：

    if (... and tool.get("server") not in turn_approved_targets):

但 `_tool_by_name()` 返回的是 LangChain 的 `StructuredTool` **对象**
（用 `getattr(t, "name")` 匹配出来的）。于是每一次 `mcp_invoke` 都在
门控这一行炸掉 —— 模型刚调完 `mcp_discover_tools` 就拿到报错，
用户看到的就是「AI 说自己没有联网能力」。

修法：目标 MCP server 从 `args["server_id"]` 取（`mcp_invoke` 是通用工具，
真正的目标在参数里）。这个文件锁住两个前提条件。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_tool_by_name_returns_object_not_dict():
    """_tool_by_name 返回的是工具对象，调用方不能对它用 .get()。"""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel

    from app.agent.nodes.answer import _tool_by_name

    class _In(BaseModel):
        server_id: str = ""

    tool = StructuredTool.from_function(
        func=lambda server_id="": "ok",
        name="mcp_invoke",
        description="invoke an MCP tool",
        args_schema=_In,
    )

    found = _tool_by_name([tool], "mcp_invoke")
    assert found is tool, "应当按 name 匹配到同一个对象"
    assert not isinstance(found, dict), "返回的是对象，不是 dict"
    # 这一条正是崩溃的原因：对象没有 .get()
    assert not hasattr(found, "get"), "StructuredTool 没有 .get()，调用方不许这么用"
    assert getattr(found, "name", None) == "mcp_invoke"
    print("PASS test_tool_by_name_returns_object_not_dict")


def test_tool_by_name_missing_returns_none():
    from app.agent.nodes.answer import _tool_by_name
    assert _tool_by_name([], "mcp_invoke") is None
    print("PASS test_tool_by_name_missing_returns_none")


def test_mcp_invoke_schema_exposes_server_id():
    """门控按 args["server_id"] 做 per-target 批准，字段名必须对得上。

    如果哪天把字段改名而忘了同步 answer.py，per-target 批准会永远匹配不上
    （表现为每次调用都弹授权框），这个断言会先炸。
    """
    from app.agent.tools.mcp_tools import _MCPInvokeInput

    fields = _MCPInvokeInput.model_fields
    assert "server_id" in fields, "mcp_invoke 的入参必须有 server_id"
    assert "tool_name" in fields
    assert "arguments" in fields
    print("PASS test_mcp_invoke_schema_exposes_server_id")


def test_permission_target_derivation():
    """复刻 answer.py 里 target 的推导逻辑，锁住 fail-closed 语义。"""
    def target_of(args):
        return (args.get("server_id") or "") if isinstance(args, dict) else ""

    assert target_of({"server_id": "playwright"}) == "playwright"
    # 缺 server_id / 非法 args -> 空串 -> 永远不等于已批准集合 -> 再问一次
    assert target_of({}) == ""
    assert target_of({"server_id": None}) == ""
    assert target_of(None) == ""
    assert target_of("not-a-dict") == ""
    print("PASS test_permission_target_derivation")


def main() -> int:
    test_names = sorted([k for k in globals().keys() if k.startswith("test_")])
    passed = 0
    failed = 0
    for name in test_names:
        try:
            globals()[name]()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print("=" * 40)
    print(f"Passed: {passed}/{len(test_names)}  Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
