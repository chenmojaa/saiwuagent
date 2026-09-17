"""长期记忆链路测试：DB 存取/去重/召回 + prompt 注入 + extract_facts 解析。

不依赖真实 LLM：extract_facts 的 LLM 调用用 fake model 替换。
"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 用临时 DB，避免污染 data/notes.db。
# 注意：真正生效的是下面那行直接改 settings.sqlite_path。
# data_dir/sqlite_path/notes_dir/chroma_dir 是四个独立字段，都默认 ./data，
# 设 HD_DATA_DIR **无效**（config.py 里没有这个名字）—— 别照抄那种写法，
# 实测会把测试数据写进开发库（候选库那次污染了 22 条）。
tmp = tempfile.mkdtemp()

from app.storage import db as dbm
dbm._engine = None  # reset engine if imported elsewhere
from app.config import settings
settings.sqlite_path = os.path.join(tmp, "test.db")

from app.storage.db import save_facts, list_facts, recall_facts, delete_fact, _norm_fact


def test_save_and_dedup():
  n1 = save_facts(["用户偏好简洁的中文回答", "用户在做 LangGraph 项目"], session_id="s_test")
  assert n1 == 2, n1
  # 完全重复（标点/空白差异）不新增
  n2 = save_facts(["用户偏好简洁的 中文回答。"])
  assert n2 == 0, n2
  # 太短 / 空 跳过
  assert save_facts(["", "ab"]) == 0
  facts = list_facts()
  assert len(facts) == 2
  assert facts[0]["session_id"] == "s_test" or facts[1]["session_id"] == "s_test"
  print("PASS save/dedup")


def test_recall_relevance():
  # 相关事实应排在前面；无关事实靠时间兜底
  save_facts(["用户对海鲜过敏", "用户住在合肥", "用户在学 Rust 语言"])
  got = recall_facts("海鲜过敏能吃什么", limit=4)
  assert any("海鲜" in f for f in got), got
  got2 = recall_facts("怎么学 Rust 快一点", limit=4)
  assert any("Rust" in f for f in got2), got2
  # 完全无关的问题：仍返回最近事实作为背景
  got3 = recall_facts("量子力学是什么", limit=4)
  assert len(got3) >= 2, got3
  print("PASS recall")


def test_delete():
  facts = list_facts()
  fid = facts[0]["id"]
  assert delete_fact(fid) is True
  assert delete_fact(99999) is False
  print("PASS delete")


def test_extract_parse():
  # fake LLM：返回 think 包裹 + JSON 数组
  class FakeResp:
    content = "<think>分析中...</think>[\"用户喜欢用 VSCode\", \"用户周末加班\"]"
  class FakeChat:
    def invoke(self, prompt):
      FakeChat.last_prompt = prompt
      return FakeResp()
  import app.agent.memory as mem
  from app.llm import factory
  orig = factory._build_model
  factory._build_model = lambda **kw: FakeChat()
  try:
    out = mem.extract_facts([
      {"role": "user", "content": "我用 VSCode 写代码，周末经常加班"},
      {"role": "assistant", "content": "了解了..."},
    ], session_id="s_x")
  finally:
    factory._build_model = orig
  assert out == ["用户喜欢用 VSCode", "用户周末加班"], out
  # 已写入 DB（去重后）
  all_facts = [f["content"] for f in list_facts()]
  assert "用户喜欢用 VSCode" in all_facts, all_facts
  # prompt 里包含已有事实列表（避免重复输出）
  assert "已有事实" in FakeChat.last_prompt
  print("PASS extract parse")


def test_extract_no_fact():
  class FakeResp:
    content = "没有新事实，输出 []"
  class FakeChat:
    def invoke(self, prompt): return FakeResp()
  import app.agent.memory as mem
  from app.llm import factory
  orig = factory._build_model
  factory._build_model = lambda **kw: FakeChat()
  try:
    out = mem.extract_facts([{"role": "user", "content": "什么是 RAG 技术" * 5},
                             {"role": "assistant", "content": "RAG 是..." * 10}])
  finally:
    factory._build_model = orig
  assert out == [], out
  print("PASS extract no-fact")


def test_extract_disabled():
  import app.agent.memory as mem
  settings.memory_extraction_enabled = False
  try:
    out = mem.extract_facts([{"role": "user", "content": "x" * 100}])
    assert out == []
  finally:
    settings.memory_extraction_enabled = True
  print("PASS extract disabled")


def test_build_messages_injects_facts():
  from app.agent.context import build_messages
  msgs = build_messages(
    instructions="You are helpful. <<CONTEXT>> <<QUESTION>>",
    chunks=[], history=[], question="hi",
    summary="", profile=None,
    memory_facts=["用户对海鲜过敏", "用户在学 Rust"],
  )
  sys = msgs[0].content
  assert "[long-term memory about this user]" in sys
  assert "用户对海鲜过敏" in sys and "用户在学 Rust" in sys
  # 无事实时不注入块
  msgs2 = build_messages(
    instructions="You are helpful. <<CONTEXT>> <<QUESTION>>",
    chunks=[], history=[], question="hi", memory_facts=[],
  )
  assert "[long-term memory" not in msgs2[0].content
  print("PASS build_messages inject")


if __name__ == "__main__":
  test_save_and_dedup()
  test_recall_relevance()
  test_delete()
  test_extract_parse()
  test_extract_no_fact()
  test_extract_disabled()
  test_build_messages_injects_facts()
  print("\nALL MEMORY TESTS PASSED")
