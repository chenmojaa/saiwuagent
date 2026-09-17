"""联网校验节点（策略 A）的测试。

锁定三件事：
  1. **分支判定**：consistent / conflict / kb_stale / web_only / unverified /
     skipped / disabled 七种 status 各自在什么条件下产生。
  2. **防污染边界**：联网结果**绝不**写主知识库（notes / chunk_fts / 向量库），
     只能进候选库。这是产品边界，比功能本身更重要 —— 一旦破掉，联网的
     噪声就会被永久固化成「用户资料」。
  3. **失败不阻断**：联网挂掉、裁决挂掉、模型返回脏 JSON，都必须降级成
     unverified 继续作答，而不是抛异常把整轮问答打死。

用假 chat 对象注入裁决结果，不打真实模型；web_search 也全部替换掉，
所以这个文件离线可跑。
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---- 测试隔离（必须在 import app.* 之前）----
# 注意：data_dir / sqlite_path / notes_dir / chroma_dir 是**四个独立字段**，
# 都默认指向 ./data，没有共同的前缀。只设 HD_DATA_DIR 是无效的 —— 那个名字
# 根本不存在（config.py 里这几个字段没用 _hd() 包一层），结果就是测试直接
# 往真实的 backend/data/notes.db 写候选，污染开发库。
# 这正是本文件早期版本踩过的坑：22 条测试候选进了真实库，得手工清。
_TMP = tempfile.mkdtemp(prefix="hd_verify_test_")
os.environ["DATA_DIR"] = _TMP
os.environ["SQLITE_PATH"] = os.path.join(_TMP, "notes.db")
os.environ["NOTES_DIR"] = os.path.join(_TMP, "notes")
os.environ["CHROMA_DIR"] = os.path.join(_TMP, "chroma")


# ---------- 测试脚手架 ----------

class _FakeChat:
  """假的裁决模型：invoke 返回预设文本。"""

  def __init__(self, reply: str):
    self.reply = reply
    self.calls = 0

  def invoke(self, payload: str):
    self.calls += 1
    self.payload = payload
    return type("_R", (), {"content": self.reply})()


@contextlib.contextmanager
def _override(**kw):
  """临时改 settings 上的开关。pydantic-settings 的实例是可变对象。"""
  from app.config import settings
  old = {k: getattr(settings, k) for k in kw}
  for k, v in kw.items():
    object.__setattr__(settings, k, v)
  try:
    yield
  finally:
    for k, v in old.items():
      object.__setattr__(settings, k, v)


def _kb_chunk(note_id: str, text: str, title: str = "知识库文档"):
  return {"note_id": note_id, "title": title, "text": text,
          "source_type": "note", "chunk_index": 0, "final_score": 0.9}


def _web_chunk(url: str, text: str, title: str = "网页"):
  return {"note_id": "web:0:" + url, "title": title, "text": text,
          "source_type": "web", "source_url": url, "chunk_index": 0,
          "final_score": 0.5}


def _run_verify(state: dict, *, web=None, verdict=None, search_calls=None):
  """跑一次 verify_node，web_search / 裁决模型都用假的。

  web: 联网返回的 chunk 列表（None 表示联网无结果）
  verdict: 裁决模型返回的字符串；None 表示裁决模型不可用
  """
  import app.agent.nodes.verify as V

  def fake_search(q, max_results=5, fetch_top=2):
    if search_calls is not None:
      search_calls.append(q)
    return list(web or [])

  import app.tools.web_search as WS
  orig_search = WS.web_search
  orig_builder = V._build_adjudicator
  WS.web_search = fake_search
  V._build_adjudicator = lambda st: (_FakeChat(verdict) if verdict is not None else None)
  try:
    return V.verify_node(state)
  finally:
    WS.web_search = orig_search
    V._build_adjudicator = orig_builder


# ============ 1. 开关与模式 ============

def test_disabled_when_switch_off():
  """总开关关闭 -> disabled，且**不联网**。"""
  calls: list = []
  with _override(web_verify_enabled=False):
    out = _run_verify({"query": "腾讯营收", "retrieved_chunks": []},
                      web=[_web_chunk("https://a.com", "x")], search_calls=calls)
  assert out["web_verify_status"] == "disabled", out
  assert calls == [], "开关关闭时不应发起联网"


def test_stale_only_skips_non_time_sensitive():
  """stale_only 模式：问题与材料都无时效信号 -> skipped，不联网。"""
  calls: list = []
  with _override(web_verify_mode="stale_only"):
    out = _run_verify(
      {"query": "什么是光合作用", "retrieved_chunks": [_kb_chunk("n1", "植物把光能转化为化学能。")]},
      web=[_web_chunk("https://a.com", "x")], search_calls=calls)
  assert out["web_verify_status"] == "skipped", out
  assert calls == [], "非时效敏感问题不应联网"


def test_stale_only_triggers_on_year_in_kb():
  """stale_only 模式：知识库片段里出现年份 -> 联网核对。"""
  calls: list = []
  with _override(web_verify_mode="stale_only"):
    _run_verify(
      {"query": "介绍一下这个政策",
       "retrieved_chunks": [_kb_chunk("n1", "2021 年的补贴标准是每台 300 元。")]},
      web=[_web_chunk("https://a.com", "2026 年起取消补贴")], search_calls=calls)
  assert calls, "知识库含年份时应触发联网核对"


def test_always_mode_verifies_even_when_kb_has_hits():
  """always 模式：知识库有结果**也**联网 —— 这是策略 A 相对旧实现的核心改动。"""
  calls: list = []
  with _override(web_verify_mode="always"):
    _run_verify(
      {"query": "什么是光合作用",
       "retrieved_chunks": [_kb_chunk("n1", "植物把光能转化为化学能。")]},
      web=[_web_chunk("https://a.com", "光合作用是把光能变成化学能")],
      verdict=json.dumps({"status": "consistent", "conflicts": [],
                          "stale_kb_refs": [], "note": "一致"}),
      search_calls=calls)
  assert calls, "always 模式下知识库有命中时仍必须联网核对"


# ============ 2. 七种 status 的分支 ============

def test_web_only_when_kb_empty():
  """知识库无材料 -> web_only，联网片段进入上下文。"""
  out = _run_verify({"query": "今天天气", "retrieved_chunks": []},
                    web=[_web_chunk("https://a.com", "今天天气晴，气温 25 度")])
  assert out["web_verify_status"] == "web_only", out
  assert len(out["retrieved_chunks"]) == 1
  assert out["retrieved_chunks"][0]["source_type"] == "web"


def test_unverified_when_web_empty():
  """联网无结果 -> unverified，且**不覆盖**知识库材料。

  注意：节点在不改动片段时干脆不返回 retrieved_chunks 键，靠 LangGraph
  的 state 合并保留原值。所以这里断言的是「没被清空」，而不是「键存在」。
  """
  kb = _kb_chunk("n1", "旧材料")
  out = _run_verify({"query": "q", "retrieved_chunks": [kb]}, web=[])
  assert out["web_verify_status"] == "unverified", out
  assert out.get("retrieved_chunks", [kb]) == [kb], "未能核对时不应丢掉知识库材料"


def test_unverified_when_adjudicator_unavailable():
  """裁决模型起不来 -> unverified，联网材料仍附加进上下文。"""
  kb = _kb_chunk("n1", "旧材料")
  web = [_web_chunk("https://a.com", "新材料")]
  out = _run_verify({"query": "q", "retrieved_chunks": [kb]}, web=web, verdict=None)
  assert out["web_verify_status"] == "unverified", out
  assert len(out["retrieved_chunks"]) == 2


def test_consistent_keeps_both_sources():
  """一致 -> consistent，知识库与联网都在上下文里（以知识库为主由提示词约束）。"""
  kb = _kb_chunk("n1", "腾讯 2025 营收 6600 亿")
  web = [_web_chunk("https://a.com", "腾讯 2025 年营收 6600 亿元")]
  out = _run_verify(
    {"query": "腾讯营收", "retrieved_chunks": [kb]}, web=web,
    verdict=json.dumps({"status": "consistent", "conflicts": [],
                        "stale_kb_refs": [], "note": "数值一致"}))
  assert out["web_verify_status"] == "consistent", out
  assert len(out["retrieved_chunks"]) == 2


def test_conflict_carries_normalized_sources():
  """冲突 -> conflict，且 refs 被翻译成可展示的来源信息。"""
  kb = _kb_chunk("n1", "补贴每台 300 元", title="2021 政策文件")
  web = [_web_chunk("https://gov.cn/x", "补贴已取消", title="2026 新政策")]
  out = _run_verify(
    {"query": "补贴多少", "retrieved_chunks": [kb]}, web=web,
    verdict=json.dumps({
      "status": "conflict",
      "conflicts": [{"claim": "补贴是否仍有效", "kb_says": "每台 300 元",
                     "web_says": "已取消", "kb_refs": [1], "web_refs": ["A"]}],
      "stale_kb_refs": [], "note": "政策口径不一致"}))
  assert out["web_verify_status"] == "conflict", out
  cs = out["web_verify_conflicts"]
  assert len(cs) == 1
  assert cs[0]["claim"] == "补贴是否仍有效"
  assert cs[0]["kb_sources"][0]["note_id"] == "n1"
  assert cs[0]["web_sources"][0]["url"] == "https://gov.cn/x"


def test_kb_stale_drops_whole_note():
  """过期 -> kb_stale，且同 note 的**所有**片段一并剔除（不只被点名那段）。"""
  stale_a = _kb_chunk("old-note", "2021 年标准：300 元")
  stale_b = _kb_chunk("old-note", "2021 年适用范围：全国")
  fresh = _kb_chunk("new-note", "其它无关材料")
  web = [_web_chunk("https://a.com", "补贴已取消，不再发放")]
  out = _run_verify(
    {"query": "补贴", "retrieved_chunks": [stale_a, stale_b, fresh]}, web=web,
    verdict=json.dumps({"status": "kb_stale", "conflicts": [],
                        "stale_kb_refs": [1], "note": "已被新政策取代"}))
  assert out["web_verify_status"] == "kb_stale", out
  assert out["kb_stale_note_ids"] == ["old-note"], out
  kept_ids = [c["note_id"] for c in out["retrieved_chunks"]]
  assert "old-note" not in kept_ids, "同一 note 的其它片段也必须剔除"
  assert "new-note" in kept_ids


def test_kb_stale_falls_back_to_web_only_when_all_dropped():
  """过期片段被剔光 -> 降级为 web_only，不能留下空上下文。"""
  web = [_web_chunk("https://a.com", "补贴新政策已生效")]
  out = _run_verify(
    {"query": "补贴", "retrieved_chunks": [_kb_chunk("old", "2021 旧政策")]}, web=web,
    verdict=json.dumps({"status": "kb_stale", "conflicts": [],
                        "stale_kb_refs": [1], "note": ""}))
  assert out["web_verify_status"] == "web_only", out
  assert len(out["retrieved_chunks"]) == 1


def test_insufficient_maps_to_unverified():
  """裁决说 insufficient（联网材料没用）-> unverified，不硬编造结论。"""
  kb = _kb_chunk("n1", "材料")
  web = [_web_chunk("https://ad.com", "广告")]
  out = _run_verify(
    {"query": "q", "retrieved_chunks": [kb]}, web=web,
    verdict=json.dumps({"status": "insufficient", "conflicts": [],
                        "stale_kb_refs": [], "note": "全是广告"}))
  assert out["web_verify_status"] == "unverified", out


def test_unknown_status_falls_back_to_unverified():
  """模型给了没见过的 status -> 按 unverified 处理，不猜。"""
  out = _run_verify(
    {"query": "q", "retrieved_chunks": [_kb_chunk("n1", "m")]},
    web=[_web_chunk("https://a.com", "w")],
    verdict=json.dumps({"status": "banana", "note": ""}))
  assert out["web_verify_status"] == "unverified", out


# ============ 3. 防污染边界（最关键） ============

def test_web_results_never_touch_main_kb():
  """联网结果绝不写 notes / chunk_fts / 向量库，只进候选库。"""
  import app.storage.db as DB
  touched: list[str] = []

  def _boom(*a, **k):
    touched.append("called")
    raise AssertionError("联网结果不得写入主知识库")

  orig_add_fts = DB.add_fts
  DB.add_fts = _boom
  # 向量库走 vector 模块，一并堵死
  try:
    import app.storage.vector as VEC
    orig_add = getattr(VEC, "add_chunks", None)
    if orig_add is not None:
      VEC.add_chunks = _boom
  except Exception:
    VEC, orig_add = None, None

  # 候选库替换成记录器
  import app.storage.candidates as C
  captured: list = []
  orig_add_cand = C.add_candidates
  C.add_candidates = lambda chunks, query="", session_id=None: (
    captured.extend(chunks) or [1])

  try:
    out = _run_verify(
      {"query": "q", "retrieved_chunks": [_kb_chunk("n1", "kb")]},
      web=[_web_chunk("https://a.com", "web 材料")],
      verdict=json.dumps({"status": "consistent", "conflicts": [],
                          "stale_kb_refs": [], "note": ""}))
  finally:
    DB.add_fts = orig_add_fts
    C.add_candidates = orig_add_cand
    if VEC is not None and orig_add is not None:
      VEC.add_chunks = orig_add

  assert touched == [], "联网结果触碰了主知识库写入路径"
  assert len(captured) == 1, "联网结果应写入候选库等待人工审批"
  assert captured[0]["source_url"] == "https://a.com"
  assert out["web_verify_status"] == "consistent"


def test_candidates_written_even_when_adjudication_fails():
  """裁决失败时候选也要落库 —— 否则这次联网白跑，人工也没得审。"""
  import app.storage.candidates as C
  captured: list = []
  orig = C.add_candidates
  C.add_candidates = lambda chunks, query="", session_id=None: (captured.extend(chunks) or [1])
  try:
    _run_verify({"query": "q", "retrieved_chunks": [_kb_chunk("n1", "kb")]},
                web=[_web_chunk("https://a.com", "w")], verdict=None)
  finally:
    C.add_candidates = orig
  assert len(captured) == 1, "裁决不可用时候选仍应入库"


# ============ 4. 解析健壮性 ============

def test_extract_json_strips_think_block():
  """推理模型的思维链里常带示例 JSON，必须剥掉后才取结论。"""
  from app.agent.nodes.verify import _extract_json
  raw = ('<think>我应该输出 {"status": "conflict"} 这种格式</think>\n'
         '{"status": "consistent", "conflicts": [], "stale_kb_refs": [], "note": "ok"}')
  got = _extract_json(raw)
  assert got is not None and got["status"] == "consistent", got


def test_extract_json_tolerates_code_fence():
  """模型有时会包一层 markdown 代码块。"""
  from app.agent.nodes.verify import _extract_json
  raw = '```json\n{"status": "kb_stale", "stale_kb_refs": [1]}\n```'
  got = _extract_json(raw)
  assert got is not None and got["status"] == "kb_stale", got


def test_extract_json_returns_none_on_garbage():
  """完全解析不出 -> None（上层降级为 unverified），不抛异常。"""
  from app.agent.nodes.verify import _extract_json
  assert _extract_json("我不知道该怎么判断") is None
  assert _extract_json("") is None
  assert _extract_json("{不完整的 json") is None


def test_refs_to_indices_drops_out_of_range_and_dirty():
  """refs 脏数据一律丢弃，越界不能变成 IndexError。"""
  from app.agent.nodes.verify import _refs_to_indices
  assert _refs_to_indices([1, 2], 5) == [0, 1]
  assert _refs_to_indices([99], 5) == [], "越界必须丢弃"
  assert _refs_to_indices([0, -1], 5) == [], "0 和负数都不是合法 1-based 编号"
  assert _refs_to_indices(["第2条"], 5) == [1], "应能从脏字符串里抠出数字"
  assert _refs_to_indices("不是列表", 5) == []
  assert _refs_to_indices(None, 5) == []


def test_time_sensitive_detector():
  """时效敏感判定：命中年份/价格/政策等，放行纯概念问题。"""
  from app.agent.nodes.verify import _looks_time_sensitive
  assert _looks_time_sensitive("最新政策是什么", []) is True
  assert _looks_time_sensitive("q", [_kb_chunk("n", "2021 年标准")]) is True
  assert _looks_time_sensitive("q", [_kb_chunk("n", "股价 12.3 元")]) is True
  assert _looks_time_sensitive("什么是光合作用", [_kb_chunk("n", "植物转化光能")]) is False


# ============ 5. 跨模块链路：verify -> 提示词 -> 横幅 ============

def test_conflict_flows_through_prompt_and_banner():
  """冲突状态要同时体现在两处，且分工不重叠：

    - `_forced_notice`（硬保证）：服务端强制插到答案最前面的横幅
    - `format_verify_block`（软约束）：告诉模型正文该怎么写

  两处都必须在，且提示词要明确叫模型**不要重复**横幅 —— 否则会出现
  同一条警告在回答里出现两次。
  """
  from app.agent.context import format_verify_block
  from app.agent.nodes.answer import _forced_notice

  out = _run_verify(
    {"query": "补贴多少", "retrieved_chunks": [_kb_chunk("n1", "每台 300 元")]},
    web=[_web_chunk("https://gov.cn/x", "补贴多少：2026 起已取消")],
    verdict=json.dumps({
      "status": "conflict",
      "conflicts": [{"claim": "补贴是否仍有效", "kb_says": "每台 300 元",
                     "web_says": "2026 起已取消", "kb_refs": [1], "web_refs": ["A"]}],
      "stale_kb_refs": [], "note": "政策口径不一致"}))

  state = {**out, "query": "补贴多少"}

  banner = _forced_notice(state)
  assert "未经裁定" in banner, banner
  assert "补贴是否仍有效" in banner
  assert "每台 300 元" in banner and "已取消" in banner

  block = format_verify_block(state["web_verify_status"], state["web_verify_note"],
                              state["web_verify_conflicts"], [])
  assert "不要重复它" in block, "提示词必须阻止模型重复横幅，否则警告会出现两次"
  assert "不要自行裁定" in block, "必须禁止模型自行判定谁对谁错"

  # 提示词与横幅不能是同一段文本（否则就是重复注入）
  assert banner.strip() not in block


def test_kb_stale_flows_through_prompt_and_banner():
  """过期状态：横幅告知用户「未采用旧内容」，提示词约束模型改用联网口径。

  注意要放一个**不过期**的知识库片段：如果所有知识库片段都被判过期，
  verify 会降级成 web_only（上下文里只剩联网材料），那时不该再挂
  「知识库过期」横幅，而是走 web_only 的说明。
  """
  from app.agent.context import format_verify_block
  from app.agent.nodes.answer import _forced_notice

  out = _run_verify(
    {"query": "补贴",
     "retrieved_chunks": [_kb_chunk("old-doc", "2021 年标准 300 元"),
                          _kb_chunk("keep-doc", "与时效无关的说明材料")]},
    web=[_web_chunk("https://gov.cn/x", "补贴已取消")],
    verdict=json.dumps({"status": "kb_stale", "conflicts": [],
                        "stale_kb_refs": [1], "note": "已被新政策取代"}))
  assert out["web_verify_status"] == "kb_stale", out
  state = {**out, "query": "补贴"}

  banner = _forced_notice(state)
  assert "已过期" in banner and "未采用" in banner, banner
  assert "old-doc" in banner, "横幅应告知剔除了哪些来源，便于用户核对"

  block = format_verify_block(state["web_verify_status"], state["web_verify_note"],
                              [], state["kb_stale_note_ids"])
  assert "过期" in block and "联网结果为准" in block, block


def test_all_kb_stale_degrades_to_web_only_without_stale_banner():
  """知识库片段全被判过期 -> 降级 web_only，且不再挂「知识库过期」横幅。"""
  from app.agent.nodes.answer import _forced_notice

  out = _run_verify(
    {"query": "补贴", "retrieved_chunks": [_kb_chunk("old-doc", "2021 年标准")]},
    web=[_web_chunk("https://gov.cn/x", "补贴已取消")],
    verdict=json.dumps({"status": "kb_stale", "conflicts": [],
                        "stale_kb_refs": [1], "note": ""}))
  assert out["web_verify_status"] == "web_only", out
  assert _forced_notice({**out, "query": "补贴"}) == "", \
    "已降级为 web_only，不应再挂知识库过期横幅"


def test_consistent_gives_soft_guidance_without_banner():
  """一致：只给软约束（以知识库为主），不挂横幅。"""
  from app.agent.context import format_verify_block
  from app.agent.nodes.answer import _forced_notice

  out = _run_verify(
    {"query": "腾讯营收", "retrieved_chunks": [_kb_chunk("n1", "6600 亿")]},
    web=[_web_chunk("https://a.com", "腾讯营收 6600 亿元")],
    verdict=json.dumps({"status": "consistent", "conflicts": [],
                        "stale_kb_refs": [], "note": "数值一致"}))
  assert _forced_notice({**out, "query": "q"}) == "", "一致时不该有横幅"
  block = format_verify_block(out["web_verify_status"], out["web_verify_note"], [], [])
  assert "以知识库材料为主" in block, "一致时应要求以知识库为主输出"


def test_build_messages_injects_verify_block():
  """verify_block 必须真的进了系统提示，否则整套校验结论到不了模型。"""
  from app.agent.context import build_messages, format_verify_block
  block = format_verify_block("conflict", "n", [{"claim": "X", "kb_says": "a",
                                                 "web_says": "b"}], [])
  msgs = build_messages(instructions="INST <<CONTEXT>> <<QUESTION>>",
                        chunks=[_kb_chunk("n1", "材料")], history=[],
                        question="q", verify_block=block)
  sys_text = msgs[0].content
  assert "web verification" in sys_text, "校验指令没有注入系统提示"
  assert "不要自行裁定" in sys_text


def test_verify_block_absent_when_disabled():
  """关掉校验时不应往提示词里塞任何东西。"""
  from app.agent.context import format_verify_block
  assert format_verify_block("disabled") == ""
  assert format_verify_block("skipped") == ""
  assert format_verify_block("") == ""


# ============ 6. 联网结果相关性闸门 ============
#
# 背景（2026-09-17 实测）：Bing 的 HTML 抓取对无 cookie 客户端会返回泛化结果 ——
# 查「腾讯控股 2025 年营收」返回腾讯视频/腾讯网，查「新能源汽车补贴政策 最新」
# 返回汉字「新」的百科词条。query 编码、mkt 参数、绕代理都不解决。
# 这个闸门守的是：**别让无关网页导致误报冲突** —— 那比不校验更糟，
# 会让用户开始怀疑本来正确的知识库内容。

def test_irrelevant_web_results_are_not_adjudicated():
  """Bing 返回泛化结果时 -> unverified，而不是拿垃圾去裁决。"""
  kb = _kb_chunk("n1", "腾讯控股 2025 年营收 6600 亿元")
  # 复刻实测的垃圾结果：只有「腾讯」，没有「控股」「营收」
  web = [_web_chunk("https://v.qq.com", "腾讯视频 - 热门电影电视剧综艺动漫在线观看"),
         _web_chunk("https://www.qq.com", "腾讯网 - 首页")]
  out = _run_verify({"query": "腾讯控股 2025 年营收", "retrieved_chunks": [kb]},
                    web=web, verdict=json.dumps({"status": "conflict", "conflicts": [
                      {"claim": "营收不一致", "kb_says": "6600 亿", "web_says": "?"}]}))
  assert out["web_verify_status"] == "unverified", \
    "无关网页不得进入裁决（否则会误报冲突），实际=%s" % out


def test_irrelevant_web_results_are_not_stored_as_candidates():
  """无关结果也不该占人工审核队列。"""
  import app.storage.candidates as C
  captured: list = []
  orig = C.add_candidates
  C.add_candidates = lambda chunks, query="", session_id=None: (captured.extend(chunks) or [1])
  try:
    _run_verify(
      {"query": "新能源汽车补贴政策 最新", "retrieved_chunks": [_kb_chunk("n1", "旧政策")]},
      web=[_web_chunk("https://baike.baidu.com/x", "新 （汉语汉字）_百度百科")],
      verdict=json.dumps({"status": "conflict", "conflicts": []}))
  finally:
    C.add_candidates = orig
  assert captured == [], "无关的联网结果不应进候选库"


def test_relevant_web_results_still_go_through():
  """真相关的材料必须放行，不能因为闸门过严把功能挡死。"""
  kb = _kb_chunk("n1", "腾讯控股 2025 年营收 6600 亿元")
  web = [_web_chunk("https://finance.example.com/t", "腾讯控股 2025 年营收 6580 亿元，同比增长 8%")]
  out = _run_verify(
    {"query": "腾讯控股 2025 年营收", "retrieved_chunks": [kb]}, web=web,
    verdict=json.dumps({"status": "consistent", "conflicts": [],
                        "stale_kb_refs": [], "note": "数值接近"}))
  assert out["web_verify_status"] == "consistent", out


def test_relevance_gate_is_lenient_for_short_queries():
  """短查询命中 1/3 词要放行（阈值 0.25 是实测标定出来的）。"""
  from app.agent.nodes.verify import _web_material_is_relevant
  assert _web_material_is_relevant(
    "补贴多少", [_web_chunk("https://a.com", "补贴已取消")]) is True


def test_query_terms_extraction():
  """2-gram 抽取：中文切 bigram，英文/数字整词保留。"""
  from app.agent.nodes.verify import _query_terms
  terms = _query_terms("腾讯控股 2025 年营收")
  assert "腾讯" in terms and "控股" in terms and "营收" in terms, terms
  assert "2025" in terms, terms
  assert "tencent" in _query_terms("Tencent revenue"), "英文应整词保留"
  assert _query_terms("！！！") == [], "纯符号应抽不出词"


def test_relevance_gate_passes_when_no_terms():
  """抽不出词时（纯符号查询）不拦，避免闸门误伤。"""
  from app.agent.nodes.verify import _web_material_is_relevant
  assert _web_material_is_relevant("？？？", [_web_chunk("https://a.com", "任意内容")]) is True
  assert _web_material_is_relevant("q", [_web_chunk("https://a.com", "任意内容")]) is True


# ============ runner ============

def main() -> int:
  names = sorted(k for k in list(globals()) if k.startswith("test_"))
  passed = failed = 0
  for name in names:
    try:
      globals()[name]()
      passed += 1
      print("PASS %s" % name)
    except AssertionError as e:
      print("FAIL %s: %s" % (name, e))
      failed += 1
    except Exception as e:
      print("ERROR %s: %s: %s" % (name, type(e).__name__, e))
      failed += 1
  print("=" * 40)
  print("Passed: %d/%d  Failed: %d" % (passed, len(names), failed))
  return 0 if failed == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
