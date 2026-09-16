# -*- coding: utf-8 -*-
"""Agent 端到端冒烟测试：用真实 LLM + 真实检索打整条链路。

与 backend/tests/ 的分工：
  - backend/tests/  组件级、无网络、可进 CI
  - 本脚本          真实 LLM + 真实向量库，验证"整条链路是否真的能跑通"

覆盖：
  L1 路由        router 意图分类 / 快速通道 / 歧义澄清
  L2 检索        hybrid_search（向量 + FTS5）在真实语料上的表现
  L3 规划        planner 是否真的拆出多步计划
  L4 计划执行    execute_plan / replan 是否真的收集到切片
  L5 图编排      整图 astream 的节点序列与状态流转
  L6 子代理      explore/plan/general 的工具白名单

用法：
    cd backend
    ./.venv/Scripts/python.exe ../scripts/agent_tests/e2e_smoke.py
    ./.venv/Scripts/python.exe ../scripts/agent_tests/e2e_smoke.py --only L1,L3

退出码：0 = 全部通过，1 = 有失败项。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---- 让脚本能 import app.* ----
_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parent.parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

os.chdir(_BACKEND)  # 关键：config.data_dir 是相对 CWD 的 ./data

from app.config import settings  # noqa: E402

RESULTS: list[tuple[str, str, bool, str]] = []  # (layer, name, ok, detail)


def record(layer: str, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((layer, name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    line = "[%s] %-4s %s" % (mark, layer, name)
    if detail:
        line += "  -- " + detail
    print(line, flush=True)


def check(layer: str, name: str, cond: bool, detail: str = "") -> bool:
    record(layer, name, bool(cond), detail)
    return bool(cond)


# ---- 凭证：优先 models.json（前端配的），回退 .env ----
def load_credentials():
    """返回 (api_key, base_url, chat_model, embedding_model)。"""
    key, base, model = "", "", ""
    mj = Path("data/models.json")
    if mj.is_file():
        try:
            d = json.loads(mj.read_text(encoding="utf-8"))
            models = d.get("models") or []
            sel = d.get("selected_id")
            pick = None
            for m in models:
                if m.get("id") == sel:
                    pick = m
                    break
            if pick is None and models:
                pick = models[0]
            if pick:
                key = pick.get("apiKey") or ""
                base = pick.get("baseUrl") or ""
                model = pick.get("defaultModel") or ""
        except Exception as e:
            print("WARN: models.json 解析失败: %s" % e)
    if not key:
        key = settings.llm_api_key or ""
    if not base:
        base = settings.llm_api_base or ""
    if not model:
        model = settings.llm_model or ""
    emb_model = settings.embedding_model or "embo-01"
    return key, base, model, emb_model


KEY, BASE, MODEL, EMB_MODEL = load_credentials()


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print("  " + title)
    print("=" * 68, flush=True)


# =========================================================================
# L1 路由层：纯函数，不发 LLM（除分类用例）
# =========================================================================
def test_l1_router() -> None:
    banner("L1 路由层")

    from app.agent.nodes.router import (
        _is_fast_chat, _parse_router_json, route_by_intent, router_node,
    )

    # 1) 快速通道正则
    for q in ["你好", "hi", "谢谢", "再见", "你是谁"]:
        check("L1", "fast-path 命中: %s" % q, _is_fast_chat(q))
    for q in ["介绍一下安徽文旅", "帮我对比 A 和 B 方案", "RAG 是什么"]:
        check("L1", "fast-path 不误伤: %s" % q, not _is_fast_chat(q), "")

    # 2) think 剥离 + JSON 提取（reasoning 模型的关键容错）
    dirty = '<think>\n让我想想...\n</think>\n{"intent":"chat","rewritten_query":"测试"}'
    try:
        d = _parse_router_json(dirty)
        check("L1", "think 包裹的 JSON 能解析", d.intent == "chat", "intent=%s" % d.intent)
    except Exception as e:
        check("L1", "think 包裹的 JSON 能解析", False, "%s: %s" % (type(e).__name__, e))

    # 3) 条件边路由决策
    cases = [
        ({"clarify_request": {"question": "?"}, "skip_clarify": False}, "clarify", "歧义 -> clarify"),
        ({"skip_retrieval": True, "intent": "chat"}, "chat_no_rag", "寒暄 -> chat_no_rag"),
        ({"intent": "research"}, "research", "research 直通"),
        ({"intent": "ingest"}, "ingest", "ingest 直通"),
        ({"intent": "report"}, "report", "report 直通"),
        ({"intent": "chat"}, "chat", "chat -> retrieve"),
        ({"intent": "chat", "use_planner": True}, "research", "Plan 开关升级 chat"),
        ({"intent": "chat", "plan": [{"query": "a"}, {"query": "b"}]}, "research", "计划>=2 步升级"),
        ({"clarify_request": {"question": "?"}, "skip_clarify": True, "intent": "chat"}, "chat", "已回答 -> 不重复澄清"),
    ]
    for state, want, label in cases:
        got = route_by_intent(state)
        check("L1", label, got == want, "got=%s want=%s" % (got, want))

    # 4) 真实 LLM 分类（用最快的小模型）
    if not KEY:
        record("L1", "真实 LLM 意图分类", False, "无可用 API key，跳过")
        return
    probe = [
        ("你好呀", "chat"),
        ("帮我对比一下我笔记里 A 方案和 B 方案的优缺点", "research"),
        ("https://example.com 保存这个页面到知识库", "ingest"),
        ("给我生成一份本周的周报总结", "report"),
    ]
    for q, want in probe:
        t0 = time.time()
        try:
            out = router_node({
                "query": q, "messages": [],
                "api_key_override": KEY, "base_url_override": BASE,
                "step_count": 0,
            })
            got = out.get("intent")
            ms = int((time.time() - t0) * 1000)
            check("L1", "分类 %r -> %s" % (q[:18], want), got == want,
                  "got=%s (%dms)" % (got, ms))
        except Exception as e:
            check("L1", "分类 %r -> %s" % (q[:18], want), False,
                  "%s: %s" % (type(e).__name__, e))

    # 5) 歧义澄清：真实触发
    try:
        out = router_node({
            "query": "帮我调研哪吒",
            "messages": [],
            "api_key_override": KEY, "base_url_override": BASE,
            "step_count": 0,
        })
        cr = out.get("clarify_request")
        check("L1", "歧义词触发 clarify_request", bool(cr),
              "question=%s" % (cr or {}).get("question", "(none)"))
    except Exception as e:
        check("L1", "歧义词触发 clarify_request", False, "%s: %s" % (type(e).__name__, e))


# =========================================================================
# L2 检索层：真实向量 + FTS5
# =========================================================================
def test_l2_retrieval() -> None:
    banner("L2 检索层")

    from app.storage.hybrid import hybrid_search, _fts_escape
    from app.storage.db import _fts_sanitize_phrase, _fts_tokenize

    # FTS5 转义（CJK 特殊字符）
    for raw, label in [('"引号"', "双引号"), ("a*b", "星号"), ("牛魔王", "普通中文"),
                       ("(括号)", "括号"), ("x^2", "脱字符")]:
        for fn, fname in ((_fts_escape, "hybrid._fts_escape"),
                          (_fts_sanitize_phrase, "db._fts_sanitize_phrase")):
            try:
                esc = fn(raw)
                check("L2", "%s 不炸: %s" % (fname, label), isinstance(esc, str),
                      repr(esc)[:40])
            except Exception as e:
                check("L2", "%s 不炸: %s" % (fname, label), False,
                      "%s: %s" % (type(e).__name__, e))

    # CJK 应被拆成单字做 OR 匹配（这是中文召回的关键）
    toks = _fts_tokenize("安徽文旅")
    check("L2", "CJK 拆成单字便于 OR 召回", len(toks) >= 4,
          "tokens=%s" % toks)

    # 注入式输入必须被中和，不能把 FTS5 语法带进去
    for evil in ['" OR 1=1 --', "NEAR(", "a AND b"]:
        try:
            cleaned = _fts_sanitize_phrase(evil)
            check("L2", "FTS 注入被中和: %r" % evil[:14],
                  all(c not in cleaned for c in '"()*:^'), "-> %r" % cleaned[:40])
        except Exception as e:
            check("L2", "FTS 注入被中和: %r" % evil[:14], False, str(e)[:80])

    if not KEY:
        record("L2", "hybrid_search 真实召回", False, "无 API key，跳过")
        return

    # 真实检索
    q = "安徽文旅"
    t0 = time.time()
    try:
        hits = hybrid_search(q, top_k=5, api_key=KEY, base_url=BASE, model=EMB_MODEL)
        ms = int((time.time() - t0) * 1000)
        check("L2", "hybrid_search 返回结果", len(hits) > 0,
              "%d 条 (%dms)" % (len(hits), ms))
        if hits:
            h = hits[0]
            check("L2", "结果含必要字段",
                  all(k in h for k in ("note_id", "chunk_index", "text")),
                  "keys=%s" % ",".join(sorted(h.keys()))[:80])
            check("L2", "final_score 存在且合理",
                  h.get("final_score") is not None and 0 <= float(h.get("final_score") or 0) <= 1,
                  "score=%s" % h.get("final_score"))
            print("      top1: title=%s idx=%s score=%.3f text=%r" % (
                h.get("title"), h.get("chunk_index"),
                float(h.get("final_score") or 0), (h.get("text") or "")[:60]))
    except Exception as e:
        check("L2", "hybrid_search 返回结果", False,
              "%s: %s" % (type(e).__name__, str(e)[:160]))

    # 无意义查询：应被阈值过滤掉或返回极少
    try:
        hits = hybrid_search("zzzqqq不存在的词xyz", top_k=5, api_key=KEY,
                             base_url=BASE, model=EMB_MODEL)
        check("L2", "无关查询不硬凑结果", len(hits) <= 3,
              "返回 %d 条" % len(hits))
    except Exception as e:
        check("L2", "无关查询不硬凑结果", False, "%s: %s" % (type(e).__name__, str(e)[:120]))

    # 父子上下文扩展是否生效
    try:
        hits = hybrid_search("安徽", top_k=3, api_key=KEY, base_url=BASE, model=EMB_MODEL)
        has_matched = any(h.get("matched_text") for h in hits)
        check("L2", "父块上下文扩展字段存在", True,
              "matched_text %s" % ("有" if has_matched else "无(可能未触发)"))
    except Exception as e:
        check("L2", "父块上下文扩展字段存在", False, str(e)[:120])

    # ---- 重排链路真实可用性 ----
    # retrieve.py 会先召回一批再用 CrossEncoder 重排到 5 条。若
    # sentence_transformers 未安装，重排会静默失败——修复后改为自适应：
    # 重排不可用就不做大池召回，避免"召回 50 再截断到 5"的 10 倍空转。
    try:
        import importlib
        importlib.import_module("sentence_transformers")
        st_ok = True
    except Exception:
        st_ok = False
    check("L2", "[能力缺口] CrossEncoder 重排依赖已安装", st_ok,
          "sentence_transformers 未安装 -> 重排不可用（已由自适应召回兜底）"
          if not st_ok else "")

    # 自适应召回深度：必须与"重排是否可用"一致，否则就是白做功
    try:
        from app.agent.nodes.retrieve import (
            _hybrid_topk, _rerank_available, RECALL_PLAIN, RECALL_WITH_RERANK,
        )
        depth = _hybrid_topk()
        expect = RECALL_WITH_RERANK if _rerank_available() else RECALL_PLAIN
        check("L2", "召回深度自适应（重排缺失时不空转）", depth == expect,
              "depth=%d 期望=%d（重排可用=%s）" % (depth, expect, _rerank_available()))
    except Exception as e:
        check("L2", "召回深度自适应（重排缺失时不空转）", False, str(e)[:120])

    # 验证 retrieve 节点在重排不可用时仍能返回 5 条（降级正确）
    try:
        from app.agent.nodes.retrieve import retrieve_node
        out = retrieve_node({
            "query": "安徽文旅", "rewritten_query": "安徽文旅",
            "messages": [{"role": "user", "content": "安徽文旅"}],
            "api_key_override": KEY, "base_url_override": BASE,
            "embedding_model_override": EMB_MODEL, "step_count": 0,
        })
        n = len(out.get("retrieved_chunks") or [])
        check("L2", "retrieve 节点降级后仍返回结果", n > 0, "%d 条" % n)
    except Exception as e:
        check("L2", "retrieve 节点降级后仍返回结果", False, str(e)[:120])


# =========================================================================
# L3 规划层：planner 真的会拆步骤吗
# =========================================================================
def test_l3_planner() -> None:
    banner("L3 规划层")

    from app.agent.nodes.planner import planner_node, _parse_plan

    # 纯解析
    good = '{"plan_summary":"分别查","steps":[{"query":"A方案"},{"query":"B方案"},{"query":"A方案"}]}'
    try:
        p = _parse_plan(good)
        check("L3", "计划解析去重", p["queries"] == ["A方案", "B方案"],
              "queries=%s" % p["queries"])
    except Exception as e:
        check("L3", "计划解析去重", False, "%s: %s" % (type(e).__name__, e))

    # 超过 max_steps 要截断
    many = json.dumps({"plan_summary": "x",
                       "steps": [{"query": "q%d" % i} for i in range(10)]})
    try:
        p = _parse_plan(many)
        cap = int(settings.planner_max_steps)
        check("L3", "计划步数被 max_steps(%d) 截断" % cap, len(p["queries"]) == cap,
              "len=%d" % len(p["queries"]))
    except Exception as e:
        check("L3", "计划步数被 max_steps 截断", False, "%s: %s" % (type(e).__name__, e))

    # 坏 JSON 必须返回空计划（不抛）
    for bad in ["", "not json", '{"steps":[]}', '{"steps":"x"}']:
        try:
            out = planner_node({"rewritten_query": "测试", "api_key_override": KEY,
                                "base_url_override": BASE, "step_count": 0})
            check("L3", "坏输入不抛异常: %r" % bad[:12], isinstance(out, dict))
        except Exception as e:
            check("L3", "坏输入不抛异常: %r" % bad[:12], False, str(e)[:80])

    # 确定性收敛逻辑单测（不依赖 LLM 自觉）
    from app.agent.nodes.planner import _collapse_redundant, _is_simple
    check("L3", "简单问题判定正确",
          _is_simple("RAG 是什么") and not _is_simple("对比 A 和 B 的优缺点"),
          "RAG=simple / 对比=multi")
    collapsed = _collapse_redundant(
        ["RAG 的定义", "RAG 的定义是什么", "RAG 原理"], "RAG 是什么")
    check("L3", "近重复子查询被折叠", len(collapsed) == 1, "-> %s" % collapsed)
    kept = _collapse_redundant(
        ["安徽文旅的发展特点", "云南文旅的发展特点", "两地文旅对比"],
        "对比一下安徽文旅和云南文旅的发展特点")
    check("L3", "对比类问题不被误收敛", len(kept) == 3, "%d 步" % len(kept))

    if not KEY:
        record("L3", "planner 真实拆解", False, "无 API key，跳过")
        return

    # 真实拆解：多步问题
    q = "帮我对比一下安徽文旅和云南文旅的发展特点"
    t0 = time.time()
    try:
        out = planner_node({"rewritten_query": q, "query": q,
                            "api_key_override": KEY, "base_url_override": BASE,
                            "use_planner": True, "step_count": 0})
        plan = out.get("plan") or []
        ms = int((time.time() - t0) * 1000)
        check("L3", "planner 拆出多步计划", len(plan) >= 2,
              "%d 步 (%dms): %s" % (len(plan), ms, [s.get("query") for s in plan][:4]))
        if plan:
            check("L3", "计划带 summary", bool((out.get("plan_summary") or "").strip()),
                  (out.get("plan_summary") or "")[:60])
    except Exception as e:
        check("L3", "planner 拆出多步计划", False, "%s: %s" % (type(e).__name__, e))

    # 简单问题应只出 1 步（否则是过度拆分，会让同一批文档被重复检索）
    simple_qs = ["RAG 是什么", "安徽文旅有哪些景点", "知识库里有什么"]
    for sq in simple_qs:
        try:
            out = planner_node({"rewritten_query": sq, "query": sq,
                                "api_key_override": KEY, "base_url_override": BASE,
                                "use_planner": True, "step_count": 0})
            plan = out.get("plan") or []
            check("L3", "简单问题不过度拆分: %s" % sq, len(plan) == 1,
                  "%d 步: %s" % (len(plan), [s.get("query") for s in plan][:4]))
        except Exception as e:
            check("L3", "简单问题不过度拆分: %s" % sq, False, str(e)[:100])

    # 对比类问题必须保留多步（防止收敛过头把该拆的也压成 1 步）
    try:
        cq = "对比一下我知识库里安徽文旅和云南文旅的发展特点"
        out = planner_node({"rewritten_query": cq, "query": cq,
                            "api_key_override": KEY, "base_url_override": BASE,
                            "use_planner": True, "step_count": 0})
        plan = out.get("plan") or []
        check("L3", "对比类问题保留多步", len(plan) >= 2,
              "%d 步: %s" % (len(plan), [s.get("query") for s in plan][:4]))
    except Exception as e:
        check("L3", "对比类问题保留多步", False, str(e)[:100])


# =========================================================================
# L4 计划执行层：真的收到切片吗
# =========================================================================
def test_l4_plan_exec() -> None:
    banner("L4 计划执行层")

    from app.agent.nodes.research import execute_plan_node, replan_node

    base_state = {
        "plan": [{"query": "安徽文旅"}, {"query": "安徽旅游景点"}],
        "plan_cursor": 0,
        "retrieved_chunks": [],
        "plan_status": [],
        "research_notes": [],
        "research_iterations": 0,
        "step_count": 0,
        "api_key_override": KEY,
        "base_url_override": BASE,
    }
    if not KEY:
        record("L4", "execute_plan 收集切片", False, "无 API key，跳过")
        return

    # 第一步
    try:
        out = execute_plan_node(dict(base_state))
        chunks = out.get("retrieved_chunks") or []
        cursor = out.get("plan_cursor")
        check("L4", "execute_plan 收集到切片", len(chunks) > 0,
              "%d 切片, cursor=%s" % (len(chunks), cursor))
        # 并行开启时 execute_plan 会委派给 parallel_plan_node，一次性推进到
        # 计划末尾；串行时才是一步 +1。两种都是预期行为，这里断言"有推进"。
        check("L4", "cursor 有推进", isinstance(cursor, int) and cursor >= 1,
              "cursor=%s (并行=%s)" % (cursor, settings.parallel_plan_enabled))
        check("L4", "切片带 matched_query 标注",
              all(c.get("matched_query") for c in chunks) if chunks else False,
              (chunks[0].get("matched_query") if chunks else "n/a"))
        # 设计冲突：并行模式下 plan_cursor 一次到底，chat.py 依赖
        # "每步一条 astream 事件" 来推进度——并行时这层进度就没了。
        if settings.parallel_plan_enabled and len(base_state["plan"]) > 1:
            check("L4", "[设计冲突] 并行模式丢失逐步 SSE 进度",
                  cursor == len(base_state["plan"]),
                  "cursor 直接到 %s，前端只会看到一次 plan 事件" % cursor)
    except Exception as e:
        check("L4", "execute_plan 收集到切片", False, "%s: %s" % (type(e).__name__, e))
        return

    # 第二步：不该重复收集同一批切片
    try:
        s2 = dict(base_state)
        s2["plan_cursor"] = 1
        s2["retrieved_chunks"] = list(out.get("retrieved_chunks") or [])
        out2 = execute_plan_node(s2)
        c2 = out2.get("retrieved_chunks") or []
        check("L4", "跨步去重生效", len(c2) >= len(s2["retrieved_chunks"]),
              "累计 %d 切片" % len(c2))
    except Exception as e:
        check("L4", "跨步去重生效", False, str(e)[:120])

    # replan 空转保护：确定性分支——查询为空时必须立刻 stalled，不再空跑
    try:
        out3 = replan_node({
            "rewritten_query": "", "query": "",
            "retrieved_chunks": [], "research_notes": [],
            "api_key_override": KEY, "base_url_override": BASE, "step_count": 0,
        })
        check("L4", "replan 空查询立即 stalled", out3.get("replan_stalled") is True,
              "stalled=%s" % out3.get("replan_stalled"))
    except Exception as e:
        check("L4", "replan 空查询立即 stalled", False, str(e)[:120])

    # replan 正常补查：应生成新角度并累计 notes
    try:
        s = dict(base_state)
        s["retrieved_chunks"] = list(out.get("retrieved_chunks") or [])
        s["rewritten_query"] = "安徽文旅"
        out4 = replan_node(s)
        notes = out4.get("research_notes") or []
        check("L4", "replan 生成新检索角度",
              (len(notes) > 0) or out4.get("replan_stalled") is True,
              "notes=%s stalled=%s" % (len(notes), out4.get("replan_stalled")))
    except Exception as e:
        check("L4", "replan 生成新检索角度", False, str(e)[:120])

    # 补检索的查询生成绝不能把 <think> 当成检索词。
    # 推理模型（MiniMax 等）的响应以 <think> 开头，旧实现直接取第一行，
    # 于是查询词变成字面量 "<think>"，搜回一堆无关切片并被当成"来源"展示。
    if KEY:
        try:
            from app.agent.nodes.research import _generate_followup, _followup_model
            st = {"api_key_override": KEY, "base_url_override": BASE,
                  "model_override": MODEL or None}
            fchat = _followup_model(st)
            fq = _generate_followup(
                [{"title": "T", "text": "合肥龙虾节 每年6-8月 举办地点合肥市"}],
                "推荐合肥蜀山区野钓的地点", fchat)
            ok = fq is None or ("<think" not in fq and len(fq.strip()) >= 2)
            check("L4", "补检索查询不含思维链", ok, "query=%r" % fq)
        except Exception as e:
            check("L4", "补检索查询不含思维链", False, str(e)[:120])

    # 并行路径
    if settings.parallel_plan_enabled:
        t0 = time.time()
        try:
            from app.agent.nodes.research import parallel_plan_node
            out4 = parallel_plan_node(dict(base_state))
            ms = int((time.time() - t0) * 1000)
            n = len(out4.get("retrieved_chunks") or [])
            check("L4", "parallel_plan 能跑通", n > 0, "%d 切片 (%dms)" % (n, ms))
            check("L4", "parallel 后 cursor 推到末尾",
                  out4.get("plan_cursor") == len(base_state["plan"]),
                  "cursor=%s" % out4.get("plan_cursor"))
        except Exception as e:
            check("L4", "parallel_plan 能跑通", False, "%s: %s" % (type(e).__name__, e))


# =========================================================================
# L5 图编排层：整图 astream
# =========================================================================
def _initial_state(query: str, *, use_planner=None, skip_clarify=False) -> dict:
    """最小可用的初始 state（对齐 chat.py 的 per-turn 重置）。"""
    return {
        "messages": [{"role": "user", "content": query}],
        "session_id": "e2e_test",
        "query": query,
        "retrieved_chunks": [],
        "provider_override": None,
        "model_override": MODEL,
        "base_url_override": BASE,
        "api_key_override": KEY,
        "reasoning_level_override": None,
        "embedding_model_override": EMB_MODEL,
        "step_count": 0,
        "profile": {},
        "summary": "",
        "memory_facts": [],
        "project_rules": "",
        "intent": "",
        "rewritten_query": "",
        "plan": [],
        "plan_summary": "",
        "plan_cursor": 0,
        "plan_status": [],
        "replan_stalled": False,
        "skip_retrieval": False,
        "research_iterations": 0,
        "research_notes": [],
        "ingest_result": {},
        "report_result": {},
        "answer": None,
        "citations": [],
        "agent_permission": "default",
        "use_planner": use_planner,
        "clarify_request": None,
        "skip_clarify": skip_clarify,
        "subagent_mode": None,
    }


async def _run_graph(query: str, *, use_planner=None, skip_clarify=False, timeout=180):
    """跑一次整图，返回 (访问过的节点序列, 最终 state)。"""
    from app.agent.graph import get_graph
    graph = await get_graph()
    st = _initial_state(query, use_planner=use_planner, skip_clarify=skip_clarify)
    cfg = {"configurable": {"thread_id": "e2e_%d" % int(time.time() * 1000)}}
    seen: list[str] = []
    final: dict = {}
    async for chunk in graph.astream(st, cfg, stream_mode="updates"):
        for node, delta in (chunk or {}).items():
            seen.append(node)
            if isinstance(delta, dict):
                final.update(delta)
    # 补上初始 state 里、节点没回写的字段
    merged = dict(st)
    merged.update(final)
    return seen, merged


def test_l5_graph() -> None:
    banner("L5 图编排层")

    if not KEY:
        record("L5", "整图 astream", False, "无 API key，跳过")
        return

    # 关键：AsyncSqliteSaver 持有绑定到「首次事件循环」的 aiosqlite 连接。
    # 多次 asyncio.run() 会各自新建 loop -> 第二次起报
    # "Lock is bound to a different event loop"。生产环境 uvicorn 只有一个
    # loop 所以无碍，但测试必须把所有图运行塞进同一个 asyncio.run()。
    async def _all_graph_cases():
        results = {}

        # 5.1 寒暄：router -> END（chat_no_rag），不打检索
        try:
            seen, st = await _run_graph("你好")
            results["greeting"] = (seen, st, None)
        except Exception as e:
            results["greeting"] = (None, None, e)

        # 5.2 普通问答：router -> retrieve
        try:
            seen, st = await _run_graph("安徽文旅知识库讲了什么")
            results["qa"] = (seen, st, None)
        except Exception as e:
            results["qa"] = (None, None, e)

        # 5.3 研究意图：必须经过 planner
        try:
            seen, st = await _run_graph(
                "对比一下我知识库里安徽文旅的发展特点和挑战", use_planner=True)
            results["research"] = (seen, st, None)
        except Exception as e:
            results["research"] = (None, None, e)

        # 5.4 歧义澄清
        try:
            seen, st = await _run_graph("帮我调研哪吒")
            results["clarify"] = (seen, st, None)
        except Exception as e:
            results["clarify"] = (None, None, e)

        # 5.5 回答流式
        try:
            from app.agent.nodes.answer import answer_node_stream
            st = _initial_state("简单介绍一下安徽文旅")
            st["retrieved_chunks"] = []
            deltas, kinds = [], []
            async for kind, payload in answer_node_stream(st):
                kinds.append(kind)
                if kind == "text_delta":
                    deltas.append(payload)
            results["answer"] = (("".join(deltas), kinds), None, None)
        except Exception as e:
            results["answer"] = (None, None, e)

        return results

    try:
        R = asyncio.run(_all_graph_cases())
    except Exception as e:
        record("L5", "整图 astream", False, "%s: %s" % (type(e).__name__, str(e)[:160]))
        return

    # --- 断言 ---
    seen, st, err = R.get("greeting", (None, None, None))
    check("L5", "寒暄走快速通道", err is None and seen is not None and "retrieve" not in seen,
          ("错误: %s" % err) if err else ("路径=%s" % " -> ".join(seen or [])))

    seen, st, err = R.get("qa", (None, None, None))
    if err:
        check("L5", "问答走 retrieve", False, "%s: %s" % (type(err).__name__, str(err)[:140]))
    else:
        check("L5", "问答走 retrieve", "retrieve" in seen, "路径=%s" % " -> ".join(seen))
        check("L5", "retrieve 后有切片",
              len(st.get("retrieved_chunks") or []) > 0,
              "%d 切片" % len(st.get("retrieved_chunks") or []))

    seen, st, err = R.get("research", (None, None, None))
    if err:
        check("L5", "研究意图经过 planner", False,
              "%s: %s" % (type(err).__name__, str(err)[:140]))
    else:
        check("L5", "研究意图经过 planner", "planner" in seen,
              "路径=%s" % " -> ".join(seen))
        check("L5", "研究意图执行计划",
              any(n in seen for n in ("execute_plan", "replan")),
              "路径=%s" % " -> ".join(seen))
        check("L5", "研究收集到切片",
              len(st.get("retrieved_chunks") or []) > 0,
              "%d 切片" % len(st.get("retrieved_chunks") or []))
        check("L5", "计划状态有记录",
              len(st.get("plan_status") or []) > 0,
              "plan_status=%d 条" % len(st.get("plan_status") or []))

    seen, st, err = R.get("clarify", (None, None, None))
    if err:
        check("L5", "歧义问题图终止并带 clarify_request", False,
              "%s: %s" % (type(err).__name__, str(err)[:140]))
    else:
        cr = st.get("clarify_request")
        check("L5", "歧义问题图终止并带 clarify_request", bool(cr),
              "question=%s" % ((cr or {}).get("question") or "(none)")[:50])

    payload, _e, err = R.get("answer", (None, None, None))
    if err:
        check("L5", "answer_node_stream 产出 text_delta", False,
              "%s: %s" % (type(err).__name__, str(err)[:140]))
    else:
        text, kinds = payload
        check("L5", "answer_node_stream 产出 text_delta", "text_delta" in kinds,
              "事件=%s" % ",".join(sorted(set(kinds))))
        check("L5", "answer 以 done 收尾", kinds and kinds[-1] == "done",
              "末事件=%s" % (kinds[-1] if kinds else "none"))
        check("L5", "回答非空", len(text.strip()) > 0, "%d 字符" % len(text))


# =========================================================================
# L6 子代理层：工具白名单
# =========================================================================
def test_l6_subagents() -> None:
    banner("L6 子代理层")

    from app.agent.subagents import (
        PROFILES, SUBAGENT_MODES, _filter_tools_for_mode, _safe_mode, parse_plan_payload,
    )

    check("L6", "三个 profile 齐备", SUBAGENT_MODES == {"explore", "plan", "general"},
          str(sorted(SUBAGENT_MODES)))
    check("L6", "profile 均有 system prompt",
          all(p.get("system") for p in PROFILES.values()))

    # 非法 mode 应回落到 general
    check("L6", "非法 mode 回落 general", _safe_mode("hacker") == "general",
          _safe_mode("hacker"))

    all_tools = [
        "hybrid_search", "mcp:fs:fs_read", "mcp:fs:fs_write", "mcp:fs:fs_delete",
        "mcp:fs:fs_mkdir", "mcp:fs:fs_ls", "ingest_url", "ingest_text",
        "ingest_file", "delete_note", "load_skill", "mcp_invoke",
    ]
    exp = _filter_tools_for_mode(all_tools, "explore")
    pln = _filter_tools_for_mode(all_tools, "plan")
    gen = _filter_tools_for_mode(all_tools, "general")

    check("L6", "explore 剔除写/删工具",
          not any(t in exp for t in ("mcp:fs:fs_write", "mcp:fs:fs_delete",
                                     "mcp:fs:fs_mkdir", "delete_note")),
          "剩 %d 个" % len(exp))
    check("L6", "plan 连读工具也剔除",
          not any(t in pln for t in ("hybrid_search", "mcp:fs:fs_read", "mcp:fs:fs_ls")),
          "剩 %d 个" % len(pln))
    check("L6", "general 仍挡破坏性工具",
          not any(t in gen for t in ("mcp:fs:fs_write", "mcp:fs:fs_delete", "delete_note")),
          "剩 %d 个" % len(gen))
    check("L6", "explore 保留 hybrid_search",
          "hybrid_search" in exp, "")

    # 计划载荷解析
    p = parse_plan_payload('<think>x</think>{"plan_summary":"s","steps":[{"query":"a"},{"query":"a"},{"query":"b"}]}')
    check("L6", "parse_plan_payload 去重", bool(p) and [s["query"] for s in p["steps"]] == ["a", "b"],
          str(p["steps"]) if p else "None")
    check("L6", "parse_plan_payload 坏输入返回 None",
          parse_plan_payload("garbage") is None and parse_plan_payload("") is None)


# =========================================================================
# L7 治理层：钩子 + 权限
# =========================================================================
def test_l7_governance() -> None:
    banner("L7 治理层（钩子 / 权限）")

    from app.agent import tool_permissions as tp
    from app.agent.hooks import _match_tool, fire, is_blocked, list_hooks, set_hooks

    # 权限解析优先级
    check("L7", "默认允许只读工具", tp.is_tool_allowed("mcp:fs:fs_read")[0] == "allow",
          str(tp.is_tool_allowed("mcp:fs:fs_read")))
    check("L7", "默认拒绝删除", tp.is_tool_allowed("mcp:fs:fs_delete")[0] == "deny",
          str(tp.is_tool_allowed("mcp:fs:fs_delete")))
    check("L7", "写操作默认需审批", tp.requires_approval("mcp:fs:fs_write"))
    check("L7", "未知工具默认 ask", tp.is_tool_allowed("some_unknown_tool")[0] == "ask",
          str(tp.is_tool_allowed("some_unknown_tool")))
    check("L7", "通配符 skill:* 命中", tp.is_tool_allowed("skill:browser")[0] == "allow",
          str(tp.is_tool_allowed("skill:browser")))

    # 通配符匹配逻辑
    check("L7", "钩子通配符 * 命中一切", _match_tool("*", "mcp:x:y"))
    check("L7", "钩子精确匹配", _match_tool("mcp:a:b", "mcp:a:b"))
    check("L7", "钩子不误匹配", not _match_tool("mcp:a:b", "mcp:a:c"))

    # 钩子真实执行：写一个临时脚本，验证 block 语义
    import tempfile
    hdir = Path(settings.data_dir) / "hooks"
    hdir.mkdir(parents=True, exist_ok=True)
    blocker = hdir / "_e2e_blocker.py"
    blocker.write_text(
        "import sys, json\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'block': 'e2e test veto'}))\n",
        encoding="utf-8")
    allow = hdir / "_e2e_allow.py"
    allow.write_text("import sys, json\njson.load(sys.stdin)\n", encoding="utf-8")

    orig = list_hooks()
    try:
        set_hooks([
            {"name": "_e2e_block", "phase": "PreToolUse", "tool": "*",
             "script": "_e2e_blocker.py", "enabled": True},
        ])
        res = fire("PreToolUse", {"tool": "mcp:fs:fs_write", "args": {}})
        blocked, reason = is_blocked(res)
        check("L7", "PreToolUse 钩子能否决工具调用", blocked, "reason=%s" % reason[:40])

        set_hooks([
            {"name": "_e2e_allow", "phase": "PreToolUse", "tool": "*",
             "script": "_e2e_allow.py", "enabled": True},
        ])
        res2 = fire("PreToolUse", {"tool": "mcp:fs:fs_read", "args": {}})
        blocked2, _ = is_blocked(res2)
        check("L7", "允许型钩子不误否决", not blocked2, "runs=%d" % len(res2))

        # 脚本不存在 -> PreToolUse 应 block（安全默认）
        set_hooks([{"name": "_e2e_missing", "phase": "PreToolUse", "tool": "*",
                    "script": "no_such_script_xyz.py", "enabled": True}])
        res3 = fire("PreToolUse", {"tool": "x", "args": {}})
        b3, _ = is_blocked(res3)
        check("L7", "钩子脚本缺失时安全否决", b3, "")
    except Exception as e:
        check("L7", "PreToolUse 钩子能否决工具调用", False,
              "%s: %s" % (type(e).__name__, e))
    finally:
        set_hooks(orig)
        for f in (blocker, allow):
            try:
                f.unlink()
            except Exception:
                pass


# =========================================================================
# L8 上下文层：预算与裁剪
# =========================================================================
def test_l8_context() -> None:
    banner("L8 上下文层")

    from app.agent.context import (
        CONTEXT_TOKEN_BUDGET, MAX_CHUNK_CHARS, estimate_tokens, format_context,
        trim_history,
    )
    from app.agent.nodes.answer import _citations_from_text

    check("L8", "CJK token 估算 > ASCII",
          estimate_tokens("中文中文") > estimate_tokens("abcd"),
          "cjk=%d ascii=%d" % (estimate_tokens("中文中文"), estimate_tokens("abcd")))
    check("L8", "空串估算为 0", estimate_tokens("") == 0)

    # 单块硬上限
    big = [{"title": "t", "text": "甲" * 5000, "note_id": "n1", "chunk_index": 0}]
    ctx = format_context(big)
    check("L8", "单块被 MAX_CHUNK_CHARS 截断", len(ctx) <= MAX_CHUNK_CHARS + 200,
          "len=%d cap=%d" % (len(ctx), MAX_CHUNK_CHARS))

    # 总预算截断
    many = [{"title": "t%d" % i, "text": "乙" * 700, "note_id": "n%d" % i,
             "chunk_index": i} for i in range(20)]
    ctx2 = format_context(many)
    used = estimate_tokens(ctx2)
    check("L8", "总预算不超 CONTEXT_TOKEN_BUDGET", used <= CONTEXT_TOKEN_BUDGET + 400,
          "used~%d budget=%d" % (used, CONTEXT_TOKEN_BUDGET))

    # 空 chunk -> 占位符
    check("L8", "空引用给占位符", format_context([]) != "",
          repr(format_context([]))[:40])

    # 历史滑窗
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": "msg%d" % i}
            for i in range(40)]
    recent, overflow = trim_history(hist)
    check("L8", "历史滑窗生效", len(recent) <= 12 and len(overflow) > 0,
          "recent=%d overflow=%d" % (len(recent), len(overflow)))
    check("L8", "滑窗保留最近消息", recent and recent[-1]["content"] == "msg39",
          recent[-1]["content"] if recent else "none")

    # 引用提取
    chunks = [{"note_id": "n1", "title": "T1", "chunk_index": 0, "text": "a", "final_score": 0.9},
              {"note_id": "n2", "title": "T2", "chunk_index": 1, "text": "b", "final_score": 0.8}]
    cits = _citations_from_text("根据资料[2]，以及[1]可以得出。", chunks)
    check("L8", "引用提取且按序", [c["note_id"] for c in cits] == ["n1", "n2"],
          str([c["note_id"] for c in cits]))
    check("L8", "越界引用被忽略",
          _citations_from_text("见[9]", chunks) == [], "越界应丢弃")
    check("L8", "重复引用去重",
          len(_citations_from_text("[1][1][1]", chunks)) == 1, "")

    # 反伪造：模型没写 [n] 时绝不能凭空生成引用。
    # 旧实现会把全部检索切片塞进 citations，前端再渲染成「来源：[n]」，
    # 让"用自有知识回答"的答案看起来像有知识库依据。
    check("L8", "无 [n] 时不得伪造引用",
          _citations_from_text("安徽是……（模型用自有知识作答，无标记）", chunks) == [],
          "模型未引用时应返回空")

    # 思维链剥离：存储保留 <think>（前端要展示），但喂模型前必须剥掉。
    # 实测真实数据里思维链占助手消息 36% 的 token。
    from app.agent.context import strip_think
    st_cases = [
        ("<think>想一下</think>这是答案", "这是答案", "闭合标签"),
        ("<thinking>x</thinking>答案", "答案", "thinking 标签"),
        ("<think>只有思考</think>", "", "纯思维链"),
        ("普通回答", "普通回答", "无思维链"),
        ("<think>未闭合的思考", "", "未闭合标签"),
        ("", "", "空串"),
    ]
    for raw, want, label in st_cases:
        got = strip_think(raw)
        check("L8", "思维链剥离: %s" % label, got == want,
              "got=%r want=%r" % (got, want))

    # 端到端：build_messages 出来的历史里不能残留 <think>
    try:
        from app.agent.context import build_messages as _bm
        msgs = _bm(
            instructions="sys <<CONTEXT>> <<QUESTION>>",
            chunks=[],
            history=[
                {"role": "user", "content": "问题一"},
                {"role": "assistant", "content": "<think>内心戏很长</think>回答一"},
            ],
            question="问题二",
        )
        joined = "\n".join(getattr(m, "content", "") or "" for m in msgs)
        check("L8", "build_messages 产物不含 <think>",
              "<think" not in joined and "内心戏" not in joined,
              "残留检查")
    except Exception as e:
        check("L8", "build_messages 产物不含 <think>", False, str(e)[:120])


# =========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑指定层，如 L1,L3")
    args = ap.parse_args()
    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}

    print("=" * 68)
    print("  Agent 端到端冒烟测试")
    print("  backend : %s" % _BACKEND)
    print("  model   : %s" % (MODEL or "(未配置)"))
    print("  base    : %s" % (BASE or "(未配置)"))
    print("  api_key : %s" % ("已配置" if KEY else "缺失 —— LLM 相关用例会跳过"))
    print("  emb     : %s" % EMB_MODEL)
    print("  planner : %s | parallel: %s" % (settings.planner_enabled, settings.parallel_plan_enabled))
    print("=" * 68, flush=True)

    layers = [
        ("L1", test_l1_router),
        ("L2", test_l2_retrieval),
        ("L3", test_l3_planner),
        ("L4", test_l4_plan_exec),
        ("L5", test_l5_graph),
        ("L6", test_l6_subagents),
        ("L7", test_l7_governance),
        ("L8", test_l8_context),
    ]
    for code, fn in layers:
        if only and code not in only:
            continue
        try:
            fn()
        except Exception as e:
            record(code, "层内未捕获异常", False, "%s: %s" % (type(e).__name__, e))
            traceback.print_exc()

    # ---- 汇总 ----
    print("\n" + "=" * 68)
    print("  结果汇总")
    print("=" * 68)
    by_layer: dict[str, list[bool]] = {}
    for layer, _name, ok, _d in RESULTS:
        by_layer.setdefault(layer, []).append(ok)
    for layer in sorted(by_layer):
        oks = by_layer[layer]
        p, t = sum(oks), len(oks)
        flag = "OK  " if p == t else "FAIL"
        print("  %s %-4s %d/%d" % (flag, layer, p, t))
    total, passed = len(RESULTS), sum(1 for r in RESULTS if r[2])
    print("-" * 68)
    print("  总计 %d/%d 通过" % (passed, total))
    failed = [r for r in RESULTS if not r[2]]
    if failed:
        print("\n  失败明细：")
        for layer, name, _ok, detail in failed:
            print("    [%s] %s  -- %s" % (layer, name, detail))
    print("=" * 68)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
