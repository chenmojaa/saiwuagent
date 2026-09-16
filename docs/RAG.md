# Smallhouse RAG（检索增强生成）— 完整说明

> 最后更新：2026-09-11。当前实现已从"单轮 hybrid_search"演进为 **plan-and-execute**：检索由 LangGraph 的 planner 拆成 1-4 步，可并行执行、可 replan 补查。

## 一句话总结
RAG = 用户发消息时，先把你知识库里**最相关的几段文本**（chunk）捞出来，塞进 LLM 的 prompt，让 LLM 基于这些上下文回答，而不是凭空瞎编。

## 三层链路

```
[知识库内容]
   PDFs / DOCX / PPTX / XLSX / CSV / HTML / TXT / MD / 图片(OCR)
        │
        ▼  ingest (Phase 1: 把内容入库)
backend/app/tools/ingest.py
       → parse_* 读出文字
       → chunk_text(s, 500, 80)  <- 500 字符一段，重叠 80，优先按段落/句切
       → embed_texts(chunks)     <- 调 embedding API 拿到向量
       → add_chunks(note_id, chunks, embeddings)
            ├─ Chroma collection "notes" (cosine)
            └─ SQLite FTS5 `chunk_fts` (bm25)
       → Note.embedded=True, Note.chunk_count=N

[用户发问]
chat page  ->  POST /api/chat (SSE)
                    │
                    ▼  api/chat.py
   _build_initial_state()   # 每轮显式重置 intent/plan/clarify 等字段
   if body.use_rag and query: initial_state["use_rag"] = True
                    │
                    ▼  graph.astream(initial_state)   [AsyncSqliteSaver]
   router_node
        ├─ ambiguous  -> clarify 节点终止，SSE 推 clarify 等前端回话
        ├─ research   -> planner_node
        │                    → {"plan_summary": ..., "steps": [{"query": ...}, ...]}
        │                 execute_plan_node  (可并行：parallel_plan_node)
        │                    ↺ replan_node  ← 材料不足时补查，最多 HD_RESEARCH_MAX_ITER 轮
        └─ chat       -> retrieve_node

backend/app/storage/hybrid.py  (真正的检索实现)
  ├─ emb = embed_texts([query], mode="query")      <- 把用户问句也 embed
  │       （MiniMax 按 URL 预判 mode，不支持 query 时直接走 db，避免每次重试）
  ├─ vec_hits = vector_search(emb, top_k=10)       <- Chroma cosine 邻居
  │    for h: h["vec_score"] = max(0, 1 - distance)
  ├─ kw_hits = fts_search(query, top_k=10)          <- SQLite FTS5 BM25
  │    _fts_escape 剥除 " ( ) * ^ : 后包成短语，CJK 友好
  │    for h: h["kw_score"]  = 1/(1+|bm25|)
  ├─ merge dict by (note_id, chunk_index) - 去重
  ├─ ranked: 0.7 * vec + 0.3 * kw
  ├─ 双阈值过滤: MIN_FINAL_SCORE=0.18 / MIN_DIM_SCORE=0.18
  ├─ merge_neighboring_hits()   <- 同 note 内相邻 chunk 聚簇 (window=2)
  │        再从 FTS5 回捞 [center-2, center+2] 拼成完整 context，
  │        原命中片段另存 matched_text
  └─ return top_k

backend/app/agent/nodes/answer.py  (LangGraph 节点)
  reads state.retrieved_chunks
  builds LLM messages (agent/context.py 负责预算与裁剪):
   SYSTEM:
    "基于以下参考文本回答用户问题。如果参考与问题无关，就说不知道。"
    ---
    [1] title=xxx score=0.812
        内容...
    [2] title=yyy score=0.654
        内容...
  + 历史（agent/context.py: trim_history，滑窗 12 条 / 2000 tokens）
  -> LLM stream -> SSE -> 前端
```

## 上下文预算（`agent/context.py`）

| 常量 | 值 | 说明 |
|---|---|---|
| `CONTEXT_TOKEN_BUDGET` | 3000 | 引用 chunks 总预算，超出按分数截断 |
| `MAX_CHUNK_CHARS` | 800 | 单 chunk 硬上限 |
| `HISTORY_TOKEN_BUDGET` | 2000 | 历史消息预算 |
| 历史滑窗 | 12 条 | 超出走 `summarize_overflow` 摘要（≤150 字） |

`estimate_tokens` 用启发式估算（CJK ×1.5 + ASCII ×0.35），不引入 tokenizer 依赖。当本轮**没有任何 chunk** 时，`_strip_citation_rules` 会剥掉引用规则，防止模型输出"来源：无"。

## 为什么 RAG 之前没生效
1. embedding 入库时 key = .env 占位符 -> 401 -> 笔记从未成功 embed -> Chroma 为空
   -> 修复：在设置页添加真实 MiniMax entry（baseUrl + apiKey + embeddingModel="embo-01"）
   -> 这样每次入库 reembed 时用真实 key 走 MiniMax body `{"model": "embo-01", "type": "db", "texts":[...]}`

2. 检索时 hybrid_search 用 .env 占位符 -> 同样 401 -> vec_hits 空 -> 只剩 keyword
   -> 修复：前端 store/chat.ts send() 在 use_rag=true 时同时把
       embedding_model + embedding_base_url + api_key 发给后端
       （这就是给 /api/chat 加的 embedding_model / embedding_base_url 字段）

3. FTS5 中文分词是 unicode61 -> 命中"文化和旅游厅"作为整体
   -> 已改进：`_fts_escape` 转义 + 双 pass（短语匹配 → per-token OR → LIKE 兜底）

## 关键约束 / 副作用
- 推理等级（reasoning_level）目前 OpenAI / MiniMax 未统一识别；后端对 MiniMax 用 `extra_body` 字段；可忽略
- top_k 默认 5；如果你问的问题需更宽召回，可在 settings 里调（暂未暴露给前端）
- embedding 模型必须和入库时一致：如果入库用了 embo-01，检索也必须是 embo-01；否则向量空间错位
- **planner 与 replan 预算是两套**：计划步数上限 = `HD_PLANNER_MAX_STEPS`（默认 4），replan 轮数上限 = `HD_RESEARCH_MAX_ITER`（默认 3）。旧版用 `plan_queries[:max_iter]` 会把 4 步计划砍成 3 步，已修复
- **并行执行**：`HD_PARALLEL_PLAN_ENABLED=true` 时多步并发跑（worker 数 = min(步数, 4)，单步失败被隔离不影响其余步骤）
- **空转保护**：某轮 replan 未产生新 chunk 时置 `replan_stalled`，立即收尾，避免耗预算

## 一键诊断
```powershell
# 1. 入库向量库实际有几条？
cd D:\one_agent\backend
.\.venv\Scripts\python.exe -c "import chromadb; from chromadb.config import Settings; c = chromadb.PersistentClient(path='data/chroma', settings=Settings(anonymized_telemetry=False)); print(c.get_collection('notes').count())"

# 2. SQLite 总条目 + 已嵌入条目
.\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('data/notes.db'); print(c.execute('SELECT count(*), sum(embedded) FROM notes').fetchone())"

# 3. FTS5 总条目
.\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('data/notes.db'); print(c.execute('SELECT count(*) FROM chunk_fts').fetchone())"

# 4. 启用知识库开关后，发消息，浏览器 devtools 看 fetch body：
#    有 embedding_model / embedding_base_url 字段 = OK
#    有 use_rag: true = OK
#    浏览器 console 看 SSE：
#    "event: citations
#     data: [...3 chunks]"  -> RAG 命中
#     没出 citations event   -> retrieved_chunks 空，prompt 无引用
#    "event: plan  data: {...steps}" -> planner 已接管，逐步看 stage 事件
```

## 让 RAG 工作的具体步骤
1. 打开设置页（顶栏"设置"）
2. 添加自定义模型：
   - 名称：任意
   - Base URL：https://api.minimax.chat/v1
   - API Key：你真实的 MiniMax key
   - 自动识别模型列表 -> 勾选你想用的 chat 模型
   - 在 embedding_model 栏填 `embo-01`（前端 fallback: 如果 baseUrl 含 "minimax" 且留空会默认填 embo-01）
   - 保存
3. 知识库页 -> 上传几个 PDF / 笔记 -> 等左下"未 embedding"变"N chunks"
4. 聊天页 -> 打开"知识库"开关 -> 输入"文化和旅游厅"或相似问句 -> 看到 LLM 答案下方有 CitationsCard 引用

## 没看到引用卡时的排查清单
- [ ] 浏览器 devtools 看 fetch /api/chat 的 body 有没有 `embedding_model: "embo-01"`
- [ ] 服务端日志 `D:\one_agent\backend\logs\hd.log` 有没有 "401 invalid api key" 关键词
- [ ] Chroma count > 0 吗（见上面诊断 1）
- [ ] FTS5 中匹配关键词的 chunk > 0 吗（见诊断 3，临时 SELECT for 测试）
- [ ] 前端 store/chat.ts useRag 为 true 吗（devtools 看）