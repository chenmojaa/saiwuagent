# Smallhouse — 功能现状 (v0.6)

> 本地优先的个人知识库 Agent：把散落的文章 / 文件 / 对话 / 飞书文档统一塞进一个可检索的知识库，用自然语言向 LLM 提问，答案带原文引用。
> 最近更新：2026-09-11（同步 plan-and-execute 循环、子代理、hooks、项目规则、长期记忆抽取、权限规则）

---

## 0. 一句话定位

**Smallhouse** = 本地优先的个人 Second Brain。多 LLM 自由切换 + 混合检索（向量 0.7 + 关键词 0.3）+ plan-and-execute 编排 + MCP 工具调用 + 飞书知识库同步。

---

## 1. 技术栈

| 层 | 技术 | 备注 |
|---|---|---|
| 后端框架 | FastAPI 0.115+ | 异步 REST + SSE 流式 |
| Agent / LLM 编排 | LangGraph 1.2 + LangChain 1.3 | router → (planner → execute_plan ⟲ replan) / retrieve / ingest / report |
| LLM 抽象 | `langchain-openai` + 工厂路由 | OpenAI / Anthropic + 7 家 OpenAI-compatible |
| Embedding | `httpx` 直连 `/v1/embeddings`，MiniMax 原生 body | OpenAI / MiniMax / 其他 OpenAI-compatible |
| 向量库 | ChromaDB（持久化到 `data/chroma/`） | cosine + HNSW |
| 关系库 | SQLite + SQLModel + FTS5 全文索引 | `data/notes.db` |
| 检查点 | `AsyncSqliteSaver`（惰性异步单例） | `data/checkpoints.sqlite` |
| 文件解析 | pdfplumber、openpyxl、python-docx、python-pptx、pypdf、trafilatura、BeautifulSoup、pytesseract | 文本 / Office / PDF / OCR / URL |
| 飞书 | httpx + 官方 Open API | OAuth + DFS walk + bitable 记录 |
| MCP | stdio 通信 + 自写 MCP 客户端 | 支持 npx 启动 MCP server，会话复用 |
| 前端 | Vue 3 + Vite 6 + TypeScript | `<script setup>` + SFC |
| 状态管理 | Pinia 3 | chat / notes / sessions / models / settings |
| UI 库 | Naive UI 2.41 | 蓝色主题（`#3b82f6`） |
| 路由 | vue-router 4 | createWebHistory |
| SSE 解析 | 自写 `streamSse` | fetch + ReadableStream |
| 部署 | uvicorn 127.0.0.1:5006 + Vite 127.0.0.1:5174 | `scripts/start-all.ps1` |

---

## 2. 核心功能清单

### 2.1 LLM 与模型管理

- **多 provider 路由**：OpenAI / Anthropic 原生 + DeepSeek / 智谱 / 月之暗面 / SiliconFlow / Ollama / MiniMax 走 OpenAI-compatible `base_url`（`backend/app/llm/factory.py`）
- **Auto 模式**：未指定 provider/model 时退回 `.env` 默认；用户 key 优先级 `header > body > .env`
- **每请求覆盖**：`base_url` / `api_key` / `provider` / `model` / `reasoning_level` 五个字段可临时覆盖
- **推理程度 4 档**：`low | medium | high | xhigh`
- **自动识别模型清单**：`POST /api/custom-models` → httpx 调 `<base_url>/models`，按 URL 关键字推断 provider
- **API key 安全**：列表接口返回脱敏 key（`sk****xx`），真实 key 只存服务端 `data/models.json`，前端不回传脱敏值
- **前端持久化**：localStorage 存自定义模型 + 选中模型 + 推理程度（`stores/models.ts`）

### 2.2 知识库入库

支持 11 种入口（`backend/app/tools/ingest.py`）：

| 入口 | 解析器 | 输出 |
|---|---|---|
| PDF 文本型 | `parse_pdf.py` (pdfplumber) | Markdown + 表格块 |
| PDF 扫描型 | `parse_pdf.py` (pdfplumber render + Tesseract OCR) | Markdown + 表格块 |
| DOCX | `parse_doc.py` (python-docx + 原生 XML) | Markdown + 表格块（保留合并） |
| PPTX | `parse_pptx.py` (python-pptx + 原生 XML) | Markdown + 表格块（保留合并） |
| XLSX | `parse_xlsx.py` (openpyxl) | Markdown + 表格块（保留合并） |
| CSV | `parse_csv.py` | Markdown 表格 |
| HTML | `parse_html.py` (BeautifulSoup / trafilatura) | Markdown |
| TXT / MD | 直读 | 原文 |
| 图片 | `ocr.py` (Tesseract `chi_sim`) | OCR 文本 |
| URL | `fetch_url.py` (trafilatura) | Markdown |
| 飞书文档 / 多维表格 | `parse_feishu_doc.py` | Markdown |

入库链路：`parse -> chunk_text(500/80) -> embed_texts -> Chroma add_chunks + FTS5 add_fts -> SQLite.embedded=true, chunk_count=N`。

### 2.3 表格识别（三层框架）

> 核心思想：**别找"一个全能的表格库"，把识别拆成「版面分析 -> 结构还原 -> 文字识别」三层，按文件类型决定在哪层发力**。

| 文件类型 | 走的层 | 实现 | 输出 |
|---|---|---|---|
| **XLSX / DOCX / PPTX** | 结构还原 | 直接吃 XML 语义 | Markdown 表格，**零信息损失** |
| **CSV** | 结构还原 | 行解析 | Markdown 表格 |
| **PDF 文本型** | 版面 + 结构 | pdfplumber `lines` / `text` 策略 | Markdown 表格 + 跨页表头去重续接 |
| **PDF 扫描型** | 三层全走 | pdfplumber 渲染 -> Tesseract PSM 12 -> 列对齐 + 行间距启发式 | Markdown 表格 |
| **飞书 docx** | 结构还原 | `parse_feishu_doc.py` raw_content 轻清洗 | Markdown |
| **飞书 bitable** | 结构还原 | fields + records -> 表格（10 种 ui_type） | Markdown 表格 |

**测试覆盖**：`scripts/table_tests/` 6 个 fixture，当前 **6/6 PASS**。

### 2.4 RAG 检索与对话

**三层链路**：

| 层 | 文件 | 关键逻辑 |
|---|---|---|
| 1. 入库 | `api/notes.py` + `storage/vector.py` | parse -> chunk -> embed -> Chroma + FTS5 |
| 2. 检索 | `api/chat.py` hybrid_search | `0.7 * vec_score + 0.3 * kw_score`，top_k=5 |
| 3. prompt | `agent/nodes/answer.py` | chunks -> system prompt，引用 `[n]` 标注 |

**检索细节**：
- 向量：Chroma cosine，distance -> 0~1 分
- 关键词：SQLite FTS5 BM25，**CJK 友好**：双 pass（phrase match + per-token OR + LIKE 兜底）
- **MiniMax 兼容**：embedding 先 `mode="query"`，上游拒则自动 fallback 到 `mode="db"`
- 阈值过滤：final_score < 0.18 或单维度 < 0.18 的 chunk 不进 prompt，避免闲聊被强行附假引用
- 上下文预算：reference chunks 限 3000 tokens，单 chunk 硬限 800 字符

**SSE 事件流**（`api/chat.py` 权威定义）：
```
event: session       -> 新会话时推 session_id
event: stage         -> 阶段（router / agent / rag_search / llm_stream / memory_shortcut）
event: plan          -> 计划卡片（phase=created / step_done / replan / done）
event: clarify       -> 歧义澄清请求（request_id + 问题 + 候选）
event: citations     -> [{note_id, chunk_index, title, snippet, score}, ...]
（无 event 名，默认 message）-> 正文 token delta（逐行 data:）
event: answer        -> 非流式整段回答（记忆快捷指令的 ack）
event: tool          -> MCP 工具调用（phase=start / call / result）
event: permission    -> 权限审批（phase=request / result）
event: subagent      -> 子代理执行进度
event: ingest        -> 入库结果
event: report        -> 周报结果
event: error         -> 错误（不中断连接语义，前端可展示）
event: done          -> 结束
```

**前端 UX**：
- 输入框：知识库开关 + 模型选择器 + 推理等级 + Plan 开关 + 权限控制
- 流中：阶段标签 + 已用秒数 + 思考过程折叠显示 + **计划进度条**（每个步骤 pending → running → done，补检索步骤虚线标出）
- 答案下方：引用卡片，列编号 / 标题 / chunk 索引 / score / 片段，可点击回原文
- 歧义问题：弹窗让用户选择（纵向排列 + "其他：___" 自定义输入）
- 工具审批：破坏性工具调用前弹窗授权，用户答复后继续
- Mermaid 图表：卡片化渲染（标题栏 + 折叠/展开 + 复制源码 + 下载 PNG）

### 2.5 Agent 编排（LangGraph：plan-and-execute）

**图结构**（`agent/graph.py`）——共 7 个节点：

```
router ──┬─ chat        ──> retrieve ──> answer
         ├─ ambiguous   ──> clarify（终止，等前端回话）
         ├─ research    ──> planner ──> execute_plan ⟲ replan ──> answer
         ├─ ingest      ──> ingest
         └─ report      ──> report
```

- **router** (`nodes/router.py`)：意图分类 + query 改写 + 歧义检测。三层判定：
  1. **正则快通道**（`_FAST_CHAT_PATTERNS`）：问候 / 闲聊 / 搜索类关键词直接判定 `chat`，跳过 LLM（省一次 10s+ 延迟）
  2. **LLM 分类**：输出 JSON（`intent` + `rewritten_query` + `ambiguous` + `clarify`）；解析时先剥 `<think>` 包裹再用正则抓 JSON，避免模型带思考前缀时整体失败
  3. **兜底降级**：解析失败 / LLM 报错 → `intent=chat`，不阻断主流程
- **planner** (`nodes/planner.py`)：把复杂问题拆成 1-4 步检索计划（`plan_summary` + `steps[].query`）。解析失败返回空计划，自动回退到 research 单轮链路。
- **execute_plan / replan** (`nodes/research.py`)：逐步执行计划；材料不足则 replan 生成补查角度，最多 `HD_RESEARCH_MAX_ITER` 轮（默认 3）。若某轮没有新增切片则置 `replan_stalled` 提前终止，避免空转。
  - **并行执行**：`parallel_plan_node` 用 `ThreadPoolExecutor` 并发跑多步（worker 数取 `min(步数, 4)`，读取时用 `getattr` 兜底默认 4），单步异常被捕获并记日志，其余步骤结果照常返回。
  - **预算设计**：计划步数上限 = `HD_PLANNER_MAX_STEPS`（默认 4），replan 轮数上限 = `HD_RESEARCH_MAX_ITER`（默认 3）——两者独立，旧版"计划步数被 max_iter 截断"的问题已修复。
- **ingest** (`nodes/ingest.py`)：检测 URL / 触发词，调用入库 + LLM 提取标题/标签/摘要 + FTS5 软查重
- **report** (`nodes/report.py`)：拉取最近 N 天笔记，按 tag 分组，LLM 生成 Markdown 周报

**启发式升级**：`route_by_intent` 在计划步数 ≥ 2 时把 `chat` 自动升级为 `research`；`_AUTO_PLAN_KEYWORDS`（对比 / 比较 / vs / 汇总…）命中时自动开启 planner。

**状态与 checkpoint**：
- `AsyncSqliteSaver` 是**惰性异步单例**——构造依赖运行中的事件循环，所以 `get_graph()` 在首次异步调用时才建并缓存，落盘 `data/checkpoints.sqlite`
- `_build_initial_state` 每轮显式重置全部 per-turn 字段（`intent` / `plan` / `clarify_request` 等），防止上一轮 checkpoint 残留（如 stale `intent="ingest"`）串味
- `AgentState.messages` 刻意**不加 `operator.add` reducer**：chat.py 每轮从 DB 重新播种完整历史，服务端是唯一事实源

**歧义澄清（clarify）**：router 判 `ambiguous` → 图终止 → SSE 推 `clarify` 事件 → 前端弹卡片 → `POST /api/chat/clarify` 唤醒等待的 future → 带 `skip_clarify=True` 二次跑图（`clarify.py`，等待超时 300 秒）。

**环境变量**：`HD_USE_GRAPH=false` 一键回退旧直调链路；`HD_PLANNER_ENABLED=false` 关掉规划器。

### 2.6 子代理与工具治理

- **子代理**（`agent/subagents.py`）：`explore` / `plan` / `general` 三个 profile，各自独立 system prompt + 工具白名单；`_filter_tools_for_mode` 会剔除破坏性工具（写文件、执行命令等），`run_subagent_stream` 流式回传，SSE `event: subagent` 可见。
- **Hooks**（`agent/hooks.py`）：`PreToolUse` / `PostToolUse` 两阶段。脚本走 stdin 收 JSON、stdout 写 JSON，返回 `{"block": "reason"}` 可否决该次工具调用；超时默认 5 秒（夹在 0.5~30s 之间）。管理接口 `api/hooks.py`。
- **权限规则**（`api/permissions_rules.py`）：按工具名定义"需审批 / 直接放行"，`AgentState.agent_permission` 传递；MCP 破坏性工具默认走弹窗。
- **项目规则**（`agent/agents_md.py` + `api/project_rules.py`）：候选文件名 `AGENTS.md` / `.agents.md` / `CLAUDE.md` / `.claude.md`，**target-first 向上最多 6 层**收集，单文件 32KB 上限、最多 8 个来源；内容注入系统提示，也可在设置页直接编辑。

### 2.7 长期记忆

- **抽取**：`extract_facts`（`agent/memory.py`）每轮后台跑，只看最近 6 条消息，用 `<<EXISTING>>` 占位把已有事实喂给模型做去重
- **召回**：`recall_facts` 按相关性挑 `HD_MEMORY_MAX_FACTS` 条（默认 8）注入上下文
- **压缩**：事实超过 12 条时 `summarize_overflow` 压成 ≤150 字摘要
- **短路指令**：用户说"记住…/忘记…"时 chat.py 走 memory shortcut 直接落库，不进 LLM
- 开关 `HD_MEMORY_EXTRACTION_ENABLED`，管理接口 `api/memory.py`

### 2.8 上下文预算（`agent/context.py`）

| 常量 | 值 | 作用 |
|---|---|---|
| `CONTEXT_TOKEN_BUDGET` | 3000 | 引用 chunks 总预算 |
| `MAX_CHUNK_CHARS` | 800 | 单 chunk 硬上限 |
| `HISTORY_TOKEN_BUDGET` | 2000 | 历史消息预算 |
| 历史滑窗 | 12 条 | 超出走摘要 |

`estimate_tokens` 用启发式估算（CJK ×1.5 + ASCII ×0.35），不依赖 tokenizer。无检索结果时 `_strip_citation_rules` 会剥掉引用规则，防止模型输出"来源：无"。

### 2.9 MCP 工具调用

- **协议**：stdio 通信（不支持 http），通过 `npx` 启动 MCP server
- **会话复用**：同一轮对话共享一个 MCP server 进程（`MCPSession` 类），进程死亡自动重启
- **工具发现**：模型先调 `mcp_discover_tools` 获取工具清单，再决定调用哪些
- **流式传输**：工具调用响应 token-by-token 流式转发（`astream`），不阻塞
- **权限控制**：default=每次访问前询问；full=完全访问不询问
- **预设**：`npx @modelcontextprotocol/server-filesystem`，默认授权 backend 工作目录
- **Windows 适配**：`npx` → `npx.CMD`（`shutil.which`），`taskkill /F /T /PID` 杀子进程树

### 2.10 会话管理

- `ChatSession` + `ChatMessage` 持久化（SQLModel）
- 标题自动取首条 user 消息前 50 字
- 无 `session_id` 时后端自动建会话，SSE `event: session` 推回
- 侧栏搜索：标题 / 内容模糊匹配
- 删除会话同时清空消息
- 上下文管理：历史滑窗 12 条 / 2000 tokens，溢出摘要到 ≤150 字符

### 2.11 飞书知识库集成

> 把飞书 wiki 空间里的文档 + 多维表格自动同步到本地 KB。

| 模块 | 文件 | 职责 |
|---|---|---|
| OAuth 客户端 | `tools/feishu_client.py` | tenant_access_token 缓存 2h，线程安全 |
| Wiki 节点 DFS | `tools/feishu_client.py` | `walk_nodes()` 递归所有子节点 |
| 文档解析 | `tools/parse_feishu_doc.py` | docx `raw_content` 轻清洗 + bitable records->Markdown |
| 同步编排 | `feishu_sync.py` | `sync_space()` / `sync_all()` / `SyncResult` 计数 |
| 路由 | `api/feishu.py` | `GET /status` / `GET /spaces` / `POST /sync` |
| 后台循环 | `main.py` startup | `FEISHU_SYNC_INTERVAL_MIN=15`（可设为 0 关掉） |

**增量同步**：Note 表加 `source_revision` + `source_updated_at`，对比 Feishu `obj_edit_time`，相同 → skipped，不同 → 删除旧 chunks → 重走入库。

**API**：
```
GET  /api/feishu/status         # 配置状态 + interval
GET  /api/feishu/spaces         # 列出可见 wiki 空间
POST /api/feishu/sync           # body 可选 {"space_id": "..."}，触发同步
```

### 2.12 前端 UX

- **蓝色主题**：naive-ui `themeOverrides` primary = `#3b82f6`
- **页面**：ChatView / NotesView / LoginView / SkillsMcpView
- **组件**：
  - `ChatHistory` — 侧栏：会话列表 + 搜索 + 新建 + 跳知识库
  - `MessageBubble` — 消息气泡（含思考过程折叠 + Mermaid 图表卡片）
  - `CitationPreview` — 引用卡片
  - `ModelSelector` — 模型 + 推理等级
  - `StreamingIndicator` — 流式阶段指示
  - `ThinkingIndicator` — 思考过程指示
  - `McpPanel` — MCP 工具调用历史
  - `McpCallHistory` — MCP 调用记录
  - `SkillsPanel` — 技能面板
  - `CommandPalette` — 命令面板
  - `IngestResultCard` — 入库结果卡片
  - `SettingsDrawer` — 设置抽屉（Tab 按钮右对齐）
  - `SettingsOperations` — 设置操作区
  - 计划进度条（plan-strip）/ 澄清卡片 — 计划执行进度可视化与歧义澄清交互
  - 设置页新增面板 — 项目规则 / 钩子 / 子代理 / 权限规则 / 长期记忆
- **Pinia stores**：chat / notes / sessions / models / settings / auth
- **i18n**：`t('key', zh, en)` 动态语言切换，默认中文
- **HTTP 客户端**：fetch + 自动 `X-API-Key` header + 自写 `streamSse`

### 2.13 部署

- 后端：`uvicorn 127.0.0.1:5006`（同时托管前端 dist 静态文件）
- 前端 dev：vite `127.0.0.1:5174`
- 公网访问：花生壳 `https://11gv92qt74799.vicp.fun` → `192.168.1.7:5006`
- **一键启动**：`scripts/start-all.ps1`（杀旧 -> 起后端 -> 起前端 -> 探活）
- 上传限制：50MB（`HD_MAX_UPLOAD_BYTES` 可调）
- 鉴权：`Authorization: Bearer <token>` 或 `X-Auth-Token: <token>`，密码 PBKDF2 哈希存储

---

## 3. 系统架构

```
+----------------------+      SSE      +-----------------------+
|  Vue 3 + naive-ui    | <-----------> |   FastAPI (5006)     |
|  Chat / Notes /      |  REST / SSE   |   - chat (SSE+RAG)    |
|  Skills / MCP        |               |   - notes (ingest)    |
+----------------------+               |   - search            |
                                       |   - sessions          |
                                       |   - settings          |
                                       |   - feishu            |
                                       |   - mcp               |
                                       |   - agents            |
                                       |   - hooks             |
                                       |   - permissions       |
                                       |   - project-rules     |
                                       |   - memory            |
                                       |   - background        |
                                       |   - auth              |
                                       +----------+------------+
                                                  |
                                                  v
                                       +-----------------------+
                                       |   LangGraph (7 nodes) |
                                       |   router → planner →  |
                                       |   execute_plan ⟲ replan|
                                       |   / retrieve / ingest |
                                       |   / report            |
                                       +----+--------------+----+
                                            |              |
                          +-----------------+              +-----------------+
                          v                                               v
              +------------+----------+                      +-----------+-----------+
              |   Hybrid Search       |                      |   LLM (factory)      |
              |   0.7 * vec + 0.3*kw  |                      |   7+ providers        |
              +----+----+-------------+                      +-----------+-----------+
                   |    |                                            |
                   v    v                                            v
            +------+    +------+                            +---------+---------+
            |Chroma|    | FTS5 |                            | OpenAI / Anthropic|
            +------+    +------+                            | DeepSeek / Zhipu  |
                                                            | Moonshot / Ollama |
                                                            | MiniMax / ...     |
                                                            +-------------------+
```

---

## 4. 数据流（RAG 三层）

### 入库

```
PDF/DOCX/PPTX/XLSX/CSV/HTML/TXT/图片/URL/飞书docx/飞书bitable
    -> parse_*  -> {title, content}
    -> chunk_text(500/80)
    -> embed_texts (OpenAI-compat / MiniMax native)
    -> Chroma.add + SQLite FTS5.add
    -> Note {embedded=True, chunk_count=N}
```

### 检索 + 问答

```
ChatView -> chat.send() -> chatStream() (api/chat.ts) -> POST /api/chat (SSE)
   |
   v
api/chat.py:  _build_initial_state()  # 每轮重置 per-turn 字段
   |            use_rag=True?
   -> graph.astream(initial_state)   # AsyncSqliteSaver 加载 checkpoint
        -> router_node (intent 分类 + query 改写 + ambiguous)
             ├─ ambiguous -> clarify（终止，SSE 推事件等前端）
             ├─ research  -> planner_node -> execute_plan_node ⟲ replan_node
             ├─ chat      -> retrieve_node（内部 hybrid_search）
             ├─ ingest    -> ingest_node
             └─ report    -> report_node
        -> answer_node: build_messages(messages + chunks) -> LLM stream
   -> SSE: session -> stage -> plan -> clarify -> citations
           -> message delta -> tool / permission / subagent -> done
   -> 前端 stream 累加 + stripThink + 渲染 + CitationPreview
```

`hybrid_search` 内部（`storage/hybrid.py`）：
```
embed_texts(query, mode="query")
vector_search(emb, top_k=10)      [Chroma cosine]
fts_search(query, top_k=10)       [SQLite FTS5 BM25, CJK 双 pass]
merge + dedupe by (note_id, chunk_index)
score = 0.7 * vec + 0.3 * kw
阈值过滤 (MIN_FINAL_SCORE=0.18, MIN_DIM_SCORE=0.18)
merge_neighboring_hits()   # 同笔记相邻 chunk 聚簇 + FTS5 回捞上下文
```

---

## 5. API 端点表

当前运行实例共 **84 个端点**（`GET /openapi.json` 实测）。按模块分组：

### 系统 / 鉴权

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| POST | `/api/auth/login` | 登录 |
| POST | `/api/auth/register` | 注册 |
| POST | `/api/auth/logout` | 登出 |
| GET | `/api/auth/me` | 当前用户 |
| DELETE | `/api/auth/me` | 注销账号 |
| GET | `/api/auth/me/export` | 导出个人数据 |
| POST | `/api/auth/change-password` | 改密码（含未登录态） |
| POST | `/api/auth/change-password-authed` | 改密码（已登录） |

### 聊天

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/chat` | SSE 聊天（含 RAG / plan / clarify） |
| POST | `/api/chat/clarify` | 提交澄清回答，唤醒等待中的图 |
| POST | `/api/chat/permission` | 提交工具权限审批结果 |

### 知识与检索

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/notes` | 笔记列表 |
| GET | `/api/notes/{note_id}` | 笔记详情 |
| DELETE | `/api/notes/{note_id}` | 删除笔记 |
| GET | `/api/notes/{note_id}/download` | 下载磁盘 .md |
| POST | `/api/notes/{note_id}/reembed` | 重跑 embedding |
| POST | `/api/notes/file` | 文件上传入库 |
| POST | `/api/notes/image` | 图片 OCR 入库 |
| POST | `/api/notes/pdf` | PDF 上传入库 |
| POST | `/api/notes/text` | 文本入库 |
| POST | `/api/notes/url` | URL 抓取入库 |
| GET | `/api/notes-stats` | 笔记 + Chroma 计数 |
| POST | `/api/search` | 独立 hybrid 搜索（不调 LLM） |
| GET | `/api/ocr-status` | OCR / Tesseract 可用性 |

### 会话

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/sessions` | 会话列表 |
| POST | `/api/sessions` | 新建会话 |
| GET | `/api/sessions/{session_id}` | 会话详情 |
| PATCH | `/api/sessions/{session_id}` | 重命名 / 更新 |
| DELETE | `/api/sessions/{session_id}` | 删会话 |
| POST | `/api/sessions/{session_id}/fork` | 从该会话分叉出新会话 |

### 模型与设置

| 方法 | 路径 | 用途 |
|---|---|---|
| GET / POST | `/api/settings/models` / `/api/settings/custom-models` | 模型配置读写 |
| GET / POST | `/api/settings/llm-config` | LLM 全局配置 |
| POST | `/api/settings/custom-models` | httpx 探测 `<base_url>/models` |
| GET / POST | `/api/custom-models` | 自定义模型列表 / 新增 |
| GET / POST | `/api/custom-models/selected` | 当前选中模型读写 |
| PATCH / DELETE | `/api/custom-models/{model_id}` | 修改 / 删除模型 |

### Agent 治理

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/agents/run` | 手动跑子代理 |
| POST | `/api/agents/plan-suggest` | 生成计划建议 |
| GET / POST | `/api/hooks` | 钩子列表 / 新增 |
| POST | `/api/hooks/test` | 测试钩子脚本 |
| GET | `/api/permissions/rules` | 权限规则列表 |
| PATCH | `/api/permissions/rules` | 更新权限规则 |
| DELETE | `/api/permissions/rules/{tool_pattern}` | 删除某条规则 |
| GET / POST | `/api/project-rules` | 项目规则读写（AGENTS.md） |
| POST | `/api/project-rules/resolve` | 解析某目录命中的规则集 |

### 长期记忆

| 方法 | 路径 | 用途 |
|---|---|---|
| GET / POST | `/api/memory/facts` | 事实列表 / 新增 |
| PUT / DELETE | `/api/memory/facts/{fact_id}` | 修改 / 删除事实 |
| GET | `/api/memory/recall` | 按查询相关性召回事实 |

### MCP

| 方法 | 路径 | 用途 |
|---|---|---|
| GET / POST | `/api/mcp` | MCP server 列表 / 新增 |
| PATCH / DELETE | `/api/mcp/{server_id}` | 修改 / 删除 server |
| POST | `/api/mcp/{server_id}/test` | 连通性测试 |
| GET | `/api/mcp/presets` | 预设列表 |
| POST | `/api/mcp/presets/{preset_id}` | 一键安装预设 |
| GET | `/api/mcp/calls` | 调用记录 |
| DELETE | `/api/mcp/calls` | 清空调用记录 |
| GET | `/api/mcp/calls/stats` | 调用统计 |

### 技能

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/skills` | 已装技能列表 |
| GET | `/api/skills/recommended` | 推荐技能 |
| GET | `/api/skills/{skill_id}/detail` | 技能详情 |
| POST | `/api/skills/install/{skill_id}` | 安装技能 |
| POST | `/api/skills/upload` | 上传技能包 |
| GET | `/api/skills/{skill_id}/download` | 下载技能包 |
| DELETE | `/api/skills/{skill_id}` | 卸载技能 |

### 飞书

| 方法 | 路径 | 用途 |
|---|---|---|
| GET / POST | `/api/feishu/config` | 飞书配置读写 |
| GET | `/api/feishu/status` | 配置状态 + interval |
| GET | `/api/feishu/spaces` | 列出可见 wiki 空间 |
| POST | `/api/feishu/sync` | 触发同步 |
| POST | `/api/feishu/test` | 连通性测试 |

### 后台任务

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/background/kinds` | 可跑的任务类型（reindex / ocr_all / sync_feishu / embed_all） |
| GET | `/api/background/jobs` | 任务状态列表 |
| POST | `/api/background/start` | 启动任务（SSE 子进程产出） |
| POST | `/api/background/reindex` | 触发全量重建索引 |

---

## 6. 目录结构

```
one_agent/
|-- docs/
|   |-- PLAN.md                # 项目计划书（P0-P9）
|   |-- RAG.md                 # RAG 三层链路详解
|   |-- SKILLS.md              # MCP 技能体系
|   |-- FEATURES.md            # ★ 本文档：功能现状总览
|   |-- CONTEXT_UPGRADE.md     # 上下文管理升级实施记录
|   \-- file-writing-policy.md # UTF-8 无 BOM 写入规范
|-- backend/
|   \-- app/
|       |-- main.py            # FastAPI 入口 + 中间件 + Feishu 后台循环
|       |-- config.py          # pydantic-settings（planner / memory / parallel 等开关）
|       |-- feishu_sync.py     # 飞书同步编排
|       |-- api/               # 19 个 REST 路由模块 / 84 个端点
|       |   |-- agents.py  auth.py  background.py  chat.py
|       |   |-- custom_models.py  feishu.py  health.py
|       |   |-- hooks.py  mcp.py  memory.py  notes.py
|       |   |-- permissions_rules.py  project_rules.py
|       |   |-- search.py  sessions.py  settings.py  skills.py
|       |-- agent/             # LangGraph（7 节点）
|       |   |-- graph.py       # router → planner → execute_plan ⟲ replan
|       |   |-- state.py       # AgentState（含 plan / clarify / permission 字段）
|       |   |-- schemas.py
|       |   |-- clarify.py     # 歧义澄清：future 等待 + 300s 超时
|       |   |-- hooks.py       # PreToolUse / PostToolUse 钩子生命周期
|       |   |-- subagents.py   # explore / plan / general profile + 工具白名单
|       |   |-- memory.py      # extract_facts / recall_facts / summarize_overflow
|       |   |-- context.py     # 上下文预算 + trim_history + build_messages
|       |   |-- agents_md.py   # AGENTS.md 分层规则加载（target-first，6 层）
|       |   |-- prompts/
|       |   |   \-- config.yaml  # 提示词配置（热加载，mtime 缓存）
|       |   \-- nodes/
|       |       |-- router.py      # 意图分类 + query 改写 + 歧义检测
|       |       |-- planner.py     # 复杂问题拆 1-4 步检索计划
|       |       |-- research.py    # execute_plan / replan / parallel_plan
|       |       |-- retrieve.py    # hybrid_search 包装
|       |       |-- ingest.py      # URL/文本检测 + LLM 元数据 + FTS5 软查重
|       |       |-- report.py      # 时间窗拉取 + 按 tag 分组 + LLM 周报
|       |       \-- answer.py      # 提示词组装 + citations 收集
|       |-- embeddings/factory.py   # OpenAI-compat + MiniMax 原生 body
|       |-- llm/factory.py          # 7+ provider + reasoning + base_url override
|       |-- storage/
|       |   |-- db.py          # SQLModel + FTS5 + ChatSession/ChatMessage
|       |   |-- vector.py      # Chroma collection 封装
|       |   |-- hybrid.py      # 向量 0.7 + FTS5 0.3 + 父块上下文扩展
|       |   \-- models_store.py  # 模型配置持久化（原子写 + 跨进程锁）
|       \-- tools/             # 11 个解析器 + ingest + ocr + feishu
|           |-- chunk.py  fetch_url.py  ocr.py  ingest.py
|           |-- parse_pdf.py  parse_doc.py  parse_pptx.py  parse_xlsx.py
|           |-- parse_csv.py  parse_html.py
|           |-- parse_feishu_doc.py
|           \-- feishu_client.py
|-- data/                      # 运行时数据（勿手改）
|   |-- notes.db               # 实际在用的是 backend/data/notes.db
|   |-- chroma/                # 向量库
|   \-- checkpoints.sqlite     # LangGraph 检查点
|-- frontend/
|   \-- src/
|       |-- main.ts  App.vue  style.css  router/index.ts  i18n.ts
|       |-- api/       # client.ts + chat/notes/sessions/settings/custom-models.ts
|       |-- stores/    # chat / notes / sessions / models / settings / auth (Pinia)
|       |-- views/     # ChatView / NotesView / LoginView / SkillsMcpView
|       \-- components/
|           |-- ChatHistory.vue       # 侧栏会话列表
|           |-- MessageBubble.vue     # 消息气泡 + Mermaid 图表卡片
|           |-- CitationPreview.vue   # 引用卡片
|           |-- ModelSelector.vue     # 模型 + 推理等级
|           |-- StreamingIndicator.vue # 流式阶段指示
|           |-- ThinkingIndicator.vue  # 思考过程指示
|           |-- McpPanel.vue          # MCP 工具调用面板
|           |-- McpCallHistory.vue    # MCP 调用记录
|           |-- SkillsPanel.vue       # 技能面板
|           |-- CommandPalette.vue    # 命令面板
|           |-- IngestResultCard.vue  # 入库结果卡片
|           |-- SettingsDrawer.vue    # 设置抽屉
|           \-- SettingsOperations.vue # 设置操作区
|-- scripts/
|   |-- start-all.ps1         # 一键启动
|   |-- install-service.ps1   # NSSM 安装 backend 为 Windows 服务
|   |-- backup.ps1            # SQLite VACUUM INTO + robocopy + 14 份轮转
|   |-- rag_eval/             # RAG 评测套件
|   |   |-- golden.jsonl      # 6 条 golden 样例
|   |   \-- run.py            # runner (Recall@K / MRR / 闲聊误命中率)
|   \-- table_tests/          # 表格识别测试套件（6/6 PASS）
|       |-- run_all.py
|       \-- sample.*          # fixture 文件
\-- logs/                     # hd.log（RotatingFileHandler, 10MB x 5）
```

---

## 7. 启动 & 自检

### 7.1 一键启动

```powershell
D:\one_agent\scripts\start-all.ps1
```

### 7.2 手动启动

```powershell
# 后端
cd D:\one_agent\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 5006

# 前端（新 shell）
cd D:\one_agent\frontend
npm install
npm run dev                   # http://127.0.0.1:5174
```

### 7.3 表格识别自检

```powershell
cd D:\one_agent\scripts\table_tests
$env:PATH += ";C:\Program Files\Tesseract-OCR"
D:\one_agent\backend\.venv\Scripts\python.exe run_all.py
```

**当前基线：6/6 PASS**。

### 7.4 RAG 自检

```powershell
cd D:\one_agent\backend
.\.venv\Scripts\python.exe -c "from app.storage.hybrid import hybrid_search; print(hybrid_search('牛魔王的来历', top_k=3, base_url='https://api.minimax.chat/v1', api_key='<key>', model='embo-01'))"
```

### 7.5 飞书自检

```powershell
# 配置：backend/.env 加 FEISHU_ENABLED=true + FEISHU_APP_ID + FEISHU_APP_SECRET
curl http://127.0.0.1:5006/api/feishu/status
curl http://127.0.0.1:5006/api/feishu/spaces
curl -X POST http://127.0.0.1:5006/api/feishu/sync
```

---

## 8. 已知约束 / 注意事项

1. **embedding 模型必须和入库时一致**：换 embedding 后旧的 Chroma 向量会失效，要用「重跑 embedding」按钮重建
2. **PowerShell UTF-8 BOM**：`docs/file-writing-policy.md` 是踩坑记录，写代码文件必须用 `[System.Text.UTF8Encoding]::new($false)`
3. **RAG 阈值**：`min_final_score=0.18`、`min_dim_score=0.18`，低于则不进 prompt
4. **MiniMax embedding**：必须显式传 `mode="db"`（存储）/ `mode="query"`（检索），否则会拿到 2013 invalid params；factory 已自动 fallback
5. **API key 脱敏**：列表接口返回 `sk****xx`，前端不回传含 `*` 的值，后端从 `models.json` 取真实 key
6. **飞书 App Secret**：写在 `backend/.env`（在 `.gitignore`），建议定期去 open.feishu.cn 重新生成
7. **上传大小限制**：默认 50MB，`HD_MAX_UPLOAD_BYTES` 环境变量可调
8. **OCR 依赖**：Tesseract（`chi_sim.traineddata` 需单独装），`ocr.py` 启动时会自动检测并提示
9. **MCP stdio**：不支持 http 传输，Windows 上 `npx` 需解析到 `npx.CMD`，杀进程用 `taskkill /F /T /PID`
10. **提示词热加载**：`config.yaml` 修改后自动生效（mtime 缓存），无需重启后端
11. **花生壳 UDP 中继**：Clash TUN 模式可能掐断 UDP 通道，需在 Clash 中给 `phtunnel.exe` 加直连规则
12. **数据目录**：`config.py` 的 `data_dir` 默认是相对 CWD 的 `./data`。真实在用的是 `backend/data/`（notes.db ≈ 2.1MB）；从仓库根目录启动后端会落到根目录的 `data/`，导致"看不到已有数据"。启动请用 `dev.py` 或 `start-all.ps1`
13. **同步 `hybrid_search` 跑在 async 端点里**：检索是同步 IO（Chroma + SQLite），高并发下会阻塞事件循环。当前单用户场景可接受，多用户需改 `run_in_executor`（见路线图）
14. **增强模块皆为"可失败的可选项"**：planner / clarify / hooks / memory / rerank 任一失败都不会阻断主流程，只会静默降级
15. **Plan 与 replan 是两套独立预算**：计划步数受 `HD_PLANNER_MAX_STEPS` 约束、replan 轮数受 `HD_RESEARCH_MAX_ITER` 约束，不要混用

---

## 9. 路线图（PLAN.md P0-P9 当前进度）

| 阶段 | 内容 | 状态 | 备注 |
|---|---|---|---|
| **P0** | 计划书 | OK | `docs/PLAN.md` |
| **P1** | 后端骨架 | OK | FastAPI + LangGraph 空图 + 模型工厂 |
| **P2** | 入库链路 | OK | URL / text / 11 种文件格式 + 飞书 |
| **P3** | 检索 + 问答 | OK | 向量 + 关键词混合 + 引用卡片 |
| **P4** | Agent 化 | OK | Router + 7 节点图全部跑通 |
| **P5** | 前端骨架 | OK | Vue 3 + Pinia + Naive UI |
| **P6** | 端到端 | OK | SSE / 引用卡片 / 入库对话框 |
| **P7** | 多模型切换 | OK | + Auto + 自定义 base_url + 推理 4 档 |
| **P8** | 体验打磨 | OK | 主题色 / 侧栏搜索 / 流式阶段 / 思考过程 / Mermaid 图表 |
| **P9** | 高级 | 部分 | OK MCP / OK Plan / OK 澄清 / OK 飞书增量 / OK 子代理 / OK Hooks / OK 项目规则 / OK 长期记忆 / 缺 RSS |

**已完成的高级能力**：MCP 工具调用、plan-and-execute 编排（含并行执行）、歧义澄清、子代理（explore/plan/general）、PreToolUse/PostToolUse 钩子、项目规则分层加载、长期记忆抽取与召回、权限规则、Mermaid 图表卡片化、提示词热加载、表格识别三层框架、飞书知识库增量同步、LangGraph checkpoint 持久化。

**未做的（可选下一步）**：
- RSS 定时抓取 + 周报推送
- 接入 PaddleOCR PP-StructureV2 提升扫描 PDF 表格识别精度
- 知识图谱可视化
- 多用户 / 团队知识库
- 移动端 App
- `hybrid_search` 移出事件循环（`run_in_executor`）
- `main.py` 的 `@app.on_event("startup")` 迁移到 lifespan

---

## 10. 相关文档索引

- `docs/PLAN.md` — 项目原始计划书（P0-P9 路线图 + 技术选型）
- `docs/RAG.md` — RAG 三层链路详解（含 CJK 查询修复记录）
- `docs/SKILLS.md` — MCP 技能体系说明
- `docs/CONTEXT_UPGRADE.md` — 上下文管理升级实施记录（Phase 1-4）
- `docs/file-writing-policy.md` — UTF-8 无 BOM 写入规范（踩坑记录）
- `README.md` / `README.zh.md` — 项目 README + 亮色主题截图
