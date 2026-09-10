# Smallhouse — 功能现状 (v0.6)

> 本地优先的个人知识库 Agent：把散落的文章 / 文件 / 对话 / 飞书文档统一塞进一个可检索的知识库，用自然语言向 LLM 提问，答案带原文引用。
> 最近更新：2026-09-10（含 MCP 工具调用、Plan 任务规划、歧义澄清、Mermaid 图表卡片、提示词热加载）

---

## 0. 一句话定位

**Smallhouse** = 本地优先的个人 Second Brain。多 LLM 自由切换 + 混合检索（向量 0.7 + 关键词 0.3）+ MCP 工具调用 + 飞书知识库同步。

---

## 1. 技术栈

| 层 | 技术 | 备注 |
|---|---|---|
| 后端框架 | FastAPI 0.115+ | 异步 REST + SSE 流式 |
| Agent / LLM 编排 | LangGraph 1.2 + LangChain 1.3 | Router → (retrieve/research/ingest/report) → answer |
| LLM 抽象 | `langchain-openai` + 工厂路由 | OpenAI / Anthropic + 7 家 OpenAI-compatible |
| Embedding | `httpx` 直连 `/v1/embeddings`，MiniMax 原生 body | OpenAI / MiniMax / 其他 OpenAI-compatible |
| 向量库 | ChromaDB（持久化到 `data/chroma/`） | cosine + HNSW |
| 关系库 | SQLite + SQLModel + FTS5 全文索引 | `data/notes.db` |
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

**SSE 事件流**：
```
event: session       -> 新会话时推 session_id
event: stage         -> 阶段（router / rag_search / llm_stream / done）
event: citations     -> [{note_id, chunk_index, title, snippet, score}, ...]
event: message       -> 流式 token delta
event: tool          -> MCP 工具调用状态
event: clarify       -> 歧义澄清选项
event: ingest        -> 入库结果
event: report        -> 周报结果
data: [DONE]
```

**前端 UX**：
- 输入框：知识库开关 + 模型选择器 + 推理等级 + Plan 开关 + 权限控制
- 流中：阶段标签 + 已用秒数 + 思考过程折叠显示
- 答案下方：引用卡片，列编号 / 标题 / chunk 索引 / score / 片段，可点击回原文
- 歧义问题：弹窗让用户选择（纵向排列 + "其他：___" 自定义输入）
- Mermaid 图表：卡片化渲染（标题栏 + 折叠/展开 + 复制源码 + 下载 PNG）

### 2.5 Agent 多节点编排（LangGraph + Router）

- **架构**：LangGraph StateGraph，入口走 Router 意图分类，按结果分支到不同节点
- **意图分类三层**：
  1. **正则快通道**：问候/闲聊关键词直接判定 `chat`，跳过 LLM（避免 12s 延迟）
  2. **LLM 分类**：输出 JSON（intent + rewritten_query + ambiguous + clarify），同时做查询改写和歧义检测
  3. **兜底降级**：JSON 解析失败/LLM 报错 → 降级 `intent=chat`，不阻断主流程
- **4 个子 agent**：
  - **Router** (`agent/nodes/router.py`)：意图分类 + query 改写 + 歧义检测
  - **研究 agent** (`agent/nodes/research.py`)：多轮 hybrid_search，最多 3 轮，每轮生成下一个检索角度，凑够 8 个 chunks 提前停
  - **入库管家** (`agent/nodes/ingest.py`)：检测 URL / 触发词，调用入库 + LLM 提取标题/标签/摘要 + FTS5 软查重
  - **周报 agent** (`agent/nodes/report.py`)：拉取最近 N 天笔记，按 tag 分组，LLM 生成 Markdown 周报
- **环境变量**：`HD_USE_GRAPH=false` 一键回退旧直调链路

### 2.6 MCP 工具调用

- **协议**：stdio 通信（不支持 http），通过 `npx` 启动 MCP server
- **会话复用**：同一轮对话共享一个 MCP server 进程（`MCPSession` 类），进程死亡自动重启
- **工具发现**：模型先调 `mcp_discover_tools` 获取工具清单，再决定调用哪些
- **流式传输**：工具调用响应 token-by-token 流式转发（`astream`），不阻塞
- **权限控制**：default=每次访问前询问；full=完全访问不询问
- **预设**：`npx @modelcontextprotocol/server-filesystem`，默认授权 backend 工作目录
- **Windows 适配**：`npx` → `npx.CMD`（`shutil.which`），`taskkill /F /T /PID` 杀子进程树

### 2.7 会话管理

- `ChatSession` + `ChatMessage` 持久化（SQLModel）
- 标题自动取首条 user 消息前 50 字
- 无 `session_id` 时后端自动建会话，SSE `event: session` 推回
- 侧栏搜索：标题 / 内容模糊匹配
- 删除会话同时清空消息
- 上下文管理：历史滑窗 max 20 条 / 6000 tokens，溢出摘要到 ≤300 字符

### 2.8 飞书知识库集成

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

### 2.9 前端 UX

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
- **Pinia stores**：chat / notes / sessions / models / settings / auth
- **i18n**：`t('key', zh, en)` 动态语言切换，默认中文
- **HTTP 客户端**：fetch + 自动 `X-API-Key` header + 自写 `streamSse`

### 2.10 部署

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
                                       |   - auth              |
                                       +----------+------------+
                                                  |
                                                  v
                                       +-----------------------+
                                       |   LangGraph           |
                                       |   Router -> 4 agents  |
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
api/chat.py:  use_rag=True?
   -> hybrid_search(query, top_k=5, embedding_model, embedding_base_url, api_key)
        -> embed_texts(query)              [OpenAI-compat]
        -> vector_search(emb, top_k=10)    [Chroma cosine]
        -> fts_search(query, top_k=10)     [SQLite FTS5 BM25, CJK 双 pass]
        -> merge + dedupe by (note_id#chunk_index)
        -> score = 0.7 * vec + 0.3 * kw
        -> 阈值过滤 (min_score=0.18, min_dim_score=0.18)
        -> trim top_k
   -> graph.stream(initial_state)
        -> router_node (intent 分类 + query 改写)
        -> retrieve_node / research_node / ingest_node / report_node
        -> answer_node
   -> answer_node: build_prompt(messages + chunks) -> LLM stream
   -> SSE: session -> stage -> message delta -> citations -> done
   -> 前端 stream 累加 + stripThink + 渲染 + CitationPreview
```

---

## 5. API 端点表

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| POST | `/api/auth/login` | 登录 |
| POST | `/api/auth/register` | 注册 |
| POST | `/api/chat` | SSE 聊天（含 RAG） |
| POST | `/api/search` | 独立 hybrid 搜索（无需 LLM） |
| GET / POST / DELETE | `/api/notes` | 笔记 CRUD |
| POST | `/api/notes/file` | 文件上传入库 |
| POST | `/api/notes/image` | 图片 OCR 入库 |
| POST | `/api/notes/text` | 文本入库 |
| POST | `/api/notes/url` | URL 抓取入库 |
| POST | `/api/notes/{id}/download` | 下载磁盘 .md |
| POST | `/api/notes/{id}/reembed` | 重新跑 embedding |
| GET | `/api/notes-stats` | 笔记 + Chroma 计数 |
| GET | `/api/sessions` | 会话列表 |
| POST | `/api/sessions` | 新建会话 |
| GET | `/api/sessions/{id}` | 会话详情 |
| DELETE | `/api/sessions/{id}` | 删会话 |
| GET / POST | `/api/settings/models` | 列出 / 写模型配置 |
| POST | `/api/settings/custom-models` | httpx 探测 `<base_url>/models` |
| GET | `/api/feishu/status` | 飞书配置状态 |
| GET | `/api/feishu/spaces` | 列出可见 wiki 空间 |
| POST | `/api/feishu/sync` | 触发同步 |
| POST | `/api/mcp/servers` | MCP server 管理 |
| POST | `/api/mcp/invoke` | MCP 工具调用 |
| GET | `/api/agents` | Agent 列表 |
| GET | `/api/memory` | 记忆管理 |
| POST | `/api/skills` | 技能管理 |

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
|       |-- config.py          # pydantic-settings
|       |-- feishu_sync.py     # 飞书同步编排
|       |-- api/               # 19 个 REST 路由
|       |   |-- agents.py  auth.py  background.py  chat.py
|       |   |-- custom_models.py  feishu.py  health.py
|       |   |-- hooks.py  mcp.py  memory.py  notes.py
|       |   |-- permissions_rules.py  project_rules.py
|       |   |-- search.py  sessions.py  settings.py  skills.py
|       |-- agent/             # LangGraph
|       |   |-- graph.py       # router -> 4 agents -> answer
|       |   |-- state.py
|       |   |-- schemas.py
|       |   |-- prompts/
|       |   |   \-- config.yaml  # 提示词配置（热加载，mtime 缓存）
|       |   \-- nodes/
|       |       |-- router.py      # 意图分类 + query 改写 + 歧义检测
|       |       |-- retrieve.py    # hybrid_search 包装
|       |       |-- research.py    # 多轮 hybrid_search + follow-up 生成
|       |       |-- ingest.py      # URL/文本检测 + LLM 元数据 + FTS5 软查重
|       |       |-- report.py      # 时间窗拉取 + 按 tag 分组 + LLM 周报
|       |       \-- answer.py      # 提示词组装 + citations 收集
|       |-- embeddings/factory.py   # OpenAI-compat + MiniMax 原生 body
|       |-- llm/factory.py          # 7+ provider + reasoning + base_url override
|       |-- storage/
|       |   |-- db.py          # SQLModel + FTS5 + ChatSession/ChatMessage
|       |   |-- vector.py      # Chroma collection 封装
|       |   |-- hybrid.py      # 向量 0.7 + FTS5 0.3 加权
|       |   \-- models_store.py  # 模型配置持久化（原子写 + 跨进程锁）
|       \-- tools/             # 11 个解析器 + ingest + ocr + feishu
|           |-- chunk.py  fetch_url.py  ocr.py  ingest.py
|           |-- parse_pdf.py  parse_doc.py  parse_pptx.py  parse_xlsx.py
|           |-- parse_csv.py  parse_html.py
|           |-- parse_feishu_doc.py
|           \-- feishu_client.py
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

---

## 9. 路线图（PLAN.md P0-P9 当前进度）

| 阶段 | 内容 | 状态 | 备注 |
|---|---|---|---|
| **P0** | 计划书 | OK | `docs/PLAN.md` |
| **P1** | 后端骨架 | OK | FastAPI + LangGraph 空图 + 模型工厂 |
| **P2** | 入库链路 | OK | URL / text / 11 种文件格式 + 飞书 |
| **P3** | 检索 + 问答 | OK | 向量 + 关键词混合 + 引用卡片 |
| **P4** | Agent 化 | OK | Router + 4 个子 agent 全部跑通 |
| **P5** | 前端骨架 | OK | Vue 3 + Pinia + Naive UI |
| **P6** | 端到端 | OK | SSE / 引用卡片 / 入库对话框 |
| **P7** | 多模型切换 | OK | + Auto + 自定义 base_url + 推理 4 档 |
| **P8** | 体验打磨 | OK | 主题色 / 侧栏搜索 / 流式阶段 / 思考过程 / Mermaid 图表 |
| **P9** | 高级 | 部分 | OK MCP / OK Plan / OK 澄清弹窗 / OK 飞书增量同步 / 缺 RSS |

**已完成的高级能力**：MCP 工具调用、Plan 任务规划、歧义澄清弹窗、Mermaid 图表卡片化、提示词热加载、表格识别三层框架、飞书知识库增量同步。

**未做的（可选下一步）**：
- RSS 定时抓取 + 周报推送
- 接入 PaddleOCR PP-StructureV2 提升扫描 PDF 表格识别精度
- 知识图谱可视化
- 多用户 / 团队知识库
- 移动端 App

---

## 10. 相关文档索引

- `docs/PLAN.md` — 项目原始计划书（P0-P9 路线图 + 技术选型）
- `docs/RAG.md` — RAG 三层链路详解（含 CJK 查询修复记录）
- `docs/SKILLS.md` — MCP 技能体系说明
- `docs/CONTEXT_UPGRADE.md` — 上下文管理升级实施记录（Phase 1-4）
- `docs/file-writing-policy.md` — UTF-8 无 BOM 写入规范（踩坑记录）
- `README.md` / `README.zh.md` — 项目 README + 亮色主题截图
