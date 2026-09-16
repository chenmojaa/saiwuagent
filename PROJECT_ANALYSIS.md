# 个人知识小助手 —— Agent 项目技术分析文档

> 本文档全面梳理项目的功能实现与技术方案，可直接作为简历项目的素材来源。

---

## 一、项目概览

**个人知识小助手** 是一个全栈 AI Agent 应用：用户通过自然语言对话，系统自动完成意图识别、歧义澄清、计划拆解、知识检索、多步研究、知识入库、周报生成等任务，并将答案连同可点击的引用来源一起流式返回。

- **后端**：Python 3.11+ / FastAPI / LangGraph / LangChain / SQLite (SQLModel) / ChromaDB / FTS5
- **前端**：Vue 3 / TypeScript / Pinia / Vue Router / Naive UI / marked + DOMPurify + mermaid
- **通信**：REST + SSE（Server-Sent Events）流式传输
- **大模型**：支持 OpenAI / Anthropic / MiniMax 等多 Provider 可切换（含 reasoning 模型）
- **规模**：后端 92 个 Python 文件（约 12,860 行）、前端约 8,971 行、84 个 API 端点

### 核心架构图（文字版）

```
用户提问
   │
   ▼
┌─────────────────────────── LangGraph 状态机（7 节点）─────────────────┐
│                                                                       │
│  router ──┬─ ambiguous   → clarify（推卡片，图终止等前端回话）→ END   │
│           ├─ chat        → retrieve（混合检索）→ answer → END         │
│           ├─ chat_no_rag → END（寒暄快速通道，跳过检索）              │
│           ├─ research    → planner → execute_plan ⟲ replan → answer   │
│           ├─ ingest      → ingest（文档解析入库）→ END                │
│           └─ report      → report（周报生成）→ END                    │
│                                                                       │
│  Checkpointer: AsyncSqliteSaver（data/checkpoints.sqlite，重启不丢状态）│
└───────────────────────────────────────────────────────────────────────┘
   │
   ▼
SSE 流式返回（session / stage / plan / clarify / message delta /
             citations / tool / permission / subagent / ingest / report / done）
```

> 除主图外还有三条治理侧链：**子代理**（explore / plan / general）、**工具钩子**（PreToolUse / PostToolUse）、**长期记忆**（每轮后台抽取事实 + 按需召回）。

---

## 二、后端功能与实现方式

### 2.1 Agent 核心：LangGraph 状态机编排

**文件**：`backend/app/agent/graph.py`、`state.py`

- 使用 **LangGraph StateGraph** 构建 **7 节点工作流**：`router` / `planner` / `execute_plan` / `replan` / `retrieve` / `ingest` / `report`，通过 `add_conditional_edges` 按 `route_by_intent` 分发；`route_after_planner` 决定计划是执行还是直接补查。
- 状态用 `TypedDict`（`AgentState`）定义，包含会话消息、用户画像、历史摘要、意图、改写查询、检索分块、计划（`plan` / `plan_cursor` / `plan_status` / `replan_stalled` / `use_planner`）、澄清请求（`clarify_request` / `skip_clarify`）、权限（`agent_permission`）等 25+ 字段。
- **关键设计决策**：`messages` 字段不使用 `operator.add` reducer——因为服务端每次从 DB 播种完整历史（服务端为唯一事实源），若用累加语义会与 checkpoint 历史叠加导致消息重复。
- **Checkpointer 持久化**：`AsyncSqliteSaver` 落盘至 `data/checkpoints.sqlite`，进程重启后对话状态不丢失。由于 saver 构造依赖运行中的事件循环，采用**惰性异步单例**模式在首次请求时构建并缓存。
- **per-turn 字段重置**：`_build_initial_state` 每轮显式重置 intent / plan / clarify_request 等字段，防止上一轮 checkpoint 残留（例如上一轮是 ingest，这一轮寒暄却仍被当作入库）。
- **plan-and-execute 循环**：`planner → execute_plan ⟲ replan → END`。循环的每一步都会产出一条 astream 事件，chat.py 转成 SSE 逐步推送——这解决了旧版"研究阶段是个黑盒、用户只能干等"的体验问题。

### 2.2 意图路由节点（Router）

**文件**：`backend/app/agent/nodes/router.py`

- **正则快速通道**（`_FAST_CHAT_PATTERNS`）：对「你好 / hi / 谢谢 / 你是谁」等寒暄意图直接正则匹配，绕过 LLM 分类（此前用 MiniMax 做分类产生 12s 延迟，改为正则后毫秒级响应）。
- **LLM 路由**：输出意图（chat / research / ingest / report）+ **查询改写** `rewritten_query`（结合最近 3 轮历史消解指代，如「它」→ 具体实体）+ **歧义标记** `ambiguous` / `clarify`。
- **踩坑经验**：结构化输出（structured output）会被 reasoning 模型的 `<think>` 包裹破坏，因此改为**原始 JSON 文本解析**（`_parse_router_json` 先剥 think 再用正则抓 JSON），容错更强。
- **启发式升级**：`route_by_intent` 在计划步数 ≥ 2 时把 `chat` 自动升级为 `research`；`_AUTO_PLAN_KEYWORDS`（对比 / 比较 / vs / 汇总 / 分析…）命中时自动开 planner。
- 提示词中约束改写查询「与用户最新消息保持同一语言」，避免改写成英文导致中文检索失效。

### 2.3 计划器与多步执行（Planner / execute_plan / replan）

**文件**：`backend/app/agent/nodes/planner.py`、`research.py`

- **planner**：`PLANNER_PROMPT` 要求模型把复杂问题拆成 1-4 步检索计划，输出 `{"plan_summary": ..., "steps": [{"query": ...}, ...]}`；解析失败返回空计划，让 research 自动回退到单轮链路。
- **execute_plan**：按 `plan_cursor` 逐步执行，每步走 hybrid_search，并把已收集 chunk 追加到状态。
- **replan**：材料不足（收集数 < `research_target_chunks` 且轮数 < `research_max_iter`）时生成下一个检索角度继续补查；某轮无新增则置 `replan_stalled` 提前收尾，避免空转耗预算。
- **预算设计（修复旧版冲突）**：计划步数上限 = `planner_max_steps`（默认 4），不再被 `research_max_iter` 截断——旧版 `plan_queries[:max_iter]` 会把 4 步计划砍成 3 步。replan 轮数上限 = `research_max_iter`（默认 3）。两套预算相互独立。
- **并行执行**：`parallel_plan_node` 用 `ThreadPoolExecutor`（`parallel_plan_max_workers`，默认 4）并发跑多步，单步失败被隔离，不影响其余步骤。
- **占位符 bug 修复**：旧版 `<<ORIG` 少了 `>>` 导致原始问题从未注入后续提示词，现通过 `<<ORIG>>` 确保每轮都锚定用户原问题。

### 2.4 歧义澄清（Clarify）

**文件**：`backend/app/agent/clarify.py`、`api/chat.py`

- router 判定 `ambiguous` → 图在 clarify 节点终止 → SSE 推 `clarify` 事件（问题 + 候选选项）→ 前端弹卡片 → `POST /api/chat/clarify` 提交答案并唤醒等待中的 future → 带 `skip_clarify=True` 二次跑图。
- 等待超时 `TIMEOUT_SEC = 300.0` 秒，超时后按原问题继续，不阻塞用户。
- 这是"模型驱动的澄清"而非关键词硬编码——由 router 在理解语义后决定是否需要追问。

### 2.5 混合检索（Hybrid Retrieval）

**文件**：`backend/app/storage/hybrid.py`、`vector.py`、`db.py`

- **双路召回**：
  - **向量检索**：ChromaDB，`cosine` 相似度 + HNSW 索引，本地持久化于 `backend/data/chroma/`；
  - **关键词检索**：SQLite **FTS5** 全文索引（`chunk_fts` 虚拟表，unicode61 tokenizer），弥补向量检索对「牛魔王」等 CJK 专有名词的弱点。
- **融合排序**：`final = 0.7 × 向量分 + 0.3 × 关键词分`，并设最终分数与维度分数双阈值过滤（`MIN_FINAL_SCORE` / `MIN_DIM_SCORE`，均 0.18）。
- **Embedding**：MiniMax `embo-01` API，文档入库用 `db` 模式、用户查询用 `query` 模式（同模型不同模式保证向量空间一致性），按 32 条/批批量请求；**按 base_url 预判 mode**，避免每次调用先试错再重试。
- **CJK 友好 FTS5**：`_fts_escape` 剥除 `" ( ) * ^ :` 等特殊字符后包成短语，双 pass（短语匹配 → per-token OR → LIKE 兜底）。
- **父块上下文扩展**：`merge_neighboring_hits` 把同一 note 内相邻 chunk 聚簇（窗口 2），再从 FTS5 回捞 `[center-2, center+2]` 拼成完整上下文，原命中片段另存 `matched_text`——命中不止是切片。
- Embedding API Key 缺失时自动降级为纯 FTS5 检索，保证服务可用。

### 2.6 文档解析与分块（Ingest）

**文件**：`backend/app/agent/nodes/ingest.py`、`backend/app/tools/chunk.py`

- 支持多种来源：**飞书文档、网页（trafilatura 抓取）、纯文本、上传文件**（docx / pptx / xlsx / PDF，PDF 用 pypdf + pdfplumber 双引擎，含 pytesseract OCR 兜底）。
- **分块策略**：500 字符窗口 + 80 字符重叠，优先在段落边界切分，其次句子边界——平衡召回粒度与上下文完整性。
- 入库时对内容做 **Unicode 转义序列解码**（修复 Excel 来源标题乱码问题）。
- Ingest 节点由 LLM 自动生成标题、标签、摘要，支持重复检测（duplicate_of）。

### 2.7 流式作答与引用（Answer）

**文件**：`backend/app/api/chat.py`、`backend/app/agent/nodes/answer.py`

- **SSE 事件分级推送**：`session`（会话 id）/ `stage`（router/agent/rag_search/llm_stream 各阶段进度）/ `plan`（计划卡片 created/step_done/replan/done）/ `clarify`（澄清请求）/ `citations`（引用列表）/ `message`（正文增量，默认事件名）/ `answer`（记忆快捷指令的整段答复）/ `tool`（MCP 调用 start/call/result）/ `permission`（权限审批 request/result）/ `subagent`（子代理进度）/ `ingest` / `report` / `error` / `done`，前端可展示全流程进度条。
- **上下文组装固定顺序**：system（人设 + 历史摘要 + 用户画像 + 参考材料）→ 对话历史 → 当前问题。
- **Token 预算控制**：参考材料总预算 3000 tokens（低分块先截断），单块硬上限 800 字符。
- **历史滑动窗口**：最近 12 条消息 / 2000 tokens，窗口外内容压缩为 ≤150 字摘要注入 system。
- **引用系统**：提示词要求 LLM 在正文中插入 `[n]` 标记；后端按 index 排序 citations 保证与前端按钮映射正确；正则提取引用（已知对 LLM 格式波动敏感，是待改进点）。
- **提示词工程**：人设「个人知识小助手」；语言约束（与提问同语言）；空参考兜底（无参考资料时用自身知识回答且绝不提参考材料，由 `_strip_citation_rules` 剥离引用规则）。

### 2.8 上下文预算（Context）

**文件**：`backend/app/agent/context.py`

- `CONTEXT_TOKEN_BUDGET = 3000`（引用材料总预算）、`MAX_CHUNK_CHARS = 800`（单块硬限）、`HISTORY_TOKEN_BUDGET = 2000`（历史预算）、历史滑窗 12 条。
- `estimate_tokens` 用启发式估算（CJK ×1.5 + ASCII ×0.35），不引入 tokenizer 依赖。
- `format_context` / `trim_history` / `build_messages` 三个函数串起完整的 prompt 组装链。

### 2.9 子代理（Sub-agents）

**文件**：`backend/app/agent/subagents.py`

- 三个内置 profile：**explore**（探索检索）、**plan**（规划）、**general**（通用），各自独立 system prompt + 工具白名单。
- `_filter_tools_for_mode` 按 profile 剔除破坏性工具（写文件 / 执行命令等），确保子代理不会越权。
- `run_subagent_stream` 流式回传执行过程，SSE `event: subagent` 可见；手动触发接口 `POST /api/agents/run`。

### 2.10 工具钩子（Hooks）

**文件**：`backend/app/agent/hooks.py`、`api/hooks.py`

- 两阶段：**PreToolUse**（调用前）/ **PostToolUse**（调用后），脚本从 stdin 读 JSON、往 stdout 写 JSON。
- **可否决**：PreToolUse 返回 `{"block": "reason"}` 即拒绝该次工具调用，并把 reason 反馈给模型。
- 超时默认 5 秒（`_DEFAULT_TIMEOUT_S = 5.0`，夹在 0.5~30s 之间），钩子失败不阻断主流程。
- 可在设置页编写、测试（`POST /api/hooks/test`）。

### 2.11 权限规则（Permissions）

**文件**：`backend/app/api/permissions_rules.py`

- 按**工具名模式**定义"需审批 / 直接放行"，规则落库后通过 `AgentState.agent_permission` 传给图。
- MCP 破坏性工具默认走弹窗审批，前端 `permission` SSE 事件驱动，用户答复走 `POST /api/chat/permission`。

### 2.12 项目规则（AGENTS.md 分层加载）

**文件**：`backend/app/agent/agents_md.py`、`api/project_rules.py`

- 候选文件名 `AGENTS.md` / `.agents.md` / `CLAUDE.md` / `.claude.md`。
- **target-first 向上最多 6 层**收集，单文件 32KB 上限、最多 8 个来源；解析为 `RuleSet`（含每个 `RuleSource` 的路径与内容），注入 system prompt。
- 支持在设置页直接编辑，`POST /api/project-rules/resolve` 可预览某目录实际命中的规则集。

### 2.13 长期记忆（Long-term Memory）

**文件**：`backend/app/agent/memory.py`、`api/memory.py`

- **抽取** `extract_facts`：每轮后台跑，只看最近 6 条消息，用 `<<EXISTING>>` 把已有事实喂给模型做去重，避免重复落库。
- **召回** `recall_facts`：按相关性挑 `memory_max_facts`（默认 8）条注入上下文。
- **压缩** `summarize_overflow`：事实超过 12 条时压成 ≤150 字摘要。
- **短路指令**：用户说"记住…/忘记…"时 chat.py 走 memory shortcut 直接落库，不进 LLM，响应更快。
- 开关 `HD_MEMORY_EXTRACTION_ENABLED`；CRUD 走 `/api/memory/facts`。

### 2.14 用户认证（Auth）

**文件**：`backend/app/api/auth.py`

- **注册/登录**：手机号 + 密码，`users` 表存储 `password_salt` + PBKDF2 哈希（**不存明文**），phone 唯一索引。
- **Token 机制**：自签格式 `userId.exp.HMAC签名`，密钥持久化于 `backend/data/auth_secret` 文件（重启不掉线），TTL 7 天。
- API：`/register`、`/login`、`/change-password`、`/me`（服务端验签）。
- 中间件自动将请求中的 X-API-Key 持久化到服务端（解决 Embedding Key 首次配置问题）。

### 2.15 飞书知识库同步

**文件**：`backend/app/feishu_sync.py`

- 基于 `obj_edit_time` 的**增量同步**：仅拉取上次同步后变更的文档。
- **失败重试机制**（关键修复）：初始同步时 Embedding 失败（缺 API Key）的文档曾因 revision 已推进而被永久跳过（「待索引」状态）。修复为：跳过条件改为「revision 匹配 **且** embedded=True」，且**仅在 embedding 成功后才推进 revision**，未索引文档每轮自动重试。

### 2.16 技能系统（Skills）

**文件**：`backend/app/api/skills.py`

- 支持上传**文件夹或 zip 压缩包**，后端统一处理：
  - **安全校验**：限制压缩包大小、文件数、路径穿越（zip slip）防护；
  - 自动解压，要求根目录含 `SKILL.md` 才视为合法技能；
  - 注册写入 `installed.json`，支持查看技能文件结构、打包下载 zip。
- 技能中心分「推荐首页」与「我的技能」两个视图，支持搜索、介绍展开/收起。

### 2.17 MCP 服务管理与工具调用

**文件**：`backend/app/api/mcp.py`、`backend/app/agent/tools/mcp_client.py`、`mcp_tools.py`

- **配置管理**：MCP 服务 CRUD，支持 stdio / http（SSE / Streamable HTTP）两种传输方式，启动参数与环境变量 JSON 配置，连接测试，内置预设一键导入。
- **Agent 真实调用 MCP（全链路已打通）**：
  - **自研轻量 MCP 客户端**（不依赖 `mcp`/`langchain-mcp-adapters`）：子进程拉起 stdio server → JSON-RPC 2.0 握手（initialize）→ `tools/list` 工具发现 → `tools/call` 工具执行；
  - **协议细节**：MCP stdio 传输为**换行分隔 JSON**（非 LSP Content-Length 帧）；Windows 下用**每进程单例读线程 + queue** 读取（`select()` 不支持 Windows 管道，且多读线程会互抢响应行）；`npx` 需解析为 `npx.CMD`；进程清理用 `taskkill /T` 杀整棵进程树（terminate 只杀 cmd 壳会留下 node 僵尸）；
  - **工具抽象**：暴露 `mcp_list_servers` / `mcp_discover_tools` / `mcp_invoke` 三个聚合工具（而非每工具一个），模型先发现再调用；能力清单注入 system prompt；
  - **tool-call 循环**：answer 节点 `bind_tools` 后最多 4 步循环（可配），每次调用的发起/结果通过 SSE `tool` 事件推给前端，渲染为 running→ok/failed 状态 chips。

### 2.18 周报代理（Report）

**文件**：`backend/app/agent/nodes/report.py`

- 拉取最近 N 天笔记 → 按标签分组 → LLM 汇总生成结构化周报 → **周报本身再作为一条笔记入库**，形成知识闭环。

### 2.19 提示词与配置管理

- 提示词外置于 `config.yaml`，支持多模型配置；
- 基于 **mtime 的提示词缓存自动失效**（`__init__.py`），改提示词无需重启后端；
- 每请求级参数覆盖（provider / model / base_url / api_key / reasoning_level / embedding_model override），支持运行时切换模型。

---

## 三、前端功能与实现方式

### 3.1 整体架构

- **Vue 3 组合式 API + TypeScript + Pinia + Vue Router + Naive UI**，Vite 构建（HMR 热更新）。
- 页面：登录页 `/login`、聊天 `/chat`（含会话 ID 路由）、知识库 `/notes`、设置 `/settings`、MCP 管理 `/mcp`、技能页 `/skills`。
- **明暗主题**：CSS 变量体系（`--bg-app` / `--brand-blue` / `--text-primary` 等），组件统一引用变量实现一键换肤。

### 3.2 SSE 流式传输

**文件**：`frontend/src/api/client.ts`、`chat.ts`

- `streamSse()` 基于 fetch POST 流式读取，手动按 `event:` / `data:` / 空行解析 SSE 协议（EventSource 不支持 POST 的替代方案）。
- 将后端事件映射为 `session / stage / plan / clarify / citations / message / answer / tool / permission / subagent / ingest / report / error / done` 前端事件。

### 3.3 思考过程（Thinking）流式展示

**文件**：`frontend/src/stores/chat.ts`

- 流式期间维护 `visible` / `thinkBuf` / `inThink` 三个缓冲区，实时从 delta 中分离 `<think>...</think>` 内容，思考过程**全程流式可见**（默认折叠，可展开）。
- **踩坑修复**：
  - `'</think>'` 偏移 bug（close 标签 8 字符曾被 +9，吃掉答案首字）；
  - reasoning 模型流式输出偶尔不闭合 `</think>`，正则改为**闭合标签可选**，保证未闭合时也能提取已有思考内容；
  - 渲染条件改为「只要有内容（含未完成思考块）就渲染气泡」，修复首次不显示、切页返回才显示的问题。

### 3.4 消息渲染（MessageBubble）

**文件**：`frontend/src/components/MessageBubble.vue`

- **三级解析**：`splitThink()` 提取思考块 → `splitSourceLine()` 提取尾部「来源：[n]」→ 正文 Markdown 渲染。
- **Markdown 渲染管线**：`marked` 渲染 + 自定义 code renderer 保留 mermaid 源码 → **DOMPurify 消毒防 XSS** → mermaid 异步渲染图表。
- **行内引用交互**：正文中 `[n]` 替换为带 `data-cite` 的可点击徽章（事件委托），点击高亮对应来源卡片。
- 思考区块展开状态持久化到 localStorage。

### 3.5 引用来源预览（CitationPreview）

- 展示 `[index]` + 标题 + 来源类型标签（飞书文档 / 网页 / 文本 / 上传文件）+ chunk/score + 摘要片段；
- 前端对 LLM 返回的 `\uXXXX` 转义标题/摘要做解码显示（配合后端入库解码，双保险）。

### 3.6 消息操作（复制 / 重新生成 / 删除 / 回退）

**文件**：`frontend/src/views/ChatView.vue`、`stores/chat.ts`

- hover 消息卡片显示操作栏：时间戳 + 复制 + 重新生成（仅助手消息）+ 删除 + 回退；
- 用户消息操作栏右对齐、助手消息左对齐（role class + flex 方向控制）；
- 点击反馈：蓝色高亮 + 0.35s 缩放动画（1→1.25→1）+ 0.8s 自动恢复；
- store 实现 `deleteMessage` / `regenerate` / `undoLast` 三个 action。

### 3.7 登录与路由守卫（安全修复）

**文件**：`frontend/src/router/index.ts`、`stores/auth.ts`

- **问题**：原守卫只查 localStorage 是否有 token 字符串，复制 URL 即可绕过登录。
- **修复后的三层校验**：
  1. 本地解析 token `userId.exp.签名` 中的 exp 检查过期；
  2. 调用 `/api/auth/me` 服务端验证 HMAC 签名真实性；
  3. 每次页面加载仅校验一次（内存缓存），登录/登出/401 时联动重置缓存与登录态。
- 未登录访问受保护路由 → 跳转 `/login`；已登录访问登录页 → 重定向 `/chat`。

### 3.8 知识库页面（NotesView）

- 本地知识库列表（标题/摘要搜索、Unicode 解码显示）、统计信息、飞书空间选择与手动同步触发。

### 3.9 MCP 管理页面（McpView）

- 三段式卡片编辑器：基本信息 → 传输方式（stdio/http 大按钮切换，选中蓝色高亮）→ 高级配置（参数/环境变量用等宽字体编辑）；
- JSON 格式前端校验、连接测试、预设一键导入、删除确认。

---

## 四、数据库设计

SQLite：`backend/data/notes.db`

| 表 | 用途 |
|---|---|
| `users` | 用户（phone 唯一索引、password_salt、PBKDF2 password_hash、时间戳） |
| `notes` | 知识条目（标题、标签、摘要、来源类型、飞书 revision、embedded 状态） |
| `chunks` / `chunk_fts` | 文本分块 + FTS5 全文索引虚拟表 |
| 会话/消息表 | 聊天会话与消息持久化（服务端为事实源，每次请求重新播种历史） |
| `memory_facts` | 长期记忆事实（每轮后台抽取，支持去重与压缩） |
| `hooks` | 工具钩子脚本配置（PreToolUse / PostToolUse） |
| `permission_rules` | 权限规则（工具名模式 → 是否审批） |

另有两个独立持久化文件：
- `data/checkpoints.sqlite`：LangGraph AsyncSqliteSaver 状态检查点；
- `data/chroma/`：ChromaDB 向量库（cosine + HNSW）；
- `data/auth_secret`：token 签名密钥。

> ⚠️ 注意：`config.py` 的 `data_dir` 默认是相对 CWD 的 `./data`。真实在用的是 `backend/data/`；从仓库根目录启动后端会落到根目录的 `data/`（那里也有一份较小的 notes.db），会造成"看不到已有数据"的困惑。请统一用 `dev.py` / `start-all.ps1` 启动。

---

## 五、技术亮点总结（简历可直接引用）

1. **基于 LangGraph 的 plan-and-execute 编排**：router / planner / execute_plan / replan / retrieve / ingest / report 七节点状态机 + 条件边分发，`execute_plan ⟲ replan` 构成可迭代、可并行的检索循环；AsyncSqliteSaver 持久化 checkpoint 保证重启不丢对话状态；深入理解 reducer 语义（避免 `operator.add` 与持久化 checkpoint 的消息叠加问题）与 **per-turn 字段重置**（隔离跨轮残留）。
2. **混合检索（Hybrid Search）**：0.7×向量（ChromaDB cosine + HNSW）+ 0.3×FTS5 关键词融合排序，解决纯向量检索对中文专名召回差的问题；db/query 双模式 embedding 保证向量空间一致（按 URL 预判 mode，免去每次重试）；父块上下文扩展让命中不止一个切片。
3. **全链路 SSE 流式体验**：14 类事件分级推送（阶段/计划/澄清/引用/正文/工具/权限/子代理…），前端手写 SSE 解析器 + 三缓冲区思考分离算法，实现 reasoning 模型思考过程全程流式可见；plan-and-execute 每步都推 `plan` 事件，聊天区实时渲染计划进度条，研究过程不再黑盒。
4. **模型驱动的歧义澄清**：router 在语义层判定 ambiguous 后主动追问，Chat 端点通过 future 等待 + 300s 超时实现"图暂停—前端作答—图恢复"的人机协同。工具侧另有权限审批 broker（`agent/tools/permissions.py`）实现"调用前弹窗—用户批准—继续执行"。
5. **RAG 工程化细节**：500 字/80 重叠的边界感知分块、3000 token 参考预算 + 低分先截断、12 条滑动窗口 + 历史压缩摘要、行内 [n] 引用标记与来源卡片联动、无引用时剥离引用规则防瞎编。
6. **Agent 治理体系**：子代理（explore/plan/general + 工具白名单）、PreToolUse/PostToolUse 钩子（可返回 `{"block": reason}` 否决调用）、工具级权限规则、AGENTS.md 分层项目规则加载（target-first 6 层）——把"能用"提升到"可控可用"。
7. **长期记忆**：每轮后台抽取事实 + `<<EXISTING>>` 去重 + 相关性召回 + 超量自动压缩；"记住/忘记"走短路指令不进 LLM。
8. **安全实践**：PBKDF2 密码哈希、HMAC 自签 token + 前端三层登录校验（本地过期 + 服务端验签 + 缓存联动）、DOMPurify 防 XSS、zip 路径穿越防护。
9. **生产级容错**：embedding 失败重试（revision 只在成功后推进）、embedding 缺 key 自动降级 FTS5、提示词 mtime 缓存失效（改词不重启）、LLM 输出容错解析（剥 think 再抓 JSON）、**所有增强模块（planner/clarify/hooks/memory）均为可失败的可选项，永不阻断主流程**。
10. **性能优化**：正则快速通道替代 LLM 意图分类（12s → 毫秒级）、embedding 批量请求（32/批）、惰性单例构建异步资源、`ThreadPoolExecutor` 并行执行计划步骤（默认 4 并发，失败隔离）、replan 空转保护。
11. **完整产品闭环**：登录注册 → 多源知识入库（飞书增量同步/网页/文件）→ 计划式检索问答 → 周报自动生成再入库，形成知识管理闭环。

---

## 六、可量化指标（简历数字素材）

- **7 个 Agent 节点**、4 类意图路由、1 条寒暄快速通道、1 条歧义澄清分支
- 计划 1-4 步（`planner_max_steps=4`），replan 最多 3 轮（`research_max_iter=3`），并行执行 4 线程
- 混合检索权重 0.7 / 0.3，双阈值过滤（0.18 / 0.18）
- 分块 500 字符 / 80 重叠；参考预算 3000 tokens；单块上限 800 字符
- 历史窗口 12 条 / 2000 tokens，溢出压缩 ≤150 字
- 长期记忆召回上限 8 条，超 12 条触发压缩
- 项目规则最多加载 8 个来源、单文件 32KB、向上 6 层
- 钩子超时默认 5s；澄清等待超时 300s
- **84 个 REST/SSE 端点**、14 类 SSE 事件
- Embedding 批量 32 条/请求；token TTL 7 天
- 意图分类延迟从 ~12s 优化到毫秒级（正则快速通道）

---

## 七、已知局限与改进方向（面试可谈）

- 引用提取依赖正则，对 LLM 输出格式波动敏感 → 可改用结构化输出或函数调用
- 同步 `hybrid_search` 直接跑在 async 端点上（Chroma + SQLite 都是同步 IO）→ 高并发会阻塞事件循环，应改为 `run_in_executor`
- `main.py` 仍用已弃用的 `@app.on_event("startup")` → 建议迁移到 lifespan 上下文管理器
- `config.py` 的 `data_dir` 是相对 CWD 的 `./data`，从根目录启动会落到错误目录 → 建议改为基于 `__file__` 的绝对路径
- FTS5 unicode61 对中文分词不理想 → 可换 jieba 分词 + 自定义 tokenizer
- localStorage 存 token 有 XSS 窃取风险 → 可升级 httpOnly cookie + CSRF 防护
- 根目录散落 `backend-uvicorn.log` / `logs/` / `__pycache__/` 等运行产物 → 应纳入 `.gitignore` 并统一到 `logs/`
