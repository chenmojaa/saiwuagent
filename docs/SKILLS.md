# MCP 与技能体系

> Smallhouse 通过 MCP（Model Context Protocol）和 Skill 包两个维度扩展 Agent 能力。
> MCP 让 Agent 调用外部工具（文件系统、GitHub、数据库等），Skill 包提供结构化的提示词指令集。

---

## 1. MCP 工具调用

### 1.1 架构

```
用户提问 → LLM 分析 → 决定调用 MCP 工具
                ↓
    mcp_discover_tools → 获取工具清单
                ↓
    mcp_invoke → stdio 通信 → 执行工具 → 返回结果
                ↓
    结果注入上下文 → LLM 继续回答
```

- **通信方式**：stdio（不支持 http）
- **会话复用**：同一轮对话共享一个 MCP server 进程（`MCPSession` 类），进程死亡自动重启
- **工具发现**：模型先调 `mcp_discover_tools` 获取工具清单，再决定调用哪些
- **流式传输**：工具调用响应 token-by-token 流式转发（`astream`），不阻塞
- **Windows 适配**：`npx` → `npx.CMD`（`shutil.which`），杀进程用 `taskkill /F /T /PID`

### 1.2 预设 MCP Server

| ID | 名称 | 说明 | 命令 |
|---|---|---|---|
| `filesystem` | 本地文件 | 按目录授权读写文件 | `npx -y @modelcontextprotocol/server-filesystem .` |
| `fetch` | 网页抓取 | 抓取网页内容转换为模型可读文本 | `npx -y @modelcontextprotocol/server-fetch` |
| `memory` | 持久记忆 | 保存跨会话的结构化记忆 | `npx -y @modelcontextprotocol/server-memory` |
| `github` | GitHub | 查询仓库、Issue 和代码（需 Token） | `npx -y @modelcontextprotocol/server-github` |
| `postgres` | PostgreSQL | 查询数据库（建议只读账号） | `npx -y @modelcontextprotocol/server-postgres` |
| `sequential-thinking` | 顺序思考 | 为复杂任务增加结构化思考步骤 | `npx -y @modelcontextprotocol/server-sequential-thinking` |

### 1.3 配置与权限

- **配置文件**：`data/mcp/servers.json`（原子写 + 跨进程锁）
- **启用/禁用**：每个 server 有 `enabled` 字段，只有 `enabled: true` 的 server 才会被加载
- **环境变量**：每个 server 可配置独立的 `env`（如 GitHub Token）
- **权限控制**：
  - `default`：每次访问前询问用户
  - `full`：完全访问不询问（localStorage 持久化）
- **最大限制**：最多 50 个 server，每个 server 最多 100 个参数、100 个环境变量

### 1.4 API

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/mcp/servers` | 列出所有 MCP server |
| POST | `/api/mcp/servers` | 添加 MCP server |
| PATCH | `/api/mcp/servers/{id}` | 更新 MCP server |
| DELETE | `/api/mcp/servers/{id}` | 删除 MCP server |
| POST | `/api/mcp/servers/{id}/toggle` | 启用/禁用 |
| POST | `/api/mcp/invoke` | 调用 MCP 工具 |
| POST | `/api/mcp/discover` | 发现工具清单 |

### 1.5 前端组件

- **McpPanel**：MCP 工具调用面板，显示可用 server 和工具
- **McpCallHistory**：MCP 调用历史记录，显示每次调用的工具名、参数、结果
- **SkillsMcpView**：技能与 MCP 管理页面

---

## 2. Skill 技能包

### 2.1 什么是 Skill

Skill 是一个包含 `SKILL.md` 的文件夹，定义了 Agent 在特定领域的行为指令。安装后，Agent 在相关场景下会自动加载对应的提示词。

```
my-skill/
├── SKILL.md          # 必需：元数据 + 指令
├── examples/         # 可选：示例文件
└── scripts/          # 可选：辅助脚本
```

`SKILL.md` 格式：
```markdown
---
name: 技能名称
description: 技能描述
---

# 技能名称

具体指令内容...
```

### 2.2 推荐技能

| ID | 名称 | 说明 | 来源 |
|---|---|---|---|
| `docx` | Word 文档专家 | 创建、编辑和批处理 .docx 文件 | anthropics/skills |
| `xlsx` | 表格数据助手 | 读取、清洗和分析 Excel 工作簿 | anthropics/skills |
| `pdf` | PDF 处理工具箱 | 提取文本与表格、合并拆分页面 | anthropics/skills |
| `webapp-testing` | Web 应用测试 | 浏览器自动化检查页面流程 | anthropics/skills |
| `frontend-design` | 前端设计审查 | 优化界面层级和设计系统一致性 | anthropics/skills |
| `skill-creator` | 技能创建器 | 按规范编写新技能包 | anthropics/skills |

### 2.3 安装方式

1. **从推荐列表安装**：点击推荐技能的"导入"按钮，自动从 GitHub 下载
2. **上传文件夹**：拖拽包含 `SKILL.md` 的文件夹到上传区域
3. **上传压缩包**：支持 ZIP / TAR / TAR.GZ 格式，自动解压并识别 `SKILL.md`

### 2.4 存储

- **安装目录**：`data/skills/installed/{skill_id}/`
- **注册表**：`data/skills/installed.json`（原子写）
- **安全限制**：
  - 压缩包最大 100MB
  - 单文件最大 80MB
  - 最多 8000 个文件
  - 路径遍历防护（拒绝 `..`、绝对路径、特殊字符）

### 2.5 API

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/skills/recommended` | 获取推荐技能列表 |
| GET | `/api/skills` | 列出已安装技能 |
| GET | `/api/skills/{id}/detail` | 技能详情（含 SKILL.md 内容 + 文件树） |
| POST | `/api/skills/upload` | 上传技能（文件夹或压缩包） |
| POST | `/api/skills/install/{id}` | 从推荐列表安装 |
| GET | `/api/skills/{id}/download` | 下载技能为 ZIP |
| DELETE | `/api/skills/{id}` | 卸载技能 |

---

## 3. MCP vs Skill 对比

| 维度 | MCP | Skill |
|---|---|---|
| **本质** | 外部工具调用协议 | 提示词指令包 |
| **能力** | 读写文件、查数据库、调 API | 指导 LLM 在特定领域的行为 |
| **运行时** | 独立进程（stdio 通信） | 注入到 system prompt |
| **配置** | `data/mcp/servers.json` | `data/skills/installed.json` |
| **安装** | 配置 server 参数（命令/环境变量） | 上传文件夹或从推荐列表安装 |
| **权限** | 每次调用需用户批准（default 模式） | 无运行时权限控制 |
| **适用场景** | 需要执行外部操作（读写/查询/调用） | 需要特定领域的专业知识或格式 |

---

## 4. 相关文件

- `backend/app/api/mcp.py` — MCP server 管理与工具调用 API
- `backend/app/api/skills.py` — Skill 包注册与安装 API
- `backend/app/agent/nodes/answer.py` — 提示词组装（含 Skill 注入）
- `frontend/src/components/McpPanel.vue` — MCP 工具面板
- `frontend/src/components/McpCallHistory.vue` — MCP 调用历史
- `frontend/src/components/SkillsPanel.vue` — 技能面板
- `frontend/src/views/SkillsMcpView.vue` — 技能与 MCP 管理页面
