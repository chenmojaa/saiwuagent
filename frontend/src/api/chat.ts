import { postJson, streamSse } from './client'

export interface ChatMessage {
  role: "user" | "assistant" | "system"
  content: string
}

export interface Citation {
  note_id: string
  title?: string
  chunk_index: number
  snippet: string
  score?: number
  source_type?: string
  source_url?: string
}

export interface ChatRequest {
  messages: ChatMessage[]
  provider?: string | null
  model?: string | null
  use_rag?: boolean
  use_planner?: boolean | null
  session_id?: string | null
  base_url?: string | null
  api_key?: string | null
  reasoning_level?: string | null
  embedding_model?: string | null
  embedding_base_url?: string | null
  agent_permission?: 'default' | 'full'
}

/** 歧义澄清请求（router 判定问题有歧义时经 SSE 下发） */
export interface ClarifyEvent {
  request_id: string
  question: string
  options: string[]
}

export interface StageEvent {
  stage: "router" | "rag_search" | "llm_stream" | "agent" | "web_verify"
  status: "started" | "done"
  ms?: number
  hits?: number
  intent?: string
  rewritten_query?: string
  agent?: string
  iterations?: number
  steps?: number
  plan_summary?: string
  step?: number | null
  total_steps?: number
  query?: string
}

export interface PlanStepItem {
  query: string
  status: "pending" | "running" | "done"
  hits?: number
  replan?: boolean
}

export interface PlanEvent {
  phase: "created" | "step_done" | "replan" | "done"
  summary?: string
  queries?: string[]
  index?: number
  query?: string
  hits?: number
  iterations?: number
}

export interface IngestResult {
  ok: boolean
  note_id?: string
  title?: string
  tags?: string[]
  summary?: string
  source_type?: string
  embedded?: boolean
  chunk_count?: number
  duplicate_of?: number | null
  error?: string
}

export interface ReportResult {
  ok: boolean
  empty?: boolean
  period_days?: number
  note_id?: string
  counts?: { notes: number; tags: number }
  summary?: string
  message?: string
}

export interface ToolEvent {
  phase: "start" | "call" | "result"
  name?: string
  calls?: string[]
  args?: unknown
  ok?: boolean
  snippet?: string
}

export interface PermissionEvent {
  phase: "request" | "result"
  request_id: string
  tool?: string
  args?: unknown
  approved?: boolean
}

/** 来源状态：这次回答有没有用上知识库 / 联网兜底 */
export interface SourceStatus {
  /** 本轮来自知识库的切片数 */
  kb_hits: number
  /** 联网兜底的结果数（本地完全没有内容时才会 > 0） */
  web_hits?: number
  /** 回答是否真的引用了材料（模型判定检索结果不相关时会为 false） */
  grounded: boolean
  /** 联网校验结论（策略 A）：consistent | conflict | kb_stale | web_only | unverified | skipped | disabled */
  verify_status?: string
  /** 校验的一句话说明 */
  verify_note?: string
  /** 冲突条目（verify_status === 'conflict' 时才有） */
  conflicts?: VerifyConflict[]
  /** 被判定过期、已从参考资料中剔除的知识库 note_id */
  stale_note_ids?: string[]
}

/** 联网校验发现的单条事实冲突 */
export interface VerifyConflict {
  claim: string
  kb_says?: string
  web_says?: string
  kb_sources?: { note_id?: string; title?: string }[]
  web_sources?: { title?: string; url?: string }[]
}

/** 联网校验事件（在答案开始流式输出之前发出） */
export interface VerifyEvent {
  status: string
  note?: string
  conflicts?: VerifyConflict[]
  stale_note_ids?: string[]
}

export interface ChatStreamEvent {
  type: "session" | "delta" | "citations" | "source_status" | "verify" | "done" | "error" | "stage" | "ingest" | "report" | "tool" | "permission" | "plan" | "clarify"
  session_id?: string
  data?: string | Citation[] | SourceStatus | VerifyEvent | StageEvent | IngestResult | ReportResult | ToolEvent | PermissionEvent | PlanEvent | ClarifyEvent
}

/** 回复 Agent 的本地访问权限请求（允许 / 拒绝） */
export async function respondPermission(requestId: string, approve: boolean): Promise<{ ok: boolean }> {
  return postJson<{ ok: boolean }>('/chat/permission', { request_id: requestId, approve })
}

/** 回复歧义澄清（点选项 / 自由输入；空串=跳过，按原问题继续） */
export async function respondClarify(requestId: string, answer: string): Promise<{ ok: boolean }> {
  return postJson<{ ok: boolean }>('/chat/clarify', { request_id: requestId, answer })
}

// 移除 <think>...</think> 思考段落（配对的 + 未闭合的尾部）
export function stripThink(s: string): string {
  if (!s) return s
  return s
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<think>[\s\S]*$/gi, '')
    .replace(/<\/think>/gi, '')
    .trim()
}

export async function* chatStream(req: ChatRequest, signal?: AbortSignal): AsyncGenerator<ChatStreamEvent> {
  const stream = streamSse("/chat", {
    messages: req.messages,
    provider: req.provider || undefined,
    model: req.model || undefined,
    use_rag: req.use_rag !== false,
    use_planner: req.use_planner === undefined ? undefined : req.use_planner,
    session_id: req.session_id || undefined,
    base_url: req.base_url || undefined,
    api_key: req.api_key || undefined,
    reasoning_level: req.reasoning_level || undefined,
    embedding_model: req.embedding_model || undefined,
    embedding_base_url: req.embedding_base_url || undefined,
    agent_permission: req.agent_permission || "default",
  }, signal)
  for await (const ev of stream) {
    if (ev.data === "[DONE]") { yield { type: "done" }; return }
    if (ev.event === "session") {
      try {
        const obj = JSON.parse(ev.data)
        yield { type: "session", session_id: obj.session_id }
      } catch {}
    } else if (ev.event === "citations") {
      try { yield { type: "citations", data: JSON.parse(ev.data) } }
      catch {}
    } else if (ev.event === "source_status") {
      try { yield { type: "source_status", data: JSON.parse(ev.data) } }
      catch {}
    } else if (ev.event === "verify") {
      try { yield { type: "verify", data: JSON.parse(ev.data) } }
      catch {}
    } else if (ev.event === "stage") {
      try { yield { type: "stage", data: JSON.parse(ev.data) } }
      catch {}
    } else if (ev.event === "tool") {
      try { yield { type: "tool", data: JSON.parse(ev.data) } } catch {}
    } else if (ev.event === "permission") {
      try { yield { type: "permission", data: JSON.parse(ev.data) } } catch {}
    } else if (ev.event === "plan") {
      try { yield { type: "plan", data: JSON.parse(ev.data) } } catch {}
    } else if (ev.event === "clarify") {
      try { yield { type: "clarify", data: JSON.parse(ev.data) } } catch {}
    } else if (ev.event === "error") {
      yield { type: "error", data: ev.data }
    } else if (ev.event === "ingest") {
      try { yield { type: "ingest", data: JSON.parse(ev.data) } } catch {}
    } else if (ev.event === "report") {
      try { yield { type: "report", data: JSON.parse(ev.data) } } catch {}
    } else if (ev.event === "message") {
      // 未具名事件 = 答案正文分片。SSE 规范里这是默认事件类型，
      // streamSse 也会把「没有 event: 行」的帧标成 "message"。
      yield { type: "delta", data: ev.data }
    } else {
      // 未知的**具名**事件：必须忽略，不能当正文。
      //
      // 这里原来是个无条件的 else 兜底，把任何未识别事件都当 delta ——
      // 于是新加的 `verify` 事件（载荷是 JSON）被原样拼进了答案，用户看到的是
      //   {"status":"conflict",...}这是正常的答案文本
      // 后端加一个事件类型就会污染所有回答，而且只在触发该事件时出现，
      // 很容易漏测。按 SSE 语义，具名事件就该由认识它的消费方处理，
      // 不认识就跳过。
      console.warn("[sse] 忽略未知事件类型:", ev.event)
    }
  }
}
