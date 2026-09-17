import { defineStore } from 'pinia'
import { chatStream, respondClarify, respondPermission, stripThink, type ChatMessage, type ChatRequest, type Citation, type ClarifyEvent, type IngestResult, type ReportResult, type ToolEvent, type PermissionEvent, type PlanStepItem, type PlanEvent, type SourceStatus, type VerifyEvent } from '@/api/chat'
import { useSettingsStore } from './settings'
import { useModelsStore } from './models'
import { useSessionsStore } from './sessions'

export interface ToolCallItem {
  name: string
  status: 'running' | 'ok' | 'failed'
  snippet?: string
}

export interface PendingPermission {
  requestId: string
  tool: string
  args: unknown
}

/** 歧义澄清弹窗数据：router 判定问题有歧义时经 SSE clarify 事件下发 */
export interface PendingClarify {
  requestId: string
  question: string
  options: string[]
}

interface Msg extends ChatMessage {
  id: string
  citations?: Citation[]
  activeCitationIndex?: number | null
  ingest?: IngestResult
  report?: ReportResult
  toolCalls?: ToolCallItem[]
  // 任务规划：计划摘要 + 步骤执行状态（plan-strip 渲染）
  planSummary?: string
  planSteps?: PlanStepItem[]
  // 来源状态：这次回答有没有用上知识库 / 联网兜底（用于提示条）
  sourceStatus?: SourceStatus
  // 联网校验（策略 A）：答案开始流式输出之前就到达，可据此在答案上方渲染
  // 冲突卡片。source_status 里也有同一份数据，但那个是流结束才发的，
  // 想边流边提示就得用这个。
  verify?: VerifyEvent
}
export interface PipelineStage {
  stage: "router" | "rag_search" | "llm_stream" | "agent" | "web_verify"
  status: "started" | "done"
  ms?: number
  hits?: number
  intent?: string
  agent?: string
  iterations?: number
  steps?: number
  plan_summary?: string
  step?: number | null
  total_steps?: number
  query?: string
  at: number
}

interface State {
  sessionId: string | null
  loadToken: number
  messages: Msg[]
  isStreaming: boolean
  error: string | null
  useRag: boolean
  // 任务规划开关：research 意图先分解子查询再检索（localStorage 持久化）
  usePlanner: boolean
  abortCtl: AbortController | null
  streamingSessionId: string | null
  streamingMessageId: string | null
  stage: PipelineStage | null
  // Saved in-flight messages when the user navigates AWAY from a streaming
  // session. Restored when they come back so the partial answer + the
  // thinking row don't disappear the moment they click another row.
  streamingSnapshot: Msg[] | null
  // True while the LLM is inside a <think>...</think> block during streaming.
  // ChatView uses this to swap the placeholder label to "思考中..." so the
  // user gets a hint that the model is reasoning (not stalling) before the
  // first body token arrives.
  thinking: boolean
  // Agent 本地访问权限：default=每次访问前询问；full=完全访问不询问
  agentPermission: 'default' | 'full'
  // 当前待用户批准的权限请求（弹窗数据）
  pendingPermission: PendingPermission | null
  // 当前待用户回答的歧义澄清（弹窗数据；模型驱动，始终开启）
  pendingClarify: PendingClarify | null
}

export const useChatStore = defineStore("chat", {
  state: (): State => ({
    sessionId: null,
    loadToken: 0,
    isStreaming: false,
    messages: [],
    error: null,
    useRag: true,
    usePlanner: ((): boolean => {
      try { return localStorage.getItem('planner-disabled') !== '1' } catch { return true }
    })(),
    abortCtl: null,
    streamingSessionId: null,
    streamingMessageId: null,
    stage: null,
    streamingSnapshot: null,
    thinking: false,
    agentPermission: ((): 'default' | 'full' => {
      try { return localStorage.getItem('agent-permission') === 'full' ? 'full' : 'default' } catch { return 'default' }
    })(),
    pendingPermission: null,
    pendingClarify: null,
  }),
  getters: {
    streamingHere: (s): boolean => s.isStreaming && s.streamingSessionId !== null && s.streamingSessionId === s.sessionId,
  },
  actions: {
    toggleRag() { this.useRag = !this.useRag },
    abortStream() {
      this.abortCtl?.abort()
      this.abortCtl = null
    },
    togglePlanner() {
      this.usePlanner = !this.usePlanner
      try {
        if (this.usePlanner) localStorage.removeItem('planner-disabled')
        else localStorage.setItem('planner-disabled', '1')
      } catch { /* localStorage 不可用时仅内存生效 */ }
    },
    setAgentPermission(mode: 'default' | 'full') {
      this.agentPermission = mode
      try {
        if (mode === 'full') localStorage.setItem('agent-permission', 'full')
        else localStorage.removeItem('agent-permission')
      } catch {}
    },
    // 用户在弹窗中回复权限请求
    async resolvePermission(approve: boolean) {
      const p = this.pendingPermission
      if (!p) return
      this.pendingPermission = null
      try { await respondPermission(p.requestId, approve) } catch {}
    },
    // 用户回答歧义澄清：点选项 / 自由输入；空串=跳过按原问题继续。
    // SSE 流仍保持打开，服务端拿到答案后继续同一条流。
    async answerClarify(answer: string) {
      const p = this.pendingClarify
      if (!p) return
      this.pendingClarify = null
      // 瞬时失败（启动期 401 等）重试一次，否则服务端会白等 300s 超时
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          await respondClarify(p.requestId, answer)
          return
        } catch (e) {
          if (attempt === 1) console.warn('clarify answer failed:', e)
          else await new Promise(r => setTimeout(r, 600))
        }
      }
    },
    /** 构造 /chat 请求 payload */
    _buildPayload(text: string, extra?: Partial<ChatRequest>): ChatRequest {
      const models = useModelsStore()
      const sel = models.selected
      // Phase 3.5: stop shipping the entire conversation history in every
      // request. The backend reads history from the DB so the only payload it
      // needs is the current turn (O(1) regardless of session length).
      return {
        messages: [{ role: 'user', content: text }],
        provider: sel?.provider ?? null,
        model: sel?.modelName ?? null,
        use_rag: this.useRag,
        use_planner: this.usePlanner,
        session_id: this.sessionId,
        base_url: sel?.baseUrl ?? null,
        // 列表接口返回的 apiKey 是脱敏值（sk****xx），回传会覆盖服务端原始
        // key 导致上游 401（MiniMax 1004 login fail）——脱敏值一律不发送
        api_key: sel?.apiKey && !sel.apiKey.includes('*') ? sel.apiKey : null,
        reasoning_level: sel?.reasoning ?? null,
        embedding_model: sel?.embeddingModel ?? null,
        embedding_base_url: sel?.baseUrl ?? null,
        agent_permission: this.agentPermission,
        ...extra,
      }
    },
    /** 消费 /chat SSE 流并更新 asstMsg */
    async _runStream(asstMsg: Msg, payload: ChatRequest) {
      const sessions = useSessionsStore()

      // Stream-time <think> state machine: preserves think content (not just
      // visible text) so MessageBubble can render the collapsible thinking
      // section during streaming — previously the think text was discarded
      // and only appeared after navigating away and back (when the full
      // content was reloaded from the backend).
      let visible = ''
      let thinkBuf = ''
      let inThink = false

      const feed = (raw: string): void => {
        let s = raw
        while (s.length) {
          if (!inThink) {
            const open = s.indexOf('<think>')
            if (open === -1) {
              visible += s
              s = ''
            } else {
              visible += s.slice(0, open)
              s = s.slice(open + 7)
              if (!inThink) this.thinking = true
              inThink = true
            }
          } else {
            const close = s.indexOf('</think>')
            if (close === -1) {
              // Still inside <think>; accumulate and wait for next delta.
              thinkBuf += s
              s = ''
              return
            }
            thinkBuf += s.slice(0, close)
            s = s.slice(close + 8)
            if (inThink) this.thinking = false
            inThink = false
          }
        }
      }

      const flushBubble = (): void => {
        // Prepend the accumulated thinking block so MessageBubble's
        // splitThink() can extract it and render the collapsible section.
        const thinkBlock = thinkBuf ? '<think>\n' + thinkBuf + '\n</think>\n' : ''
        asstMsg.content = (thinkBlock + visible).trimEnd()
        const idx = this.messages.findIndex(m => m.id === asstMsg.id)
        if (idx >= 0) this.messages[idx] = { ...asstMsg }
      }

      try {
        const stream = chatStream(payload, this.abortCtl?.signal)
        for await (const ev of stream) {
          if (ev.type === 'session' && ev.session_id) {
            this.sessionId = ev.session_id
            if (this.streamingSessionId === null) this.streamingSessionId = ev.session_id
            sessions.load()
          } else if (ev.type === 'delta' && typeof ev.data === 'string') {
            feed(ev.data)
            flushBubble()
            // 服务端已开始输出（澄清超时被跳过等场景）：关掉仍挂着的澄清弹窗
            if (this.pendingClarify) this.pendingClarify = null
          } else if (ev.type === 'stage' && ev.data && typeof ev.data === 'object') {
            this.stage = { ...(ev.data as PipelineStage), at: Date.now() }
          } else if (ev.type === 'tool' && ev.data && typeof ev.data === 'object') {
            // 工具调用进度：running -> ok/failed，最终渲染为消息上方的工具条
            const t = ev.data as ToolEvent
            if (!asstMsg.toolCalls) asstMsg.toolCalls = []
            if (t.phase === 'call' && t.name) {
              asstMsg.toolCalls.push({ name: t.name, status: 'running' })
            } else if (t.phase === 'result' && t.name) {
              const pending = [...asstMsg.toolCalls].reverse().find(x => x.name === t.name && x.status === 'running')
              if (pending) {
                pending.status = t.ok ? 'ok' : 'failed'
                pending.snippet = t.snippet
              } else {
                asstMsg.toolCalls.push({ name: t.name, status: t.ok ? 'ok' : 'failed', snippet: t.snippet })
              }
            }
            const idx = this.messages.findIndex(m => m.id === asstMsg.id)
            if (idx >= 0) this.messages[idx] = { ...asstMsg }
          } else if (ev.type === 'plan' && ev.data && typeof ev.data === 'object') {
            // 任务规划进度：created -> step_done* -> replan* -> done
            const p = ev.data as PlanEvent
            if (p.phase === 'created' && p.queries?.length) {
              asstMsg.planSummary = p.summary || ''
              asstMsg.planSteps = p.queries.map(q => ({ query: q, status: 'pending' as const }))
              // 计划生成后第一步立即视为执行中
              if (asstMsg.planSteps.length) asstMsg.planSteps[0].status = 'running'
            } else if (p.phase === 'step_done' && asstMsg.planSteps && typeof p.index === 'number') {
              const st = asstMsg.planSteps[p.index]
              if (st) { st.status = 'done'; st.hits = p.hits ?? 0 }
              const next = asstMsg.planSteps[p.index + 1]
              if (next && next.status === 'pending') next.status = 'running'
            } else if (p.phase === 'replan' && p.query) {
              // 动态补缺：追加一个「补检索」chip
              if (!asstMsg.planSteps) asstMsg.planSteps = []
              asstMsg.planSteps.push({ query: p.query, status: 'done', hits: p.hits ?? 0, replan: true })
            } else if (p.phase === 'done') {
              // 收尾：所有未完成步骤标记完成（计划提前达标终止的情形）
              if (asstMsg.planSteps) {
                for (const st of asstMsg.planSteps) {
                  if (st.status !== 'done') st.status = 'done'
                }
              }
            }
            const idx = this.messages.findIndex(m => m.id === asstMsg.id)
            if (idx >= 0) this.messages[idx] = { ...asstMsg }
          } else if (ev.type === 'permission' && ev.data && typeof ev.data === 'object') {
            // 权限请求（默认模式）：弹窗让用户批准/拒绝本地访问
            const p = ev.data as PermissionEvent
            if (p.phase === 'request') {
              this.pendingPermission = {
                requestId: p.request_id,
                tool: p.tool || 'mcp_invoke',
                args: p.args,
              }
            } else if (p.phase === 'result') {
              this.pendingPermission = null
            }
          } else if (ev.type === 'clarify' && ev.data && typeof ev.data === 'object') {
            // 歧义澄清（模型驱动）：弹窗让用户选含义 / 补充说明 / 跳过
            const c = ev.data as ClarifyEvent
            this.pendingClarify = {
              requestId: c.request_id,
              question: c.question || '',
              options: Array.isArray(c.options) ? c.options : [],
            }
          } else if (ev.type === 'citations' && Array.isArray(ev.data)) {
            asstMsg.citations = ev.data as Citation[]
            const idx = this.messages.findIndex(m => m.id === asstMsg.id)
            if (idx >= 0) this.messages[idx] = { ...asstMsg }
          } else if (ev.type === 'source_status' && ev.data && typeof ev.data === 'object') {
            // 来源状态：知识库有没有命中 / 是否走了联网兜底 / 回答有没有真的引用
            asstMsg.sourceStatus = ev.data as SourceStatus
            const idx = this.messages.findIndex(m => m.id === asstMsg.id)
            if (idx >= 0) this.messages[idx] = { ...asstMsg }
          } else if (ev.type === 'verify' && ev.data && typeof ev.data === 'object') {
            // 联网校验结论（策略 A）。比 source_status 早到 —— 在答案开始流式
            // 输出之前就发出来了，所以能用来在答案上方提前挂冲突卡片。
            // 答案正文里的强制警告横幅由后端拼接，前端不重复渲染。
            asstMsg.verify = ev.data as VerifyEvent
            const idx = this.messages.findIndex(m => m.id === asstMsg.id)
            if (idx >= 0) this.messages[idx] = { ...asstMsg }
          } else if (ev.type === 'ingest' && ev.data) {
            asstMsg.ingest = ev.data as IngestResult
            const idx = this.messages.findIndex(m => m.id === asstMsg.id)
            if (idx >= 0) this.messages[idx] = { ...asstMsg }
          } else if (ev.type === 'report' && ev.data) {
            asstMsg.report = ev.data as ReportResult
            const idx = this.messages.findIndex(m => m.id === asstMsg.id)
            if (idx >= 0) this.messages[idx] = { ...asstMsg }
          } else if (ev.type === 'error') {
            const msg = (ev.data && typeof ev.data === 'string') ? ev.data : String(ev.data ?? 'unknown error')
            visible += '\n\n[Error] ' + msg
            flushBubble()
            this.error = msg
          } else if (ev.type === 'done') {
            if (inThink) {
              // LLM ended without ever emitting a </think>; close it out so
              // the placeholder doesn't keep flashing "思考中..." forever.
              this.thinking = false
            }
            flushBubble()
            break
          }
        }
        sessions.load()
      } catch (e) {
        // Ignore aborts (user switched sessions or hit +new chat).
        if (this.abortCtl?.signal.aborted) return
        const errMsg = (e as Error).message || String(e)
        this.error = errMsg
        visible += '\n\n[Error] ' + errMsg
        flushBubble()
      } finally {
        this.isStreaming = false
        this.streamingSessionId = null
        this.streamingMessageId = null
        this.abortCtl = null
        this.stage = null
        this.thinking = false
        this.streamingSnapshot = null
        this.pendingPermission = null
        this.pendingClarify = null
      }
    },
    async send(text: string) {
      if (!text.trim()) return
      if (this.isStreaming) return
      if (this.streamingSessionId !== null && this.streamingSessionId === this.sessionId) return
      const userMsg: Msg = { id: 'u-' + String(Date.now()), role: 'user', content: text }
      const asstMsg: Msg = { id: 'a-' + String(Date.now() + 1), role: 'assistant', content: '' }
      this.messages.push(userMsg, asstMsg)
      this.isStreaming = true
      this.streamingSessionId = this.sessionId
      this.streamingMessageId = asstMsg.id
      this.error = null
      this.stage = null
      this.abortCtl = new AbortController()

      // 单段式流：router 内联做歧义检测，需要澄清时经 SSE clarify 事件
      // 弹窗追问，用户回答后同一条流继续（无需两段式 preview + override）。
      await this._runStream(asstMsg, this._buildPayload(text))
    },
    async loadFromSession(sessionId: string) {
      const token = ++this.loadToken
      this.error = null
      // Three in-flight scenarios we have to protect so the partial answer and
      // the thinking row don't disappear the moment the user navigates:
      //
      //   (A) Already on this session with in-flight messages -> keep them.
      //   (B) In welcome state (this.sessionId === null) and the freshly-created
      //       streaming session is now being opened from the sidebar -> keep
      //       in-flight, don't overwrite with stale DB rows.
      //   (C) Navigating AWAY from the streaming session -> save snapshot so
      //       we can restore when they come back.
      const inFlight =
        // Decoupled from messages.length on purpose: when the user navigates
        // away mid-stream, ChatView routes through `clear()` which (after
        // the fix below) keeps messages intact but zeros sessionId. Relying
        // on messages.length here would incorrectly classify that as "not
        // in flight" and we'd overwrite the live state with stale DB rows.
        this.isStreaming && this.streamingSessionId !== null

      const sameSessionReopen =
        inFlight &&
        this.streamingSessionId === sessionId &&
        (this.sessionId === sessionId || this.sessionId === null)

      const leavingStream =
        inFlight &&
        this.sessionId === this.streamingSessionId &&
        this.streamingSessionId !== sessionId

      if (sameSessionReopen) {
        // (A)/(B): no DB fetch needed, the live state is already the most
        // up-to-date view (DB only has the user message at this point).
        this.sessionId = sessionId
        this.error = null
        return
      }

      if (leavingStream) {
        // (C): save in-flight state for the round-trip.
        const messages = [...this.messages]
        const assistantId = this.streamingMessageId
        if (assistantId && !messages.some(m => m.id === assistantId)) {
          messages.push({ id: assistantId, role: 'assistant', content: '' })
        }
        this.streamingSnapshot = messages
      }

      const sessions = useSessionsStore()
      await sessions.loadDetail(sessionId)
      if (token !== this.loadToken) return
      const detail = sessions.currentDetail
      if (!detail || detail.id !== sessionId) { this.error = "Session not found"; return }

      // Coming back to a still-streaming session from somewhere else.
      if (this.isStreaming && this.streamingSnapshot && this.streamingSessionId === sessionId) {
        this.sessionId = detail.id
        const messages = [...this.streamingSnapshot]
        const assistantId = this.streamingMessageId
        if (assistantId && !messages.some(m => m.id === assistantId)) {
          messages.push({ id: assistantId, role: 'assistant', content: '' })
        }
        this.messages = messages
        this.streamingSnapshot = null
        this.error = null
        return
      }

      this.sessionId = detail.id
      // Keep the leading <think>...</think> block intact here; MessageBubble
      // renders it as a collapsible details section so the user can still
      // see the model's reasoning when re-opening a past conversation.
      this.messages = detail.messages.map(m => ({
        id: "h-" + String(m.id),
        role: m.role as "user" | "assistant" | "system",
        content: m.content,
        citations: m.citations || undefined,
      }))
      const snapshotBelongsToBackgroundStream =
        this.isStreaming &&
        this.streamingSnapshot !== null &&
        this.streamingSessionId !== null &&
        this.streamingSessionId !== sessionId
      if (!snapshotBelongsToBackgroundStream) {
        this.streamingSnapshot = null
      }
      this.error = null
    },
    clear() {
      // If a background stream is still running, leaving (e.g. clicking
      // "知识库" / Skill / 搜索对话) must NOT wipe messages or snapshot.
      // `loadFromSession` is the only place that knows how to restore the
      // snapshot; if we zero it here, the user's next click back into the
      // session lands on stale DB rows and the in-flight assistant bubble
      // (including the streaming "思考中..." placeholder) disappears.
      if (this.isStreaming && this.streamingSessionId !== null && this.messages.length > 0) {
        const messages = [...this.messages]
        const assistantId = this.streamingMessageId
        if (assistantId && !messages.some(m => m.id === assistantId)) {
          messages.push({ id: assistantId, role: 'assistant', content: '' })
        }
        this.streamingSnapshot = messages
        this.sessionId = null
        this.error = null
        return
      }
      this.sessionId = null
      this.messages = []
      this.streamingSnapshot = null
      this.error = null
    },
    setActiveCitation(msgId: string, idx: number | null) {
      const i = this.messages.findIndex(m => m.id === msgId)
      if (i >= 0) this.messages[i] = { ...this.messages[i], activeCitationIndex: idx }
    },
    /** 删除一条消息（及其配对消息） */
    deleteMessage(msgId: string) {
      const i = this.messages.findIndex(m => m.id === msgId)
      if (i < 0) return
      const msg = this.messages[i]
      // 删用户消息时连带后面的 assistant；删 assistant 时连带前面的 user
      if (msg.role === 'user' && i + 1 < this.messages.length && this.messages[i + 1].role === 'assistant') {
        this.messages.splice(i, 2)
      } else if (msg.role === 'assistant' && i > 0 && this.messages[i - 1].role === 'user') {
        this.messages.splice(i - 1, 2)
      } else {
        this.messages.splice(i, 1)
      }
    },
    /** 重新生成：删除最后一条 assistant，用其前一条 user 重新发送 */
    async regenerate() {
      if (this.isStreaming) return
      const last = this.messages[this.messages.length - 1]
      if (!last || last.role !== 'assistant') return
      const prev = this.messages[this.messages.length - 2]
      if (!prev || prev.role !== 'user') return
      this.messages.pop() // 删掉这条 assistant
      const text = prev.content
      await this.send(text)
    },
    /** 回退：删除最后一条 user + assistant */
    undoLast() {
      if (this.isStreaming) return
      const n = this.messages.length
      if (n >= 2 && this.messages[n - 1].role === 'assistant' && this.messages[n - 2].role === 'user') {
        this.messages.splice(n - 2, 2)
      } else if (n >= 1) {
        this.messages.pop()
      }
    },
  },
})
