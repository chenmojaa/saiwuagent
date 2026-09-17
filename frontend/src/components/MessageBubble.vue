<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { t } from '@/i18n'
import { Marked, type Tokens } from 'marked'
import DOMPurify from 'dompurify'
import mermaid from 'mermaid'
import type { Citation } from '@/api/chat'

const props = defineProps<{
  role: 'user' | 'assistant' | 'system'
  content: string
  citations?: Citation[]
  activeIndex?: number | null
}>()

const emit = defineEmits<{
  'update:activeIndex': [number]
}>()

const bubbleEl = ref<HTMLElement | null>(null)
const detailsRef = ref<HTMLDetailsElement | null>(null)

// Per-bubble thinking-process open/close state is stored in localStorage
// keyed by a hash of the think content. This survives route navigation
// (which unmounts <ChatView> and remounts every <MessageBubble>) so the
// user's manual expand/collapse choice persists across pages.
const THINK_KEY_PREFIX = 'hd:thinkOpen:'
function hashContent(s: string): string {
  let h = 5381 >>> 0
  for (let i = 0; i < s.length; i++) {
    h = (((h << 5) + h) ^ s.charCodeAt(i)) >>> 0
  }
  return h.toString(36)
}

let thinkToggleHandler: ((e: Event) => void) | null = null
async function syncThinkOpenState(): Promise<void> {
  const el = detailsRef.value
  const t = think.value
  // Detach any handler attached to a previous <details> element so we
  // never leak listeners when Vue swaps the node on content update.
  if (el && thinkToggleHandler) {
    el.removeEventListener('toggle', thinkToggleHandler)
    thinkToggleHandler = null
  }
  if (!el || !t) return
  const key = THINK_KEY_PREFIX + hashContent(t)
  // Only override the element's open state when we have an explicit saved
  // choice. A missing key means "user hasn't decided yet", so we leave the
  // element's default in place (the <details> ships with the `open` attr so
  // the thinking content is visible on first render).
  const saved = localStorage.getItem(key)
  if (saved !== null && el.open !== (saved === '1')) {
    el.open = saved === '1'
  }
  thinkToggleHandler = () => {
    localStorage.setItem(key, el.open ? '1' : '0')
  }
  el.addEventListener('toggle', thinkToggleHandler)
}


// 1) Pull the LEADING <think>...</think> block off and keep it as a separate
//    piece, so the template can render it as a collapsible "thinking
//    process" section above the actual answer. Inline (non-leading)
//    <think> blocks are still stripped from the main body to keep the rest
//    of the pipeline unchanged.
function splitThink(s: string): { think: string; rest: string } {
  // Match leading <think>... block. The closing </think> is optional because
  // during streaming the tag may not be closed yet — we still want to show
  // whatever thinking content has arrived so far.
  const m = s.match(/^\s*<think>([\s\S]*?)(?:<\/think>([\s\S]*))?\s*$/i)
  if (m) {
    return { think: m[1].trim(), rest: (m[2] || '').trim() }
  }
  const cleaned = s.replace(/<think>[\s\S]*?(<\/think>|$)/gi, '').trim()
  return { think: '', rest: cleaned }
}

// 2) Pull a trailing "来源：[n][m]..." line off the end. Keep the indices as
//    clickable buttons; the body itself never carries [n] markers.
const SOURCE_LINE_RE = /\n?\s*\u6765\u6e90\s*[:\uff1a]\s*((?:\[\s*\d+\s*\])+)\s*$/
function splitSourceLine(s: string): { body: string; tokens: number[] } {
  const m = s.match(SOURCE_LINE_RE)
  if (!m) return { body: s, tokens: [] }
  const tokens = Array.from(m[1].matchAll(/\[(\d+)\]/g)).map(x => parseInt(x[1], 10))
  return { body: s.slice(0, m.index).trimEnd(), tokens }
}

const thinkRest = computed(() => splitThink(props.content || ''))
const think = computed(() => thinkRest.value.think)
const cleaned = computed(() => thinkRest.value.rest)
const sourceLine = computed(() => splitSourceLine(cleaned.value))
const body = computed(() => sourceLine.value.body)
const sourceTokens = computed(() => sourceLine.value.tokens)

// Only show source tokens that have a matching citation in props.citations.
// Guards against two failure modes: (1) LLM hallucinating [N] with no
// retrieved chunks (citations = []) and (2) LLM citing an index out of
// range. Either way, dangling [N] buttons render as broken UI.
//
// When the LLM did NOT write an explicit "来源：[n]" line but citations
// exist (backend extracted them from the inline [n] markers the model did
// write), fall back to showing ALL citation indices so the source footer is
// still visible. 注意：后端只会在模型确实写了 [n] 时才返回 citations，
// 不会再"把检索到的切片当成引用"，所以这里列出的都是真引用。
const validSourceTokens = computed<number[]>(() => {
  const cites = props.citations
  if (!cites || cites.length === 0) return []
  const max = cites.length
  if (sourceTokens.value.length > 0) {
    return sourceTokens.value.filter(n => n >= 1 && n <= max)
  }
  return Array.from({ length: max }, (_, i) => i + 1)
})

// 3) Replace inline [n] markers with clickable citation buttons so the
//    user can tap them directly in the answer text (instead of having
//    them silently stripped). The buttons share the same toggle() handler
//    as the footer source-line buttons.
function replaceCiteMarkers(s: string): string {
  return s.replace(/\[\s*(\d+)\s*\]/g, (_m, nStr: string) => {
    const n = parseInt(nStr, 10)
    const cites = props.citations
    if (!cites || cites.length === 0) return ''
    if (n < 1 || n > cites.length) return ''
    return `<span class="cite-inline" data-cite="${n}" title="查看来源 [${n}]">[${n}]</span>`
  })
}
const bodyNoCite = computed(() => replaceCiteMarkers(body.value))

// 4) Markdown -> HTML via marked, with a custom renderer for ```mermaid blocks.
function escapeHtml(s: string): string {
  return s.replace(/[&<>"'\u00b7]/g, (c) => {
    switch (c) {
      case '&': return '&amp;'
      case '<': return '&lt;'
      case '>': return '&gt;'
      case '"': return '&quot;'
      case "'": return '&#39;'
      default: return c
    }
  })
}

const md = new Marked({
  gfm: true,
  breaks: true,
})

// 提取 mermaid 类型标题，用于卡片 header
const MMD_TITLE_RE = /^flowchart\s+(?:TD|TB|LR|RL|BT|TD)\s+(?:([\u4e00-\u9fff\w]{2,20})\s+\n|.*?\n)/i
function extractMermaidTitle(src: string): string {
  // 优先从首行注释或子图标题取
  const firstLine = src.split('\n')[0].trim()
  if (firstLine.startsWith('title ')) return firstLine.slice(6).trim()
  if (firstLine.startsWith('---') && firstLine.endsWith('---')) return firstLine.slice(3, -3).trim()
  // flowchart / sequenceDiagram 等类型名
  if (/^(flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|journey|mindmap|pie)/i.test(firstLine)) {
    // 取第二个 token 作为标题
    const parts = firstLine.split(/\s+/)
    if (parts.length >= 2) {
      const t = parts[1]
      // 如果是方向词则跳过
      if (!['TD','TB','LR','RL','BT','TD'].includes(t.toUpperCase())) return t
    }
  }
  return ''
}

// 已渲染好的 mermaid SVG，按「源码哈希」缓存。
//
// 为什么必须放进响应式状态（而不是直接 innerHTML 塞进 DOM）：
// 模板里 <div class="md-body" v-html="renderedHtml"> 由 v-html 拥有该子树。
// 原来渲染成功后用 `block.innerHTML = svg` 命令式写入，Vue 完全不知情；
// 一旦 renderedHtml 重新求值（切页面重进会话后 citations / 流式状态落定等），
// Vue 就用 innerHTML 覆盖回去 —— SVG 被抹掉、data-rendered 也没了，
// 于是出现「刚返回时正常，切页/重进变回源码」。
//
// 现在把 SVG 纳入 renderedHtml 的求值结果，DOM 与 v-html 始终一致，
// 任何重渲染都不会再把图冲掉。
// 值为空串表示「渲染过但失败了」，用于避免反复重试。
const mermaidSvgs = ref<Record<string, string>>({})

// 把已渲染的 SVG 填回对应块。在 DOMPurify 之后注入，
// 保证 DOMPurify 不会改动 mermaid 生成的 SVG（其 id/style 等属性较多）。
// 块的 innerHTML 只有一段转义文本、没有嵌套 div，所以第一个 </div> 就是它的闭合。
function injectRenderedSvg(html: string, key: string, svg: string): string {
  const marker = 'data-mmd-key="' + key + '"'
  const at = html.indexOf(marker)
  if (at < 0) return html
  const openEnd = html.indexOf('>', at)
  if (openEnd < 0) return html
  const closeAt = html.indexOf('</div>', openEnd)
  if (closeAt < 0) return html
  // 只清理**本块**开标签里的 data-rendered。
  //
  // 曾经的写法是 html.slice(0, openEnd).replace(...)，作用域覆盖了本块之前的
  // 全部内容 —— 于是一轮里注入第二个块时，会把第一个块刚打上的
  // data-rendered="1" 一并抹掉。那个块下一轮又被
  // `.mermaid-block:not([data-rendered])` 选中重新渲染，写回 mermaidSvgs
  // 触发 renderedHtml 重算，然后再次被抹掉……**无限循环 = 页面持续闪烁抖动**。
  // 只有一条 mermaid 时不暴露（head 里没有别的块），所以这个 bug 要同一气泡内
  // 出现 ≥2 个 mermaid 代码块才会显形。
  const tagStart = html.lastIndexOf('<', at)
  const tag = html.slice(tagStart, openEnd).replace(/\s*data-rendered="1"/g, '')
  const inner = svg || html.slice(openEnd + 1, closeAt)  // 空串=失败，保留源码
  return html.slice(0, tagStart) + tag + ' data-rendered="1">' + inner + html.slice(closeAt)
}

md.use({
  renderer: {
    code(token: Tokens.Code): string {
      const lang = (token.lang || '').trim()
      const text = token.text
      if (lang === 'mermaid') {
        const title = extractMermaidTitle(text) || '流程图'
        const source = encodeURIComponent(text)
        const key = hashContent(text)
        return `<div class="mermaid-card" data-source="${source}">` +
          `<div class="mermaid-card-header">` +
            `<span class="mermaid-card-title">${title}</span>` +
            `<div class="mermaid-card-toolbar">` +
              `<button class="mermaid-action" data-action="toggle" title="折叠/展开"><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="2 5 7 10 12 5"/></svg></button>` +
              `<button class="mermaid-action" data-action="copy" title="复制源码"><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="4" width="8" height="8" rx="1"/><path d="M10 4V2.5a1 1 0 0 0-1-1H4.5a1 1 0 0 0-1 1V10h1.5"/></svg></button>` +
              `<button class="mermaid-action" data-action="download" title="下载 PNG"><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M7 2v8M3 7l4 4 4-4M2 12h10"/></svg></button>` +
            `</div>` +
          `</div>` +
          `<div class="mermaid-card-body">` +
            `<div class="mermaid-block" data-source="${source}" data-mmd-key="${key}">${escapeHtml(text)}</div>` +
          `</div>` +
        `</div>`
      }
      const langClass = lang ? ' class="language-' + escapeHtml(lang) + '"' : ''
      return '<pre><code' + langClass + '>' + escapeHtml(text) + '</code></pre>'
    },
  },
})

const renderedHtml = computed(() => {
  if (props.role !== 'assistant') return ''
  const raw = md.parse(bodyNoCite.value, { async: false }) as string
  let html = DOMPurify.sanitize(raw, {
    ADD_ATTR: ['data-source', 'data-rendered', 'data-action', 'data-mmd-key', 'class', 'data-cite', 'title'],
    ADD_TAGS: ['span'],
  })
  // 读 mermaidSvgs 既是为了注入，也让本 computed 依赖它：
  // 渲染完成写入 SVG 后，这里会自动重算，v-html 与 DOM 保持一致。
  for (const [key, svg] of Object.entries(mermaidSvgs.value)) {
    html = injectRenderedSvg(html, key, svg)
  }
  return html
})

// 事件委托：点击内联 [n] 引用按钮时高亮对应来源
let mdBodyClickHandler: ((e: MouseEvent) => void) | null = null
function bindMdBodyClick(): void {
  const el = bubbleEl.value
  if (!el) return
  if (mdBodyClickHandler) el.removeEventListener('click', mdBodyClickHandler)
  mdBodyClickHandler = (e: MouseEvent) => {
    // 1) 内联 [n] 引用
    const citeTarget = (e.target as HTMLElement).closest('.cite-inline') as HTMLElement | null
    if (citeTarget) {
      const n = parseInt(citeTarget.getAttribute('data-cite') || '', 10)
      if (!isNaN(n)) toggle(n)
      return
    }
    // 2) Mermaid 卡片操作按钮（右上角 toolbar / 折叠）
    const cardAction = (e.target as HTMLElement).closest('.mermaid-action') as HTMLElement | null
    if (cardAction) {
      const card = cardAction.closest('.mermaid-card') as HTMLElement | null
      if (!card) return
      const action = cardAction.getAttribute('data-action')
      const body = card.querySelector('.mermaid-card-body') as HTMLElement | null
      if (action === 'toggle') {
        card.classList.toggle('collapsed')
      } else if (action === 'copy' && body) {
        const src = decodeURIComponent(body.querySelector('.mermaid-block')?.getAttribute('data-source') || '')
        navigator.clipboard.writeText(src).catch(() => {})
      } else if (action === 'download' && body) {
        const svgEl = body.querySelector('svg')
        if (svgEl) downloadMermaidSvg(svgEl, card.querySelector('.mermaid-card-title')?.textContent || '流程图')
      }
      return
    }
    // 3) Mermaid 卡片标题栏（整条 header 点击折叠）
    const cardHeader = (e.target as HTMLElement).closest('.mermaid-card-header') as HTMLElement | null
    if (cardHeader) {
      const card = cardHeader.closest('.mermaid-card') as HTMLElement | null
      if (card) card.classList.toggle('collapsed')
    }
  }
  el.addEventListener('click', mdBodyClickHandler)
}

function toggle(n: number): void {
  if (props.activeIndex === n) emit('update:activeIndex', -1)
  else emit('update:activeIndex', n)
}

// 把 mermaid SVG 导出为 PNG 并触发下载
function downloadMermaidSvg(svgEl: SVGElement, title: string): void {
  const clone = svgEl.cloneNode(true) as SVGElement
  // 补上 viewBox / 宽高，避免导出空白
  const bbox = svgEl.getBoundingClientRect()
  if (!clone.getAttribute('width')) clone.setAttribute('width', String(bbox.width || 600))
  if (!clone.getAttribute('height')) clone.setAttribute('height', String(bbox.height || 400))
  const svgData = new XMLSerializer().serializeToString(clone)
  const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const img = new Image()
  img.onload = () => {
    const canvas = document.createElement('canvas')
    const scale = 2 // 2x 高清
    canvas.width = (bbox.width || 600) * scale
    canvas.height = (bbox.height || 400) * scale
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    URL.revokeObjectURL(url)
    canvas.toBlob((pngBlob) => {
      if (!pngBlob) return
      const a = document.createElement('a')
      a.href = URL.createObjectURL(pngBlob)
      a.download = `${title}.png`
      a.click()
      setTimeout(() => URL.revokeObjectURL(a.href), 1000)
    }, 'image/png')
  }
  img.onerror = () => URL.revokeObjectURL(url)
  img.src = url
}

// 5) Mermaid rendering on mount + on every content change.
let mermaidReady = false
async function ensureMermaid(): Promise<void> {
  if (mermaidReady) return
  mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose', fontFamily: 'inherit' })
  mermaidReady = true
}

// LLM 生成 mermaid 时常见的两类语法小错会导致整张图回退成文字：
//  1) 一行写多条连线（"H --> K     K --> S"）——mermaid 要求一行一条语句
//  2) 源码里混入 ``` 围栏残留（如末尾 "T --> U```"）
// 这里在渲染前做保守修复：去反引号、把同行的后续语句前插入 ';'（官方分隔符）。
function repairMermaidSource(src: string): string {
  const s = src.replace(/```+/g, '').replace(/\r\n/g, '\n')
  const ARROW = '(?:-->|==>|-\\.->|--o|--x|->>|-->>|->)'
  const stmtStart = new RegExp(
    '[A-Za-z0-9_\\u4e00-\\u9fff][\\w\\u4e00-\\u9fff"-]*' + // 节点 ID
    '\\s*(?:\\[[^\\]]*\\]|\\([^)]*\\)|\\{[^}]*\\})?' +     // 可选形状 [..]/(..)/{..}
    '\\s*' + ARROW
  )
  const lines = s.split('\n').map((line) => {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('%%')) return line
    const arrows = line.match(/--?>|==>|-\.->|--o|--x|->>|-->>/g)
    if (!arrows || arrows.length < 2) return line
    // 语句边界：非行首的空白段，其后紧跟「节点ID + 可选形状 + 箭头」→ 替换为 '; '
    return line.replace(
      new RegExp('(?<=\\S)\\s+(?=' + stmtStart.source + ')', 'g'),
      '; ',
    )
  })
  return lines.join('\n').trim()
}

// 渲染结果写入 mermaidSvgs（响应式），由 renderedHtml 统一注入 DOM。
// 这里**不再**用 block.innerHTML 直接改 DOM —— 那正是「切页面回来变源码」的根因。
// 串行化仅作防御性处理（实测 mermaid 11 并发本身没问题，但保证顺序更稳）。
let mermaidRenderChain: Promise<unknown> = Promise.resolve()
function serializeMermaidRender<T>(fn: () => Promise<T>): Promise<T> {
  const run = mermaidRenderChain.then(fn, fn)
  mermaidRenderChain = run.catch(() => { /* 失败不阻断后续 */ })
  return run
}

async function renderMermaidIn(root: HTMLElement): Promise<void> {
  // 流式时序根修：marked 会把未闭合的 ```mermaid 围栏当作完整代码块交给
  // 自定义 renderer，半截源码进 mermaid.render 必然解析失败。围栏数量为
  // 奇数说明最后一个代码块还没闭合（正在流式输出），这一拍跳过渲染，
  // 等闭合后再画——既省掉大量无效渲染，也不再产生错误占位 SVG。
  const fenceCount = (bodyNoCite.value.match(/```/g) || []).length
  if (fenceCount % 2 === 1) return
  const blocks = Array.from(root.querySelectorAll<HTMLElement>('.mermaid-block:not([data-rendered])'))
  const updates: Record<string, string> = {}
  for (const block of blocks) {
    const key = block.getAttribute('data-mmd-key') || ''
    if (!key) continue
    const source = decodeURIComponent(block.getAttribute('data-source') || block.textContent || '')
    const renderOnce = async (code: string) => {
      const id = 'mmd-' + Math.random().toString(36).slice(2, 10)
      const r = await serializeMermaidRender(() => mermaid.render(id, code))
      return r.svg
    }
    // 强制 SVG 背景透明，适配暗色主题
    const cleanSvg = (svg: string) => svg
      .replace(/<rect[^>]*fill="[^"]*"[^>]*\/>/g, '')
      .replace(/fill="#fff[^"]*"/g, 'fill="transparent"')
      .replace(/fill="#ffffff[^"]*"/g, 'fill="transparent"')
    try {
      updates[key] = cleanSvg(await renderOnce(source))
    } catch (err) {
      // 原文解析失败：尝试修复语法后重试一次
      // 注意：这里一定要打日志。原实现把错误完全吞掉，导致「图变成源码」
      // 这种问题在浏览器里没有任何线索，只能靠猜。
      console.warn('[mermaid] 首次渲染失败，尝试修复后重试', err)
      try {
        const repaired = repairMermaidSource(source)
        if (repaired && repaired !== source.trim()) {
          updates[key] = cleanSvg(await renderOnce(repaired))
          continue
        }
        throw new Error('repair no-op')
      } catch (err2) {
        // 仍然失败：保留源码显示。写空串标记「已尝试」，避免每轮反复重试。
        console.warn('[mermaid] 修复后仍失败，回退为源码显示', err2)
        console.warn('[mermaid] 原始源码：\n' + source)
        updates[key] = ''
      }
    }
  }
  // 只在内容真的变了时才写回。
  // mermaidSvgs 是 renderedHtml 的依赖，而 renderedHtml 变化又会触发本函数 ——
  // 无条件赋值（{...spread} 每次都是新引用）会让这条回路永远停不下来。
  // 这里加一道闸：值全部相同就不写，从结构上杜绝「渲染 -> 重算 -> 再渲染」死循环。
  const changed = Object.entries(updates).some(([k, v]) => mermaidSvgs.value[k] !== v)
  if (changed) {
    // 一次性写回：触发 renderedHtml 重算 -> v-html 更新 -> 图出现
    mermaidSvgs.value = { ...mermaidSvgs.value, ...updates }
  }
}

// mermaid.render 解析失败时会把错误占位 SVG（炸弹）留在 document.body 末尾
// 的临时容器（id 形如 dmmd-xxxx）里，调用方 catch 之后它们也不会被清理——
// 这就是「滚动到页面底部才成片看到炸弹」的来源。主动扫掉这些孤儿容器。
function sweepMermaidErrorContainers(): void {
  document.body.querySelectorAll(':scope > [id^="dmmd-"]').forEach((el) => el.remove())
}

async function refreshMermaid(): Promise<void> {
  await ensureMermaid()
  await nextTick()
  if (bubbleEl.value) await renderMermaidIn(bubbleEl.value)
  sweepMermaidErrorContainers()
}

onMounted(() => {
  refreshMermaid()
  bindMdBodyClick()
})
watch(renderedHtml, () => {
  refreshMermaid()
  bindMdBodyClick()
}, { flush: 'post' })
// Restore + persist thinking open/close. flush:'post' ensures detailsRef is
// populated (the v-if element exists) by the time we read it.
watch([think, detailsRef], () => {
  syncThinkOpenState().catch(() => { /* localStorage may be blocked */ })
}, { flush: 'post', immediate: true })
onBeforeUnmount(() => {
  if (detailsRef.value && thinkToggleHandler) {
    detailsRef.value.removeEventListener('toggle', thinkToggleHandler)
    thinkToggleHandler = null
  }
})
</script>

<template>
  <div :class="['bubble-row', role]">
    <div :class="['bubble', role]">
      <details v-if="role === 'assistant' && think" ref="detailsRef" class="think-section">
        <summary>{{ t('ui.misc.025', '思考过程', '思考过程') }}</summary>
        <div class="think-body">{{ think }}</div>
      </details>
      <div v-if="role === 'assistant'" ref="bubbleEl" class="md-body" v-html="renderedHtml"></div>
      <div v-else class="plain-body">{{ bodyNoCite }}</div>
      <div v-if="role === 'assistant' && validSourceTokens.length" class="source-line">
        <span>{{ t('ui.misc.040', '来源：', '来源：') }}</span>
        <button
          v-for="n in validSourceTokens"
          :key="n"
          class="cite-btn"
          :class="{ active: activeIndex === n }"
          type="button"
          @click.stop="toggle(n)"
        >[{{ n }}]</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bubble-row {
  display: flex;
  margin-bottom: 12px;
}
.bubble-row.user { justify-content: flex-end; }
.bubble-row.assistant { justify-content: flex-start; }

.bubble {
  max-width: 85%;
  padding: 10px 14px;
  border-radius: 10px;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 14px;
  line-height: 1.6;
}
.bubble.user {
  background: var(--bg-bubble-user);
  color: var(--text-on-user);
}
.bubble.assistant {
  background: var(--bg-bubble-assistant);
  color: var(--text-primary);
}
.bubble.system {
  background: var(--bg-bubble-thinking);
  color: var(--text-muted);
  font-style: italic;
}

/* Markdown body (assistant only) lives inside a div that opts out of pre-wrap
   so marked's own <p>/<pre>/<ul> whitespace is respected. */
.md-body {
  white-space: normal;
}
.md-body :deep(p) { margin: 0.5em 0; }
.md-body :deep(p:first-child) { margin-top: 0; }
.md-body :deep(p:last-child) { margin-bottom: 0; }
.md-body :deep(h1),
.md-body :deep(h2),
.md-body :deep(h3),
.md-body :deep(h4) { margin: 0.8em 0 0.4em; font-weight: 600; line-height: 1.3; }
.md-body :deep(h1) { font-size: 1.4em; }
.md-body :deep(h2) { font-size: 1.25em; }
.md-body :deep(h3) { font-size: 1.1em; }
.md-body :deep(h4) { font-size: 1em; }
.md-body :deep(ul),
.md-body :deep(ol) { margin: 0.5em 0; padding-left: 1.5em; }
.md-body :deep(li) { margin: 0.2em 0; }
.md-body :deep(li > p) { margin: 0.1em 0; }
.md-body :deep(strong) { font-weight: 600; }
.md-body :deep(em) { font-style: italic; }
.md-body :deep(del) { color: var(--text-muted, #888); }
.md-body :deep(code) {
  background: rgba(0, 0, 0, 0.06);
  padding: 1px 5px;
  border-radius: 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.9em;
}
.md-body :deep(pre) {
  background: rgba(0, 0, 0, 0.04);
  padding: 10px 12px;
  border-radius: 6px;
  overflow-x: auto;
  margin: 0.6em 0;
  white-space: pre;
}
.md-body :deep(pre code) {
  background: transparent;
  padding: 0;
  font-size: 0.85em;
}
.md-body :deep(a) {
  color: var(--accent, #3b82f6);
  text-decoration: underline;
  text-underline-offset: 2px;
}
.md-body :deep(blockquote) {
  border-left: 3px solid var(--border-soft, #ddd);
  padding: 0 12px;
  margin: 0.6em 0;
  color: var(--text-muted, #666);
}
.md-body :deep(table) {
  border-collapse: collapse;
  margin: 0.6em 0;
}
.md-body :deep(th),
.md-body :deep(td) {
  border: 1px solid var(--border-soft, #ddd);
  padding: 4px 8px;
}
.md-body :deep(th) { background: rgba(0, 0, 0, 0.03); }
.md-body :deep(hr) {
  border: none;
  border-top: 1px dashed var(--border-soft, #ddd);
  margin: 0.8em 0;
}
.md-body :deep(img) {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
  margin: 0.4em 0;
}
.md-body :deep(.mermaid-block) {
  text-align: center;
  margin: 12px 0;
  padding: 12px;
  background: rgba(0, 0, 0, 0.02);
  border-radius: 8px;
  overflow-x: auto;
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.85em;
}
.md-body :deep(.mermaid-block[data-rendered]) {
  background: transparent;
  padding: 0;
  white-space: normal;
}
.md-body :deep(.mermaid-block[data-rendered] svg) {
  max-width: 100%;
  height: auto;
}

/* Mermaid 卡片：独立区域 + 标题栏 + 右上角操作按钮 */
.md-body :deep(.mermaid-card) {
  margin: 16px 0;
  border: 1px solid var(--border-soft, rgba(128, 128, 128, 0.25));
  border-radius: 12px;
  background: var(--bg-elevated, rgba(255, 255, 255, 0.03));
  overflow: hidden;
  transition: border-color 0.2s;
}
.md-body :deep(.mermaid-card:hover) {
  border-color: rgba(59, 130, 246, 0.4);
}
.md-body :deep(.mermaid-card-header) {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: rgba(59, 130, 246, 0.08);
  border-bottom: 1px solid var(--border-soft, rgba(128, 128, 128, 0.2));
  cursor: pointer;
  user-select: none;
}
.md-body :deep(.mermaid-card-title) {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary, #222);
  display: flex;
  align-items: center;
  gap: 6px;
}
.md-body :deep(.mermaid-card-title)::before {
  content: '';
  display: inline-block;
  width: 4px;
  height: 14px;
  background: var(--accent, #3b82f6);
  border-radius: 2px;
}
.md-body :deep(.mermaid-card-toolbar) {
  display: flex;
  gap: 4px;
}
.md-body :deep(.mermaid-action) {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--text-secondary, #888);
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.md-body :deep(.mermaid-action:hover) {
  background: rgba(59, 130, 246, 0.15);
  color: var(--accent, #3b82f6);
}
.md-body :deep(.mermaid-card-body) {
  padding: 16px;
  overflow-x: auto;
  transition: max-height 0.3s ease, opacity 0.3s ease, padding 0.3s ease;
}
.md-body :deep(.mermaid-card.collapsed .mermaid-card-body) {
  max-height: 0;
  padding-top: 0;
  padding-bottom: 0;
  opacity: 0;
  overflow: hidden;
}
.md-body :deep(.mermaid-card.collapsed .mermaid-action[data-action="toggle"] svg) {
  transform: rotate(180deg);
}

.think-section {
  margin: -2px 0 8px;
  padding: 6px 10px;
  border: 1px dashed var(--border-soft, #ddd);
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.03);
  font-size: 12px;
  opacity: 0.85;
}
.think-section > summary {
  cursor: pointer;
  user-select: none;
  color: var(--text-muted, #888);
  font-weight: 500;
}
.think-section > summary::marker { color: var(--text-muted, #888); }
.think-body {
  margin-top: 6px;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text-muted, #666);
  line-height: 1.55;
}

.source-line {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--border-soft, #ddd);
  font-size: 13px;
}
.source-line span {
  color: var(--text-muted, #888);
  margin-right: 4px;
}

.cite-btn {
  display: inline;
  padding: 0 4px;
  margin: 0 1px;
  border: none;
  background: transparent;
  color: var(--accent, #3b82f6);
  font: inherit;
  cursor: pointer;
  border-radius: 4px;
  text-decoration: underline;
  text-underline-offset: 2px;
  line-height: inherit;
}
.cite-btn:hover { background: rgba(59, 130, 246, 0.12); }
.cite-btn.active {
  background: var(--accent, #3b82f6);
  color: #fff;
  text-decoration: none;
}

/* 内联引用 [n] 标记：正文中直接显示的小圆角徽章 */
.cite-inline {
  display: inline-block;
  padding: 0 5px;
  margin: 0 1px;
  font-size: 0.85em;
  font-weight: 600;
  line-height: 1.5;
  color: var(--accent, #3b82f6);
  background: rgba(59, 130, 246, 0.10);
  border-radius: 4px;
  cursor: pointer;
  vertical-align: baseline;
  user-select: none;
  transition: background 0.15s, color 0.15s;
}
.cite-inline:hover {
  background: rgba(59, 130, 246, 0.22);
}
.cite-inline.active {
  background: var(--accent, #3b82f6);
  color: #fff;
}
</style>
