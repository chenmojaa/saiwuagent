<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore } from '@/stores/chat'
import { useSessionsStore } from '@/stores/sessions'

const chat = useChatStore()
const sessions = useSessionsStore()
const router = useRouter()

// Only when a stream is running AND the user is NOT looking at the session
// that's producing.
const visible = computed<boolean>(() => {
  return (
    chat.isStreaming &&
    chat.streamingSessionId !== null &&
    chat.streamingSessionId !== chat.sessionId
  )
})

const targetSession = computed(() => {
  if (!chat.streamingSessionId) return null
  return sessions.items.find(s => s.id === chat.streamingSessionId) || null
})

const label = computed<string>(() => {
  const s = chat.stage
  if (!s) return '\u601d\u8003\u4e2d...'
  if (s.stage === 'router' && s.status === 'started') return '\u8bc6\u522b\u610f\u56fe\u4e2d...'
  if (s.stage === 'router' && s.status === 'done') {
    return s.intent ? '\u8def\u7531 -> ' + s.intent : '\u8def\u7531\u5b8c\u6210'
  }
  if (s.stage === 'rag_search' && s.status === 'started') return '\u68c0\u7d22\u77e5\u8bc6\u5e93\u4e2d...'
  if (s.stage === 'rag_search' && s.status === 'done') {
    return s.hits && s.hits > 0
      ? '\u68c0\u7d22\u5230 ' + s.hits + ' \u6761'
      : '\u68c0\u7d22\u5b8c\u6210\uff08\u65e0\u5339\u914d\uff09'
  }
  if (s.stage === 'agent' && s.status === 'started') {
    return s.agent === 'research' ? '\u591a\u8f6e\u68c0\u7d22\u4e2d...' :
           s.agent === 'ingest'  ? '\u5165\u5e93\u4e2d...' :
           s.agent === 'report'  ? '\u751f\u6210\u5468\u62a5\u4e2d...' : 'agent \u8fd0\u884c\u4e2d'
  }
  if (s.stage === 'agent' && s.status === 'done') {
    if (s.agent === 'planner') {
      return s.steps && s.steps > 0
        ? '\u5df2\u5236\u5b9a ' + s.steps + ' \u6b65\u68c0\u7d22\u8ba1\u5212'
        : '\u65e0\u9700\u5206\u89e3\uff0c\u76f4\u63a5\u68c0\u7d22'
    }
    if (s.agent === 'research') return '\u7814\u7a76\u5b8c\u6210 (' + (s.iterations || 0) + ' \u8f6e)'
    if (s.agent === 'ingest')  return '\u5165\u5e93\u5b8c\u6210'
    if (s.agent === 'report')  return '\u5468\u62a5\u751f\u6210\u5b8c\u6210'
    return 'agent \u5b8c\u6210'
  }
  if (s.stage === 'web_verify') {
    // 策略 A：知识库命中后仍会联网核对。升级到真实浏览器时可能要 5-12s，
    // 必须让用户知道在等什么，否则看起来像卡死了。
    return s.status === 'done' ? '联网核对完成' : '联网核对中...'
  }
  if (s.stage === 'llm_stream' && s.status === 'started') return '\u751f\u6210\u56de\u7b54\u4e2d...'
  return '\u601d\u8003\u4e2d...'
})

const elapsed = ref(0)
let timerId: number | null = null
function startTimer() {
  stopTimer()
  elapsed.value = 0
  timerId = window.setInterval(() => { elapsed.value++ }, 1000)
}
function stopTimer() {
  if (timerId !== null) { window.clearInterval(timerId); timerId = null }
}

watch(visible, (v) => {
  if (v) startTimer()
  else { stopTimer(); elapsed.value = 0 }
}, { immediate: true })

onBeforeUnmount(stopTimer)

function switchTo() {
  if (chat.streamingSessionId) {
    router.push({ name: 'chat-id', params: { id: chat.streamingSessionId } })
  }
}
</script>

<template>
  <Transition name="slide-up">
    <div v-if="visible" class="streaming-indicator" @click="switchTo" role="button" tabindex="0">
      <div class="indicator-icon" aria-hidden="true">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
      <div class="indicator-content">
        <div class="indicator-title">{{ targetSession?.title || '未命名会话' }}</div>
        <div class="indicator-status">{{ label }}（{{ elapsed }}s）· 点击切回去</div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.streaming-indicator {
  position: fixed;
  bottom: 24px;
  right: 24px;
  background: var(--bg-bubble-assistant, #2a2a2a);
  color: var(--text-primary, #fff);
  padding: 10px 14px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  max-width: 320px;
  z-index: 1000;
  user-select: none;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
  border: 1px solid var(--border-soft, rgba(255, 255, 255, 0.08));
}
.streaming-indicator:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
}
.streaming-indicator:focus { outline: none; }
.streaming-indicator:focus-visible {
  outline: 2px solid var(--accent, #3b82f6);
  outline-offset: 2px;
}

.indicator-icon {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  flex-shrink: 0;
}
.dot {
  width: 6px;
  height: 6px;
  background: var(--accent, #3b82f6);
  border-radius: 50%;
  animation: bounce 1.2s ease-in-out infinite;
}
.dot:nth-child(2) { animation-delay: 0.15s; }
.dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.5; }
  40% { transform: translateY(-4px); opacity: 1; }
}

.indicator-content {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.indicator-title {
  font-size: 13px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 240px;
}
.indicator-status {
  font-size: 12px;
  opacity: 0.7;
}

.slide-up-enter-active,
.slide-up-leave-active {
  transition: opacity 0.22s ease, transform 0.22s ease;
}
.slide-up-enter-from,
.slide-up-leave-to {
  opacity: 0;
  transform: translateY(16px);
}
</style>
