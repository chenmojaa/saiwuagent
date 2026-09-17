<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue"
import { useChatStore } from "@/stores/chat"

const props = defineProps<{
  show: boolean
}>()

const chat = useChatStore()

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
watch(() => props.show, (v) => {
  if (v) startTimer()
  else stopTimer()
}, { immediate: true })
onBeforeUnmount(stopTimer)

const label = computed<string>(() => {
  if (chat.thinking) return "思考中..."
  const s = chat.stage
  if (!s) return "正在思考..."
  if (s.stage === "router" && s.status === "started") return "识别意图中..."
  if (s.stage === "router" && s.status === "done") {
    return s.intent ? "路由 -> " + s.intent : "路由完成"
  }
  if (s.stage === "rag_search" && s.status === "started") return "检索知识库中..."
  if (s.stage === "rag_search" && s.status === "done") {
    return s.hits && s.hits > 0
      ? "检索到 " + s.hits + " 条（" + Math.round(s.ms ?? 0) + "ms）"
      : "未找到相关内容"
  }
  if (s.stage === "agent" && s.status === "started") {
    return s.agent === "research" ? "多轮检索中..."
      : s.agent === "ingest"  ? "入库中..."
      : s.agent === "report"  ? "生成周报中..." : "agent 运行中"
  }
  if (s.stage === "agent" && s.status === "done") {
    if (s.agent === "planner") {
      return s.steps && s.steps > 0
        ? "已制定 " + s.steps + " 步检索计划"
        : "无需分解，直接检索"
    }
    if (s.agent === "research") {
      // 计划逐步执行：显示当前步骤进度
      if (s.step && s.total_steps) {
        const q = s.query ? "：" + (s.query.length > 14 ? s.query.slice(0, 14) + "…" : s.query) : ""
        return "执行计划 " + s.step + "/" + s.total_steps + q
      }
      if (s.query) return "补检索中：" + (s.query.length > 16 ? s.query.slice(0, 16) + "…" : s.query)
      if (s.iterations != null) return "研究完成 (" + s.iterations + " 轮)"
      return "多轮检索中..."
    }
    if (s.agent === "ingest")  return "入库完成"
    if (s.agent === "report")  return "周报已生成"
  }
  if (s.stage === "web_verify") {
    // 策略 A：知识库命中后仍会联网核对。升级到真实浏览器时可能要 5-12s，
    // 这里带秒数显示（外层模板会拼），用户才知道不是卡死了。
    return s.status === "done" ? "联网核对完成" : "联网核对中..."
  }
  if (s.stage === "llm_stream" && s.status === "started") return "生成回答中..."
  return "正在思考..."
})
</script>

<template>
  <div v-if="show" class="thinking-row" role="status" aria-live="polite">
    <div class="thinking-bubble">
      <span class="thinking-dots" aria-hidden="true">
        <span></span><span></span><span></span>
      </span>
      <span class="thinking-text">{{ label }}（{{ elapsed }}s）</span>
    </div>
  </div>
</template>

<style scoped>
.thinking-row {
  display: flex;
  justify-content: flex-start;
  margin-bottom: 12px;
}
.thinking-bubble {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 10px;
  background: var(--bg-bubble-thinking);
  color: var(--text-muted);
  font-size: 14px;
  line-height: 1.6;
}
.thinking-dots {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  flex-shrink: 0;
}
.thinking-dots > span {
  width: 5px;
  height: 5px;
  background: currentColor;
  border-radius: 50%;
  animation: thinking-bounce 1.2s ease-in-out infinite;
  opacity: 0.55;
}
.thinking-dots > span:nth-child(2) { animation-delay: 0.15s; }
.thinking-dots > span:nth-child(3) { animation-delay: 0.3s; }
@keyframes thinking-bounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.45; }
  40%           { transform: translateY(-3px); opacity: 1; }
}
.thinking-text { white-space: nowrap; }
</style>
