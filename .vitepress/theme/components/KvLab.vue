<script setup>
import { computed, ref } from 'vue'

const devices = [
  { id: 'rtx4090', name: 'RTX 4090', cap: 24 },
  { id: 'rtx5090', name: 'RTX 5090', cap: 32 },
  { id: 'h100', name: 'H100', cap: 80 },
  { id: 'pro6000', name: 'RTX PRO 6000', cap: 96 },
  { id: 'h200', name: 'H200', cap: 141 }
]
const contexts = [2048, 8192, 16384, 32768]
// Qwen3-8B、BF16：常驻权重 16.381 GB，每个 token 的 KV 为 147,456 bytes，工作区 2 GiB
const weights = 16.381
const workspace = (2 * 2 ** 30) / 1e9
const kvPerToken = 147456

const deviceId = ref('rtx4090')
const S = ref(8192)
const B = ref(4)
const cap = computed(() => devices.find((d) => d.id === deviceId.value).cap)

const kvOne = computed(() => (S.value * kvPerToken) / 1e9)
const kv = computed(() => B.value * kvOne.value)
const total = computed(() => weights + workspace + kv.value)
const bMax = computed(() => Math.max(0, Math.floor((cap.value - weights - workspace) / kvOne.value)))
const fits = computed(() => total.value <= cap.value)
const scale = computed(() => Math.max(cap.value, total.value))

const parts = computed(() => [
  { key: 'a', name: '权重', value: weights },
  { key: 'b', name: '工作区', value: workspace },
  { key: 'c', name: `KV（${B.value} 条请求）`, value: kv.value }
])
const pct = (v) => (v / scale.value) * 100
</script>

<template>
  <figure class="gpu-fig gpu-lab">
    <div class="gpu-tools">
      <button
        v-for="d in devices"
        :key="d.id"
        type="button"
        :aria-pressed="d.id === deviceId"
        :class="{ active: d.id === deviceId }"
        @click="deviceId = d.id"
      >
        {{ d.name }}（{{ d.cap }} GB）
      </button>
    </div>
    <div class="gpu-tools">
      <button
        v-for="c in contexts"
        :key="c"
        type="button"
        :aria-pressed="c === S"
        :class="{ active: c === S }"
        @click="S = c"
      >
        上下文 {{ c / 1024 }}K
      </button>
    </div>
    <div class="gpu-lab-row">
      <label for="kv-b">并发请求数 B = <b>{{ B }}</b></label>
      <input id="kv-b" v-model.number="B" type="range" min="1" max="64" />
    </div>
    <div class="gpu-read" aria-live="polite">
      <span><b>{{ total.toFixed(1) }}</b> GB 需要</span>
      <span><b>{{ bMax }}</b> 条请求上限</span>
      <span>{{ fits ? '放得下' : `超出 ${(total - cap).toFixed(1)} GB，放不下` }}。每条请求的 KV 占 {{ kvOne.toFixed(2) }} GB。</span>
    </div>
    <div class="gpu-legend gpu-chart-legend">
      <span v-for="p in parts" :key="p.key"><i class="swatch" :class="`s${p.key}`"></i>{{ p.name }}</span>
    </div>
    <div class="gpu-stack-wrap">
      <div class="gpu-stack" role="img" :aria-label="`权重 ${weights} GB，工作区 ${workspace.toFixed(2)} GB，KV ${kv.toFixed(2)} GB，容量 ${cap} GB。`">
        <div
          v-for="p in parts"
          :key="p.key"
          class="seg"
          :class="`s${p.key}`"
          :style="{ width: pct(p.value) + '%' }"
          :title="`${p.name}：${p.value.toFixed(2)} GB`"
        >
          <span v-if="pct(p.value) > 11">{{ p.value.toFixed(1) }}</span>
        </div>
      </div>
      <div class="gpu-cap" :style="{ left: pct(cap) + '%' }"></div>
    </div>
    <p class="gpu-cap-label" :style="{ paddingLeft: Math.min(pct(cap), 70) + '%' }">容量 {{ cap }} GB</p>
    <figcaption>模型固定为 Qwen3-8B、BF16。工作区按 2 GiB 估算，实际还需扣除操作系统、驱动和框架的占用。</figcaption>
  </figure>
</template>
