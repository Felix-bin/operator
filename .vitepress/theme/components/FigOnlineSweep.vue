<script setup>
import { computed, ref } from 'vue'

// RTX 4060 Laptop 上两次运行的平均，M × N = 2^26 个 float，Block = 256，每次取 9 次的中位数
const data = [
  { N: 256, safe: 2.38, v2: 2.83, v3: 3.19 },
  { N: 512, safe: 2.4, v2: 2.4, v3: 2.35 },
  { N: 1024, safe: 2.42, v2: 2.39, v3: 2.34 },
  { N: 4096, safe: 2.38, v2: 2.4, v3: 2.32 },
  { N: 8192, safe: 2.36, v2: 2.36, v3: 2.32 },
  { N: 16384, safe: 2.4, v2: 2.4 },
  { N: 32768, safe: 3.44, v2: 3.09 },
  { N: 65536, safe: 4.41, v2: 3.48 },
  { N: 131072, safe: 4.59, v2: 3.49 },
  { N: 262144, safe: 4.6, v2: 3.51 },
  { N: 1048576, safe: 4.62, v2: 3.52 }
]
const L = 50
const R = 650
const T = 18
const B = 278
const x = (n) => L + ((Math.log2(n) - 8) / 12) * (R - L)
const y = (v) => B - (v / 5) * (B - T)
const yTicks = [0, 1, 2, 3, 4, 5]
const xTicks = [256, 1024, 4096, 16384, 65536, 262144, 1048576]
const xLabel = (n) => (n >= 1048576 ? `${n / 1048576}M` : n >= 1024 ? `${n / 1024}K` : String(n))

const line = (key) =>
  data
    .filter((d) => d[key] !== undefined)
    .map((d, i) => `${i ? 'L' : 'M'}${x(d.N).toFixed(1)} ${y(d[key]).toFixed(1)}`)
    .join(' ')

const BYTES = 2 ** 26 * 4
const BW = 230e9
const refs = [2, 3, 4].map((k) => ({ k, v: ((k * BYTES) / BW) * 1000 }))

const hover = ref(null)
function onMove(e) {
  const box = e.currentTarget.getBoundingClientRect()
  const svgX = ((e.clientX - box.left) / box.width) * 720
  let best = 0
  data.forEach((d, i) => {
    if (Math.abs(x(d.N) - svgX) < Math.abs(x(data[best].N) - svgX)) best = i
  })
  hover.value = best
}
const shown = computed(() => data[hover.value ?? 3])
</script>

<template>
  <div>
    <div class="gpu-legend gpu-chart-legend">
      <span><i class="key ka"></i>三遍 Safe</span>
      <span><i class="key kb"></i>两遍 Online V2</span>
      <span><i class="key online-kc"></i>V3（数组暂存）</span>
    </div>
    <div class="gpu-scroll" tabindex="0">
      <svg
        class="w680 gpu-chart"
        viewBox="0 0 720 322"
        role="img"
        aria-label="256 MiB 输入的耗时随行长变化。行长不超过 16384 时三种方案都约 2.4 毫秒；行长达到 65536 以上时，Safe 约 4.6 毫秒，V2 约 3.5 毫秒。"
        @pointermove="onMove"
        @pointerdown="onMove"
        @pointerleave="hover = null"
      >
        <g v-for="t in yTicks" :key="t">
          <line class="gpu-grid" :x1="L" :x2="R" :y1="y(t)" :y2="y(t)" />
          <text class="gpu-ts gpu-end" :x="L - 8" :y="y(t) + 4">{{ t }}</text>
        </g>
        <text class="gpu-ts" x="0" y="12">毫秒</text>
        <text v-for="t in xTicks" :key="t" class="gpu-ts gpu-mid" :x="x(t)" :y="B + 20">{{ xLabel(t) }}</text>
        <text class="gpu-ts gpu-mid" :x="(L + R) / 2" :y="B + 40">每行元素数 N（对数轴）</text>
        <g v-for="r in refs" :key="r.k">
          <line class="gpu-ln gpu-dash" :x1="L" :x2="R" :y1="y(r.v)" :y2="y(r.v)" style="opacity: 0.45" />
          <text class="gpu-ts" :x="R + 8" :y="y(r.v) + 4">{{ r.k }}N 流量</text>
        </g>
        <line v-if="hover !== null" class="gpu-cross" :x1="x(shown.N)" :x2="x(shown.N)" :y1="T" :y2="B" />
        <path class="gpu-line la" :d="line('safe')" />
        <path class="gpu-line lb" :d="line('v2')" />
        <path class="gpu-line online-lc" :d="line('v3')" />
        <template v-for="d in data" :key="d.N">
          <circle class="gpu-pt pa" :cx="x(d.N)" :cy="y(d.safe)" r="4" />
          <circle class="gpu-pt pb" :cx="x(d.N)" :cy="y(d.v2)" r="4" />
          <circle v-if="d.v3 !== undefined" class="gpu-pt online-pc" :cx="x(d.N)" :cy="y(d.v3)" r="4" />
        </template>
        <rect :x="L" :y="T" :width="R - L" :height="B - T" fill="transparent" />
      </svg>
    </div>
    <div class="gpu-read online-sweep-read" aria-live="polite">
      <span>N = {{ shown.N.toLocaleString('en-US') }}</span>
      <span>三遍 Safe {{ shown.safe.toFixed(2) }} ms</span>
      <span>Online V2 {{ shown.v2.toFixed(2) }} ms</span>
      <span v-if="shown.v3 !== undefined">V3 {{ shown.v3.toFixed(2) }} ms</span>
    </div>
  </div>
</template>

<style>
.online-kc {
  border-color: var(--gpu-cc) !important;
}
.gpu-line.online-lc {
  stroke: var(--gpu-cc);
}
.gpu-pt.online-pc {
  fill: var(--gpu-cc);
}
.online-sweep-read {
  gap: 4px 18px;
  font-variant-numeric: tabular-nums;
}
</style>
