<script setup>
import { computed, ref } from 'vue'

// P：BF16 输入、FP32 累加的稠密矩阵峰值（TFLOP/s）；R：显存带宽（TB/s）
const devices = [
  { id: 'rtx3090', name: 'RTX 3090', P: 71.0, R: 0.936 },
  { id: 'rtx4090', name: 'RTX 4090', P: 165.2, R: 1.008 },
  { id: 'rtx5090', name: 'RTX 5090', P: 209.5, R: 1.792 },
  { id: 'pro6000', name: 'RTX PRO 6000', P: 503.8, R: 1.792 },
  { id: 'a100', name: 'A100 SXM', P: 312.0, R: 2.039 },
  { id: 'h100', name: 'H100 SXM', P: 989.4, R: 3.35 },
  { id: 'h200', name: 'H200 SXM', P: 989.5, R: 4.8 }
]
const d = 4096
const maxM = 512
const W = 720
const H = 300
const left = 56
const right = 24
const top = 16
const bottom = 42

const deviceId = ref('rtx4090')
const M = ref(256)
const device = computed(() => devices.find((x) => x.id === deviceId.value))

// 都按 μs 计：FLOPs / (TFLOP/s × 10^6)，bytes / (TB/s × 10^6)
const matrixTime = (m) => (2 * m * d * d) / (device.value.P * 1e6)
const memoryTime = (m) => (2 * d * d + 4 * m * d) / (device.value.R * 1e6)
const intensity = (m) => (m * d) / (d + 2 * m)

const yMax = computed(() => {
  const peak = Math.max(matrixTime(maxM), memoryTime(maxM))
  const step = 10 ** Math.floor(Math.log10(peak))
  const unit = [1, 2, 2.5, 5, 10].find((u) => u * step >= peak / 4) * step
  return Math.ceil(peak / unit) * unit
})
const yTicks = computed(() => {
  const n = 4
  return Array.from({ length: n + 1 }, (_, i) => (yMax.value / n) * i)
})
const xTicks = [1, 128, 256, 384, 512]
const x = (m) => left + ((m - 1) / (maxM - 1)) * (W - left - right)
const y = (t) => H - bottom - (t / yMax.value) * (H - top - bottom)
const linePath = (fn) => `M${x(1)} ${y(fn(1))} L${x(maxM)} ${y(fn(maxM))}`

const crossover = computed(() => {
  const Is = (device.value.P / device.value.R)
  return (Is * d) / (d - 2 * Is)
})
const dominant = computed(() => (matrixTime(M.value) > memoryTime(M.value) ? '矩阵计算' : '片外读取'))

const fmt = (t) => (t >= 100 ? t.toFixed(0) : t >= 10 ? t.toFixed(1) : t.toFixed(2))

function onPointer(e) {
  const box = e.currentTarget.getBoundingClientRect()
  const px = ((e.clientX - box.left) / box.width) * W
  const m = Math.round(1 + ((px - left) / (W - left - right)) * (maxM - 1))
  M.value = Math.min(maxM, Math.max(1, m))
}
</script>

<template>
  <figure class="gpu-fig gpu-lab">
    <div class="gpu-tools">
      <button
        v-for="item in devices"
        :key="item.id"
        type="button"
        :aria-pressed="item.id === deviceId"
        :class="{ active: item.id === deviceId }"
        @click="deviceId = item.id"
      >
        {{ item.name }}
      </button>
    </div>
    <div class="gpu-lab-row">
      <label for="roofline-m">一次处理的行数 M = <b>{{ M }}</b></label>
      <input id="roofline-m" v-model.number="M" type="range" min="1" :max="maxM" />
    </div>
    <div class="gpu-read" aria-live="polite">
      <span><b>{{ fmt(matrixTime(M)) }}</b> μs 矩阵计算</span>
      <span><b>{{ fmt(memoryTime(M)) }}</b> μs 片外读取</span>
      <span><b>{{ intensity(M).toFixed(intensity(M) < 10 ? 1 : 0) }}</b> FLOPs/byte 算术强度</span>
      <span>当前瓶颈：{{ dominant }}。从 M ≈ {{ Math.ceil(crossover) }} 起，矩阵计算更慢。</span>
    </div>
    <div class="gpu-legend gpu-chart-legend">
      <span><i class="key ka"></i>矩阵计算</span>
      <span><i class="key kb"></i>片外读取</span>
    </div>
    <div class="gpu-scroll" tabindex="0">
      <svg
        class="w680 gpu-chart"
        :viewBox="`0 0 ${W} ${H}`"
        role="img"
        :aria-label="`${device.name} 上 Q 投影的矩阵计算时间与片外读取时间，随行数变化，约在 ${Math.ceil(crossover)} 行相交。`"
        @pointermove="onPointer"
        @pointerdown="onPointer"
      >
        <g v-for="t in yTicks" :key="t">
          <line class="gpu-grid" :x1="left" :x2="W - right" :y1="y(t)" :y2="y(t)" />
          <text class="gpu-ts gpu-end" :x="left - 8" :y="y(t) + 4">{{ t }}</text>
        </g>
        <text v-for="t in xTicks" :key="t" class="gpu-ts gpu-mid" :x="x(t)" :y="H - bottom + 18">{{ t }}</text>
        <text class="gpu-ts gpu-mid" :x="(left + W - right) / 2" :y="H - 6">行数 M</text>
        <text class="gpu-ts" :x="left" :y="11">时间（μs）</text>

        <path class="gpu-line la" :d="linePath(matrixTime)" />
        <path class="gpu-line lb" :d="linePath(memoryTime)" />

        <line class="gpu-cross" :x1="x(M)" :x2="x(M)" :y1="top" :y2="H - bottom" />
        <circle class="gpu-pt pa" :cx="x(M)" :cy="y(matrixTime(M))" r="4" />
        <circle class="gpu-pt pb" :cx="x(M)" :cy="y(memoryTime(M))" r="4" />
      </svg>
    </div>
    <figcaption>
      简化模型：输入、权重和输出都读写一次显存，计算与读取可以完全重叠。矩阵速率取 BF16 输入、FP32 累加的稠密峰值。在图上移动指针，或拖动滑块，可以查看任意行数。
    </figcaption>
  </figure>
</template>
