<script setup>
import { computed, ref, useId } from 'vue'

const N = 4096
const options = [64, 128, 256]
const d = ref(128)
const W = 720
const H = 330
const left = 64
const right = 24
const top = 18
const bottom = 46
const yLo = 5
const yHi = 11
const sliderId = useId()

// 与 flashv1_verify.py 的 io_closed / io_std 同一公式
const flash = (M, dd) => {
  const Bc = Math.ceil(M / (4 * dd))
  const Tc = Math.ceil(N / Bc)
  return Tc * (3 * N * dd + 4 * N) + 2 * N * dd
}
const std = (dd) => 4 * N * N + 4 * N * dd
const floorIo = (dd) => 4 * N * dd

const lo = computed(() => d.value)
const hi = computed(() => N * d.value)
const x = (M) => left + (Math.log(M / lo.value) / Math.log(hi.value / lo.value)) * (W - left - right)
const y = (v) => H - bottom - ((Math.log10(v) - yLo) / (yHi - yLo)) * (H - top - bottom)
const flashPath = computed(() => {
  const pts = []
  const n = 320
  for (let i = 0; i <= n; i++) {
    const M = lo.value * (hi.value / lo.value) ** (i / n)
    pts.push(`${x(M).toFixed(1)} ${y(flash(M, d.value)).toFixed(1)}`)
  }
  return `M${pts.join(' L')}`
})
const stdLine = computed(() => `M${left} ${y(std(d.value))} L${W - right} ${y(std(d.value))}`)
const floorLine = computed(() => `M${left} ${y(floorIo(d.value))} L${W - right} ${y(floorIo(d.value))}`)
const xTicks = computed(() => [100, 1000, 10000, 100000].filter((t) => t >= lo.value && t <= hi.value))
const yTicks = [5, 6, 7, 8, 9, 10, 11]

// 滑块在对数轴上取值，默认停在 192 KB / 2
const pos = ref(-1)
const STEPS = 400
const defaultM = 98304
const M = computed(() => {
  if (pos.value < 0) return defaultM
  return Math.round(lo.value * (hi.value / lo.value) ** (pos.value / STEPS))
})
const sliderValue = computed(() =>
  pos.value < 0 ? Math.round((Math.log(defaultM / lo.value) / Math.log(hi.value / lo.value)) * STEPS) : pos.value
)
const Bc = computed(() => Math.ceil(M.value / (4 * d.value)))
const Tc = computed(() => Math.ceil(N / Bc.value))
const fNow = computed(() => flash(M.value, d.value))
const sNow = computed(() => std(d.value))
const ratio = computed(() => sNow.value / fNow.value)
const cross = computed(() => 3 * d.value * d.value)
const wan = (v) => (v >= 1e8 ? `${(v / 1e8).toFixed(2)} 亿` : `${(v / 1e4).toFixed(0)} 万`)
const expLabel = { 5: '10⁵', 6: '10⁶', 7: '10⁷', 8: '10⁸', 9: '10⁹', 10: '10¹⁰', 11: '10¹¹' }
const xLabel = { 100: '10²', 1000: '10³', 10000: '10⁴', 100000: '10⁵' }

function onPointer(e) {
  const box = e.currentTarget.getBoundingClientRect()
  const px = ((e.clientX - box.left) / box.width) * W
  const t = Math.max(0, Math.min(1, (px - left) / (W - left - right)))
  pos.value = Math.round(t * STEPS)
}
function pick(v) {
  d.value = v
  pos.value = -1
}
</script>

<template>
  <figure class="gpu-fig gpu-lab">
    <div class="gpu-tools">
      <button v-for="v in options" :key="v" type="button" :aria-pressed="v === d" :class="{ active: v === d }" @click="pick(v)">
        d = {{ v }}
      </button>
    </div>
    <div class="gpu-lab-row">
      <label :for="sliderId">SRAM 大小 M = <b>{{ M.toLocaleString('en-US') }}</b> 个元素</label>
      <input :id="sliderId" :value="sliderValue" type="range" min="0" :max="STEPS" @input="pos = Number($event.target.value)" />
    </div>
    <div class="gpu-read" aria-live="polite">
      <span><b>{{ wan(fNow) }}</b> FlashAttention</span>
      <span><b>{{ wan(sNow) }}</b> 标准实现</span>
      <span><b>{{ ratio.toFixed(2) }}</b> 倍</span>
      <span>B_c = {{ Bc }}，T_c = {{ Tc }}。{{ ratio >= 1 ? 'FlashAttention 读写更少。' : 'SRAM 太小，FlashAttention 反而读写更多。' }}</span>
    </div>
    <div class="gpu-legend gpu-chart-legend">
      <span><i class="key ka"></i>FlashAttention，T_c(3Nd + 4N) + 2Nd</span>
      <span><i class="key kb"></i>标准实现，4N² + 4Nd</span>
    </div>
    <div class="gpu-scroll" tabindex="0">
      <svg
        class="w680 gpu-chart"
        :viewBox="`0 0 ${W} ${H}`"
        role="img"
        :aria-label="`N = 4096、d = ${d} 时 HBM 读写量随 SRAM 大小变化。约在 M = ${cross} 处两条线相交，M 更大时 FlashAttention 更少。`"
        @pointermove="onPointer"
        @pointerdown="onPointer"
      >
        <g v-for="t in yTicks" :key="t">
          <line class="gpu-grid" :x1="left" :x2="W - right" :y1="y(10 ** t)" :y2="y(10 ** t)" />
          <text class="gpu-ts gpu-end" :x="left - 8" :y="y(10 ** t) + 4">{{ expLabel[t] }}</text>
        </g>
        <text v-for="t in xTicks" :key="t" class="gpu-ts gpu-mid" :x="x(t)" :y="H - bottom + 18">{{ xLabel[t] }}</text>
        <text class="gpu-ts gpu-mid" :x="(left + W - right) / 2" :y="H - 8">SRAM 大小 M（元素个数，对数轴，从 d 到 Nd）</text>
        <text class="gpu-ts" :x="left" y="12">HBM 读写（元素个数，对数轴）</text>

        <path class="gpu-line la" :d="flashPath" />
        <path class="gpu-line lb" :d="stdLine" />
        <path class="gpu-cross gpu-dash" :d="floorLine" />
        <text class="gpu-ts" :x="left + 6" :y="y(floorIo(d)) - 6">输入输出至少各读写一次：4Nd</text>
        <text class="gpu-ts" :x="left + 6" :y="y(std(d)) - 6">标准实现</text>
        <text class="gpu-ts" :x="left + 6" :y="y(flash(lo, d)) - 8">FlashAttention</text>

        <line class="gpu-cross" :x1="x(cross)" :x2="x(cross)" :y1="top" :y2="H - bottom" stroke-dasharray="3 3" />
        <text class="gpu-ts" :x="x(cross) + 5" :y="top + 28">M ≈ 3d²</text>

        <line class="gpu-cross" :x1="x(M)" :x2="x(M)" :y1="top" :y2="H - bottom" />
        <circle class="gpu-pt pa" :cx="x(M)" :cy="y(fNow)" r="4" />
        <circle class="gpu-pt pb" :cx="x(M)" :cy="y(sNow)" r="4" />
      </svg>
    </div>
    <figcaption>
      N = 4096。纵轴按前向传播统计，曲线上的台阶来自对 T_c 的取整。两条线约在 M = 3d² 处相交，这是计入常数后的结果，渐近式 N²d²/M 看不出来。论文证明的下界只说明曲线的最右端无法再低于虚线的量级，并不说明中段已经贴着下界。在图上移动指针或拖动滑块查看任意 M。
    </figcaption>
  </figure>
</template>
