<script setup>
import { computed, ref, useId } from 'vue'

const KB = 1024
// warp_max[32] 与 block_result 各两份共 264 字节，smem 按 16 字节对齐，所以从第 272 字节开始（见 PTX）
const STATIC_BYTES = 272
// 驻留时每个 Block 的实际占用：按 128 字节取整再加 1 KiB。由 RTX 4060 上的占用率查询拟合，假设 A100 相同
const GRANULE = 128
const BLOCK_OVERHEAD = KB
const NO_OPTIN = 48 * KB
const PER_BLOCK = 163 * KB
const PER_SM = 164 * KB
// A100（compute capability 8.0）每个 SM 最多驻留 2048 个线程，见官方 Compute Capabilities 表
const MAX_THREADS = 2048
const SCALE_MAX = 280 * KB

const PRESETS = [1024, 4096, 8192, 12220, 12224, 32768, 41660, 41664, 65536]
const BLOCKS = [128, 256, 512, 1024]

const n = ref(4096)
const block = ref(256)
const sliderId = useId()

const slider = computed({
  get: () => Math.log2(n.value),
  set: (v) => {
    n.value = Math.max(4, Math.round(2 ** Number(v) / 4) * 4)
  }
})

const dyn = computed(() => n.value * 4)
const total = computed(() => dyn.value + STATIC_BYTES)
const optIn = computed(() => total.value > NO_OPTIN)
const fits = computed(() => total.value <= PER_BLOCK)
const footprint = computed(() => Math.ceil(total.value / GRANULE) * GRANULE + BLOCK_OVERHEAD)
const bySmem = computed(() => (fits.value ? Math.floor(PER_SM / footprint.value) : 0))
const byThreads = computed(() => Math.floor(MAX_THREADS / block.value))
const resident = computed(() => Math.min(bySmem.value, byThreads.value))
const threads = computed(() => resident.value * block.value)
const perThread = computed(() => Math.ceil(n.value / block.value))
const limitNote = computed(() =>
  bySmem.value > byThreads.value
    ? `共享内存够放 ${bySmem.value} 个 Block，但线程上限 ${MAX_THREADS} 只够 ${byThreads.value} 个。`
    : ''
)

const status = computed(() => {
  if (!fits.value) return `放不下。一个 Block 要 ${(total.value / KB).toFixed(2)} KiB，超过每个 Block 163 KB 的上限，启动会失败。`
  if (optIn.value) return `放得下，但已超过 48 KB，启动前必须调用 cudaFuncSetAttribute 声明。`
  return '放得下，也不需要额外声明。'
})

const X0 = 16
const X1 = 704
const px = (bytes) => X0 + (Math.min(bytes, SCALE_MAX) / SCALE_MAX) * (X1 - X0)
const chunks = computed(() => {
  const count = fits.value ? resident.value : 1
  return Array.from({ length: count }, (_, i) => ({
    i,
    x: px(i * footprint.value),
    w: px(footprint.value * (i + 1)) - px(footprint.value * i)
  }))
})
const ariaLabel = computed(
  () => `每个 SM 有 164 KB 共享内存。N 为 ${n.value} 时每个 Block 占 ${(total.value / KB).toFixed(2)} KiB，每个 SM 最多驻留 ${resident.value} 个 Block。`
)
</script>

<template>
  <figure class="gpu-fig gpu-lab">
    <div class="gpu-tools" role="group" aria-label="常用的 N">
      <button
        v-for="p in PRESETS"
        :key="p"
        type="button"
        :class="{ active: n === p }"
        :aria-pressed="n === p"
        @click="n = p"
      >
        N = {{ p }}
      </button>
    </div>
    <div class="gpu-tools" role="group" aria-label="Block 大小">
      <button
        v-for="b in BLOCKS"
        :key="b"
        type="button"
        :class="{ active: block === b }"
        :aria-pressed="block === b"
        @click="block = b"
      >
        Block {{ b }}
      </button>
    </div>
    <div class="gpu-lab-row">
      <label :for="sliderId">每行 N = <b>{{ n }}</b> 个 float</label>
      <input :id="sliderId" v-model.number="slider" type="range" min="8" max="16" step="0.01" />
    </div>
    <div class="gpu-read" aria-live="polite">
      <span><b>{{ dyn }}</b> 启动参数（字节）</span>
      <span><b>{{ (total / KB).toFixed(2) }}</b> 每 Block 占用（KiB）</span>
      <span><b>{{ resident }}</b> 每 SM 最多驻留 Block</span>
      <span><b>{{ threads }}</b> 对应线程</span>
    </div>
    <p class="sm-explain">{{ status }}{{ limitNote }}smem 下标范围 0~{{ n - 1 }}，每个线程暂存约 {{ perThread }} 个值。</p>
    <div class="gpu-scroll" tabindex="0">
      <svg class="w680" viewBox="0 0 720 110" role="img" :aria-label="ariaLabel">
        <rect class="gpu-n" :x="X0" y="34" :width="px(PER_SM) - X0" height="34" />
        <rect
          v-for="c in chunks"
          :key="c.i"
          :class="fits ? 'gpu-m' : 'sm-zone-inf'"
          :x="c.x"
          y="34"
          :width="c.w"
          height="34"
        />
        <line class="gpu-ln" :x1="px(NO_OPTIN)" y1="22" :x2="px(NO_OPTIN)" y2="76" />
        <text class="gpu-ts gpu-mid" :x="px(NO_OPTIN)" y="16">单 Block 48 KB：超过要声明</text>
        <line class="gpu-ln" :x1="px(PER_SM)" y1="22" :x2="px(PER_SM)" y2="76" />
        <text class="gpu-ts gpu-mid" :x="px(PER_SM)" y="16">SM 164 KB（单 Block 163 KB）</text>
        <text class="gpu-tk" :x="X0" y="92">0</text>
        <text class="gpu-tk gpu-mid" :x="px(PER_SM)" y="92">164 KB</text>
        <text class="gpu-tk gpu-end" :x="X1" y="92">280 KB</text>
      </svg>
    </div>
    <figcaption>
      是否放得下，看动态 N × 4 字节加静态 272 字节。琥珀色每段是一个驻留 Block 的占用：再按 128 字节取整，并多算 1 KiB。这条规则是在 RTX 4060 上用占用率查询拟合出来的，这里假设 A100 相同，没有在 A100 上验证。1 KB 按 1024 字节算。驻留数同时受共享内存和线程上限限制，A100 每个 SM 最多 2048 个线程。这里没有算寄存器，所以读数是上限，实际驻留可能更少。
    </figcaption>
  </figure>
</template>
