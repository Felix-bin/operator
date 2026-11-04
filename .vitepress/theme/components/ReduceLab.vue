<script setup>
import { computed, ref, useId } from 'vue'

const props = defineProps({
  versions: { type: Array, default: () => ['v0', 'v1', 'v2'] },
  initial: { type: String, default: '' },
  round: { type: Number, default: 1 },
  bank: { type: Boolean, default: false }
})

const BLOCK = 256
const WARP = 32
const up = [1, 2, 4, 8, 16, 32, 64, 128]
const defs = {
  v0: {
    name: 'V0 朴素树形',
    steps: up,
    code: (s) => `if (tid % ${2 * s} == 0)  smem[tid] += smem[tid + ${s}];`,
    read: (t, s) => (t % (2 * s) === 0 ? t : -1)
  },
  v1: {
    name: 'V1 strided index',
    steps: up,
    code: (s) => `index = tid * ${2 * s};  if (index < 256)  smem[index] += smem[index + ${s}];`,
    read: (t, s) => (t * 2 * s < BLOCK ? t * 2 * s : -1)
  },
  v2: {
    name: 'V2 步长反转',
    steps: [128, 64, 32, 16, 8, 4, 2, 1],
    code: (s) => `if (tid < ${s})  smem[tid] += smem[tid + ${s}];`,
    read: (t, s) => (t < s ? t : -1)
  }
}

function bankCounts(version, s, warp) {
  const counts = Array(WARP).fill(0)
  for (let lane = 0; lane < WARP; lane++) {
    const addr = defs[version].read(warp * WARP + lane, s)
    if (addr >= 0) counts[addr % WARP]++
  }
  return counts
}
function summarize(version, s) {
  let active = 0
  let diverged = 0
  let issued = 0
  let beats = 0
  let ways = 1
  for (let w = 0; w < BLOCK / WARP; w++) {
    let n = 0
    for (let lane = 0; lane < WARP; lane++) if (defs[version].read(w * WARP + lane, s) >= 0) n++
    active += n
    if (n === 0) continue
    issued++
    if (n < WARP) diverged++
    const most = Math.max(...bankCounts(version, s, w))
    beats += most
    ways = Math.max(ways, most)
  }
  return { active, diverged, issued, beats, ways }
}

const version = ref(props.initial || props.versions[0])
const roundIdx = ref(props.round)
const sliderId = useId()

const def = computed(() => defs[version.value])
const s = computed(() => def.value.steps[roundIdx.value - 1])
const now = computed(() => summarize(version.value, s.value))
const total = computed(() =>
  def.value.steps.reduce(
    (sum, step) => {
      const r = summarize(version.value, step)
      return { issued: sum.issued + r.issued, diverged: sum.diverged + r.diverged, beats: sum.beats + r.beats }
    },
    { issued: 0, diverged: 0, beats: 0 }
  )
)

const warps = computed(() =>
  Array.from({ length: BLOCK / WARP }, (_, w) => {
    const on = Array.from({ length: WARP }, (_, lane) => def.value.read(w * WARP + lane, s.value) >= 0)
    const n = on.filter(Boolean).length
    return { w, n, cells: on.map((v) => (v ? 'rd-on' : n ? 'rd-off' : 'rd-idle')) }
  })
)

const reads = computed(() => {
  const hit = new Set()
  for (let lane = 0; lane < WARP; lane++) {
    const addr = def.value.read(lane, s.value)
    if (addr >= 0) hit.add(addr)
  }
  return hit
})
const counts = computed(() => bankCounts(version.value, s.value, 0))
const addrRows = computed(() =>
  Array.from({ length: BLOCK / WARP }, (_, r) => ({
    label: `${r * WARP}~${r * WARP + WARP - 1}`,
    cells: Array.from({ length: WARP }, (_, b) => {
      const addr = r * WARP + b
      if (!reads.value.has(addr)) return ''
      return counts.value[b] > 1 ? 'rd-clash' : 'rd-hit'
    })
  }))
)
</script>

<template>
  <figure class="gpu-fig gpu-lab">
    <div v-if="versions.length > 1" class="gpu-tools">
      <button
        v-for="v in versions"
        :key="v"
        type="button"
        :class="{ active: version === v }"
        :aria-pressed="version === v"
        @click="version = v"
      >
        {{ defs[v].name }}
      </button>
    </div>
    <div class="gpu-lab-row">
      <label :for="sliderId">第 <b>{{ roundIdx }}</b> 轮，步长 s = <b>{{ s }}</b></label>
      <input :id="sliderId" v-model.number="roundIdx" type="range" min="1" max="8" />
    </div>
    <div class="gpu-read" aria-live="polite">
      <span><b>{{ now.active }}</b> 个活跃线程</span>
      <span><b>{{ now.diverged }}</b> 个 Warp 分化</span>
      <span v-if="bank"><b>{{ now.ways > 1 ? `${now.ways} 路` : '无' }}</b> Bank 冲突</span>
      <span>
        8 轮合计发射 {{ total.issued }} 次 Warp，其中 {{ total.diverged }} 次分化<template v-if="bank">，读取折算 {{ total.beats }} 拍</template>。
      </span>
    </div>
    <p class="gpu-rd-code"><code>{{ def.code(s) }}</code></p>
    <div class="gpu-legend gpu-chart-legend">
      <span><i class="rd-on"></i>执行 if 体</span>
      <span><i class="rd-off"></i>被掩码关闭，白占车道</span>
      <span><i class="rd-idle"></i>整个 Warp 跳过</span>
      <template v-if="bank">
        <span><i class="rd-hit"></i>Warp 0 读取的地址</span>
        <span><i class="rd-clash"></i>同一 Bank 被多次读取</span>
      </template>
    </div>
    <div class="gpu-scroll" tabindex="0">
      <div
        class="gpu-rd-grid"
        role="img"
        :aria-label="`${def.name}，步长 ${s}：${now.active} 个线程活跃，${now.diverged} 个 Warp 分化。`"
      >
        <span class="head"></span>
        <span v-for="l in [0, 8, 16, 24]" :key="l" class="head span8">车道 {{ l }}</span>
        <span class="head"></span>
        <template v-for="row in warps" :key="row.w">
          <span class="label">Warp {{ row.w }}</span>
          <i v-for="(c, i) in row.cells" :key="i" :class="c"></i>
          <span class="tail">{{ row.n }}/32</span>
        </template>
      </div>
      <div
        v-if="bank"
        class="gpu-rd-grid gpu-rd-bank"
        role="img"
        :aria-label="`Warp 0 读取 smem 时，单个 Bank 最多被 ${now.ways} 个线程访问。`"
      >
        <span class="head"></span>
        <span v-for="l in [0, 8, 16, 24]" :key="l" class="head span8">Bank {{ l }}</span>
        <span class="head"></span>
        <template v-for="row in addrRows" :key="row.label">
          <span class="label">{{ row.label }}</span>
          <i v-for="(c, i) in row.cells" :key="i" :class="c || 'rd-blank'"></i>
          <span class="tail"></span>
        </template>
        <span class="label">线程数</span>
        <span v-for="(n, i) in counts" :key="i" class="count" :class="{ clash: n > 1 }">{{ n || '' }}</span>
        <span class="tail"></span>
      </div>
    </div>
    <figcaption>
      <template v-if="bank">
        上方每行是一个 Warp，下方每行是 smem 中连续的 32 个地址，每列是一个 Bank。同一列出现多个实心格，就是这些线程要排队。这里统计 Warp 0 读取第一个操作数的那条指令，读第二个操作数时冲突路数相同。“读取折算”把每个活跃 Warp 的读取计为其最大冲突路数，只是访存的粗略模型，不代表实际耗时。
      </template>
      <template v-else>
        每行是一个 Warp，每格是一条车道，Block 为 256 个线程。只要 Warp 里有一条车道要执行，整组就要占用一次发射。数字由代码中的条件直接算出。
      </template>
    </figcaption>
  </figure>
</template>
