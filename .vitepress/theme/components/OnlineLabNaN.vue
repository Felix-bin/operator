<script setup>
import { computed, ref, useId } from 'vue'

const FLT_MAX = 3.4028234663852886e38
const fr = Math.fround
const expf = (x) => fr(Math.exp(x))
const fmaxf = (a, b) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : Math.max(a, b))

const mode = ref('second')
const block = ref(256)
const filled = ref(8)
const init = ref('inf')
const sliderId = useId()
const blocks = [32, 64, 128, 256, 512, 1024]

const m0 = computed(() => (init.value === 'inf' ? -Infinity : -FLT_MAX))
// 固定的示例数值，只用来观察 NaN 怎么走，不代表某组真实输入
const sampleM = (l) => fr(2 + 3 * Math.sin(l * 1.7))
const sampleD = (l) => fr(20 + 15 * Math.cos(l * 0.9))

const start = computed(() => {
  const s = []
  for (let l = 0; l < 32; l++) {
    if (mode.value === 'second') {
      s.push(l < block.value / 32 ? [sampleM(l), sampleD(l)] : [m0.value, 0])
    } else {
      s.push(l < filled.value ? [sampleM(l), 1] : [m0.value, 0])
    }
  }
  return s
})

// 严格按 warpReduceOnline：每轮所有车道同时读取上一轮的值，越界读自己
const rounds = computed(() => {
  const out = [{ label: '初值', off: 0, s: start.value }]
  let s = start.value
  for (let off = 16; off > 0; off >>= 1) {
    const old = s
    s = old.map(([m, d], l) => {
      const [m2, d2] = l + off <= 31 ? old[l + off] : [m, d]
      const n = fmaxf(m, m2)
      return [n, fr(fr(d * expf(fr(m - n))) + fr(d2 * expf(fr(m2 - n))))]
    })
    out.push({ label: `offset ${off}`, off, s })
  }
  return out
})

const rows = computed(() =>
  rounds.value.map((r, ri) => {
    const nextOff = ri < 5 ? rounds.value[ri + 1].off : 0
    return {
      label: r.label,
      nan: r.s.filter(([, d]) => Number.isNaN(d)).length,
      cells: r.s.map(([, d], l) => ({
        cls: Number.isNaN(d) ? 'onl-nan' : d !== 0 ? 'onl-real' : 'onl-empty',
        src: nextOff !== 0 && l === nextOff
      }))
    }
  })
)

const result = computed(() => rounds.value[5].s[0])
const expected = computed(() => {
  let m = -Infinity
  let total = 0
  for (const [mi, di] of start.value) if (di !== 0) m = Math.max(m, mi)
  if (m === -Infinity) return { m: null, d: 0 }
  for (const [mi, di] of start.value) if (di !== 0) total += di * Math.exp(mi - m)
  return { m, d: total }
})
const isNaNResult = computed(() => Number.isNaN(result.value[1]))

// 沿着真实数据流往回找：车道 0 的 NaN 最初是哪一轮、哪两条车道产生的
const story = computed(() => {
  if (!isNaNResult.value) {
    return null
  }
  const hops = []
  let lane = 0
  let r = 5
  while (r > 0) {
    if (Number.isNaN(rounds.value[r - 1].s[lane][1])) {
      r--
      continue
    }
    const off = rounds.value[r].off
    const src = lane + off
    if (src <= 31 && Number.isNaN(rounds.value[r - 1].s[src][1])) {
      hops.push(`第 ${r} 轮车道 ${lane} 读到车道 ${src} 的 NaN`)
      lane = src
      r--
      continue
    }
    hops.push(`第 ${r} 轮车道 ${lane} 读车道 ${src}，两边的 m 都是 −INF，−INF − (−INF) 得 NaN`)
    break
  }
  return hops.reverse().join('；') + '。'
})

const fmtD = (v) => (Number.isNaN(v) ? 'NaN' : v.toFixed(4))
const condition = computed(() => {
  if (init.value !== 'inf') return '换成有限的初值后，空车道之间的减法得 0，乘 d = 0 还是 0，空车道成了真正的单位元。'
  if (mode.value === 'second') {
    const nw = block.value / 32
    return nw < 16
      ? `num_warps = ${nw}，小于 16，车道 ${nw}~15 和它们第一轮读取的车道 ${nw + 16}~31 都是空的。`
      : `num_warps = ${nw}，不小于 16。空车道虽然也会变成 NaN，但车道 0 读取它们时它们还没坏，后面也不会再读。`
  }
  return filled.value < 16
    ? `只有 ${filled.value} 条车道有数据，少于 16 条，车道 ${filled.value}~15 第一轮读到的也是空车道。`
    : `有 ${filled.value} 条车道有数据，车道 0~15 都不空，通往车道 0 的路径上不会出现两个 −INF 相减。`
})
</script>

<template>
  <figure class="gpu-fig gpu-lab onl-nan-lab">
    <div class="gpu-tools">
      <button type="button" :class="{ active: mode === 'second' }" :aria-pressed="mode === 'second'" @click="mode = 'second'">第二级规约（Warp 0）</button>
      <button type="button" :class="{ active: mode === 'first' }" :aria-pressed="mode === 'first'" @click="mode = 'first'">第一级：线程没分到元素</button>
    </div>
    <div class="gpu-tools online-tools2">
      <button type="button" :class="{ active: init === 'inf' }" :aria-pressed="init === 'inf'" @click="init = 'inf'">原稿：m = -INFINITY</button>
      <button type="button" :class="{ active: init === 'max' }" :aria-pressed="init === 'max'" @click="init = 'max'">修复：m = -FLT_MAX</button>
    </div>
    <div v-if="mode === 'second'" class="gpu-tools online-tools2" role="group" aria-label="Block 大小">
      <button v-for="b in blocks" :key="b" type="button" :class="{ active: block === b }" :aria-pressed="block === b" @click="block = b">Block {{ b }}</button>
    </div>
    <div v-else class="gpu-lab-row">
      <label :for="sliderId">这个 Warp 有 <b>{{ filled }}</b> 条车道分到了元素</label>
      <input :id="sliderId" v-model.number="filled" type="range" min="0" max="32" />
    </div>

    <div class="gpu-read online-read" aria-live="polite">
      <span>车道 0 的 d = <b :class="{ 'online-bad-text': isNaNResult }">{{ fmtD(result[1]) }}</b></span>
      <span>正确值 <b>{{ fmtD(expected.d) }}</b></span>
    </div>
    <p class="online-note onl-story">{{ story || condition }}<template v-if="story"> {{ condition }}</template></p>

    <div class="gpu-legend gpu-chart-legend">
      <span><i class="onl-real"></i>有数据</span>
      <span><i class="onl-empty"></i>空车道（d = 0）</span>
      <span><i class="onl-nan"></i>d 是 NaN</span>
      <span><i class="onl-src"></i>车道 0 下一轮要读的车道</span>
    </div>
    <div class="gpu-scroll" tabindex="0">
      <div class="gpu-rd-grid onl-grid" role="img" :aria-label="`五轮合并后车道 0 的 d 是 ${fmtD(result[1])}，正确值是 ${fmtD(expected.d)}。`">
        <span class="head"></span>
        <span v-for="l in [0, 8, 16, 24]" :key="l" class="head span8">车道 {{ l }}</span>
        <span class="head"></span>
        <template v-for="row in rows" :key="row.label">
          <span class="label">{{ row.label }}</span>
          <i v-for="(c, i) in row.cells" :key="i" :class="[c.cls, { 'onl-src': c.src }]"></i>
          <span class="tail">{{ row.nan }}/32</span>
        </template>
      </div>
    </div>
    <figcaption>
      每行是一轮合并后 32 条车道的状态，最右边一列是 NaN 的车道数。规则和 <code>warpReduceOnline</code> 一致：同一轮所有车道读上一轮的值，越界时读自己。各车道的 m、d 用固定的示例数值，所有状态按单精度实时算出。
    </figcaption>
  </figure>
</template>

<style>
.onl-grid {
  grid-template-columns: 68px repeat(32, minmax(0, 1fr)) 44px;
}
.onl-grid i {
  height: 18px;
  box-sizing: border-box;
}
.onl-real,
.gpu-legend i.onl-real {
  background: var(--gpu-c);
  border-color: var(--gpu-c);
}
.onl-empty,
.gpu-legend i.onl-empty {
  background: var(--gpu-n-fill);
  border-color: var(--vp-c-divider);
}
.onl-nan,
.gpu-legend i.onl-nan {
  background: repeating-linear-gradient(135deg, var(--online-bad) 0 3px, transparent 3px 5px);
  border-color: var(--online-bad);
}
.onl-grid i.onl-src {
  outline: 2px solid var(--vp-c-text-1);
  outline-offset: -1px;
}
.gpu-legend i.onl-src {
  background: transparent;
  border: 2px solid var(--vp-c-text-1);
}
.vp-doc .onl-story {
  min-height: 4.8em;
}
</style>
