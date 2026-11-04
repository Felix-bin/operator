<script setup>
import { computed, ref, useId } from 'vue'

// 舍入到 binary16：最近偶数，超过 65520 变成 inf
function f16(x) {
  if (!Number.isFinite(x) || x === 0) return x
  const sign = Math.sign(x)
  const a = Math.abs(x)
  if (a >= 65520) return sign * Infinity
  let e = Math.floor(Math.log2(a))
  if (2 ** e > a) e--
  else if (2 ** (e + 1) <= a) e++
  const ulp = 2 ** Math.max(e - 10, -24)
  const q = a / ulp
  let r = Math.floor(q)
  const frac = q - r
  if (frac > 0.5 || (frac === 0.5 && r % 2 === 1)) r++
  return sign * r * ulp
}

const FORMATS = {
  fp32: {
    name: 'FP32',
    rt: Math.fround,
    hi: Math.log(2 ** 128 * (1 - 2 ** -25)),
    sub: -126 * Math.LN2,
    lo: -150 * Math.LN2,
    span: 130
  },
  fp16: {
    name: 'FP16',
    rt: f16,
    hi: Math.log(65520),
    sub: -14 * Math.LN2,
    lo: -25 * Math.LN2,
    span: 30
  }
}

const PRESETS = [
  { name: '普通', text: '1, 2, 3, 4' },
  { name: '一项超界', text: '90, 1, 2, 3' },
  { name: '全部超界', text: '100, 100, 100' },
  { name: 'FP16 边界', text: '12, 1, 2' },
  { name: '全部很负', text: '-110, -120, -130' },
  { name: '次正规', text: '-100, -101' },
  { name: '带掩码', text: '5, 3, -inf' },
  { name: '整行掩码', text: '-inf, -inf' }
]

const format = ref('fp32')
const text = ref('90, 1, 2, 3')
const inputId = useId()

function parse(str) {
  const tokens = str.split(/[,，\s]+/).filter(Boolean)
  if (!tokens.length) return { error: '请至少输入一个分数。' }
  if (tokens.length > 12) return { error: '最多输入 12 个分数。' }
  const values = []
  for (const tok of tokens) {
    const t = tok.toLowerCase()
    if (t === 'inf' || t === '+inf') values.push(Infinity)
    else if (t === '-inf') values.push(-Infinity)
    else if (t === 'nan') values.push(NaN)
    else {
      const v = Number(t)
      if (Number.isNaN(v)) return { error: `“${tok}” 不是数字，可以写 inf、-inf 或 nan。` }
      values.push(v)
    }
  }
  return { values }
}

const parsed = computed(() => parse(text.value))
const lastValid = ref([90, 1, 2, 3])
const xs = computed(() => {
  if (parsed.value.values) lastValid.value = parsed.value.values
  return lastValid.value
})

const fmax = (a, b) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : Math.max(a, b))

const result = computed(() => {
  const f = FORMATS[format.value]
  const rt = f.rt
  const x = xs.value.map(rt)

  // 朴素：e^x / sum(e^x)
  const ne = x.map((v) => rt(Math.exp(v)))
  let nd = 0
  for (const e of ne) nd = rt(nd + e)
  const ny = ne.map((e) => rt(e / nd))

  // Safe：与 Kernel 相同，先求 m，再累加，最后乘 1/d
  let m = -Infinity
  for (const v of x) m = rt(fmax(m, v))
  const shifted = x.map((v) => rt(v - m))
  const se = shifted.map((v) => rt(Math.exp(v)))
  let sd = 0
  for (const e of se) sd = rt(sd + e)
  const inv = rt(1 / sd)
  const sy = se.map((e) => rt(e * inv))

  // 真值：双精度在减最大值之后计算
  const dm = xs.value.reduce(fmax, -Infinity)
  const de = xs.value.map((v) => Math.exp(v - dm))
  const dd = de.reduce((a, b) => a + b, 0)
  const ey = de.map((e) => e / dd)

  return { x, ne, nd, ny, m, shifted, se, sd, sy, ey }
})

const countNaN = (arr) => arr.filter((v) => Number.isNaN(v)).length
const naiveBad = computed(() => countNaN(result.value.ny))
const safeBad = computed(() => countNaN(result.value.sy))

const explain = computed(() => {
  const r = result.value
  const parts = []
  if (naiveBad.value === 0) parts.push('朴素的结果没有 NaN。')
  else if (r.nd === Infinity) parts.push('朴素：分母上溢成 inf，超界的项是 inf ÷ inf = NaN，其余项被除成 0。')
  else if (r.nd === 0) parts.push('朴素：所有项都舍入成 0，分母是 0，0 ÷ 0 = NaN。')
  else parts.push('朴素：输入本身就含 NaN。')
  if (safeBad.value === 0) parts.push(`Safe：m = ${fmt(r.m)}，最大的那一项 e⁰ = 1，所以分母不小于 1。`)
  else parts.push('Safe：x − m 已经是 NaN。输入含 +inf 或 NaN，或者整行都是 -inf，减最大值救不了这几种情况。')
  return parts.join(' ')
})

function fmt(v) {
  if (Number.isNaN(v)) return 'NaN'
  if (v === Infinity) return 'inf'
  if (v === -Infinity) return '-inf'
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1e5 || a < 1e-3) return v.toExponential(3)
  return String(Number(v.toPrecision(5)))
}
const bad = (v) => Number.isNaN(v) || !Number.isFinite(v)

// 数轴
const X0 = 96
const X1 = 704
const fmtInfo = computed(() => FORMATS[format.value])
const px = (v) => {
  const s = fmtInfo.value.span
  const c = Math.max(-s, Math.min(s, v))
  return X0 + ((c + s) / (2 * s)) * (X1 - X0)
}
const inAxis = (v) => Number.isFinite(v) && Math.abs(v) <= fmtInfo.value.span
const pointClass = (v) => {
  const f = fmtInfo.value
  if (Number.isNaN(v)) return 'sm-pt-bad'
  if (v > f.hi) return 'sm-pt-bad'
  if (v < f.lo) return 'sm-pt-zero'
  return 'sm-pt-ok'
}
const points = computed(() =>
  result.value.x
    .map((v, i) => ({ i, top: v, bottom: result.value.shifted[i] }))
    .filter((p) => !Number.isNaN(p.top) && !Number.isNaN(p.bottom))
)
const ariaLabel = computed(() => {
  const f = fmtInfo.value
  return `${f.name} 下 e 的 x 次方在 x 大于 ${f.hi.toFixed(2)} 时变成 inf，小于 ${f.lo.toFixed(2)} 时变成 0。减去最大值之后，所有指数都不大于 0。`
})
</script>

<template>
  <figure class="gpu-fig gpu-lab">
    <div class="gpu-tools" role="group" aria-label="浮点格式">
      <button
        v-for="(f, k) in FORMATS"
        :key="k"
        type="button"
        :class="{ active: format === k }"
        :aria-pressed="format === k"
        @click="format = k"
      >
        {{ f.name }}
      </button>
    </div>
    <div class="gpu-tools" role="group" aria-label="预设输入">
      <button
        v-for="p in PRESETS"
        :key="p.name"
        type="button"
        :class="{ active: text === p.text }"
        :aria-pressed="text === p.text"
        @click="text = p.text"
      >
        {{ p.name }}
      </button>
    </div>
    <div class="gpu-lab-row sm-input-row">
      <label :for="inputId">分数（最多 12 个，可写 inf、-inf、nan）</label>
      <input :id="inputId" v-model="text" class="sm-input" type="text" spellcheck="false" autocomplete="off" />
    </div>
    <p v-if="parsed.error" class="sm-error" role="alert">{{ parsed.error }}</p>

    <div class="gpu-read" aria-live="polite">
      <span><b :class="{ 'sm-bad': bad(result.nd) || result.nd === 0 }">{{ fmt(result.nd) }}</b> 朴素分母</span>
      <span><b :class="{ 'sm-bad': naiveBad > 0 }">{{ naiveBad }}</b> 个 NaN（朴素）</span>
      <span><b :class="{ 'sm-bad': bad(result.sd) }">{{ fmt(result.sd) }}</b> Safe 分母</span>
      <span><b :class="{ 'sm-bad': safeBad > 0 }">{{ safeBad }}</b> 个 NaN（Safe）</span>
    </div>
    <p class="sm-explain">{{ explain }}</p>

    <div class="gpu-scroll" tabindex="0">
      <svg class="w680" viewBox="0 0 720 190" role="img" :aria-label="ariaLabel">
        <text class="gpu-ts gpu-mid" :x="(X0 + px(fmtInfo.lo)) / 2" y="20">舍入成 0</text>
        <text class="gpu-ts gpu-mid" :x="(px(fmtInfo.sub) + px(fmtInfo.hi)) / 2" y="20">正规数范围</text>
        <text class="gpu-ts gpu-mid" :x="(px(fmtInfo.hi) + X1) / 2" y="20">变成 inf</text>

        <rect class="sm-zone-zero" :x="X0" y="30" :width="px(fmtInfo.lo) - X0" height="116" />
        <rect class="sm-zone-sub" :x="px(fmtInfo.lo)" y="30" :width="px(fmtInfo.sub) - px(fmtInfo.lo)" height="116" />
        <rect class="gpu-n" :x="px(fmtInfo.sub)" y="30" :width="px(fmtInfo.hi) - px(fmtInfo.sub)" height="116" />
        <rect class="sm-zone-inf" :x="px(fmtInfo.hi)" y="30" :width="X1 - px(fmtInfo.hi)" height="116" />

        <line class="gpu-ln gpu-dash" :x1="px(0)" y1="30" :x2="px(0)" y2="146" />

        <text class="gpu-t" x="10" y="64">朴素：x</text>
        <text class="gpu-t" x="10" y="124">Safe：x − m</text>

        <g v-for="p in points" :key="p.i">
          <line class="sm-shift" :x1="px(p.top)" y1="60" :x2="px(p.bottom)" y2="120" />
          <circle
            :class="[pointClass(p.top), { 'sm-pt-off': !inAxis(p.top) }]"
            :cx="px(p.top)"
            cy="60"
            r="5"
          />
          <circle
            :class="[pointClass(p.bottom), { 'sm-pt-off': !inAxis(p.bottom) }]"
            :cx="px(p.bottom)"
            cy="120"
            r="5"
          />
        </g>

        <text class="gpu-tk gpu-end" :x="px(fmtInfo.lo) + 2" y="164">{{ fmtInfo.lo.toFixed(2) }}</text>
        <text class="gpu-tk" :x="px(fmtInfo.sub) - 2" y="164">{{ fmtInfo.sub.toFixed(2) }}</text>
        <text class="gpu-tk gpu-mid" :x="px(0)" y="164">0</text>
        <text class="gpu-tk gpu-mid" :x="px(fmtInfo.hi)" y="164">{{ fmtInfo.hi.toFixed(2) }}</text>
        <text class="gpu-ts gpu-mid" :x="(X0 + X1) / 2" y="182">指数的值。线性刻度，范围 ±{{ fmtInfo.span }}</text>
      </svg>

      <table class="sm-table">
        <thead>
          <tr>
            <th>x</th>
            <th>e^x</th>
            <th>朴素 y</th>
            <th>x − m</th>
            <th>e^(x − m)</th>
            <th>Safe y</th>
            <th>真值</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(v, i) in result.x" :key="i">
            <td>{{ fmt(v) }}</td>
            <td :class="{ 'sm-bad': bad(result.ne[i]) }">{{ fmt(result.ne[i]) }}</td>
            <td :class="{ 'sm-bad': bad(result.ny[i]) }">{{ fmt(result.ny[i]) }}</td>
            <td :class="{ 'sm-bad': Number.isNaN(result.shifted[i]) }">{{ fmt(result.shifted[i]) }}</td>
            <td :class="{ 'sm-bad': bad(result.se[i]) }">{{ fmt(result.se[i]) }}</td>
            <td :class="{ 'sm-bad': bad(result.sy[i]) }">{{ fmt(result.sy[i]) }}</td>
            <td>{{ fmt(result.ey[i]) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <figcaption>
      输入先舍入到所选格式，e^x 用双精度算出后再舍入，分母按下标顺序累加，都在所选格式下进行。超出刻度的点贴在边上并画成空心。“真值”是双精度下减最大值后的结果，输入含 NaN 或 +inf 时也是 NaN。GPU 上的 expf 允许 2 ULP 误差，末位可能和这里不同。这里保留次正规数，用 -ftz=true 编译时，FP32 中小于 -87.34 的指数就会变成 0。
    </figcaption>
  </figure>
</template>

<style>
.sm-input-row label {
  flex: 1 1 100%;
}
.sm-input {
  flex: 1 1 200px;
  min-width: 0;
  padding: 6px 10px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  font-family: var(--vp-font-family-mono);
  font-size: 0.875rem;
  color: var(--vp-c-text-1);
}
.sm-input:focus-visible {
  outline: 2px solid var(--gpu-c);
  outline-offset: -2px;
}
.vp-doc .sm-error {
  margin: 0 16px 8px;
  font-size: 0.875rem;
  color: var(--vp-c-danger-1);
}
.vp-doc .sm-explain {
  margin: 4px 16px 0;
  font-size: 0.875rem;
  line-height: 1.6;
  color: var(--vp-c-text-2);
}
.gpu-read b.sm-bad,
.sm-table td.sm-bad {
  color: var(--vp-c-danger-1);
}
.sm-zone-inf,
.sm-zone-zero {
  fill: var(--vp-c-danger-soft);
  stroke: var(--vp-c-divider);
  stroke-width: 1.2;
}
.sm-zone-sub {
  fill: var(--gpu-n-fill);
  fill-opacity: 0.5;
  stroke: var(--vp-c-divider);
  stroke-width: 1.2;
}
.sm-shift {
  stroke: var(--vp-c-text-3);
  stroke-width: 1;
  stroke-dasharray: 3 3;
}
.sm-pt-ok {
  fill: var(--gpu-c);
  stroke: var(--gpu-c);
  stroke-width: 1.5;
}
.sm-pt-bad {
  fill: var(--vp-c-danger-1);
  stroke: var(--vp-c-danger-1);
  stroke-width: 1.5;
}
.sm-pt-zero {
  fill: var(--vp-c-text-3);
  stroke: var(--vp-c-text-3);
  stroke-width: 1.5;
}
circle.sm-pt-off {
  fill: var(--vp-c-bg-soft);
}
.gpu-page .vp-doc .sm-table {
  display: table;
  width: 100%;
  min-width: 620px;
  margin: 12px 0 0;
  border-collapse: collapse;
  font-family: var(--vp-font-family-mono);
  font-size: 0.8125rem;
  font-variant-numeric: tabular-nums;
}
.gpu-page .vp-doc .sm-table tr {
  background: transparent;
  border: 0;
}
.gpu-page .vp-doc .sm-table th,
.gpu-page .vp-doc .sm-table td {
  padding: 4px 8px;
  border: 1px solid var(--vp-c-divider);
  background: transparent;
  font-size: inherit;
  font-weight: 400;
  text-align: right;
  white-space: nowrap;
}
.gpu-page .vp-doc .sm-table th {
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
}
.gpu-page .vp-doc .sm-table td.sm-bad {
  color: var(--vp-c-danger-1);
}
</style>
