<script setup>
import { computed, ref, useId } from 'vue'

const FLT_MAX = 3.4028234663852886e38
const fr = Math.fround
const expf = (x) => fr(Math.exp(x))
// fmaxf：有一边是 NaN 时返回另一边
const fmaxf = (a, b) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : Math.max(a, b))

const presets = [
  { id: 'mixed', name: '中途出现新高', xs: [1, 0.5, 3, 2, 4.5, -1, 4.5, 2.5] },
  { id: 'asc', name: '升序，每步都修正', xs: [-1, 0, 1, 2, 3, 4, 5, 6] },
  { id: 'first', name: '最大值在最前', xs: [6, 2, 3, 1, 0, 2, -1, 4] },
  { id: 'mask', name: '第一个元素被掩码', xs: [-Infinity, 1, -Infinity, 3, 2, -Infinity, 4, 0] }
]
const fmtInput = (v) => (v === -Infinity ? '-inf' : v === Infinity ? 'inf' : String(v))

const presetId = ref('mixed')
const xs = ref(presets[0].xs.map(fr))
const text = ref(presets[0].xs.map(fmtInput).join(', '))
const inputError = ref('')
const init = ref('inf')
const k = ref(0)
const fieldId = useId()

function usePreset(p) {
  presetId.value = p.id
  xs.value = p.xs.map(fr)
  text.value = p.xs.map(fmtInput).join(', ')
  inputError.value = ''
  k.value = 0
}
function parseInput() {
  const parts = text.value.split(/[\s,，]+/).filter(Boolean)
  const vals = parts.map((p) => {
    const t = p.toLowerCase().replace('−', '-').replace('∞', 'inf')
    if (t === '-inf') return -Infinity
    return Number(t)
  })
  if (!vals.length || vals.some((v) => Number.isNaN(v) || v === Infinity)) {
    inputError.value = '只能填数字或 -inf，用逗号隔开。'
    return
  }
  if (vals.length > 12) {
    inputError.value = '最多 12 个元素。'
    return
  }
  inputError.value = ''
  presetId.value = ''
  xs.value = vals.map(fr)
  k.value = 0
}

const m0 = computed(() => (init.value === 'inf' ? -Infinity : -FLT_MAX))

// 逐步递推，和 Kernel 一样用单精度
const trace = computed(() => {
  const out = []
  let m = m0.value
  let d = 0
  xs.value.forEach((x, j) => {
    const mNew = fmaxf(m, x)
    const factor = expf(fr(m - mNew))
    const scaled = fr(d * factor)
    const term = expf(fr(x - mNew))
    const dNew = fr(scaled + term)
    out.push({ j, x, mOld: m, dOld: d, mNew, factor, scaled, term, d: dNew, isNew: mNew > m })
    m = mNew
    d = dNew
  })
  return out
})

// 从头直接计算前 k 个元素，用双精度做参照
function direct(count) {
  const part = xs.value.slice(0, count).filter((v) => v !== -Infinity)
  if (!part.length) return { m: -Infinity, d: 0 }
  const m = Math.max(...part)
  return { m, d: part.reduce((s, v) => s + Math.exp(v - m), 0) }
}

const n = computed(() => xs.value.length)
const cur = computed(() => (k.value > 0 ? trace.value[k.value - 1] : null))
const state = computed(() => (cur.value ? { m: cur.value.mNew, d: cur.value.d } : { m: m0.value, d: 0 }))
const ref_ = computed(() => direct(k.value))
const isNaNNow = computed(() => Number.isNaN(state.value.d))
const diff = computed(() => Math.abs(state.value.d - ref_.value.d))

const maxD = computed(() => {
  let top = 1
  for (let i = 1; i <= n.value; i++) top = Math.max(top, direct(i).d)
  return top
})
const pct = (v) => `${(v / maxD.value) * 100}%`

// 上一步结束时每个元素的贡献，以及这一步之后的贡献
function contribs(count, m) {
  return xs.value.slice(0, count).map((x, j) => ({ j, v: x === -Infinity ? 0 : Math.exp(x - m) }))
}
const before = computed(() => {
  if (!cur.value || isNaNNow.value || cur.value.dOld === 0) return []
  return contribs(k.value - 1, cur.value.mOld)
})
const after = computed(() => {
  if (!cur.value || isNaNNow.value) return []
  return contribs(k.value, cur.value.mNew).map((c) => ({ ...c, fresh: c.j === k.value - 1 }))
})

function fmt(v) {
  if (Number.isNaN(v)) return 'NaN'
  if (v === -Infinity) return '−∞'
  if (v === Infinity) return '+∞'
  if (v <= -1e30) return '−FLT_MAX'
  return v.toFixed(4)
}
const short = (v) => (v === -Infinity ? '−∞' : Number(v.toFixed(2)).toString().replace('-', '−'))

const formula = computed(() => {
  const c = cur.value
  if (!c) return `初始状态：m = ${fmt(m0.value)}，d = 0`
  return `d = ${fmt(c.dOld)} × ${fmt(c.factor)} + ${fmt(c.term)} = ${fmt(c.d)}`
})
const note = computed(() => {
  const c = cur.value
  if (!c) return '点“下一步”读入第一个元素。'
  if (isNaNNow.value) {
    if (Number.isNaN(c.dOld)) return 'NaN 一旦出现就会一直传下去，后面每一步都是 NaN。'
    return `旧 m 和新 m 都是 −∞，m − m_new 是 −∞ − (−∞)，得 NaN。d = 0 也救不了它，因为 0 × NaN 仍是 NaN。把 m₀ 换成 −FLT_MAX 再试一次。`
  }
  if (c.dOld === 0 && !c.isNew) return `这个元素是 −∞，贡献为 0，状态仍然是“空”。`
  if (c.dOld === 0) return `这是第一个有效元素：旧的和是 0，修正因子只是乘在 0 上。`
  if (c.isNew) return `${short(c.x)} 超过了旧最大值 ${short(c.mOld)}。之前的每一项都要改用新基准，整体乘以 e^(${short(c.mOld)} − ${short(c.mNew)}) = ${fmt(c.factor)}。`
  return `${short(c.x)} 没有超过当前最大值 ${short(c.mOld)}，修正因子是 e⁰ = 1，公式退化成直接累加。代码里 expf 仍然要算。`
})

const outputs = computed(() => {
  if (k.value !== n.value || !n.value) return null
  const { m, d } = state.value
  const safe = direct(n.value)
  const rows = xs.value.map((x, j) => ({
    j,
    online: fr(expf(fr(x - m)) * fr(1 / d)),
    safe: x === -Infinity ? 0 : Math.exp(x - safe.m) / safe.d
  }))
  const worst = rows.reduce((w, r) => Math.max(w, Math.abs(r.online - r.safe)), 0)
  return { rows, worst }
})
</script>

<template>
  <figure class="gpu-fig gpu-lab online-lab">
    <div class="gpu-tools">
      <button
        v-for="p in presets"
        :key="p.id"
        type="button"
        :class="{ active: presetId === p.id }"
        :aria-pressed="presetId === p.id"
        @click="usePreset(p)"
      >
        {{ p.name }}
      </button>
    </div>
    <div class="gpu-lab-row online-field">
      <label :for="fieldId">输入</label>
      <input :id="fieldId" v-model="text" type="text" spellcheck="false" @change="parseInput" @keyup.enter="parseInput" />
      <span v-if="inputError" class="online-error" role="alert">{{ inputError }}</span>
    </div>
    <div class="gpu-tools online-tools2">
      <button type="button" :class="{ active: init === 'inf' }" :aria-pressed="init === 'inf'" @click="init = 'inf'">m₀ = −∞（论文）</button>
      <button type="button" :class="{ active: init === 'max' }" :aria-pressed="init === 'max'" @click="init = 'max'">m₀ = −FLT_MAX（本文修复）</button>
    </div>
    <div class="gpu-tools online-tools2">
      <button type="button" class="online-step" :disabled="k >= n" @click="k++">下一步</button>
      <button type="button" :disabled="k <= 0" @click="k--">上一步</button>
      <button type="button" :disabled="k >= n" @click="k = n">走到最后</button>
      <button type="button" :disabled="k <= 0" @click="k = 0">重置</button>
      <span class="online-progress">已读入 {{ k }} / {{ n }}</span>
    </div>

    <div class="online-cells" :style="{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }" role="img" :aria-label="`输入共 ${n} 个元素，已读入 ${k} 个。`">
      <div
        v-for="(x, j) in xs"
        :key="j"
        class="cell"
        :class="{ cur: j === k - 1, done: j < k - 1 }"
      >
        <span class="val">{{ short(x) }}</span>
        <span class="tag">{{ k > 0 && !isNaNNow && x === state.m && j < k ? 'max' : `x${j}` }}</span>
      </div>
    </div>

    <div class="gpu-read online-read" aria-live="polite">
      <span>m = <b>{{ fmt(state.m) }}</b></span>
      <span>d = <b :class="{ 'online-bad-text': isNaNNow }">{{ fmt(state.d) }}</b></span>
      <span v-if="cur">修正因子 <b>{{ fmt(cur.factor) }}</b></span>
    </div>
    <p class="online-formula"><code>{{ formula }}</code></p>

    <div class="online-bars">
      <div class="row">
        <span class="name">修正前</span>
        <div class="track">
          <i v-for="c in before" :key="c.j" class="seg old" :style="{ width: pct(c.v) }"><em v-if="c.v / maxD > 0.075">x{{ c.j }}</em></i>
        </div>
      </div>
      <div class="row">
        <span class="name">修正并加入后</span>
        <div class="track">
          <i v-for="c in after" :key="c.j" class="seg" :class="c.fresh ? 'fresh' : 'old'" :style="{ width: pct(c.v) }"><em v-if="c.v / maxD > 0.075">x{{ c.j }}</em></i>
        </div>
      </div>
    </div>
    <p class="online-note">{{ note }}</p>

    <div class="gpu-read online-read online-check">
      <span>从头直接计算 d = <b>{{ fmt(ref_.d) }}</b></span>
      <span>
        <template v-if="isNaNNow">递推结果已是 NaN，无法比较</template>
        <template v-else>和递推结果相差 {{ diff.toExponential(2) }}</template>
      </span>
    </div>

    <div v-if="outputs" class="gpu-scroll" tabindex="0">
      <table class="online-out">
        <tbody>
          <tr>
            <th scope="row">第二遍输出 y</th>
            <td v-for="r in outputs.rows" :key="r.j">{{ fmt(r.online) }}</td>
          </tr>
          <tr>
            <th scope="row">三遍 Safe 结果</th>
            <td v-for="r in outputs.rows" :key="r.j">{{ fmt(r.safe) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p v-if="outputs" class="online-note">
      <template v-if="Number.isNaN(outputs.rows[0].online)">这一行的输出全部是 NaN。</template>
      <template v-else>两种算法的输出最大相差 {{ outputs.worst.toExponential(2) }}。这个差距来自单精度舍入，公式本身是等价的。</template>
    </p>

    <figcaption>
      条形的总长就是 d，每一段是一个元素的贡献 e^(x − m)。递推按单精度计算，直接计算按双精度，都由当前输入实时算出。可以改写上面的输入，负无穷写作 -inf。
    </figcaption>
  </figure>
</template>

<style>
.gpu-page {
  --online-bad: #b3261e;
}
.dark .gpu-page {
  --online-bad: #f2867e;
}
.online-bad-text {
  color: var(--online-bad) !important;
}
.online-tools2 {
  padding-top: 8px;
  align-items: center;
}
.online-lab button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.online-lab .gpu-tools button.online-step:not(:disabled) {
  border-color: var(--gpu-ca);
  background: var(--vp-c-brand-soft);
  font-weight: 600;
}
.online-progress {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
}
.online-field {
  padding-bottom: 0;
}
.online-field input {
  flex: 1 1 220px;
  min-width: 0;
  padding: 3px 8px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  font-family: var(--vp-font-family-mono);
  font-size: 0.8125rem;
}
.online-field input:focus-visible {
  outline: 2px solid var(--gpu-c);
  outline-offset: -2px;
}
.online-error {
  font-size: 0.8125rem;
  color: var(--online-bad);
}
.online-cells {
  display: grid;
  gap: 2px;
  margin: 14px 16px 12px;
}
.online-cells .cell {
  min-width: 0;
  padding: 5px 0 3px;
  border: 1px solid var(--vp-c-divider);
  background: var(--gpu-n-fill);
  text-align: center;
  line-height: 1.25;
}
.online-cells .cell.done {
  border-color: var(--gpu-c);
  background: var(--gpu-c-fill);
}
.online-cells .cell.cur {
  border-color: var(--gpu-m);
  background: var(--gpu-m-fill);
}
.online-cells .val {
  display: block;
  font-size: 0.8125rem;
  font-variant-numeric: tabular-nums;
}
.online-cells .tag {
  display: block;
  font-size: 0.6875rem;
  color: var(--vp-c-text-2);
}
.online-read {
  padding-bottom: 2px;
}
.vp-doc .online-formula {
  margin: 4px 16px 10px;
  font-size: 0.8125rem;
  line-height: 1.6;
}
.online-bars {
  margin: 0 16px;
  font-size: 0.8125rem;
}
.online-bars .row {
  display: grid;
  grid-template-columns: 88px minmax(0, 1fr);
  gap: 8px;
  align-items: center;
  margin-bottom: 6px;
}
.online-bars .name {
  color: var(--vp-c-text-2);
}
.online-bars .track {
  display: flex;
  height: 26px;
  overflow: hidden;
  box-shadow: inset 0 0 0 1px var(--vp-c-divider);
}
.online-bars .seg {
  display: flex;
  flex: none;
  align-items: center;
  justify-content: center;
  box-sizing: border-box;
  min-width: 0;
  overflow: hidden;
  border-right: 2px solid var(--vp-c-bg-soft);
  font-size: 0.6875rem;
  font-style: normal;
  transition: width 0.25s ease;
}
.online-bars .seg em {
  font-style: normal;
}
.online-bars .seg.old {
  background: var(--gpu-c-fill);
}
.online-bars .seg.fresh {
  background: var(--gpu-m-fill);
}
@media (prefers-reduced-motion: reduce) {
  .online-bars .seg {
    transition: none;
  }
}
.vp-doc .online-note {
  margin: 8px 16px;
  min-height: 3em;
  font-size: 0.875rem;
  line-height: 1.6;
}
.online-check {
  padding-top: 0;
  font-size: 0.8125rem;
}
.online-check b {
  font-size: 1rem;
}
.gpu-scroll.online-bars-wrap {
  padding: 0;
}
.online-bars-wrap .gpu-bars {
  min-width: 560px;
}
.gpu-page .vp-doc .online-out {
  display: table;
  width: 100%;
  min-width: 560px;
  margin: 0;
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
}
.gpu-page .vp-doc .online-out th,
.gpu-page .vp-doc .online-out td {
  padding: 4px 6px;
  text-align: right;
  white-space: nowrap;
}
.gpu-page .vp-doc .online-out th {
  text-align: left;
  font-weight: 600;
}
</style>
