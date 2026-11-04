<script setup>
// 一行分数按 2 个一块流式处理，数值全部现算，最后一行是一次性算出的 softmax
const xs = [1, 0.2, 3, 0.5, 2, 4]
const vs = [1, 2, 3, 4, 5, 6]
const B = 2
const cellW = 50
const cellH = 30
const cx = (i) => 92 + i * 54 + Math.floor(i / B) * 14
const statX = [492, 552, 622, 708]

const rows = []
let m = -Infinity
let l = 0
let o = 0
for (let b = 0; b < xs.length; b += B) {
  const end = b + B
  const mNew = Math.max(m, ...xs.slice(b, end))
  const alpha = Math.exp(m - mNew)
  const lNew = alpha * l + xs.slice(b, end).reduce((s, x) => s + Math.exp(x - mNew), 0)
  const add = xs.slice(b, end).reduce((s, x, k) => s + Math.exp(x - mNew) * vs[b + k], 0)
  const oNew = (o * alpha * l) / lNew + add / lNew
  rows.push({
    label: `第 ${b / B + 1} 块后`,
    cells: xs.map((x, i) => ({
      cls: i >= end ? 'gpu-n' : i >= b ? 'gpu-c' : 'gpu-m',
      t: i >= end ? '–' : (Math.exp(x - mNew) / lNew).toFixed(3)
    })),
    stats: [mNew.toFixed(2), alpha.toFixed(3), lNew.toFixed(3), oNew.toFixed(3)]
  })
  m = mNew
  l = lNew
  o = oNew
}
const sum = xs.reduce((s, x) => s + Math.exp(x), 0)
const direct = xs.map((x) => Math.exp(x) / sum)
const directO = direct.reduce((s, p, i) => s + p * vs[i], 0)
const rowY = (r) => 52 + r * 40
</script>

<template>
  <svg class="w680" viewBox="0 0 800 216" role="img" :aria-label="`一行 6 个分数分 3 块处理。每读一块，都把已有权重按新的最大值和分母重新缩放。最终 O 为 ${o.toFixed(3)}，与一次性计算的 ${directO.toFixed(3)} 相同。`">
    <text class="gpu-ts" x="16" y="18">分数 x</text>
    <text class="gpu-ts" x="16" y="38">值 v</text>
    <g v-for="(x, i) in xs" :key="i">
      <text class="gpu-tk gpu-mid" :x="cx(i) + cellW / 2" y="18">{{ x.toFixed(1) }}</text>
      <text class="gpu-tk gpu-mid" :x="cx(i) + cellW / 2" y="38">{{ vs[i] }}</text>
    </g>
    <text v-for="(t, i) in ['m', 'α', 'l', 'O']" :key="t" class="gpu-t gpu-tb gpu-mid" :x="statX[i]" y="38">{{ t }}</text>
    <g v-for="(r, ri) in rows" :key="r.label">
      <text class="gpu-t" x="16" :y="rowY(ri) + 20">{{ r.label }}</text>
      <g v-for="(c, i) in r.cells" :key="i">
        <rect :class="c.cls" :x="cx(i)" :y="rowY(ri)" :width="cellW" :height="cellH" />
        <text class="gpu-tk gpu-mid" :x="cx(i) + cellW / 2" :y="rowY(ri) + 20">{{ c.t }}</text>
      </g>
      <text v-for="(t, i) in r.stats" :key="i" class="gpu-tk gpu-mid" :x="statX[i]" :y="rowY(ri) + 20">{{ t }}</text>
    </g>
    <text class="gpu-t" x="16" :y="rowY(3) + 20">一次算完</text>
    <g v-for="(p, i) in direct" :key="'d' + i">
      <rect class="gpu-n" :x="cx(i)" :y="rowY(3)" :width="cellW" :height="cellH" />
      <text class="gpu-tk gpu-mid" :x="cx(i) + cellW / 2" :y="rowY(3) + 20">{{ p.toFixed(3) }}</text>
    </g>
    <text class="gpu-tk gpu-mid" :x="statX[3]" :y="rowY(3) + 20">{{ directO.toFixed(3) }}</text>
  </svg>
</template>
