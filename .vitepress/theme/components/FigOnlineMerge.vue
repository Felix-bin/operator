<script setup>
const fr = Math.fround
const expf = (x) => fr(Math.exp(x))
const xs = [1, 0.5, 3, 2, 4.5, -1, 0, 2.5].map(fr)

function merge(a, b) {
  const m = Math.max(a.m, b.m)
  return { m, d: fr(fr(a.d * expf(fr(a.m - m))) + fr(b.d * expf(fr(b.m - m)))) }
}

const BW = 84
const BH = 40
const leafX = (i) => 52 + i * 94
const levelY = [20, 96, 172, 248]

const levels = [xs.map((x, i) => ({ m: x, d: 1, cx: leafX(i), y: levelY[0] }))]
const links = []
for (let l = 1; l < 4; l++) {
  const prev = levels[l - 1]
  const cur = []
  for (let i = 0; i < prev.length; i += 2) {
    const a = prev[i]
    const b = prev[i + 1]
    const node = { ...merge(a, b), cx: (a.cx + b.cx) / 2, y: levelY[l] }
    cur.push(node)
    for (const c of [a, b]) {
      links.push({
        d: `M${c.cx} ${c.y + BH} L${node.cx + (c === a ? -12 : 12)} ${node.y - 1}`,
        scaled: c.m < node.m
      })
    }
  }
  levels.push(cur)
}
const nodes = levels.flat()
const root = levels[3][0]

let sm = -Infinity
let sd = 0
for (const x of xs) {
  const n = Math.max(sm, x)
  sd = fr(fr(sd * expf(fr(sm - n))) + expf(fr(x - n)))
  sm = n
}
const same = sd === root.d
const f = (v) => v.toFixed(4)
const fm = (v) => String(Number(v.toFixed(2))).replace('-', '−')
</script>

<template>
  <svg class="w680" viewBox="0 0 762 346" role="img" :aria-label="`8 个元素两两合并，三层后得到 m = ${fm(root.m)}，d = ${f(root.d)}，与串行递推结果一致。`">
    <defs>
      <marker id="online-merge-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor" /></marker>
    </defs>
    <path v-for="(l, i) in links" :key="i" class="gpu-ln" :class="{ 'gpu-dash': l.scaled }" :d="l.d" marker-end="url(#online-merge-arrow)" />
    <g v-for="(n, i) in nodes" :key="i">
      <rect :class="i < 8 ? 'gpu-m' : 'gpu-c'" :x="n.cx - BW / 2" :y="n.y" :width="BW" :height="BH" />
      <text class="gpu-tk gpu-mid" :x="n.cx" :y="n.y + 16">m = {{ fm(n.m) }}</text>
      <text class="gpu-tk gpu-mid" :x="n.cx" :y="n.y + 32">d = {{ f(n.d) }}</text>
    </g>
    <text class="gpu-t gpu-mid" x="381" y="314">串行递推同一组数：m = {{ fm(sm) }}，d = {{ f(sd) }}</text>
    <text class="gpu-ts gpu-mid" x="381" y="334">
      {{ same ? '和树形合并的单精度结果完全相同' : `和树形合并相差 ${Math.abs(sd - root.d).toExponential(2)}，来自单精度舍入` }}
    </text>
  </svg>
</template>
