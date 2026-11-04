<script setup>
const W = 34
const H = 24
const PITCH = 40
const rowY = (r) => 62 + r * 62

function panel(x0, title, pairs) {
  return {
    x0,
    title,
    rows: pairs.map(({ s, active }, r) => {
      const dest = new Set(active)
      const src = new Set(active.map((d) => d + s))
      return {
        s,
        y: rowY(r),
        cells: Array.from({ length: 8 }, (_, i) => ({
          i,
          cls: dest.has(i) ? 'gpu-c' : src.has(i) ? 'gpu-m' : 'gpu-n'
        })),
        arcs: active.map((d) => {
          const from = x0 + (d + s) * PITCH + W / 2
          const to = x0 + d * PITCH + W / 2
          return `M${from} ${rowY(r)} Q${(from + to) / 2} ${rowY(r) - 34} ${to + 2} ${rowY(r) - 1}`
        })
      }
    })
  }
}

const panels = [
  panel(60, '步长从小到大（V0、V1）', [
    { s: 1, active: [0, 2, 4, 6] },
    { s: 2, active: [0, 4] },
    { s: 4, active: [0] }
  ]),
  panel(470, '步长从大到小（V2）', [
    { s: 4, active: [0, 1, 2, 3] },
    { s: 2, active: [0, 1] },
    { s: 1, active: [0] }
  ])
]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 232" role="img" aria-label="8 个槽位的两种规约顺序。步长从小到大时，结果写入的槽位是分散的 0、2、4、6；步长从大到小时，结果写入的槽位是连续的 0、1、2、3。">
    <defs>
      <marker id="rd-order" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor"/></marker>
    </defs>
    <g v-for="p in panels" :key="p.x0">
      <text class="gpu-tt" :x="p.x0" y="20">{{ p.title }}</text>
      <g v-for="row in p.rows" :key="row.s">
        <text class="gpu-ts" :x="p.x0 - 44" :y="row.y + 16">s = {{ row.s }}</text>
        <g v-for="c in row.cells" :key="c.i">
          <rect :class="c.cls" :x="p.x0 + c.i * PITCH" :y="row.y" :width="W" :height="H" />
          <text class="gpu-tk gpu-mid" :x="p.x0 + c.i * PITCH + W / 2" :y="row.y + 16">{{ c.i }}</text>
        </g>
        <path v-for="(d, i) in row.arcs" :key="i" class="gpu-ln" :d="d" marker-end="url(#rd-order)" />
      </g>
    </g>
  </svg>
</template>
