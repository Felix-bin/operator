<script setup>
const LANES = 8
const X0 = 110
const PITCH = 72
const W = 56
const H = 26
const rowY = (r) => 30 + r * 58
const cx = (i) => X0 + i * PITCH + W / 2

// 与 __shfl_down_sync 相同：源车道越界时读到自己的值
const rows = [{ label: '输入', values: [1, 2, 3, 4, 5, 6, 7, 8], useful: LANES, offset: 0 }]
for (const offset of [4, 2, 1]) {
  const prev = rows[rows.length - 1].values
  rows.push({
    label: `offset ${offset}`,
    offset,
    useful: offset,
    values: prev.map((v, i) => v + (i + offset < LANES ? prev[i + offset] : v))
  })
}
</script>

<template>
  <svg class="w680" viewBox="0 0 800 240" role="img" aria-label="8 条车道用 shuffle 规约：offset 依次为 4、2、1，每轮让前半车道读取后半车道的值，三轮后车道 0 得到总和 36。">
    <defs>
      <marker id="rd-shfl" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <text v-for="i in LANES" :key="i" class="gpu-ts gpu-mid" :x="cx(i - 1)" y="20">车道 {{ i - 1 }}</text>

    <g v-for="(row, r) in rows" :key="row.label">
      <text class="gpu-t" x="16" :y="rowY(r) + 18">{{ row.label }}</text>
      <template v-if="r > 0">
        <g v-for="i in row.useful" :key="i">
          <line class="gpu-ln" :x1="cx(i - 1)" :y1="rowY(r - 1) + H" :x2="cx(i - 1)" :y2="rowY(r) - 1" marker-end="url(#rd-shfl)" />
          <line class="gpu-ln" :x1="cx(i - 1 + row.offset)" :y1="rowY(r - 1) + H" :x2="cx(i - 1) + 6" :y2="rowY(r) - 1" marker-end="url(#rd-shfl)" />
        </g>
      </template>
      <g v-for="(v, i) in row.values" :key="i">
        <rect :class="i < row.useful ? 'gpu-c' : 'gpu-n'" :x="X0 + i * PITCH" :y="rowY(r)" :width="W" :height="H" />
        <text class="gpu-tk gpu-mid" :class="{ 'gpu-tb': r === 3 && i === 0 }" :x="cx(i)" :y="rowY(r) + 18">{{ v }}</text>
      </g>
    </g>
  </svg>
</template>
