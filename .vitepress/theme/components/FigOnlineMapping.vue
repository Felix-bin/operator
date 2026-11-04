<script setup>
const M = 12
const GRID = 4
const RH = 22
const Y0 = 44
const rowY = (r) => Y0 + r * RH
const HI = 1

function panel(x0, blocks, mapBlock) {
  const rowsX = x0 + 200
  const blockH = (M * RH - 2 - (blocks - 1) * 2) / blocks
  const blockY = (b) => Y0 + b * (blockH + 2)
  return {
    x0,
    rowsX,
    blocks: Array.from({ length: blocks }, (_, b) => ({ b, y: blockY(b), h: blockH, hi: b === HI })),
    links: Array.from({ length: M }, (_, r) => {
      const b = mapBlock(r)
      const y1 = blockY(b) + blockH / 2
      const y2 = rowY(r) + (RH - 2) / 2
      return { r, hi: b === HI, d: `M${x0 + 72} ${y1} C${x0 + 130} ${y1} ${rowsX - 60} ${y2} ${rowsX - 1} ${y2}` }
    }).sort((a, b) => a.hi - b.hi),
    rows: Array.from({ length: M }, (_, r) => ({ r, hi: mapBlock(r) === HI }))
  }
}
const panels = [
  { title: 'V2：Block 与行 1:1', sub: `启动 ${M} 个 Block，每个跑完一行就退出`, ...panel(0, M, (r) => r) },
  { title: 'V4：Block 与行 1:多', sub: `启动 ${GRID} 个 Block，row += gridDim.x`, ...panel(400, GRID, (r) => r % GRID) }
]
</script>

<template>
  <svg class="w680" viewBox="0 0 760 316" role="img" aria-label="12 行数据。V2 启动 12 个 Block，每个 Block 处理一行；V4 启动 4 个 Block，Block 1 依次处理第 1、5、9 行。">
    <g v-for="p in panels" :key="p.x0">
      <text class="gpu-t gpu-tb" :x="p.x0" y="16">{{ p.title }}</text>
      <text class="gpu-ts" :x="p.x0" y="34">{{ p.sub }}</text>
      <path v-for="l in p.links" :key="l.r" class="gpu-ln" :class="{ 'online-map-hi': l.hi }" :d="l.d" />
      <g v-for="b in p.blocks" :key="b.b">
        <rect :class="b.hi ? 'gpu-m' : 'gpu-c'" :x="p.x0" :y="b.y" width="72" :height="b.h" />
        <text class="gpu-tk gpu-mid" :x="p.x0 + 36" :y="b.y + b.h / 2 + 4">Block {{ b.b }}</text>
      </g>
      <g v-for="r in p.rows" :key="r.r">
        <rect :class="r.hi ? 'gpu-m' : 'gpu-n'" :x="p.rowsX" :y="rowY(r.r)" width="120" :height="RH - 2" />
        <text class="gpu-tk gpu-mid" :x="p.rowsX + 60" :y="rowY(r.r) + 14">第 {{ r.r }} 行</text>
      </g>
    </g>
  </svg>
</template>

<style>
.gpu-ln.online-map-hi {
  stroke: var(--gpu-m);
  stroke-width: 2;
}
</style>
