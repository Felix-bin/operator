<script setup>
const W = 40
const H = 32
const panels = [
  {
    x0: 10,
    title: '行主序：quarter-warp 形状 1 × 8',
    map: (l) => [l >> 3, l & 7]
  },
  {
    x0: 420,
    title: 'Z-order：quarter-warp 形状 2 × 4',
    map: (l) => [(l & 1) + ((l >> 4) << 1), (l & 15) >> 1]
  }
].map((p) => {
  const grid = Array.from({ length: 4 }, () => Array(8).fill(0))
  const cells = Array.from({ length: 32 }, (_, lane) => {
    const [r, c] = p.map(lane)
    grid[r][c] = lane
    return { lane, r, c, q: lane >> 3 }
  })
  const edges = []
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 8; c++) {
      if (c < 7 && grid[r][c] >> 3 !== grid[r][c + 1] >> 3) edges.push([c + 1, r, c + 1, r + 1])
      if (r < 3 && grid[r][c] >> 3 !== grid[r + 1][c] >> 3) edges.push([c, r + 1, c + 1, r + 1])
    }
  }
  return { ...p, cells, edges }
})
const shade = [1, 0.62, 0.3, 0.08]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 216" role="img" aria-label="行主序下每个 quarter-warp 占一整行。Z-order 下每个 quarter-warp 占两行四列，lane i 和 lane i 异或 1 同列，和 lane i 异或 2 同行。">
    <g v-for="p in panels" :key="p.x0">
      <text class="gpu-t gpu-tb" :x="p.x0" y="20">{{ p.title }}</text>
      <rect
        v-for="c in p.cells"
        :key="c.lane"
        class="gpu-c"
        :fill-opacity="shade[c.q]"
        stroke-width="0.6"
        :x="p.x0 + c.c * W"
        :y="36 + c.r * H"
        :width="W"
        :height="H"
      />
      <text
        v-for="c in p.cells"
        :key="`t${c.lane}`"
        class="gpu-t gpu-mid"
        :x="p.x0 + (c.c + 0.5) * W"
        :y="36 + (c.r + 0.5) * H + 5"
      >{{ c.lane }}</text>
      <line
        v-for="(e, i) in p.edges"
        :key="`e${i}`"
        class="gemm-qw"
        :x1="p.x0 + e[0] * W"
        :y1="36 + e[1] * H"
        :x2="p.x0 + e[2] * W"
        :y2="36 + e[3] * H"
      />
      <rect class="gemm-qw" :x="p.x0" y="36" :width="8 * W" :height="4 * H" />
    </g>
    <text class="gpu-ts" x="10" y="192">方格内是 lane_id，粗线圈出 4 个 quarter-warp，深浅依次为 lane 0~7、8~15、16~23、24~31。</text>
  </svg>
</template>

<style>
.gemm-qw {
  fill: none;
  stroke: currentColor;
  stroke-width: 2.4;
}
</style>
