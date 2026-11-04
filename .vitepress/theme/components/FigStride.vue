<script setup>
// 示意用的小规模：gridDim = 3，blockDim = 4，每趟 12 个元素，共 3 趟
const GRID = 3
const BLOCK = 4
const STRIDE = GRID * BLOCK
const X0 = 40
const PITCH = 20
const Y = 54
const cells = Array.from({ length: STRIDE * 3 }, (_, idx) => ({
  idx,
  block: Math.floor(idx / BLOCK) % GRID,
  mine: idx % STRIDE === 6
}))
const cx = (idx) => X0 + idx * PITCH + 9
const blocks = Array.from({ length: GRID * 3 }, (_, i) => ({ i, x: X0 + i * BLOCK * PITCH, id: i % GRID }))
const hops = [6, 18].map((from) => `M${cx(from)} ${Y} Q${cx(from + 6)} ${Y - 30} ${cx(from + 12) - 1} ${Y - 1}`)
</script>

<template>
  <svg class="w680" viewBox="0 0 800 226" role="img" aria-label="Grid Stride Loop：Block 1 的 2 号线程依次处理第 6、18、30 个元素，每次跨过 12 个元素。下方对比 float 和 float4：一个 Warp 的一次加载分别取 128 字节和 512 字节。">
    <defs>
      <marker id="rd-stride" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor"/></marker>
    </defs>

    <text class="gpu-t gpu-tb" x="40" y="14">Grid Stride Loop（示意：3 个 Block，每 Block 4 个线程）</text>
    <path v-for="(d, i) in hops" :key="i" class="gpu-ln" :d="d" marker-end="url(#rd-stride)" />
    <text class="gpu-tk gpu-mid" :x="cx(12)" y="32">+12</text>
    <text class="gpu-tk gpu-mid" :x="cx(24)" y="32">+12</text>
    <g v-for="c in cells" :key="c.idx">
      <rect :class="c.mine ? 'gpu-m' : c.block === 1 ? 'gpu-c' : 'gpu-n'" :x="X0 + c.idx * PITCH" :y="Y" width="18" height="24" />
      <text v-if="c.mine" class="gpu-tk gpu-mid gpu-tb" :x="cx(c.idx)" :y="Y + 16">{{ c.idx }}</text>
    </g>
    <g v-for="b in blocks" :key="b.i">
      <line class="gpu-ln" :x1="b.x" :y1="Y + 30" :x2="b.x + BLOCK * PITCH - 2" :y2="Y + 30" />
      <text class="gpu-ts gpu-mid" :x="b.x + BLOCK * PITCH / 2" :y="Y + 46">Block {{ b.id }}</text>
    </g>

    <text class="gpu-t gpu-tb" x="40" y="138">一个 Warp 的一次加载</text>
    <text class="gpu-t" x="40" y="168">float</text>
    <rect class="gpu-m" x="100" y="150" width="160" height="26"/>
    <text class="gpu-tk gpu-mid" x="180" y="168">32 × 4 = 128 字节</text>
    <text class="gpu-t" x="40" y="206">float4</text>
    <rect class="gpu-m" x="100" y="188" width="640" height="26"/>
    <text class="gpu-tk gpu-mid" x="420" y="206">32 × 16 = 512 字节，指令数不变</text>
  </svg>
</template>
