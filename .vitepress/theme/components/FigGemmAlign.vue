<script setup>
const CELL = 23
const X0 = 20
const COUNT = 32
const list = (n) => Array.from({ length: n }, (_, i) => i)
function strip(K, row) {
  const start = row * K
  return { start, cells: new Set([start, start + 1, start + 2, start + 3]), ok: start % 4 === 0 }
}
const rows = [
  { y: 50, K: 8, ...strip(8, 1) },
  { y: 142, K: 6, ...strip(6, 1) }
]
</script>

<template>
  <svg class="w680" viewBox="0 0 780 226" role="img" aria-label="K 是 8 时，第 1 行从下标 8 开始，float4 落在 16 字节边界上。K 是 6 时，第 1 行从下标 6 开始，float4 跨过 16 字节边界，运行时报 misaligned address。">
    <g v-for="r in rows" :key="r.K">
      <text class="gpu-t gpu-tb" :x="X0" :y="r.y - 16">K = {{ r.K }}：A[1 × K + 0] 的下标是 {{ r.start }}</text>
      <rect
        v-for="a in list(COUNT)"
        :key="a"
        :class="r.cells.has(a) ? (r.ok ? 'gpu-c' : 'gpu-m') : 'gpu-n'"
        :x="X0 + a * CELL"
        :y="r.y"
        :width="CELL"
        height="26"
      />
      <text
        v-for="a in [0, 4, 8, 12, 16, 20, 24, 28]"
        :key="`t${a}`"
        class="gpu-ts gpu-mid"
        :x="X0 + (a + 0.5) * CELL"
        :y="r.y + 42"
      >{{ a }}</text>
      <line
        v-for="a in [4, 8, 12, 16, 20, 24, 28]"
        :key="`l${a}`"
        class="gpu-ln"
        :x1="X0 + a * CELL"
        :y1="r.y - 5"
        :x2="X0 + a * CELL"
        :y2="r.y + 31"
      />
      <text class="gpu-ts" :x="X0 + 11 * CELL + 90" :y="r.y - 16">{{ r.ok ? '对齐：一条 128 位读取' : '不对齐：跨过 16 字节边界' }}</text>
    </g>
  </svg>
</template>
