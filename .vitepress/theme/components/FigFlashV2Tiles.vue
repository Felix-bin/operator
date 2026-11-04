<script setup>
const C = 28
const X = 26
const tiles = [
  { title: '完整', row: 8, col: 4, note: ['最小行 8 ≥ 最大列 7'] },
  { title: '完整（边界）', row: 7, col: 4, note: ['最小行 7 = 最大列 7'] },
  { title: '部分', row: 8, col: 8, note: ['最小行 8 < 最大列 11'] },
  { title: '跳过', row: 8, col: 12, note: ['最大行 11 < 最小列 12'] }
].map((t) => ({
  ...t,
  cells: Array.from({ length: 16 }, (_, k) => {
    const r = Math.floor(k / 4)
    const c = k % 4
    return { r, c, on: t.row + r >= t.col + c }
  })
}))
</script>

<template>
  <div class="flv2-tiles">
    <svg
      v-for="t in tiles"
      :key="t.title"
      viewBox="0 0 160 192"
      role="img"
      :aria-label="`${t.title}：行 ${t.row} 到 ${t.row + 3}，列 ${t.col} 到 ${t.col + 3}，${t.note[0]}。`"
    >
      <text class="gpu-t gpu-tb" :x="X" y="18">{{ t.title }}</text>
      <text class="gpu-ts" :x="X" y="40">列 {{ t.col }}~{{ t.col + 3 }}</text>
      <text class="gpu-ts gpu-end" :x="X - 6" y="68">{{ t.row }}</text>
      <text class="gpu-ts gpu-end" :x="X - 6" :y="68 + 3 * C">{{ t.row + 3 }}</text>
      <rect
        v-for="c in t.cells"
        :key="`${c.r}-${c.c}`"
        :class="c.on ? 'gpu-c' : 'gpu-n'"
        :x="X + c.c * C"
        :y="48 + c.r * C"
        :width="C"
        :height="C"
      />
      <text class="gpu-ts" x="4" y="182">{{ t.note[0] }}</text>
    </svg>
  </div>
  <div class="gpu-legend gpu-chart-legend">
    <span><i class="lc"></i>行 ≥ 列，可见</span>
    <span><i class="flv2-k-masked"></i>行 &lt; 列，被屏蔽</span>
  </div>
</template>

<style>
.flv2-tiles {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
@media (max-width: 560px) {
  .flv2-tiles {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
.gpu-legend i.flv2-k-masked {
  background: var(--gpu-n-fill);
  border-color: var(--vp-c-divider);
}
</style>
