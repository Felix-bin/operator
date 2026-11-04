<script setup>
const X0 = 170
const CELL = 24
const SLOTS = 24

function makePanel(y0, title, sub, N, pitch, note) {
  const cells = []
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < N; c++) {
      const slot = r * pitch + c
      if (slot < SLOTS) cells.push({ slot, cls: r === 1 ? 'gpu-m' : 'gpu-n', label: c === 0 ? `行${r}` : '' })
    }
    for (let c = N; c < pitch; c++) {
      const slot = r * pitch + c
      if (slot < SLOTS) cells.push({ slot, cls: 'sm-pad', label: '' })
    }
  }
  const start = pitch
  const offset = (start * 4) % 16
  return { y0, title, sub, cells, start, offset, note }
}

const panels = [
  makePanel(30, 'N = 8', '行 1 从第 32 字节开始', 8, 8, 'x4[0] 读第 32~47 字节，落在一个 16 字节块内。'),
  makePanel(126, 'N = 6', '行 1 从第 24 字节开始', 6, 6, 'x4[0] 读第 24~39 字节，跨过了第 32 字节的边界，未对齐。'),
  makePanel(222, 'N = 6，pitch = 8', '每行补 2 个空位', 6, 8, 'x4[0] 读第 32~47 字节，重新对齐。尾部元素仍走标量循环。')
]
const ticks = Array.from({ length: SLOTS / 4 + 1 }, (_, i) => i * 4)
</script>

<template>
  <svg class="w680" viewBox="0 0 800 292" role="img" aria-label="三种布局下的行首地址。N 为 8 时每行都从 16 字节边界开始。N 为 6 时第 1 行从第 24 字节开始，float4 跨过 16 字节边界。把每行补到 8 个位置后，第 1 行重新从第 32 字节开始。">
    <g v-for="p in panels" :key="p.y0">
      <text class="gpu-t gpu-tb" x="16" :y="p.y0 + 28">{{ p.title }}</text>
      <text class="gpu-ts" x="16" :y="p.y0 + 46">{{ p.sub }}</text>

      <text v-for="t in ticks" :key="t" class="gpu-ts gpu-mid" :x="X0 + t * CELL" :y="p.y0 - 2">{{ t * 4 }}</text>

      <g v-for="c in p.cells" :key="c.slot">
        <rect :class="c.cls" :x="X0 + c.slot * CELL" :y="p.y0 + 6" :width="CELL" height="28" />
        <text v-if="c.label" class="gpu-tk gpu-mid" :x="X0 + c.slot * CELL + CELL / 2" :y="p.y0 + 24">{{ c.label.slice(1) }}</text>
      </g>

      <line v-for="t in ticks" :key="`l${t}`" class="gpu-ln" :x1="X0 + t * CELL" :y1="p.y0 + 2" :x2="X0 + t * CELL" :y2="p.y0 + 38" />

      <rect
        class="gpu-ln sm-f4"
        :class="{ 'sm-f4-bad': p.offset !== 0 }"
        :x="X0 + p.start * CELL"
        :y="p.y0 + 2"
        :width="4 * CELL"
        height="36"
      />
      <text class="gpu-ts" :x="X0" :y="p.y0 + 56">{{ p.note }}</text>
    </g>
  </svg>
</template>

<style>
.sm-pad {
  fill: none;
  stroke: var(--vp-c-divider);
  stroke-width: 1.2;
  stroke-dasharray: 3 3;
}
rect.sm-f4 {
  stroke-width: 2.5;
}
rect.sm-f4-bad {
  stroke: var(--vp-c-danger-1);
}
</style>
