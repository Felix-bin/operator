<script setup>
const CELL = 18
const COLS = 16
const ROWS = 8
const Y0 = 50
const LANES = 8

// 每个格子属于哪条车道（-1 表示这条读指令不读它）
const panels = [
  {
    x0: 56,
    title: 'V0：一个线程一行',
    lane: (r, c) => (c === 0 && r < LANES ? r : -1),
    // 每条车道各自落在一个 32 字节事务里
    txns: Array.from({ length: LANES }, (_, r) => ({ r, c: 0, w: 8 })),
    lines: ['每条车道各发一次 32 字节事务', '32 条车道：32 次事务，1024 字节，要用的只有 128 字节']
  },
  {
    x0: 464,
    title: 'V1：一个 Block 一行',
    lane: (r, c) => (r === 0 && c < LANES ? c : -1),
    txns: [{ r: 0, c: 0, w: 8 }],
    lines: ['8 条车道合用一次 32 字节事务', '32 条车道：4 次事务，128 字节，全部用到']
  }
]

const cells = (p) => {
  const out = []
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) out.push({ r, c, lane: p.lane(r, c) })
  }
  return out
}
</script>

<template>
  <svg class="w680" viewBox="0 0 800 256" role="img" aria-label="同一条读指令中各车道访问的地址。V0 的车道落在不同行的同一列，每条车道一个事务。V1 的车道落在同一行的相邻位置，8 条车道共用一个事务。">
    <g v-for="p in panels" :key="p.x0">
      <text class="gpu-tt" :x="p.x0" y="20">{{ p.title }}</text>
      <text class="gpu-ts" :x="p.x0" y="42">列 0</text>
      <text class="gpu-ts gpu-end" :x="p.x0 + COLS * CELL" y="42">列 15</text>
      <text v-for="r in ROWS" :key="r" class="gpu-ts gpu-end" :x="p.x0 - 8" :y="Y0 + (r - 1) * CELL + 13">行 {{ r - 1 }}</text>
      <g v-for="c in cells(p)" :key="`${c.r}-${c.c}`">
        <rect
          :class="c.lane >= 0 ? 'gpu-m' : 'gpu-n'"
          :x="p.x0 + c.c * CELL"
          :y="Y0 + c.r * CELL"
          :width="CELL"
          :height="CELL"
        />
        <text
          v-if="c.lane >= 0"
          class="gpu-tk gpu-mid"
          :x="p.x0 + c.c * CELL + CELL / 2"
          :y="Y0 + c.r * CELL + 13"
        >{{ c.lane }}</text>
      </g>
      <rect
        v-for="t in p.txns"
        :key="t.r"
        class="gpu-ln sm-txn"
        :x="p.x0 + t.c * CELL"
        :y="Y0 + t.r * CELL"
        :width="t.w * CELL"
        :height="CELL"
      />
      <text class="gpu-t" :x="p.x0" y="222">{{ p.lines[0] }}</text>
      <text class="gpu-ts" :x="p.x0" y="242">{{ p.lines[1] }}</text>
    </g>
  </svg>
</template>

<style>
rect.sm-txn {
  stroke-width: 2;
}
</style>
