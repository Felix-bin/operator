<script setup>
const COL = 80
const rowY = { 0: 44, 16: 98 }
const panels = [
  {
    x0: 16,
    title: '锁步：结果正确',
    ops: [
      { lane: 0, t: 0, a: '读 smem[16]', b: '得到 a', cls: 'gpu-m' },
      { lane: 0, t: 1, a: '写 smem[0]', b: '', cls: 'gpu-c' },
      { lane: 16, t: 0, a: '读 smem[32]', b: '得到 b', cls: 'gpu-m' },
      { lane: 16, t: 1, a: '写 smem[16]', b: '变成 a+b', cls: 'gpu-c' }
    ],
    verdict: '车道 0 读到的是 a，符合预期。'
  },
  {
    x0: 416,
    title: '车道 16 抢先：结果错误',
    ops: [
      { lane: 16, t: 0, a: '读 smem[32]', b: '得到 b', cls: 'gpu-m' },
      { lane: 16, t: 1, a: '写 smem[16]', b: '变成 a+b', cls: 'gpu-c' },
      { lane: 0, t: 2, a: '读 smem[16]', b: '得到 a+b', cls: 'gpu-m' },
      { lane: 0, t: 3, a: '写 smem[0]', b: '', cls: 'gpu-c' }
    ],
    arrow: true,
    verdict: 'b 上一轮已计入 smem[0]，这里又加了一次。'
  }
]
const opX = (p, t) => p.x0 + 62 + t * COL
</script>

<template>
  <svg class="w680" viewBox="0 0 800 186" role="img" aria-label="同一条语句在车道 0 和车道 16 上的两种执行顺序。锁步时车道 0 读到旧值，结果正确。车道 16 抢先时，车道 0 读到已被改写的值，同一个数被加了两次。">
    <defs>
      <marker id="rd-lock" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <g v-for="p in panels" :key="p.x0">
      <text class="gpu-tt" :x="p.x0" y="20">{{ p.title }}</text>
      <text class="gpu-t" :x="p.x0" :y="rowY[0] + 26">车道 0</text>
      <text class="gpu-t" :x="p.x0" :y="rowY[16] + 26">车道 16</text>
      <g v-for="(op, i) in p.ops" :key="i">
        <rect :class="op.cls" :x="opX(p, op.t)" :y="rowY[op.lane]" :width="COL - 4" height="44" />
        <text class="gpu-tk gpu-mid" :x="opX(p, op.t) + (COL - 4) / 2" :y="rowY[op.lane] + (op.b ? 19 : 26)">{{ op.a }}</text>
        <text v-if="op.b" class="gpu-tk gpu-mid gpu-tb" :x="opX(p, op.t) + (COL - 4) / 2" :y="rowY[op.lane] + 35">{{ op.b }}</text>
      </g>
      <path
        v-if="p.arrow"
        class="gpu-ln"
        :d="`M${opX(p, 1) + COL - 4} ${rowY[16] + 10} L${opX(p, 2) + 24} ${rowY[0] + 45}`"
        marker-end="url(#rd-lock)"
      />
      <text class="gpu-t" :x="p.x0" y="172">{{ p.verdict }}</text>
    </g>
  </svg>
</template>
