<script setup>
const rowY = (i) => 40 + i * 34
const panels = [
  {
    x0: 16,
    title: 'V2，s = 128：各走各的 Bank',
    lanes: [
      { tid: 0, addr: 0, bank: 0 },
      { tid: 1, addr: 1, bank: 1 },
      { tid: 2, addr: 2, bank: 2 },
      { tid: 3, addr: 3, bank: 3 }
    ],
    slots: ['32 条车道同一拍完成'],
    slotW: 300,
    busy: [0, 1, 2, 3]
  },
  {
    x0: 416,
    title: 'V1，s = 2：都挤到 Bank 0',
    lanes: [
      { tid: 0, addr: 0, bank: 0 },
      { tid: 8, addr: 32, bank: 0 },
      { tid: 16, addr: 64, bank: 0 },
      { tid: 24, addr: 96, bank: 0 }
    ],
    slots: ['tid 0', 'tid 8', 'tid 16', 'tid 24'],
    slotW: 75,
    busy: [0]
  }
]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 232" role="img" aria-label="左图中四个线程访问四个不同的 Bank，同一拍完成。右图中四个线程访问同一个 Bank 的不同地址，只能分四拍依次完成。">
    <defs>
      <marker id="rd-bank" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <g v-for="p in panels" :key="p.x0">
      <text class="gpu-tt" :x="p.x0" y="20">{{ p.title }}</text>
      <g v-for="(l, i) in p.lanes" :key="l.tid">
        <rect class="gpu-c" :x="p.x0" :y="rowY(i)" width="150" height="26" />
        <text class="gpu-tk gpu-mid" :x="p.x0 + 75" :y="rowY(i) + 17">tid {{ l.tid }} 读 smem[{{ l.addr }}]</text>
        <line
          class="gpu-ln"
          :x1="p.x0 + 150"
          :y1="rowY(i) + 13"
          :x2="p.x0 + 268"
          :y2="rowY(l.bank) + 13 + (p.busy.length === 1 ? (i - 1.5) * 4 : 0)"
          marker-end="url(#rd-bank)"
        />
      </g>
      <g v-for="b in 4" :key="b">
        <rect
          :class="p.busy.includes(b - 1) ? 'gpu-m' : 'gpu-n'"
          :x="p.x0 + 270"
          :y="rowY(b - 1)"
          width="84"
          height="26"
        />
        <text class="gpu-tk gpu-mid" :x="p.x0 + 312" :y="rowY(b - 1) + 17">Bank {{ b - 1 }}</text>
      </g>
      <text class="gpu-ts" :x="p.x0" y="194">这条读指令的完成顺序</text>
      <g v-for="(slot, i) in p.slots" :key="slot">
        <rect class="gpu-m" :x="p.x0 + i * p.slotW" y="202" :width="p.slotW - 2" height="24" />
        <text class="gpu-tk gpu-mid" :x="p.x0 + i * p.slotW + (p.slotW - 2) / 2" y="218">{{ slot }}</text>
      </g>
    </g>
  </svg>
</template>
