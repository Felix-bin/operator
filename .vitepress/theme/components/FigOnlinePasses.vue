<script setup>
const X0 = 190
const W = 78
const GAP = 6
const rows = [
  {
    name: '三遍 Safe',
    sub: '先 max，再 sum',
    cells: [
      { t: '读', s: '求 max' },
      { t: '读', s: '求 sum' },
      { t: '读', s: '算 y' },
      { t: '写', s: 'y' }
    ],
    n: '4 次',
    note: '基准'
  },
  {
    name: '两遍 Online',
    sub: 'V0、V1、V2、V4',
    cells: [
      { t: '读', s: '求 m 和 d' },
      { t: '读', s: '算 y' },
      { t: '写', s: 'y' }
    ],
    n: '3 次',
    note: '显存受限时至多 1.33 倍'
  },
  {
    name: '一遍读取',
    sub: 'V3，整行暂存',
    cells: [
      { t: '读', s: '求 m 和 d' },
      { t: '写', s: 'y' },
      { t: '存', s: '暂存', dash: true },
      { t: '取', s: '暂存', dash: true }
    ],
    n: '2 次',
    note: '另有暂存的存取，见 V3'
  }
]
const rowY = (r) => 40 + r * 62
</script>

<template>
  <svg class="w680" viewBox="0 0 760 232" role="img" aria-label="每个元素的访存次数：三遍 Safe 读 3 次写 1 次，共 4 次；两遍 Online 读 2 次写 1 次，共 3 次；一遍读取的 V3 读 1 次写 1 次，共 2 次，另外还有暂存的存取。">
    <text class="gpu-ts" :x="X0" y="20">每个元素对全局内存的访问，按发生顺序</text>
    <text class="gpu-ts" x="556" y="20">合计</text>
    <g v-for="(r, i) in rows" :key="r.name">
      <text class="gpu-t gpu-tb" x="0" :y="rowY(i) + 18">{{ r.name }}</text>
      <text class="gpu-ts" x="0" :y="rowY(i) + 36">{{ r.sub }}</text>
      <g v-for="(c, j) in r.cells" :key="j">
        <rect :class="c.dash ? 'gpu-wait' : 'gpu-m'" :x="X0 + j * (W + GAP)" :y="rowY(i)" :width="W" height="46" />
        <text class="gpu-t gpu-mid gpu-tb" :x="X0 + j * (W + GAP) + W / 2" :y="rowY(i) + 19">{{ c.t }}</text>
        <text class="gpu-tk gpu-mid" :x="X0 + j * (W + GAP) + W / 2" :y="rowY(i) + 36">{{ c.s }}</text>
      </g>
      <text class="gpu-tt" x="556" :y="rowY(i) + 19">{{ r.n }}</text>
      <text class="gpu-ts" x="556" :y="rowY(i) + 38">{{ r.note }}</text>
    </g>
  </svg>
</template>
