<script setup>
// 四块输入依次经过：传输 64 tick，额外等待 128 tick，计算 128 tick
const TRANSFER = 64
const READY = 192
const COMPUTE = 128
const BLOCKS = 4

function simulate(slots) {
  const blocks = []
  let lastIssue = -TRANSFER
  let computeFree = 0
  for (let k = 0; k < BLOCKS; k++) {
    const slotFree = k < slots ? 0 : blocks[k - slots].end
    const issue = Math.max(lastIssue + TRANSFER, slotFree)
    const ready = issue + READY
    const start = Math.max(ready, computeFree)
    const end = start + COMPUTE
    blocks.push({ issue, ready, start, end })
    lastIssue = issue
    computeFree = end
  }
  return { slots, blocks, finish: computeFree }
}

const groups = [1, 2, 3].map(simulate)
const W = 760
const left = 74
const right = 74
const rowH = 13
const rowGap = 4
const groupGap = 22
const tickMax = 1280
const groupH = BLOCKS * (rowH + rowGap)
const top = 8
const H = top + groups.length * (groupH + groupGap) + 22
const x = (t) => left + (t / tickMax) * (W - left - right)
const groupY = (g) => top + g * (groupH + groupGap)
const axisTicks = [0, 320, 640, 960, 1280]
</script>

<template>
  <figure class="gpu-fig">
    <div class="gpu-legend gpu-chart-legend">
      <span><i class="swatch sb"></i>传输</span>
      <span><i class="swatch sn"></i>槽被占用（等待返回或等待计算）</span>
      <span><i class="swatch sa"></i>计算</span>
    </div>
    <div class="gpu-scroll" tabindex="0">
      <svg
        class="w680 gpu-chart"
        :viewBox="`0 0 ${W} ${H}`"
        role="img"
        :aria-label="`四块输入在不同缓冲槽数下的时间线：1 槽 ${groups[0].finish} tick，2 槽 ${groups[1].finish} tick，3 槽 ${groups[2].finish} tick。`"
      >
        <line v-for="t in axisTicks" :key="t" class="gpu-grid" :x1="x(t)" :x2="x(t)" :y1="top" :y2="H - 20" />
        <g v-for="(g, gi) in groups" :key="g.slots">
          <text class="gpu-t gpu-tb" x="0" :y="groupY(gi) + 12">{{ g.slots }} 个槽</text>
          <g v-for="(b, k) in g.blocks" :key="k">
            <rect class="gpu-seg-n" :x="x(b.issue)" :y="groupY(gi) + k * (rowH + rowGap)" :width="x(b.end) - x(b.issue)" :height="rowH">
              <title>第 {{ k }} 块：{{ b.issue }} tick 发起，{{ b.end }} tick 释放</title>
            </rect>
            <rect class="gpu-seg-b" :x="x(b.issue)" :y="groupY(gi) + k * (rowH + rowGap)" :width="x(b.issue + TRANSFER) - x(b.issue)" :height="rowH">
              <title>第 {{ k }} 块传输：{{ b.issue }} 至 {{ b.issue + TRANSFER }} tick</title>
            </rect>
            <rect class="gpu-seg-a" :x="x(b.start)" :y="groupY(gi) + k * (rowH + rowGap)" :width="x(b.end) - x(b.start)" :height="rowH">
              <title>第 {{ k }} 块计算：{{ b.start }} 至 {{ b.end }} tick</title>
            </rect>
          </g>
          <text class="gpu-ts" :x="x(g.finish) + 6" :y="groupY(gi) + (BLOCKS - 1) * (rowH + rowGap) + 11">{{ g.finish }} tick</text>
        </g>
        <text v-for="t in axisTicks" :key="t" class="gpu-ts gpu-mid" :x="x(t)" :y="H - 6">{{ t }}</text>
      </svg>
    </div>
    <figcaption>每组四行对应四块输入，横轴是 B200 SM 的时钟周期（tick）。3 个槽时计算段首尾相接，矩阵单元不再闲置；再加第 4 个槽，完成时间不变。</figcaption>
  </figure>
</template>
