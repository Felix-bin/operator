<script setup>
const LOOP = 14
const ROUND = 46
const BAR = 6
const X0 = 62
const steps = [128, 64, 32, 16, 8, 4, 2, 1]

function row(name, y, keepLoop, keepBarrier, note) {
  let x = X0
  const items = []
  for (const s of steps) {
    const big = s > 32
    if (keepLoop(big)) {
      items.push({ kind: 'gpu-n', x, w: LOOP })
      x += LOOP
    }
    items.push({ kind: 'gpu-c', x, w: ROUND, s })
    x += ROUND
    if (keepBarrier(big)) {
      items.push({ kind: 'gpu-dot', x, w: BAR })
      x += BAR
    }
  }
  return { name, y, items, note }
}

const rows = [
  row('V3', 14, () => true, () => true, ['8 次循环判断', '8 次屏障']),
  row('V4', 72, (big) => big, (big) => big, ['2 次循环判断', '2 次屏障']),
  row('V5', 130, () => false, (big) => big, ['没有循环判断', '2 次屏障'])
]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 212" role="img" aria-label="三个版本规约阶段的结构。V3 有 8 次循环判断和 8 次屏障，V4 剩 2 次循环判断和 2 次屏障，V5 去掉循环判断，保留 2 次屏障。">
    <g v-for="r in rows" :key="r.name">
      <text class="gpu-t gpu-tb" x="16" :y="r.y + 21">{{ r.name }}</text>
      <g v-for="(it, i) in r.items" :key="i">
        <rect :class="it.kind" :x="it.x" :y="r.y" :width="it.w" height="32" />
        <text v-if="it.s" class="gpu-tk gpu-mid" :x="it.x + it.w / 2" :y="r.y + 21">s={{ it.s }}</text>
      </g>
      <text class="gpu-t" x="640" :y="r.y + 13">{{ r.note[0] }}</text>
      <text class="gpu-t" x="640" :y="r.y + 31">{{ r.note[1] }}</text>
    </g>
    <rect class="gpu-c" x="62" y="186" width="16" height="16"/>
    <text class="gpu-ts" x="84" y="199">一轮规约</text>
    <rect class="gpu-n" x="160" y="186" width="16" height="16"/>
    <text class="gpu-ts" x="182" y="199">循环判断与跳转</text>
    <rect class="gpu-dot" x="300" y="186" width="6" height="16"/>
    <text class="gpu-ts" x="314" y="199">__syncthreads()</text>
    <text class="gpu-ts gpu-end" x="784" y="199">结构示意，宽度不代表耗时</text>
  </svg>
</template>
