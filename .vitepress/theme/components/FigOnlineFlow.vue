<script setup>
const X0 = 58
const H = 50
const SEG = {
  scan: { w: 108, cls: 'gpu-m' },
  warp: { w: 62, cls: 'gpu-c', a: 'Warp 内', b: 'Shuffle' },
  smem: { w: 44, cls: 'gpu-m', a: '写', b: 'smem' },
  w0: { w: 62, cls: 'gpu-c', a: 'Warp 0', b: 'Shuffle' },
  bar: { w: 14, bar: true }
}
const build = (list) => {
  let x = X0
  return list.map((s) => {
    const base = typeof s === 'string' ? SEG[s] : { ...SEG.scan, ...s }
    const item = { ...base, x }
    x += base.w + (base.bar ? 0 : 3)
    return item
  })
}
const reduce = ['warp', 'smem', 'bar', 'w0', 'bar']
const safe = build([{ a: '第 1 遍：读 x', b: '求局部 max' }, ...reduce, { a: '第 2 遍：读 x', b: '求局部 sum' }, ...reduce, { a: '第 3 遍：读 x', b: '写 y' }])
const online = build([{ a: '第 1 遍：读 x', b: '递推 (m, d)' }, ...reduce, { a: '第 2 遍：读 x', b: '写 y' }])
const end = online[online.length - 1]
const freeX = end.x + end.w + 10
const rows = [
  { name: '三遍', y: 30, segs: safe },
  { name: '两遍', y: 122, segs: online }
]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 206" role="img" aria-label="一个 Block 处理一行的时间线。三遍 Safe 有三次全局读取和两次两级规约，共 4 道屏障；两遍 Online 有两次全局读取和一次两级规约，共 2 道屏障。">
    <g v-for="r in rows" :key="r.name">
      <text class="gpu-t gpu-tb" x="0" :y="r.y + 30">{{ r.name }}</text>
      <g v-for="(s, i) in r.segs" :key="i">
        <template v-if="s.bar">
          <line class="gpu-ln" :x1="s.x + 5" :x2="s.x + 5" :y1="r.y - 8" :y2="r.y + H + 8" stroke-width="2.4" style="stroke-width: 2.4" />
        </template>
        <template v-else>
          <rect :class="s.cls" :x="s.x" :y="r.y" :width="s.w" :height="H" />
          <text class="gpu-tk gpu-mid" :x="s.x + s.w / 2" :y="r.y + 21">{{ s.a }}</text>
          <text class="gpu-tk gpu-mid" :x="s.x + s.w / 2" :y="r.y + 38">{{ s.b }}</text>
        </template>
      </g>
    </g>
    <rect class="gpu-wait" :x="freeX" y="122" :width="X0 + 742 - freeX" :height="H" />
    <text class="gpu-t gpu-mid" :x="(freeX + X0 + 742) / 2" y="143">省下一遍读取、一次规约</text>
    <text class="gpu-ts gpu-mid" :x="(freeX + X0 + 742) / 2" y="161">屏障从 4 道减到 2 道</text>
    <text class="gpu-ts" x="58" y="198">竖线是 __syncthreads()：先到的线程都要在这里等最慢的线程。</text>
  </svg>
</template>
