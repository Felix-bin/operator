<script setup>
const values = [3, 1, 5, 2, 4, 7, 2, 6]
const W = 34
const H = 24
const PITCH = 40
const Y0 = 34
const ROW = 36
const y = (r) => Y0 + r * ROW

// 串行：每轮把下一个输入加到累计值上
const SX = 62
const sx = (i) => SX + i * PITCH + W / 2
let running = values[0]
const serial = values.slice(1).map((v, i) => {
  running += v
  return { r: i + 1, value: running }
})

// 树形：每轮两两相加，数量减半
const TX = 470
let level = values.map((value, i) => ({ value, x: TX + i * PITCH + W / 2 }))
const tree = []
for (let r = 1; level.length > 1; r++) {
  const next = []
  for (let i = 0; i < level.length; i += 2) {
    next.push({ r, value: level[i].value + level[i + 1].value, x: (level[i].x + level[i + 1].x) / 2, a: level[i], b: level[i + 1] })
  }
  tree.push(...next)
  level = next
}
</script>

<template>
  <svg class="w680" viewBox="0 0 800 336" role="img" aria-label="8 个数求和：串行做法要 7 轮，树形规约只要 3 轮，加法次数都是 7 次，结果都是 30。">
    <text class="gpu-tt" x="62" y="20">串行：7 轮</text>
    <text class="gpu-tt" x="470" y="20">树形规约：3 轮</text>

    <text class="gpu-ts" x="16" :y="y(0) + 16">输入</text>
    <text v-for="r in 7" :key="`sl${r}`" class="gpu-ts" x="16" :y="y(r) + 16">第 {{ r }} 轮</text>
    <text class="gpu-ts" x="424" :y="y(0) + 16">输入</text>
    <text v-for="r in 3" :key="`tl${r}`" class="gpu-ts" x="424" :y="y(r) + 16">第 {{ r }} 轮</text>

    <!-- 串行 -->
    <g v-for="c in serial" :key="`sc${c.r}`">
      <line class="gpu-ln" :x1="sx(c.r)" :y1="y(0) + H" :x2="sx(c.r)" :y2="y(c.r)" />
      <line class="gpu-ln" :x1="sx(c.r - 1)" :y1="y(c.r - 1) + H" :x2="sx(c.r) - 6" :y2="y(c.r)" />
    </g>
    <g v-for="(v, i) in values" :key="`si${i}`">
      <rect class="gpu-n" :x="SX + i * PITCH" :y="y(0)" :width="W" :height="H" />
      <text class="gpu-tk gpu-mid" :x="sx(i)" :y="y(0) + 16">{{ v }}</text>
    </g>
    <g v-for="c in serial" :key="`sb${c.r}`">
      <rect class="gpu-c" :x="sx(c.r) - W / 2" :y="y(c.r)" :width="W" :height="H" />
      <text class="gpu-tk gpu-mid" :class="{ 'gpu-tb': c.r === 7 }" :x="sx(c.r)" :y="y(c.r) + 16">{{ c.value }}</text>
    </g>

    <!-- 树形 -->
    <g v-for="(c, i) in tree" :key="`tc${i}`">
      <line class="gpu-ln" :x1="c.a.x" :y1="y(c.r - 1) + H" :x2="c.x - 6" :y2="y(c.r)" />
      <line class="gpu-ln" :x1="c.b.x" :y1="y(c.r - 1) + H" :x2="c.x + 6" :y2="y(c.r)" />
    </g>
    <g v-for="(v, i) in values" :key="`ti${i}`">
      <rect class="gpu-n" :x="TX + i * PITCH" :y="y(0)" :width="W" :height="H" />
      <text class="gpu-tk gpu-mid" :x="TX + i * PITCH + W / 2" :y="y(0) + 16">{{ v }}</text>
    </g>
    <g v-for="(c, i) in tree" :key="`tb${i}`">
      <rect class="gpu-c" :x="c.x - W / 2" :y="y(c.r)" :width="W" :height="H" />
      <text class="gpu-tk gpu-mid" :class="{ 'gpu-tb': c.r === 3 }" :x="c.x" :y="y(c.r) + 16">{{ c.value }}</text>
    </g>

    <text class="gpu-t" x="470" :y="y(4) + 16">加法次数没有变，仍是 7 次。</text>
    <text class="gpu-t" x="470" :y="y(5) + 16">变的是能同时做的加法：第 1 轮 4 个。</text>
    <text class="gpu-t" x="470" :y="y(6) + 16">N 个数只需要 log₂N 轮。</text>
  </svg>
</template>
