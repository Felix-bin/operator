<script setup>
const CELL = 17
const X0 = 190
const list = (n) => Array.from({ length: n }, (_, i) => i)
const schemes = [
  { name: '原样存放', sub: 'As[k][row]', addr: (k, r) => k * 128 + r },
  { name: 'Padding', sub: 'As[k][row]，每层 132', addr: (k, r) => k * 132 + r },
  { name: 'XOR', sub: 'As[k][row ^ (k*4)]', addr: (k, r) => k * 128 + (r ^ (k * 4)) }
].map((s, i) => {
  // Warp 0 执行 As[k][row] = a.x 这条写：lane 负责 row = tid / 2，k = (tid % 2) * 4
  const a = Array(32).fill(0)
  const b = Array(32).fill(0)
  for (let tid = 0; tid < 32; tid++) {
    const row = tid >> 1
    const k = (tid & 1) * 4
    const bank = s.addr(k, row) % 32
    if (k === 0) a[bank]++
    else b[bank]++
  }
  const ways = Math.max(...a.map((v, j) => v + b[j]))
  return { ...s, a, b, y: 44 + i * 74, ways }
})
</script>

<template>
  <svg class="w680" viewBox="0 0 800 268" role="img" aria-label="加载 tileA 时同一行的两个线程分别写 k 等于 0 和 k 等于 4 的层。原样存放时它们落在同一个 Bank，形成 2 路写冲突。Padding 和 XOR 都把 k 等于 4 的线程移到另外 16 个 Bank，冲突消失。">
    <text
      v-for="b in [0, 8, 16, 24]"
      :key="b"
      class="gpu-ts"
      :x="X0 + b * CELL"
      y="26"
    >Bank {{ b }}</text>
    <g v-for="s in schemes" :key="s.name">
      <text class="gpu-t gpu-tb" x="10" :y="s.y + 15">{{ s.name }}</text>
      <text class="gpu-ts" x="10" :y="s.y + 34">{{ s.sub }}</text>
      <text class="gpu-ts" :x="X0 - 6" :y="s.y + 14" text-anchor="end">k=0</text>
      <text class="gpu-ts" :x="X0 - 6" :y="s.y + 34" text-anchor="end">k=4</text>
      <g v-for="b in list(32)" :key="b">
        <rect
          :class="s.a[b] ? (s.b[b] ? 'gemm-clash' : 'gpu-m') : 'gpu-n'"
          :x="X0 + b * CELL"
          :y="s.y"
          :width="CELL"
          height="20"
        />
        <rect
          :class="s.b[b] ? (s.a[b] ? 'gemm-clash' : 'gpu-m') : 'gpu-n'"
          :x="X0 + b * CELL"
          :y="s.y + 20"
          :width="CELL"
          height="20"
        />
      </g>
      <text class="gpu-t" :x="X0 + 32 * CELL + 10" :y="s.y + 26">{{ s.ways > 1 ? `${s.ways} 路` : '无' }}</text>
    </g>
  </svg>
</template>
