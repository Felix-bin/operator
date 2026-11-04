<script setup>
const CELL = 11
const X0 = 16
const COUNT = 64
const K = 2
const list = (n) => Array.from({ length: n }, (_, i) => i)
// 线程要读 M 方向的 8 个值。as：As[BM][BK]，地址 = row * 8 + k；ts：As[BK][BM]，地址 = k * 128 + row
const asRead = new Set(list(8).map((row) => row * 8 + K))
const tsBase = K * 128
const tsRead = new Set(list(8).map((row) => row))
</script>

<template>
  <svg class="w680" viewBox="0 0 800 244" role="img" aria-label="未转置时，同一列的 8 个元素在地址上间隔 8，只能逐个读取。转置后 8 个元素连续，可以用两次 float4 读取。">
    <text class="gpu-tt" :x="X0" y="22">As[BM][BK]：同一列的 8 个元素相隔 8 个 float</text>
    <text class="gpu-ts" :x="X0" y="42">地址 = row × 8 + k，这里 k = {{ K }}。只能用 8 条 32 位读取。</text>
    <rect
      v-for="a in list(COUNT)"
      :key="`a${a}`"
      :class="asRead.has(a) ? 'gpu-m' : 'gpu-n'"
      :x="X0 + a * CELL"
      y="54"
      :width="CELL"
      height="24"
    />
    <g v-for="a in [0, 8, 16, 24, 32, 40, 48, 56]" :key="`t${a}`">
      <text class="gpu-ts gpu-mid" :x="X0 + (a + K + 0.5) * CELL" y="94">{{ a + K }}</text>
    </g>

    <text class="gpu-tt" :x="X0" y="136">As[BK][BM]：转置后，这 8 个元素连续存放</text>
    <text class="gpu-ts" :x="X0" y="156">地址 = k × 128 + row，这里从 {{ tsBase }} 开始。两条 float4 读取就够了。</text>
    <rect
      v-for="a in list(COUNT)"
      :key="`b${a}`"
      :class="tsRead.has(a) ? 'gpu-m' : 'gpu-n'"
      :x="X0 + a * CELL"
      y="168"
      :width="CELL"
      height="24"
    />
    <path class="gpu-ln" :d="`M${X0} 198 v6 h${4 * CELL - 2} v-6`" />
    <path class="gpu-ln" :d="`M${X0 + 4 * CELL} 198 v6 h${4 * CELL - 2} v-6`" />
    <text class="gpu-ts gpu-mid" :x="X0 + 2 * CELL" y="222">float4</text>
    <text class="gpu-ts gpu-mid" :x="X0 + 6 * CELL" y="222">float4</text>
  </svg>
</template>
