<script setup>
const X0 = 150
const CW = 36
const CH = 28
const THREAD = 1
const BLOCK = 4
const scalar = Array.from({ length: 16 }, (_, i) => ({ i, t: i % BLOCK }))
const vec = Array.from({ length: 4 }, (_, i) => ({ i, t: i % BLOCK }))

// 第三排：N = 6 时三行的布局
const N = 6
const AW = 34
const AX = 76
const aligned = Array.from({ length: 18 }, (_, i) => ({ i, row: Math.floor(i / N) }))
const rowStart = 1 * N
const misX = AX + rowStart * AW
const offset = (rowStart * 4) % 16
</script>

<template>
  <svg class="w680" viewBox="0 0 760 322" role="img" aria-label="V1 每个线程隔 4 个元素读一个 float；V2 每个线程隔 4 个 float4 读 16 字节。N 等于 6 时，第 1 行从字节偏移 24 开始，第一次 float4 读取跨过 16 字节边界。">
    <text class="gpu-t gpu-tb" x="0" y="39">V1：float</text>
    <text class="gpu-ts" x="0" y="56">一条指令 4 字节</text>
    <g v-for="c in scalar" :key="c.i">
      <rect :class="c.t === THREAD ? 'gpu-m' : 'gpu-n'" :x="X0 + c.i * CW" y="22" :width="CW" :height="CH" />
      <text class="gpu-tk gpu-mid" :x="X0 + c.i * CW + CW / 2" y="41">t{{ c.t }}</text>
    </g>

    <text class="gpu-t gpu-tb" x="0" y="115">V2：float4</text>
    <text class="gpu-ts" x="0" y="132">一条指令 16 字节</text>
    <g v-for="c in vec" :key="c.i">
      <rect :class="c.t === THREAD ? 'gpu-m' : 'gpu-n'" :x="X0 + c.i * 4 * CW" y="98" :width="4 * CW" :height="CH" />
      <text class="gpu-tk gpu-mid" :x="X0 + c.i * 4 * CW + 2 * CW" y="117">t{{ c.t }}：x4[{{ c.i }}]</text>
    </g>

    <text class="gpu-t gpu-tb" x="0" y="178">N = 6 时三行首尾相接，每格一个 float，竖线是 16 字节边界</text>
    <g v-for="c in aligned" :key="c.i">
      <rect :class="c.row === 1 ? 'gpu-m' : 'gpu-n'" :x="AX + c.i * AW" y="214" :width="AW" :height="CH" />
      <text class="gpu-tk gpu-mid" :x="AX + c.i * AW + AW / 2" y="233">{{ c.i }}</text>
    </g>
    <g v-for="b in 5" :key="b">
      <line class="gpu-ln" :x1="AX + (b - 1) * 4 * AW" :x2="AX + (b - 1) * 4 * AW" y1="202" y2="250" />
      <text class="gpu-ts gpu-mid" :x="AX + (b - 1) * 4 * AW" y="196">{{ (b - 1) * 16 }}</text>
    </g>
    <text class="gpu-ts" x="0" y="196">字节偏移</text>
    <rect class="online-misalign" :x="misX" y="210" :width="4 * AW" :height="CH + 8" />
    <path class="gpu-ln" :d="`M${misX + 2 * AW} 246 L${misX + 2 * AW} 268`" />
    <text class="gpu-t gpu-mid" :x="misX + 2 * AW" y="285">行 1 的 x4[0] 从第 {{ rowStart * 4 }} 字节开始，除以 16 余 {{ offset }}</text>
    <text class="gpu-ts gpu-mid" :x="misX + 2 * AW" y="304">跨过了第 32 字节的边界，不满足 float4 的对齐要求</text>
  </svg>
</template>

<style>
.online-misalign {
  fill: none;
  stroke: var(--online-bad);
  stroke-width: 2.4;
}
</style>
