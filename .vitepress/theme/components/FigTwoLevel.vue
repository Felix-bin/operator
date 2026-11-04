<script setup>
const rowY = (i) => 34 + i * 24
const rows = Array.from({ length: 8 }, (_, i) => i)
</script>

<template>
  <svg class="w680" viewBox="0 0 800 236" role="img" aria-label="两级规约：8 个 Warp 各自用 shuffle 得到一个值，写入共享内存的 8 个位置，Warp 0 把它们读入寄存器后再做一次 shuffle，得到 Block 的总和。">
    <defs>
      <marker id="rd-two" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor"/></marker>
    </defs>
    <text class="gpu-t gpu-tb" x="16" y="20">第一级：各 Warp 内 5 次 shuffle</text>
    <text class="gpu-t gpu-tb gpu-mid" x="360" y="20">共享内存</text>
    <text class="gpu-t gpu-tb" x="470" y="20">第二级：Warp 0 内 5 次 shuffle</text>

    <g v-for="i in rows" :key="i">
      <rect class="gpu-c" x="16" :y="rowY(i)" width="196" height="20" />
      <text class="gpu-tk gpu-mid" x="114" :y="rowY(i) + 14">Warp {{ i }}：32 个值 → 车道 0</text>
      <line class="gpu-ln" x1="212" :y1="rowY(i) + 10" x2="309" :y2="rowY(i) + 10" marker-end="url(#rd-two)" />

      <rect class="gpu-m" x="310" :y="rowY(i)" width="100" height="20" />
      <text class="gpu-tk gpu-mid" x="360" :y="rowY(i) + 14">smem[{{ i }}]</text>
      <line class="gpu-ln" x1="410" :y1="rowY(i) + 10" x2="469" :y2="rowY(i) + 10" marker-end="url(#rd-two)" />

      <rect class="gpu-c" x="470" :y="rowY(i)" width="120" height="20" />
      <text class="gpu-tk gpu-mid" x="530" :y="rowY(i) + 14">Warp 0 车道 {{ i }}</text>
      <line class="gpu-ln" x1="590" :y1="rowY(i) + 10" x2="659" y2="130" marker-end="url(#rd-two)" />
    </g>

    <rect class="gpu-c" x="660" y="104" width="124" height="52"/>
    <text class="gpu-t gpu-tb gpu-mid" x="722" y="126">车道 0</text>
    <text class="gpu-tk gpu-mid" x="722" y="144">Block 的总和</text>
  </svg>
</template>
