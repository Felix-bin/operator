<script setup>
const rowY = (i) => 36 + i * 24
const rows = Array.from({ length: 8 }, (_, i) => i)
</script>

<template>
  <svg class="w680" viewBox="0 0 800 256" role="img" aria-label="两级规约加广播。8 个 Warp 各自用 Shuffle 得到局部结果，写入共享内存 warp_max。屏障之后，Warp 0 读取这 8 个值再做一次 Shuffle，车道 0 把结果写入 block_result。第二次屏障之后，全部 256 个线程读到结果。">
    <defs>
      <marker id="sm-two" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor"/></marker>
    </defs>
    <text class="gpu-t gpu-tb" x="16" y="20">第一级 Shuffle</text>
    <text class="gpu-t gpu-tb gpu-mid" x="228" y="20">共享内存</text>
    <text class="gpu-t gpu-tb gpu-mid" x="418" y="20">第二级 Shuffle</text>
    <text class="gpu-t gpu-tb gpu-mid" x="596" y="20">共享内存</text>
    <text class="gpu-t gpu-tb gpu-mid" x="734" y="20">广播</text>

    <g v-for="i in rows" :key="i">
      <rect class="gpu-c" x="16" :y="rowY(i)" width="130" height="20" />
      <text class="gpu-tk gpu-mid" x="81" :y="rowY(i) + 14">Warp {{ i }}：5 次</text>
      <line class="gpu-ln" x1="146" :y1="rowY(i) + 10" x2="177" :y2="rowY(i) + 10" marker-end="url(#sm-two)" />

      <rect class="gpu-m" x="178" :y="rowY(i)" width="100" height="20" />
      <text class="gpu-tk gpu-mid" x="228" :y="rowY(i) + 14">warp_max[{{ i }}]</text>
      <line class="gpu-ln" x1="278" :y1="rowY(i) + 10" x2="359" :y2="rowY(i) + 10" marker-end="url(#sm-two)" />

      <rect class="gpu-c" x="360" :y="rowY(i)" width="116" height="20" />
      <text class="gpu-tk gpu-mid" x="418" :y="rowY(i) + 14">Warp 0 车道 {{ i }}</text>
      <line class="gpu-ln" x1="476" :y1="rowY(i) + 10" x2="541" y2="130" marker-end="url(#sm-two)" />
    </g>

    <rect class="gpu-m" x="542" y="104" width="108" height="52" />
    <text class="gpu-tk gpu-mid" x="596" y="126">block_result</text>
    <text class="gpu-ts gpu-mid" x="596" y="144">车道 0 写入</text>

    <line class="gpu-ln" x1="650" y1="130" x2="683" y2="130" marker-end="url(#sm-two)" />
    <rect class="gpu-c" x="684" y="104" width="100" height="52" />
    <text class="gpu-tk gpu-mid" x="734" y="126">全部 256 个</text>
    <text class="gpu-tk gpu-mid" x="734" y="144">线程读取</text>

    <line class="gpu-ln gpu-dash" x1="318" y1="28" x2="318" y2="236" />
    <text class="gpu-ts gpu-mid" x="318" y="250">__syncthreads()</text>
    <line class="gpu-ln gpu-dash" x1="667" y1="28" x2="667" y2="236" />
    <text class="gpu-ts gpu-mid" x="667" y="250">__syncthreads()</text>
  </svg>
</template>
