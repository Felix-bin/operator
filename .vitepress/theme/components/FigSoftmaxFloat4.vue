<script setup>
const X0 = 150
const CELL = 19
const LANES = 8

const single = Array.from({ length: LANES }, (_, i) => i)
const iterV2 = Array.from({ length: 16 }, (_, i) => i)
const iterV3 = Array.from({ length: 4 }, (_, i) => i)
</script>

<template>
  <svg class="w680" viewBox="0 0 800 318" role="img" aria-label="float 与 float4 的对比。同一条读指令，float 每条车道带回 4 字节，float4 每条车道带回 16 字节。一个线程读完自己负责的 64 字节，float 要 16 条读指令，float4 只要 4 条。">
    <text class="gpu-tt" x="16" y="20">一条读指令带回多少数据（画出 8 条车道，实际 32 条）</text>

    <text class="gpu-t" x="16" y="56">V2：float</text>
    <g v-for="i in single" :key="`a${i}`">
      <rect class="gpu-m" :x="X0 + i * CELL" y="40" :width="CELL" height="24" />
      <text class="gpu-tk gpu-mid" :x="X0 + i * CELL + CELL / 2" y="57">{{ i }}</text>
    </g>
    <text class="gpu-ts" :x="X0" y="82">32 条车道：128 字节，4 次 32 字节事务</text>

    <text class="gpu-t" x="16" y="116">V3：float4</text>
    <g v-for="i in single" :key="`b${i}`">
      <rect class="gpu-m" :x="X0 + i * CELL * 4" y="100" :width="CELL * 4" height="24" />
      <text class="gpu-tk gpu-mid" :x="X0 + i * CELL * 4 + CELL / 2" y="117">{{ i }}</text>
      <line v-for="k in 3" :key="k" class="gpu-grid" :x1="X0 + i * CELL * 4 + k * CELL" y1="100" :x2="X0 + i * CELL * 4 + k * CELL" y2="124" />
    </g>
    <text class="gpu-ts" :x="X0" y="142">32 条车道：512 字节，16 次 32 字节事务</text>

    <text class="gpu-tt" x="16" y="186">线程 0 读完自己负责的 64 字节（N = 4096，Block = 256）</text>

    <text class="gpu-t" x="16" y="222">V2：16 次循环</text>
    <g v-for="i in iterV2" :key="`c${i}`">
      <rect class="gpu-c" :x="X0 + i * 38" y="206" width="36" height="24" />
    </g>
    <text class="gpu-t" x="16" y="282">V3：4 次循环</text>
    <g v-for="i in iterV3" :key="`d${i}`">
      <rect class="gpu-c" :x="X0 + i * 152" y="266" width="150" height="24" />
    </g>
    <text class="gpu-ts" :x="X0" y="248">每格是一条读指令和一轮循环开销，读回 4 字节</text>
    <text class="gpu-ts" :x="X0" y="308">总字节数不变，读指令和循环开销减到 1/4</text>
  </svg>
</template>
