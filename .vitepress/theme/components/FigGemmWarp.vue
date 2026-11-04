<script setup>
const TM = 8
const TN = 8
const SC = 2
const BX = 30
const BY = 46
const warps = Array.from({ length: 8 }, (_, w) => ({ w, r: w >> 1, c: w & 1 }))

const LX = 400
const LY = 70
const LW = 44
const LH = 38
const TID = 18
const lanes = Array.from({ length: 32 }, (_, lane) => ({ lane, r: lane >> 3, c: lane & 7 }))
const pick = { warp: TID >> 5, lane: TID & 31 }
const rowC = ((pick.warp >> 1) * 4 + (pick.lane >> 3)) * TM
const colC = ((pick.warp & 1) * 8 + (pick.lane & 7)) * TN
</script>

<template>
  <svg class="w680" viewBox="0 0 800 346" role="img" aria-label="256 个线程的 Block 分成 8 个 Warp，按 4 行 2 列排布，每个 Warp 负责 32 乘 64 的子块。一个 Warp 内 32 条车道排成 4 行 8 列，tid 18 负责第 16 到 23 行、第 16 到 23 列。">
    <text class="gpu-tt" :x="BX" y="24">Block 的 tileC：128 × 128</text>
    <g v-for="x in warps" :key="x.w">
      <rect :class="x.w === 0 ? 'gpu-c' : 'gpu-n'" :x="BX + x.c * 64 * SC" :y="BY + x.r * 32 * SC" :width="64 * SC" :height="32 * SC" />
      <text class="gpu-t gpu-mid" :x="BX + (x.c + 0.5) * 64 * SC" :y="BY + (x.r + 0.5) * 32 * SC + 5">Warp {{ x.w }}</text>
    </g>

    <text class="gpu-tt" :x="LX" y="24">Warp 0：每条车道负责 8 × 8</text>
    <text class="gpu-ts" :x="LX" y="50">方格内是 lane_id，第 0 行第 0 列对应 tileC 的前 8 行、前 8 列</text>
    <g v-for="l in lanes" :key="l.lane">
      <rect :class="l.lane === pick.lane ? 'gpu-m' : 'gpu-c'" :x="LX + l.c * LW" :y="LY + l.r * LH" :width="LW" :height="LH" />
      <text class="gpu-t gpu-mid" :x="LX + (l.c + 0.5) * LW" :y="LY + (l.r + 0.5) * LH + 5">{{ l.lane }}</text>
    </g>
    <text class="gpu-t" :x="LX" :y="LY + 4 * LH + 28">tid {{ TID }}：lane_row = {{ pick.lane >> 3 }}，lane_col = {{ pick.lane & 7 }}</text>
    <text class="gpu-t" :x="LX" :y="LY + 4 * LH + 50">负责 tileC 第 {{ rowC }}~{{ rowC + TM - 1 }} 行、第 {{ colC }}~{{ colC + TN - 1 }} 列</text>
    <text class="gpu-ts" :x="BX" y="332">256 个线程各负责 8 × 8，4 个 Warp 行 × 2 个 Warp 列，覆盖 128 × 128。</text>
  </svg>
</template>
