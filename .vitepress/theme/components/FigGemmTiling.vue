<script setup>
const TILE = 26
const STEP = 15
const G = 6
const KS = 10
const R = 2
const C = 3
const NOW = 2
const top = 44
const ax = 30
const bx = ax + KS * STEP + 50
const cx = bx + G * TILE + 50
const list = (n) => Array.from({ length: n }, (_, i) => i)
function aCls(r, k) {
  if (r !== R) return 'gpu-n'
  return k === NOW ? 'gemm-now' : 'gemm-soft'
}
function bCls(k, c) {
  if (c !== C) return 'gpu-n'
  return k === NOW ? 'gemm-now' : 'gemm-soft'
}
const ay = top + R * TILE
const by = top + NOW * STEP
const asX = ax + NOW * STEP
const bsX = bx + C * TILE
</script>

<template>
  <svg class="w680" viewBox="0 0 800 336" role="img" aria-label="C 按 BM 乘 BN 切成小块，一个 Block 负责一块。计算这一块需要 A 中 BM 行和 B 中 BN 列，它们沿 K 方向切成宽 BK 的小段，逐段搬进共享内存。">
    <defs>
      <marker id="gemm-tiling" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <text class="gpu-tt" :x="ax" y="24">A：每格 BM 行、BK 列</text>
    <g v-for="r in list(G)" :key="`a${r}`">
      <rect v-for="k in list(KS)" :key="k" :class="aCls(r, k)" :x="ax + k * STEP" :y="top + r * TILE" :width="STEP" :height="TILE" />
    </g>
    <text class="gpu-t gpu-mid" :x="bx - 25" :y="top + 3 * TILE + 4">×</text>

    <text class="gpu-tt" :x="bx" y="24">B：每格 BK 行、BN 列</text>
    <g v-for="k in list(KS)" :key="`b${k}`">
      <rect v-for="c in list(G)" :key="c" :class="bCls(k, c)" :x="bx + c * TILE" :y="top + k * STEP" :width="TILE" :height="STEP" />
    </g>
    <text class="gpu-t gpu-mid" :x="cx - 25" :y="top + 3 * TILE + 4">=</text>

    <text class="gpu-tt" :x="cx" y="24">C：一个 Block 负责 BM × BN</text>
    <g v-for="r in list(G)" :key="`c${r}`">
      <rect v-for="c in list(G)" :key="c" :class="r === R && c === C ? 'gpu-c' : 'gpu-n'" :x="cx + c * TILE" :y="top + r * TILE" :width="TILE" :height="TILE" />
    </g>
    <text class="gpu-ts" :x="cx + G * TILE + 8" :y="top + R * TILE + 17">累加</text>

    <rect class="gpu-m" x="30" y="264" width="500" height="58" />
    <text class="gpu-t gpu-tb" x="44" y="286">当前 K 步：一个 Block 的共享内存</text>
    <text class="gpu-ts" x="44" y="308">As[BM][BK] 与 Bs[BK][BN]，128 × 8 × 4 = 4 KB 各一份，共 8 KB</text>
    <line class="gpu-ln" :x1="asX + STEP / 2" :y1="ay + TILE + 2" :x2="asX + STEP / 2" y2="262" marker-end="url(#gemm-tiling)" />
    <line class="gpu-ln" :x1="bsX + TILE / 2" :y1="by + STEP + 2" :x2="bsX + TILE / 2" y2="262" marker-end="url(#gemm-tiling)" />
    <line class="gpu-ln" :x1="cx + (C + 0.5) * TILE" y1="262" :x2="cx + (C + 0.5) * TILE" :y2="top + (R + 1) * TILE + 2" marker-end="url(#gemm-tiling)" />
    <text class="gpu-ts" x="546" y="314">累加 BK 步后加载下一段</text>
  </svg>
</template>

<style>
.gemm-now {
  fill: var(--gpu-m);
  stroke: var(--gpu-m);
  stroke-width: 1;
}
.gemm-soft {
  fill: var(--gpu-m-fill);
  fill-opacity: 0.4;
  stroke: var(--gpu-m);
  stroke-width: 1;
}
</style>
