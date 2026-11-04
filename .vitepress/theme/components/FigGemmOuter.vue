<script setup>
const TM = 8
const TN = 8
const list = (n) => Array.from({ length: n }, (_, i) => i)
const CELL = 19
const ax = 420
const ay = 74
const bx = ax + 42
const by = ay - 30
const fma = TM * TN
const reads = TM + TN
const flopPerByte = (2 * fma) / (4 * reads)
const gain = flopPerByte / 0.25
</script>

<template>
  <svg class="w680" viewBox="0 0 800 306" role="img" aria-label="内积每做 1 次乘加就要读 2 个 float。外积先读 8 个 a 和 8 个 b 到寄存器，再做 64 次乘加。">
    <defs>
      <marker id="gemm-outer" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <text class="gpu-tt" x="16" y="22">内积：一次累加一个输出</text>
    <rect class="gpu-m" x="30" y="60" width="86" height="34" />
    <text class="gpu-t gpu-mid" x="73" y="82">As[i][k]</text>
    <rect class="gpu-m" x="30" y="120" width="86" height="34" />
    <text class="gpu-t gpu-mid" x="73" y="142">Bs[k][j]</text>
    <line class="gpu-ln" x1="118" y1="77" x2="208" y2="104" marker-end="url(#gemm-outer)" />
    <line class="gpu-ln" x1="118" y1="137" x2="208" y2="112" marker-end="url(#gemm-outer)" />
    <rect class="gpu-c" x="212" y="90" width="110" height="34" />
    <text class="gpu-t gpu-mid" x="267" y="112">1 次 FMA</text>
    <text class="gpu-t" x="30" y="196">读 2 个 float，做 1 次 FMA</text>
    <text class="gpu-t" x="30" y="218">0.5 次 FMA / float</text>
    <text class="gpu-t gpu-tb" x="30" y="240">0.25 FLOP/byte</text>

    <text class="gpu-tt" :x="ax - 20" y="22">外积：寄存器上累加一块</text>
    <text class="gpu-ts" :x="ax - 20" :y="ay + TM * CELL / 2 + 4">a</text>
    <rect v-for="i in list(TM)" :key="`a${i}`" class="gpu-m" :x="ax" :y="ay + i * CELL" width="18" :height="CELL" />
    <text class="gpu-ts gpu-mid" :x="bx + TN * CELL / 2" :y="by - 4">b</text>
    <rect v-for="j in list(TN)" :key="`b${j}`" class="gpu-m" :x="bx + j * CELL" :y="by" :width="CELL" height="18" />
    <g v-for="i in list(TM)" :key="`c${i}`">
      <rect v-for="j in list(TN)" :key="j" class="gpu-c" :x="bx + j * CELL" :y="ay + i * CELL" :width="CELL" :height="CELL" />
    </g>
    <text class="gpu-t" :x="bx + TN * CELL + 18" :y="ay + 30">c_frag</text>
    <text class="gpu-ts" :x="bx + TN * CELL + 18" :y="ay + 52">{{ fma }} 个寄存器</text>
    <text class="gpu-ts" :x="bx + TN * CELL + 18" :y="ay + 72">每个独立累加</text>
    <text class="gpu-t" :x="ax - 20" y="258">读 {{ reads }} 个 float，做 {{ fma }} 次 FMA</text>
    <text class="gpu-t" :x="ax - 20" y="280">{{ fma / reads }} 次 FMA / float</text>
    <text class="gpu-t gpu-tb" :x="ax - 20" y="300">{{ flopPerByte }} FLOP/byte，是内积的 {{ gain }} 倍</text>
  </svg>
</template>
