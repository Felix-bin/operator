<script setup>
const CELL = 26
const rows = (n) => Array.from({ length: n }, (_, i) => i)
const M = 5
const K = 4
const N = 6
const I = 2
const J = 3
const top = 44
const ax = 46
const bx = ax + K * CELL + 60
const cx = bx + N * CELL + 60
</script>

<template>
  <svg class="w680" viewBox="0 0 640 248" role="img" aria-label="C 的第 i 行第 j 列由 A 的第 i 行和 B 的第 j 列做内积得到。A 是 M 行 K 列，B 是 K 行 N 列，C 是 M 行 N 列。">
    <text class="gpu-tt" :x="ax" y="24">A（M × K）</text>
    <g v-for="r in rows(M)" :key="`a${r}`">
      <rect
        v-for="c in rows(K)"
        :key="c"
        :class="r === I ? 'gpu-m' : 'gpu-n'"
        :x="ax + c * CELL"
        :y="top + r * CELL"
        :width="CELL"
        :height="CELL"
      />
    </g>
    <text class="gpu-t gpu-mid" :x="ax + K * CELL + 30" :y="top + 2.6 * CELL">×</text>

    <text class="gpu-tt" :x="bx" y="24">B（K × N）</text>
    <g v-for="r in rows(K)" :key="`b${r}`">
      <rect
        v-for="c in rows(N)"
        :key="c"
        :class="c === J ? 'gpu-m' : 'gpu-n'"
        :x="bx + c * CELL"
        :y="top + r * CELL"
        :width="CELL"
        :height="CELL"
      />
    </g>
    <text class="gpu-t gpu-mid" :x="bx + N * CELL + 30" :y="top + 2.6 * CELL">=</text>

    <text class="gpu-tt" :x="cx" y="24">C（M × N）</text>
    <g v-for="r in rows(M)" :key="`c${r}`">
      <rect
        v-for="c in rows(N)"
        :key="c"
        :class="r === I && c === J ? 'gpu-c' : 'gpu-n'"
        :x="cx + c * CELL"
        :y="top + r * CELL"
        :width="CELL"
        :height="CELL"
      />
    </g>

    <text class="gpu-ts" :x="ax - 8" :y="top + I * CELL + 17" text-anchor="end">i</text>
    <text class="gpu-ts gpu-mid" :x="bx + J * CELL + 13" :y="top - 8">j</text>
    <text class="gpu-ts" :x="cx + N * CELL + 8" :y="top + I * CELL + 17">C[i][j]</text>
    <text class="gpu-t" :x="ax" y="216">C[i][j] = A[i][0]·B[0][j] + A[i][1]·B[1][j] + … + A[i][K−1]·B[K−1][j]</text>
    <text class="gpu-ts" :x="ax" y="238">每个输出元素做 K 次乘加，读 A 的一行和 B 的一列。</text>
  </svg>
</template>
