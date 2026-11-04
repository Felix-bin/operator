<script setup>
const BM = 128
const BN = 128
const BK = 8
const BS = 256

// tileA：第 0 号 Warp（tid 0~31）负责的格子，由 §4.5 的循环算出
const aCells = []
for (let t = 0; t < 32; t++) {
  const x = t % BK
  for (let i = Math.floor(t / BK), round = 0; i < BM; i += BS / BK, round++) aCells.push({ x, i, round })
}
// tileB：第 0 号 Warp
const bCells = []
for (let t = 0; t < 32; t++) {
  const y = Math.floor(t / 32)
  for (let j = t % 32, round = 0; j < BN; j += 32, round++) bCells.push({ y, j, round })
}
// tileC：tid = 0 与 tid = 21 负责的元素
const ownC = (t) => {
  const out = []
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) out.push({ r: Math.floor(t / 16) + i * 16, c: t % 16 + j * 16 })
  return out
}
const c0 = ownC(0)
const c21 = ownC(21)

const Bx = 40
const By = 44
const BCW = 5.2
const BRH = 9
const Ax = 40
const Ay = 200
const ACW = 9
const ARH = 2
const Cx = 330
const Cy = 200
const CSZ = 2
const roundFill = (r) => [1, 0.72, 0.5, 0.32][r]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 482" role="img" aria-label="加载 tileB 时一个 Warp 铺满一行的 32 个连续元素，循环 4 次覆盖 128 列。加载 tileA 时一个 Warp 覆盖 4 行，每行 8 个连续元素，循环 4 次覆盖 128 行。计算 tileC 时每个线程负责 8 乘 8 个元素，行列间距都是 16。">
    <text class="gpu-tt" :x="Bx" y="24">加载 tileB（8 × 128）：Warp 0 负责第 0 行</text>
    <rect class="gpu-n" :x="Bx" :y="By" :width="BN * BCW" :height="BK * BRH" />
    <g v-for="(c, i) in bCells" :key="i">
      <rect class="gemm-rd" :x="Bx + c.j * BCW" :y="By + c.y * BRH" :width="BCW" :height="BRH" :style="{ opacity: roundFill(c.round) }" />
    </g>
    <line v-for="n in 3" :key="n" class="gpu-ln gpu-dash" :x1="Bx + n * 32 * BCW" :y1="By - 6" :x2="Bx + n * 32 * BCW" :y2="By + BK * BRH + 6" />
    <text class="gpu-ts" :x="Bx + BN * BCW + 12" :y="By + 14">深浅表示</text>
    <text class="gpu-ts" :x="Bx + BN * BCW + 12" :y="By + 32">第几轮循环</text>
    <text class="gpu-ts" :x="Bx" :y="By + BK * BRH + 22">32 个相邻线程读 32 个连续 float，同一 Warp 的访问完全连续。下一个 Warp 负责下一行。</text>

    <text class="gpu-tt" :x="Ax" :y="Ay - 22">加载 tileA（128 × 8）</text>
    <rect class="gpu-n" :x="Ax" :y="Ay" :width="BK * ACW" :height="BM * ARH" />
    <g v-for="(c, i) in aCells" :key="i">
      <rect class="gemm-rd" :x="Ax + c.x * ACW" :y="Ay + c.i * ARH" :width="ACW" :height="ARH" :style="{ opacity: roundFill(c.round) }" />
    </g>
    <line v-for="n in 3" :key="n" class="gpu-ln gpu-dash" :x1="Ax - 6" :y1="Ay + n * 32 * ARH" :x2="Ax + BK * ACW + 6" :y2="Ay + n * 32 * ARH" />
    <text class="gpu-ts" :x="Ax + BK * ACW + 14" :y="Ay + 14">Warp 0 每轮</text>
    <text class="gpu-ts" :x="Ax + BK * ACW + 14" :y="Ay + 32">读 4 行 × 8 列</text>
    <text class="gpu-ts" :x="Ax + BK * ACW + 14" :y="Ay + 50">每轮跨 32 行</text>
    <text class="gpu-ts" :x="Ax + BK * ACW + 14" :y="Ay + 68">4 轮覆盖 128 行</text>

    <text class="gpu-tt" :x="Cx" :y="Cy - 22">累加 tileC（128 × 128）</text>
    <rect class="gpu-n" :x="Cx" :y="Cy" :width="BN * CSZ" :height="BM * CSZ" />
    <rect v-for="(p, i) in c0" :key="`a${i}`" class="gpu-c" :x="Cx + p.c * CSZ" :y="Cy + p.r * CSZ" :width="CSZ * 3" :height="CSZ * 3" />
    <rect v-for="(p, i) in c21" :key="`b${i}`" class="gemm-own2" :x="Cx + p.c * CSZ" :y="Cy + p.r * CSZ" :width="CSZ * 3" :height="CSZ * 3" />
    <rect class="gpu-c" :x="Cx + BN * CSZ + 20" :y="Cy + 6" width="14" height="14" />
    <text class="gpu-ts" :x="Cx + BN * CSZ + 42" :y="Cy + 18">tid 0</text>
    <rect class="gemm-own2" :x="Cx + BN * CSZ + 20" :y="Cy + 30" width="14" height="14" />
    <text class="gpu-ts" :x="Cx + BN * CSZ + 42" :y="Cy + 42">tid 21</text>
    <text class="gpu-ts" :x="Cx + BN * CSZ + 20" :y="Cy + 74">每个线程 8 × 8 个</text>
    <text class="gpu-ts" :x="Cx + BN * CSZ + 20" :y="Cy + 92">输出，行列都跳 16</text>
    <text class="gpu-ts" :x="Cx + BN * CSZ + 20" :y="Cy + 110">256 个线程正好</text>
    <text class="gpu-ts" :x="Cx + BN * CSZ + 20" :y="Cy + 128">不重不漏</text>
  </svg>
</template>

<style>
.gemm-rd {
  fill: var(--gpu-m);
  stroke: none;
}
.gemm-own2 {
  fill: var(--gpu-f-fill);
  stroke: var(--gpu-f);
  stroke-width: 1.2;
}
</style>
