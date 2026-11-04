<script setup>
// 按真实比例：N = 4096，d = 128，M = 98304，由论文公式得 B_c = 192，B_r = 128
const N = 4096
const Bc = 192
const Br = 128
const mx = 440
const my = 84
const mw = 200
const mh = 220
const stripW = 20
const cx = (r) => mx + (Math.min(r, N) / N) * mw
const cy = (r) => my + (Math.min(r, N) / N) * mh
const j = 7
const i = 12
const tile = { x: cx(j * Bc), y: cy(i * Br), w: cx((j + 1) * Bc) - cx(j * Bc), h: cy((i + 1) * Br) - cy(i * Br) }
const vLines = []
for (let r = Bc; r < N; r += Bc) vLines.push(cx(r))
const hLines = []
for (let r = Br; r < N; r += Br) hLines.push(cy(r))
const gridPath = [
  ...vLines.map((x) => `M${x} ${my} V${my + mh}`),
  ...hLines.map((y) => `M${mx} ${y} H${mx + mw}`)
].join(' ')
const kvTicks = vLines.map((x) => `M${x} ${my - 30} V${my - 10}`).join(' ')
const qTicks = hLines.map((y) => `M${mx - 30} ${y} H${mx - 10}`).join(' ')
</script>

<template>
  <svg class="w680" viewBox="0 0 800 330" role="img" aria-label="左图：标准实现在 HBM 中存放完整的 N 乘 N 矩阵 S 和 P。右图：FlashAttention 把矩阵切成 32 行 22 列的 Tile，每次只有一个 Tile 在 SRAM 中计算，矩阵本身从不存在。">
    <defs>
      <marker id="fv1-til" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor" /></marker>
    </defs>

    <text class="gpu-tt" x="16" y="22">标准：矩阵真实存在于 HBM</text>
    <rect class="gpu-m" x="70" y="92" width="190" height="190" />
    <text class="gpu-t gpu-mid" x="165" y="270">P，N × N，32 MiB</text>
    <rect class="gpu-m" x="40" y="62" width="190" height="190" />
    <text class="gpu-t gpu-mid" x="135" y="160">S，N × N，32 MiB</text>
    <text class="gpu-ts" x="40" y="304">N = 4096、FP16，和 d 无关</text>

    <text class="gpu-tt" x="330" y="22">FlashAttention：矩阵只是一张计算顺序表</text>
    <rect class="gpu-m" :x="mx" :y="my - 30" :width="mw" :height="stripW" />
    <path class="gpu-thin" :d="kvTicks" />
    <rect class="gpu-m" :x="mx - 30" :y="my" :width="stripW" :height="mh" />
    <path class="gpu-thin" :d="qTicks" />
    <text class="gpu-ts gpu-end" :x="mx - 8" :y="my - 15">K、V</text>
    <text class="gpu-ts gpu-end" :x="mx - 36" :y="my + 14">Q、O</text>
    <text class="gpu-ts gpu-end" :x="mx - 36" :y="my + 30">m、l</text>

    <path class="gpu-grid" :d="gridPath" fill="none" />
    <rect class="gpu-o gpu-dash" :x="mx" :y="my" :width="mw" :height="mh" />
    <rect :x="tile.x" :y="my - 30" :width="tile.w" :height="stripW" fill="var(--gpu-m)" />
    <rect :x="mx - 30" :y="tile.y" :width="stripW" :height="tile.h" fill="var(--gpu-m)" />
    <rect :x="tile.x" :y="tile.y" :width="tile.w" :height="tile.h" fill="var(--gpu-c)" />

    <rect class="gpu-c" x="668" y="130" width="120" height="122" />
    <text class="gpu-t gpu-tb gpu-mid" x="728" y="150">SRAM</text>
    <text class="gpu-tk gpu-mid" x="728" y="172">Kⱼ、Vⱼ</text>
    <text class="gpu-tk gpu-mid" x="728" y="190">Qᵢ、Oᵢ、mᵢ、lᵢ</text>
    <text class="gpu-tk gpu-mid" x="728" y="208">Sᵢⱼ：128 × 192</text>
    <text class="gpu-ts gpu-mid" x="728" y="236">用完即弃</text>
    <path class="gpu-ln" :d="`M${tile.x + tile.w} ${tile.y + tile.h / 2} L666 ${tile.y + tile.h / 2}`" marker-end="url(#fv1-til)" />
    <text class="gpu-ts" :x="mx" :y="my + mh + 18">虚线框不占存储：32 × 22 个 Tile 依次算完</text>
  </svg>
</template>
