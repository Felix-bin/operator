<script setup>
const lanes = [0, 1, 2, 3]
const S = 128
const CELL = S / 4
</script>

<template>
  <div class="flv2-pair">
    <svg viewBox="0 0 384 326" role="img" aria-label="V1 把 K 和 V 切给 4 个 Warp，Q 由所有 Warp 共享，每个 Warp 只算出 S 的若干列，同一行分散在 4 个 Warp 中，部分结果要写入共享内存、同步、再相加。">
      <text class="gpu-tt" x="10" y="22">V1：split-K，沿 K/V 切</text>

      <rect class="gpu-m" x="10" y="96" width="34" :height="S" />
      <text class="gpu-tk gpu-mid" x="27" y="164">Q</text>

      <g v-for="w in lanes" :key="`k${w}`">
        <rect class="gpu-c" :x="56 + w * CELL" y="50" :width="CELL" height="34" />
        <text class="gpu-tk gpu-mid" :x="56 + w * CELL + CELL / 2" y="71">W{{ w }}</text>
        <rect class="gpu-c" :x="56 + w * CELL" y="96" :width="CELL" :height="S" />
        <text class="gpu-tk gpu-mid" :x="56 + w * CELL + CELL / 2" y="136">W{{ w }}</text>
      </g>
      <text class="gpu-ts gpu-mid" x="120" y="44">Kᵀ</text>
      <line class="gpu-ln gpu-dash" x1="60" y1="172" x2="180" y2="172" />
      <rect x="68" y="181" width="104" height="20" style="fill: var(--gpu-c-fill)" />
      <text class="gpu-tk gpu-mid" x="120" y="196">一行跨 4 个 Warp</text>

      <g v-for="w in lanes" :key="`v${w}`">
        <rect class="gpu-c" x="198" :y="96 + w * CELL" width="34" :height="CELL" />
        <text class="gpu-tk gpu-mid" x="215" :y="96 + w * CELL + 20">W{{ w }}</text>
      </g>
      <text class="gpu-ts gpu-mid" x="215" y="88">V</text>

      <text class="gpu-t gpu-mid" x="248" y="164">=</text>

      <g v-for="w in lanes" :key="`o${w}`">
        <rect class="gpu-c" :x="262 + (3 - w) * 7" :y="96 + w * 7" width="60" :height="S - 21" />
      </g>
      <text class="gpu-tk gpu-mid" x="292" y="174">4 份</text>
      <text class="gpu-ts gpu-mid" x="302" y="88">部分 O</text>

      <rect class="gpu-m" x="10" y="240" width="362" height="28" />
      <text class="gpu-tk gpu-mid" x="191" y="258">写入共享内存 → __syncthreads → 读回相加 → 一行 O</text>
      <text class="gpu-ts" x="10" y="292">row-max 和 row-sum 也只有部分列，同样要合并</text>
      <text class="gpu-ts" x="10" y="314">Bc = 128 时，每个 Warp 负责 32 列</text>
    </svg>

    <svg viewBox="0 0 384 326" role="img" aria-label="V2 把 Q 切给 4 个 Warp，K 和 V 由所有 Warp 共享，每个 Warp 算出自己的若干行，并独占对应的输出行，不用合并结果。">
      <text class="gpu-tt" x="10" y="22">V2：沿 Q 切</text>

      <g v-for="w in lanes" :key="`q${w}`">
        <rect class="gpu-c" x="10" :y="96 + w * CELL" width="34" :height="CELL" />
        <text class="gpu-tk gpu-mid" x="27" :y="96 + w * CELL + 20">W{{ w }}</text>
        <rect class="gpu-c" x="56" :y="96 + w * CELL" :width="S" :height="CELL" />
        <text class="gpu-tk gpu-mid" x="120" :y="96 + w * CELL + 20">W{{ w }}</text>
        <rect class="gpu-c" x="262" :y="96 + w * CELL" width="60" :height="CELL" />
        <text class="gpu-tk gpu-mid" x="292" :y="96 + w * CELL + 20">W{{ w }}</text>
      </g>
      <text class="gpu-ts gpu-mid" x="27" y="88">Q</text>

      <rect class="gpu-m" x="56" y="50" :width="S" height="34" />
      <text class="gpu-tk gpu-mid" x="120" y="71">所有 Warp 共享</text>
      <text class="gpu-ts gpu-mid" x="120" y="44">Kᵀ</text>

      <rect class="gpu-m" x="198" y="96" width="34" :height="S" />
      <text class="gpu-tk gpu-mid" x="215" y="150">共享</text>
      <text class="gpu-tk gpu-mid" x="215" y="168">V</text>
      <text class="gpu-ts gpu-mid" x="215" y="88">V</text>

      <text class="gpu-t gpu-mid" x="248" y="164">=</text>
      <text class="gpu-ts gpu-mid" x="292" y="88">O</text>

      <rect class="gpu-n" x="10" y="240" width="362" height="28" />
      <text class="gpu-tk gpu-mid" x="191" y="258">row-max、row-sum 和输出都在本 Warp 内完成，不合并</text>
      <text class="gpu-ts" x="10" y="292">Warp 之间仍要 __syncthreads，等 K、V 载入共享内存</text>
      <text class="gpu-ts" x="10" y="314">Br = 128 时，每个 Warp 负责 32 行</text>
    </svg>
  </div>
</template>
