<script setup>
const SM = 108
const PITCH = 5.3
const cells = Array.from({ length: SM }, (_, i) => i)
const X0 = 10
const v1Used = 16 // b × h = 1 × 16
const rowsY = [56, 110, 164]
const labels = ['Q₀', 'Q₁', 'Q 末块']
const outs = ['O₀ L₀', 'O₁ L₁', 'O、L 末块']
const names = ['Block 0', 'Block 1', '末 Block']
</script>

<template>
  <svg class="flv2-w560" viewBox="0 0 590 382" role="img" aria-label="每个 Q 块对应一个 Thread Block。Block 把 Q 块和当前的 K、V 块放进共享内存，累积量放在寄存器，循环结束后把 O 和 L 写回 HBM。各 Block 之间不通信。下方对比一个 batch、16 个头时，只沿 batch 和头并行只有 16 个 Block，占 108 个 SM 中的 16 个；再沿序列并行有 1024 个 Block，可以占满所有 SM。">
    <defs>
      <marker id="flv2-sched" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor" /></marker>
    </defs>
    <text class="gpu-tt" x="10" y="22">前向：一个 Q 块，一个 Thread Block</text>
    <text class="gpu-ts gpu-mid" x="38" y="46">HBM</text>
    <text class="gpu-ts gpu-mid" x="290" y="46">SM 上的 Thread Block（4 个 Warp）</text>
    <text class="gpu-ts gpu-mid" x="546" y="46">HBM</text>

    <g v-for="(y, i) in rowsY" :key="i">
      <rect class="gpu-m" x="10" :y="y" width="56" height="44" />
      <text class="gpu-tk gpu-mid" x="38" :y="y + 26">{{ labels[i] }}</text>
      <line class="gpu-ln" x1="66" :y1="y + 28" x2="114" :y2="y + 28" marker-end="url(#flv2-sched)" />
      <text class="gpu-ts gpu-mid" x="90" :y="y + 18">读 1 次</text>

      <rect class="gpu-c" x="116" :y="y" width="350" height="44" />
      <text class="gpu-tk gpu-tb" x="124" :y="y + 26">{{ names[i] }}</text>
      <rect class="gpu-m" x="178" :y="y + 6" width="160" height="32" />
      <text class="gpu-tk gpu-mid" x="258" :y="y + 26">共享内存：Qᵢ、Kⱼ、Vⱼ</text>
      <rect class="gpu-n" x="344" :y="y + 6" width="116" height="32" />
      <text class="gpu-tk gpu-mid" x="402" :y="y + 26">寄存器：Õ、m、l</text>

      <line class="gpu-ln" x1="466" :y1="y + 28" x2="512" :y2="y + 28" marker-end="url(#flv2-sched)" />
      <text class="gpu-ts gpu-mid" x="489" :y="y + 18">写 1 次</text>
      <rect class="gpu-m" x="514" :y="y" width="66" height="44" />
      <text class="gpu-tk gpu-mid" x="547" :y="y + 26">{{ outs[i] }}</text>
    </g>
    <text class="gpu-ts" x="116" y="226">K、V 在 HBM 中只读，各 Block 逐块读取，互相没有依赖</text>

    <line class="gpu-grid" x1="10" y1="240" x2="580" y2="240" />

    <text class="gpu-tt" x="10" y="266">A100 的 108 个 SM，一个 batch、16 个头</text>
    <text class="gpu-t" x="10" y="292">只沿 batch 和头并行：b × h = 16 个 Block，占 16 / 108 个 SM</text>
    <rect
      v-for="c in cells"
      :key="`a${c}`"
      :class="c < v1Used ? 'gpu-c' : 'gpu-n'"
      :x="X0 + c * PITCH"
      y="300"
      width="4.2"
      height="16"
    />
    <text class="gpu-t" x="10" y="344">再沿序列并行：b × h × Tr = 1024 个 Block（N = 8192，Br = 128）</text>
    <rect
      v-for="c in cells"
      :key="`b${c}`"
      class="gpu-c"
      :x="X0 + c * PITCH"
      y="352"
      width="4.2"
      height="16"
    />
  </svg>
</template>

<style>
.gpu-fig svg.flv2-w560 {
  min-width: 560px;
}
</style>
