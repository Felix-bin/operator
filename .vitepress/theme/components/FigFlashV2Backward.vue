<script setup>
const idx = [0, 1, 2, 3]
const SUB = ['₀', '₁', '₂', '₃']
const C = 44
const F = { x: 50, y: 66 }
const B = { x: 50, y: 66 }
</script>

<template>
  <div class="flv2-pair">
    <svg viewBox="0 0 370 288" role="img" aria-label="前向时每个 Thread Block 负责注意力矩阵的一行块，独占写 O 和 L。">
      <defs>
        <marker id="flv2-bwd-a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor" /></marker>
      </defs>
      <text class="gpu-tt" x="10" y="24">前向：每行块一个 Block</text>
      <text class="gpu-ts" :x="F.x" y="54">K/V 块 →</text>
      <g v-for="r in idx" :key="r">
        <text class="gpu-t" x="14" :y="F.y + r * C + 25">Q{{ SUB[r] }}</text>
        <rect class="gpu-c" :x="F.x" :y="F.y + r * C" :width="C * 4" :height="C - 4" />
        <text class="gpu-tk gpu-mid" :x="F.x + C * 2" :y="F.y + r * C + 25">Block {{ r }}</text>
        <line class="gpu-ln" :x1="F.x + C * 4" :y1="F.y + r * C + 20" :x2="F.x + C * 4 + 32" :y2="F.y + r * C + 20" marker-end="url(#flv2-bwd-a)" />
        <rect class="gpu-m" :x="F.x + C * 4 + 34" :y="F.y + r * C + 2" width="72" :height="C - 8" />
        <text class="gpu-tk gpu-mid" :x="F.x + C * 4 + 70" :y="F.y + r * C + 25">O{{ SUB[r] }} L{{ SUB[r] }}</text>
      </g>
      <text class="gpu-ts" x="10" y="268">每个输出块只有一个写者，不用同步</text>
    </svg>

    <svg viewBox="0 0 370 420" role="img" aria-label="反向时每个 Thread Block 负责一列块，独占写 dK 和 dV；同一个 dQ 块要接收所有列块的贡献，所以用原子加累加。反向开始前，由独立的预处理 Kernel 算好 D。">
      <defs>
        <marker id="flv2-bwd-b" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor" /></marker>
      </defs>
      <text class="gpu-tt" x="10" y="24">反向：每列块一个 Block</text>
      <g v-for="c in idx" :key="`c${c}`">
        <text class="gpu-ts gpu-mid" :x="B.x + c * C + (C - 4) / 2" y="54">K{{ SUB[c] }}V{{ SUB[c] }}</text>
        <rect class="gpu-c" :x="B.x + c * C" :y="B.y" :width="C - 4" :height="C * 4" />
        <text class="gpu-tk gpu-mid" :x="B.x + c * C + (C - 4) / 2" :y="B.y + C * 2 + 4">B{{ c }}</text>
        <line class="gpu-ln" :x1="B.x + c * C + (C - 4) / 2" :y1="B.y + C * 4" :x2="B.x + c * C + (C - 4) / 2" :y2="B.y + C * 4 + 18" marker-end="url(#flv2-bwd-b)" />
        <rect class="gpu-m" :x="B.x + c * C" :y="B.y + C * 4 + 20" :width="C - 4" height="44" />
        <text class="gpu-ts gpu-mid" :x="B.x + c * C + (C - 4) / 2" :y="B.y + C * 4 + 38">dK{{ SUB[c] }}</text>
        <text class="gpu-ts gpu-mid" :x="B.x + c * C + (C - 4) / 2" :y="B.y + C * 4 + 56">dV{{ SUB[c] }}</text>
      </g>
      <g v-for="r in idx" :key="`q${r}`">
        <text class="gpu-t" x="14" :y="B.y + r * C + 25">Q{{ SUB[r] }}</text>
        <line class="gpu-ln" :x1="B.x + C * 4" :y1="B.y + r * C + 20" :x2="B.x + C * 4 + 28" :y2="B.y + r * C + 20" marker-end="url(#flv2-bwd-b)" />
        <rect class="gpu-m" :x="B.x + C * 4 + 30" :y="B.y + r * C + 2" width="60" :height="C - 8" />
        <text class="gpu-tk gpu-mid" :x="B.x + C * 4 + 60" :y="B.y + r * C + 25">dQ{{ SUB[r] }}</text>
      </g>
      <text class="gpu-ts" :x="B.x + C * 4 + 10" :y="B.y + C * 4 + 22">所有列块都向</text>
      <text class="gpu-ts" :x="B.x + C * 4 + 10" :y="B.y + C * 4 + 40">同一个 dQ 块</text>
      <text class="gpu-ts" :x="B.x + C * 4 + 10" :y="B.y + C * 4 + 58">做原子加</text>

      <rect class="gpu-n" x="10" y="326" width="350" height="84" />
      <text class="gpu-t gpu-tb" x="22" y="350">反向开始前的预处理 Kernel</text>
      <text class="gpu-tk" x="22" y="372">D = rowsum(dO ⊙ O)，每行一个数，写入 HBM</text>
      <text class="gpu-tk" x="22" y="392">反向主循环直接读取，不用每个 Tile 现算</text>
    </svg>
  </div>
</template>
