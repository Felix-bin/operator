<script setup>
const rowY = (i) => 40 + i * 52
const H = 38
const rows = [
  { in: 'Qᵢ、Kⱼ、mᵢ、lᵢ', calc: 'Pᵢⱼ = exp(QᵢKⱼᵀ / √d − mᵢ) / lᵢ', out: '' },
  { in: 'dOᵢ', calc: 'dVⱼ += Pᵢⱼᵀ dOᵢ', out: 'dVⱼ：内循环结束写一次' },
  { in: 'dOᵢ、Vⱼ、Oᵢ', calc: 'dSᵢⱼ = Pᵢⱼ ⊙ (dOᵢVⱼᵀ − Dᵢ)，Dᵢ = rowsum(dOᵢ ⊙ Oᵢ)', out: '' },
  { in: 'dQᵢ', calc: 'dQᵢ += dSᵢⱼ Kⱼ / √d', out: 'dQᵢ：每个 Tile 写回' },
  { in: 'Qᵢ（已在片上）', calc: 'dKⱼ += dSᵢⱼᵀ Qᵢ / √d', out: 'dKⱼ：内循环结束写一次' }
]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 306" role="img" aria-label="反向传播一个 Tile 的流程，自上而下：用 Q、K、m、l 重算 P；累加 dV；用 dO、V、O 算 dS；更新 dQ 和 dK。S、P、dP、dS 都不写回 HBM。">
    <defs>
      <marker id="fv1-bwd" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor" /></marker>
    </defs>
    <text class="gpu-t gpu-tb gpu-mid" x="86" y="24">从 HBM 读入</text>
    <text class="gpu-t gpu-tb gpu-mid" x="405" y="24">SRAM 内计算</text>
    <text class="gpu-t gpu-tb gpu-mid" x="708" y="24">写回 HBM</text>
    <g v-for="(r, i) in rows" :key="i">
      <rect class="gpu-m" x="16" :y="rowY(i)" width="140" :height="H" />
      <text class="gpu-tk gpu-mid" x="86" :y="rowY(i) + 23">{{ r.in }}</text>
      <line class="gpu-ln" x1="156" :y1="rowY(i) + H / 2" x2="196" :y2="rowY(i) + H / 2" marker-end="url(#fv1-bwd)" />
      <rect class="gpu-c" x="198" :y="rowY(i)" width="414" :height="H" />
      <text class="gpu-tk gpu-mid" x="405" :y="rowY(i) + 23">{{ r.calc }}</text>
      <template v-if="r.out">
        <line class="gpu-ln" x1="612" :y1="rowY(i) + H / 2" x2="628" :y2="rowY(i) + H / 2" marker-end="url(#fv1-bwd)" />
        <rect class="gpu-m" x="630" :y="rowY(i)" width="156" :height="H" />
        <text class="gpu-tk gpu-mid" x="708" :y="rowY(i) + 23">{{ r.out }}</text>
      </template>
      <line v-if="i < rows.length - 1" class="gpu-ln" x1="405" :y1="rowY(i) + H" x2="405" :y2="rowY(i + 1) - 1" marker-end="url(#fv1-bwd)" />
    </g>
  </svg>
</template>
