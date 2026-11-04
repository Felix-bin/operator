<script setup>
const B = 1
const H = 32
const SMS = 108
const cells = Array.from({ length: B * H }, (_, i) => ({ x: 16 + (i % 8) * 30, y: 44 + Math.floor(i / 8) * 30 }))
const sms = Array.from({ length: SMS }, (_, i) => ({ x: 596 + (i % 12) * 16, y: 44 + Math.floor(i / 12) * 16, on: i < B * H }))
const pct = Math.round((100 * B * H) / SMS)
const code = [
  { t: 'for j = 1 … T_c:', x: 292 },
  { t: '读 Kⱼ、Vⱼ', x: 312 },
  { t: 'for i = 1 … T_r:', x: 312 },
  { t: '读 Qᵢ、Oᵢ、mᵢ、lᵢ', x: 332 },
  { t: '算 Sᵢⱼ，更新统计量和输出', x: 332 },
  { t: '写回 Oᵢ、mᵢ、lᵢ', x: 332 }
]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 236" role="img" :aria-label="`batch 为 1、head 为 32 时只有 32 个 Thread Block。每个 Block 内部串行走完全部 K/V 块和 Q 块。A100 的 108 个 SM 只用到 ${pct}%。`">
    <defs>
      <marker id="fv1-grid" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor" /></marker>
    </defs>
    <text class="gpu-t gpu-tb" x="16" y="24">grid = (batch, head) = (1, 32)</text>
    <rect v-for="(c, i) in cells" :key="i" :class="i === 15 ? 'gpu-m' : 'gpu-c'" :x="c.x" :y="c.y" width="26" height="26" />
    <text class="gpu-ts" x="16" y="186">每格是一个 program，即一个 Thread Block</text>
    <text class="gpu-ts" x="16" y="204">它负责一个 (batch, head) 的全部计算</text>

    <path class="gpu-ln" d="M242 87 L278 87" marker-end="url(#fv1-grid)" />

    <text class="gpu-t gpu-tb" x="280" y="24">一个 Block 内部：串行</text>
    <rect class="gpu-c" x="280" y="44" width="280" height="136" />
    <text v-for="(l, i) in code" :key="i" class="gpu-tk" :x="l.x" :y="66 + i * 21">{{ l.t }}</text>
    <text class="gpu-ts" x="280" y="204">每个 K/V 块都在这个 Block 里走一遍</text>

    <text class="gpu-t gpu-tb" x="596" y="24">A100 的 {{ SMS }} 个 SM</text>
    <rect v-for="(s, i) in sms" :key="i" :class="s.on ? 'gpu-c' : 'gpu-n'" :x="s.x" :y="s.y" width="13" height="13" />
    <text class="gpu-ts" x="596" y="204">{{ B * H }} / {{ SMS }} ≈ {{ pct }}%，其余 SM 空闲</text>
  </svg>
</template>
