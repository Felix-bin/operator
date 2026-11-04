<script setup>
const x0 = 176
const w = 116
const gap = 8
const slotX = (i) => x0 + i * (w + gap)
const slots = ['i = 1', 'i = 2', '…', 'i = T_r', '下一个 j，i = 1']
const rowY = (r) => 36 + r * 34
const rows = [
  { name: 'Kⱼ、Vⱼ', size: '2 · B_c · d' },
  { name: 'Qᵢ', size: 'B_r · d' },
  { name: 'Oᵢ', size: 'B_r · d' },
  { name: 'mᵢ、lᵢ', size: '2 · B_r' },
  { name: 'Sᵢⱼ、P̃ᵢⱼ', size: 'B_r · B_c' }
]
const perSlot = [
  { r: 1, cls: 'gpu-m', t: '读入' },
  { r: 2, cls: 'gpu-m', t: '读入、更新、写回' },
  { r: 3, cls: 'gpu-m', t: '读入、更新、写回' },
  { r: 4, cls: 'gpu-c', t: '算完就丢' }
]
const active = [0, 1, 3, 4]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 212" role="img" aria-label="内循环时间线。K_j 和 V_j 在整个内循环常驻 SRAM；每次内循环换入 Q_i、O_i、m_i、l_i，并把后三者写回；S_ij 和 P_ij 只在片上存在，从不写回。">
    <text v-for="(s, i) in slots" :key="s" class="gpu-ts gpu-mid" :x="slotX(i) + w / 2" y="22">{{ s }}</text>
    <g v-for="(r, ri) in rows" :key="r.name">
      <text class="gpu-t" x="16" :y="rowY(ri) + 18">{{ r.name }}</text>
      <text class="gpu-ts" x="90" :y="rowY(ri) + 18">{{ r.size }}</text>
    </g>
    <rect class="gpu-m" :x="slotX(0)" :y="rowY(0)" :width="4 * w + 3 * gap" height="26" />
    <text class="gpu-tk gpu-mid" :x="slotX(0) + (4 * w + 3 * gap) / 2" :y="rowY(0) + 17">外循环开始时读一次，整个内循环常驻</text>
    <rect class="gpu-m" :x="slotX(4)" :y="rowY(0)" :width="w" height="26" />
    <text class="gpu-tk gpu-mid" :x="slotX(4) + w / 2" :y="rowY(0) + 17">换入新块</text>
    <template v-for="i in active" :key="i">
      <g v-for="p in perSlot" :key="p.r">
        <rect :class="p.cls" :x="slotX(i)" :y="rowY(p.r)" :width="w" height="26" />
        <text class="gpu-tk gpu-mid" :x="slotX(i) + w / 2" :y="rowY(p.r) + 17">{{ p.t }}</text>
      </g>
    </template>
    <text v-for="r in [1, 2, 3, 4]" :key="r" class="gpu-t gpu-mid" :x="slotX(2) + w / 2" :y="rowY(r) + 18">…</text>
  </svg>
</template>
