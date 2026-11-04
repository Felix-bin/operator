<script setup>
const rows = [
  {
    y: 28,
    code: 'float cache[K];',
    sub: 'K 编译期已知，下标是常量',
    boxes: [{ x: 322, w: 412, cls: 'gpu-c', a: '寄存器', b: '不占内存，但 K = 32 时要占 128 个寄存器，挤占驻留线程数' }]
  },
  {
    y: 112,
    code: 'reg_cache[count] = xi;',
    sub: 'count 运行时才知道（原稿写法）',
    boxes: [
      { x: 322, w: 120, cls: 'gpu-m', a: 'L1', b: '192 KB/SM' },
      { x: 468, w: 120, cls: 'gpu-m', a: 'L2', b: '共 40 MB' },
      { x: 614, w: 120, cls: 'gpu-m', a: 'HBM', b: '未命中才到这里' }
    ]
  },
  {
    y: 196,
    code: '__shared__ / smem[i]',
    sub: '一行放在 Block 共享的片上内存',
    boxes: [{ x: 322, w: 412, cls: 'gpu-m', a: 'Shared Memory', b: '片上，不进显存；受 Block 容量和驻留 Block 数限制' }]
  }
]
</script>

<template>
  <svg class="w680" viewBox="0 0 744 268" role="img" aria-label="暂存一行数据的三条路径：编译期常量下标的数组进寄存器；运行时下标的数组进局部内存，经过 L1 和 L2，未命中时到显存；放进共享内存则留在片上。">
    <defs>
      <marker id="online-cache-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor" /></marker>
    </defs>
    <text class="gpu-ts" x="0" y="14">代码里怎么写</text>
    <text class="gpu-ts" x="322" y="14">数据实际放在哪（数字是 A100）</text>
    <g v-for="r in rows" :key="r.y">
      <rect class="gpu-n" x="0" :y="r.y" width="280" height="56" />
      <text class="gpu-t" x="12" :y="r.y + 23" style="font-family: var(--vp-font-family-mono)">{{ r.code }}</text>
      <text class="gpu-ts" x="12" :y="r.y + 43">{{ r.sub }}</text>
      <path class="gpu-ln" :d="`M280 ${r.y + 28} L320 ${r.y + 28}`" marker-end="url(#online-cache-arrow)" />
      <g v-for="(b, i) in r.boxes" :key="i">
        <rect :class="b.cls" :x="b.x" :y="r.y" :width="b.w" height="56" />
        <text class="gpu-t gpu-mid gpu-tb" :x="b.x + b.w / 2" :y="r.y + 23">{{ b.a }}</text>
        <text class="gpu-tk gpu-mid" :x="b.x + b.w / 2" :y="r.y + 43">{{ b.b }}</text>
        <path v-if="i > 0" class="gpu-ln" :d="`M${b.x - 26} ${r.y + 28} L${b.x - 2} ${r.y + 28}`" marker-end="url(#online-cache-arrow)" />
      </g>
    </g>
  </svg>
</template>
