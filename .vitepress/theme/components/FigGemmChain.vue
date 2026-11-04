<script setup>
const BM = 128
const BN = 128
const TM = 8
const TN = 8
const ai = (a, b) => (a * b) / (2 * (a + b))
const warpAi = (16 * TM) / (4 + 8)
const boxes = [
  { x: 10, cls: 'gpu-m', t: '显存', s: 'Global Memory' },
  { x: 228, cls: 'gpu-m', t: '共享内存', s: 'Shared Memory' },
  { x: 446, cls: 'gpu-c', t: '寄存器', s: 'a_frag / b_frag' },
  { x: 664, cls: 'gpu-c', t: 'FMA', s: 'c_frag 累加' }
]
const hops = [
  { x: 136, top: `${BM}×8 + 8×${BN}`, sub: '每个 K 步，Block 协作' },
  { x: 354, top: `${TM} + ${TN} 个 float`, sub: '每个 p 步，每线程' },
  { x: 572, top: `${TM}×${TN} 次乘加`, sub: '每个 p 步，每线程' }
]
const notes = [
  { x: 10, t: `算术强度 ${ai(BM, BN)} FLOP/byte`, s: 'A100 这一层的平衡点约 9.6' },
  { x: 304, t: `单线程 ${ai(TM, TN)}，Warp 广播后约 ${warpAi.toFixed(1)} FLOP/byte`, s: 'A100 这一层的平衡点约 1' }
]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 176" role="img" aria-label="三级数据搬运：显存到共享内存由 Block 协作加载，共享内存到寄存器由线程加载，寄存器上做乘加。">
    <defs>
      <marker id="gemm-chain" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <g v-for="b in boxes" :key="b.t">
      <rect :class="b.cls" :x="b.x" y="30" width="126" height="62" />
      <text class="gpu-t gpu-tb gpu-mid" :x="b.x + 63" y="56">{{ b.t }}</text>
      <text class="gpu-ts gpu-mid" :x="b.x + 63" y="76">{{ b.s }}</text>
    </g>
    <g v-for="h in hops" :key="h.x">
      <line class="gpu-ln" :x1="h.x + 2" y1="61" :x2="h.x + 90" y2="61" marker-end="url(#gemm-chain)" />
      <text class="gpu-tk gpu-mid" :x="h.x + 46" y="16">{{ h.top }}</text>
    </g>
    <g v-for="h in hops" :key="`s${h.x}`">
      <text class="gpu-ts gpu-mid" :x="h.x + 46" y="110">{{ h.sub }}</text>
    </g>
    <g v-for="n in notes" :key="n.x">
      <text class="gpu-t" :x="n.x" y="142">{{ n.t }}</text>
      <text class="gpu-ts" :x="n.x" y="164">{{ n.s }}</text>
    </g>
  </svg>
</template>
