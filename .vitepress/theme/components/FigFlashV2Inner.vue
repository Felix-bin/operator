<script setup>
const shared = [
  { y: 58, cls: 'gpu-c', op: 'S = Q Kᵀ', cost: 'GEMM，2·Br·Bc·d' },
  { y: 92, cls: 'gpu-n', op: 'rowmax(S)，m_new = max(m, ·)', cost: 'Br·Bc + Br' },
  { y: 126, cls: 'gpu-n', op: 'P̃ = exp(S − m_new)', cost: '2·Br·Bc' },
  { y: 160, cls: 'gpu-n', op: 'l_new = e^(m − m_new)·l + rowsum(P̃)', cost: 'Br·Bc + 3Br' },
  { y: 194, cls: 'gpu-c', op: 'P̃ V', cost: 'GEMM，2·Br·Bc·d' },
  { y: 228, cls: 'gpu-n', op: '旧 O 乘 e^(m − m_new)', cost: 'Br·d' }
]
const W = 370
const X = 10
const removed = [
  { op: '旧 O 乘 l_old', cost: 'Br·d' },
  { op: '旧 O 除 l_new', cost: 'Br·d' },
  { op: '新项 P̃V 除 l_new', cost: 'Br·d' }
]
const removedY = (i) => 262 + i * 30
</script>

<template>
  <div class="flv2-pair">
    <svg viewBox="0 0 390 392" role="img" aria-label="V1 一个 Tile 内的操作：两次 GEMM，rowmax、减法、指数、rowsum、旧 O 的指数缩放，再加三个 O 级缩放：乘 l_old、除 l_new 两次，最后把 O、m、l 写回 HBM。">
      <text class="gpu-tt" :x="X" y="24">V1：每个 Tile 都归一化</text>
      <text class="gpu-ts" :x="X" y="46">操作</text>
      <text class="gpu-ts gpu-end" :x="X + W" y="46">FLOP 数</text>
      <g v-for="s in shared" :key="s.y">
        <rect :class="s.cls" :x="X" :y="s.y" :width="W" height="28" />
        <text class="gpu-tk" :x="X + 10" :y="s.y + 18">{{ s.op }}</text>
        <text class="gpu-tk gpu-end" :x="X + W - 10" :y="s.y + 18">{{ s.cost }}</text>
      </g>
      <g v-for="(r, i) in removed" :key="i">
        <rect class="gpu-n flv2-extra" :x="X" :y="removedY(i)" :width="W" height="24" />
        <text class="gpu-tk gpu-tb" :x="X + 10" :y="removedY(i) + 16">{{ r.op }}</text>
        <text class="gpu-tk gpu-end" :x="X + W - 10" :y="removedY(i) + 16">{{ r.cost }}</text>
      </g>
      <rect class="gpu-m" :x="X" y="354" :width="W" height="28" />
      <text class="gpu-tk" :x="X + 10" y="372">写回 HBM：Oᵢ、mᵢ、lᵢ，下一格再读回</text>
    </svg>

    <svg viewBox="0 0 390 392" role="img" aria-label="V2 一个 Tile 内的操作：同样的两次 GEMM、rowmax、减法、指数、rowsum 和指数缩放。三个 O 级缩放被移走，未归一化的 O 留在芯片上，等内循环结束才除一次 l。">
      <text class="gpu-tt" :x="X" y="24">V2：内循环结束才归一化</text>
      <text class="gpu-ts" :x="X" y="46">操作</text>
      <text class="gpu-ts gpu-end" :x="X + W" y="46">FLOP 数</text>
      <g v-for="s in shared" :key="s.y">
        <rect :class="s.cls" :x="X" :y="s.y" :width="W" height="28" />
        <text class="gpu-tk" :x="X + 10" :y="s.y + 18">{{ s.op }}</text>
        <text class="gpu-tk gpu-end" :x="X + W - 10" :y="s.y + 18">{{ s.cost }}</text>
      </g>
      <rect class="gpu-o gpu-dash" :x="X" y="262" :width="W" height="84" />
      <text class="gpu-t gpu-mid" :x="X + W / 2" y="300">这三步从内循环移走</text>
      <text class="gpu-ts gpu-mid" :x="X + W / 2" y="322">Õ 和 m、l 留在寄存器里</text>
      <rect class="gpu-n" :x="X" y="354" :width="W" height="28" />
      <text class="gpu-tk" :x="X + 10" y="372">循环结束后一次：O = Õ / l，L = m + log l</text>
      <text class="gpu-tk gpu-end" :x="X + W - 10" y="372">Br·d + 2Br</text>
    </svg>
  </div>
  <div class="gpu-legend gpu-chart-legend">
    <span><i class="lc"></i>Tensor Core，312 TFLOPS</span>
    <span><i class="flv2-k-n"></i>普通算术，19.5 TFLOPS</span>
    <span><i class="flv2-k-x"></i>V2 删除的操作</span>
    <span><i class="lm"></i>HBM 访问</span>
  </div>
</template>

<style>
rect.flv2-extra {
  stroke: var(--vp-c-text-1);
  stroke-width: 1.6;
}
.gpu-legend i.flv2-k-n {
  background: var(--gpu-n-fill);
  border-color: var(--vp-c-divider);
}
.gpu-legend i.flv2-k-x {
  background: var(--gpu-n-fill);
  border-color: var(--vp-c-text-1);
}
</style>
