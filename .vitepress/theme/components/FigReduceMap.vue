<script setup>
const steps = [
  { name: '循环控制', cls: 'gpu-n', tag: 'V5', note: '编译期删除' },
  { name: 'if 判断', cls: 'gpu-n', tag: 'V1', note: '让 Warp 整组进出' },
  { name: '读两个操作数', cls: 'gpu-m', tag: 'V2', note: '消除 Bank 冲突' },
  { name: '加法', cls: 'gpu-c', tag: '', note: '始终不是瓶颈' },
  { name: '写回 smem', cls: 'gpu-m', tag: 'V6', note: '改走寄存器' },
  { name: '__syncthreads', cls: 'gpu-n', tag: 'V4', note: '单 Warp 内不用' }
]
const x = (i) => 14 + i * 130
</script>

<template>
  <svg class="w680" viewBox="0 0 800 240" role="img" aria-label="一个 Block 的执行过程：加载数据、写入共享内存、屏障、规约八轮、写回结果。规约的每一轮又分为循环控制、if 判断、读操作数、加法、写回和屏障，图中标出了各版本优化的环节。">
    <defs>
      <marker id="rd-map" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>

    <text class="gpu-ts gpu-mid" x="95" y="18">V3 每线程两个元素，V7 float4</text>
    <text class="gpu-ts gpu-mid" x="515" y="18">反复执行 8 轮（Block = 256）</text>

    <rect class="gpu-m" x="14" y="28" width="162" height="44"/>
    <text class="gpu-t gpu-mid" x="95" y="55">从显存加载 input</text>
    <rect class="gpu-m" x="198" y="28" width="92" height="44"/>
    <text class="gpu-t gpu-mid" x="244" y="55">写入 smem</text>
    <rect class="gpu-c" x="312" y="28" width="376" height="44"/>
    <text class="gpu-t gpu-mid" x="500" y="55">规约</text>
    <rect class="gpu-m" x="710" y="28" width="76" height="44"/>
    <text class="gpu-t gpu-mid" x="748" y="55">写 output</text>
    <line class="gpu-ln" x1="176" y1="50" x2="198" y2="50" marker-end="url(#rd-map)"/>
    <line class="gpu-ln" x1="290" y1="50" x2="312" y2="50" marker-end="url(#rd-map)"/>
    <line class="gpu-ln" x1="688" y1="50" x2="710" y2="50" marker-end="url(#rd-map)"/>

    <line class="gpu-ln gpu-dash" x1="312" y1="72" x2="14" y2="112"/>
    <line class="gpu-ln gpu-dash" x1="688" y1="72" x2="786" y2="112"/>

    <text class="gpu-ts gpu-mid" x="500" y="98">其中一轮的步骤</text>
    <g v-for="(s, i) in steps" :key="s.name">
      <rect :class="s.cls" :x="x(i)" y="112" width="122" height="40"/>
      <text class="gpu-tk gpu-mid" :x="x(i) + 61" y="136">{{ s.name }}</text>
      <text class="gpu-t gpu-tb gpu-mid" :x="x(i) + 61" y="178">{{ s.tag || '—' }}</text>
      <text class="gpu-ts gpu-mid" :x="x(i) + 61" y="198">{{ s.note }}</text>
    </g>
    <text class="gpu-ts" x="14" y="228">琥珀色是内存访问，青色是计算，灰色是控制和同步。V7 的 Grid Stride Loop 则改变整体结构：多次加载，只规约一次。</text>
  </svg>
</template>
