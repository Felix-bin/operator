<script setup>
const H = 52
const rows = [
  { y: 36, title: '第 1 遍：求最大值', sub: 'fmaxf', out: 'm' },
  { y: 116, title: '第 2 遍：指数求和', sub: 'expf(x − m)，累加', out: 'd' },
  { y: 196, title: '第 3 遍：归一化', sub: 'expf(x − m) ÷ d', out: null }
]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 290" role="img" aria-label="三遍扫描的数据流。第 1 遍读输入，得到最大值 m；第 2 遍再读一次输入，用 m 算出分母 d；第 3 遍第三次读输入，用 m 和 d 算出结果并写到输出。合计读 192 MiB，写 64 MiB。">
    <defs>
      <marker id="sm-pass" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor"/></marker>
    </defs>

    <rect class="gpu-m" x="16" y="36" width="130" height="212" />
    <text class="gpu-t gpu-tb gpu-mid" x="81" y="134">input</text>
    <text class="gpu-tk gpu-mid" x="81" y="154">全局内存</text>

    <g v-for="r in rows" :key="r.y">
      <line class="gpu-ln" x1="146" :y1="r.y + H / 2" x2="249" :y2="r.y + H / 2" marker-end="url(#sm-pass)" />
      <text class="gpu-ts gpu-mid" x="196" :y="r.y + H / 2 - 8">读 64 MiB</text>
      <rect class="gpu-c" x="250" :y="r.y" width="220" :height="H" />
      <text class="gpu-t gpu-tb gpu-mid" x="360" :y="r.y + 22">{{ r.title }}</text>
      <text class="gpu-tk gpu-mid" x="360" :y="r.y + 40">{{ r.sub }}</text>
      <template v-if="r.out">
        <line class="gpu-ln" x1="470" :y1="r.y + H / 2" x2="519" :y2="r.y + H / 2" marker-end="url(#sm-pass)" />
        <rect class="gpu-n" x="520" :y="r.y + 8" width="56" :height="H - 16" />
        <text class="gpu-t gpu-tb gpu-mid" x="548" :y="r.y + 31">{{ r.out }}</text>
      </template>
    </g>

    <path class="gpu-ln" d="M548 80 V102 H360 V115" marker-end="url(#sm-pass)" />
    <path class="gpu-ln" d="M548 160 V182 H360 V195" marker-end="url(#sm-pass)" />

    <line class="gpu-ln" x1="470" y1="222" x2="649" y2="222" marker-end="url(#sm-pass)" />
    <text class="gpu-ts gpu-mid" x="560" y="214">写 64 MiB</text>
    <rect class="gpu-m" x="650" y="196" width="134" height="52" />
    <text class="gpu-t gpu-tb gpu-mid" x="717" y="218">output</text>
    <text class="gpu-tk gpu-mid" x="717" y="236">全局内存</text>

    <text class="gpu-t" x="16" y="276">4096 × 4096 个 float：读 3 × 64 MiB = 192 MiB，写 64 MiB，合计 256 MiB。</text>
  </svg>
</template>
