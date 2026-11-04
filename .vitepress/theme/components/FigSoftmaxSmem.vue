<script setup>
const PER = 136 // 64 MiB 的宽度
const barX = 70
const v3 = ['读', '读', '读', '写']
const v4 = ['读', '读', '写']
const segs = Array.from({ length: 16 }, (_, i) => i)
const SEG = 48
</script>

<template>
  <svg class="w680" viewBox="0 0 800 412" role="img" aria-label="V4 的数据流和流量。第 1 遍读输入求最大值 m。第 2 遍再读一次输入，算出指数值后暂存到共享内存，并求出分母 d。最后从共享内存读回指数值，乘以 1/d 写到输出。流量从 V3 的 256 MiB 降到 192 MiB。">
    <defs>
      <marker id="sm-v4" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor"/></marker>
    </defs>

    <rect class="gpu-m" x="16" y="16" width="110" height="112" />
    <text class="gpu-t gpu-tb gpu-mid" x="71" y="68">input</text>
    <text class="gpu-tk gpu-mid" x="71" y="86">全局内存</text>

    <!-- 第 1 遍 -->
    <line class="gpu-ln" x1="126" y1="38" x2="199" y2="38" marker-end="url(#sm-v4)" />
    <text class="gpu-ts gpu-mid" x="162" y="30">读 64 MiB</text>
    <rect class="gpu-c" x="200" y="16" width="200" height="44" />
    <text class="gpu-t gpu-mid" x="300" y="43">第 1 遍：fmaxf（float4）</text>
    <line class="gpu-ln" x1="400" y1="38" x2="449" y2="38" marker-end="url(#sm-v4)" />
    <rect class="gpu-c" x="450" y="16" width="100" height="44" />
    <text class="gpu-t gpu-mid" x="500" y="43">两级规约</text>
    <line class="gpu-ln" x1="550" y1="38" x2="599" y2="38" marker-end="url(#sm-v4)" />
    <rect class="gpu-n" x="600" y="20" width="50" height="36" />
    <text class="gpu-t gpu-tb gpu-mid" x="625" y="43">m</text>

    <!-- 第 2 遍 -->
    <line class="gpu-ln" x1="126" y1="106" x2="199" y2="106" marker-end="url(#sm-v4)" />
    <text class="gpu-ts gpu-mid" x="162" y="98">读 64 MiB</text>
    <rect class="gpu-c" x="200" y="84" width="200" height="44" />
    <text class="gpu-t gpu-mid" x="300" y="111">第 2 遍：expf(x − m)</text>
    <line class="gpu-ln" x1="400" y1="106" x2="449" y2="106" marker-end="url(#sm-v4)" />
    <rect class="gpu-c" x="450" y="84" width="100" height="44" />
    <text class="gpu-t gpu-mid" x="500" y="111">两级规约</text>
    <line class="gpu-ln" x1="550" y1="106" x2="599" y2="106" marker-end="url(#sm-v4)" />
    <rect class="gpu-n" x="600" y="88" width="50" height="36" />
    <text class="gpu-t gpu-tb gpu-mid" x="625" y="111">d</text>
    <path class="gpu-ln" d="M625 56 V72 H300 V83" marker-end="url(#sm-v4)" />

    <!-- 暂存与写出 -->
    <line class="gpu-ln" x1="300" y1="128" x2="300" y2="151" marker-end="url(#sm-v4)" />
    <text class="gpu-ts" x="308" y="145">写 smem[i]</text>
    <rect class="gpu-m" x="200" y="152" width="200" height="44" />
    <text class="gpu-t gpu-mid" x="300" y="179">smem：N 个 float</text>
    <line class="gpu-ln" x1="400" y1="174" x2="449" y2="174" marker-end="url(#sm-v4)" />
    <rect class="gpu-c" x="450" y="152" width="100" height="44" />
    <text class="gpu-t gpu-mid" x="500" y="179">× 1/d</text>
    <path class="gpu-ln" d="M625 124 V140 H500 V151" marker-end="url(#sm-v4)" />
    <line class="gpu-ln" x1="550" y1="174" x2="659" y2="174" marker-end="url(#sm-v4)" />
    <text class="gpu-ts gpu-mid" x="604" y="166">写 64 MiB</text>
    <rect class="gpu-m" x="660" y="152" width="124" height="44" />
    <text class="gpu-t gpu-tb gpu-mid" x="722" y="179">output</text>

    <text class="gpu-ts" x="16" y="180">第三段不碰 input</text>

    <!-- 流量 -->
    <text class="gpu-tt" x="16" y="232">全局内存流量</text>
    <text class="gpu-t" x="16" y="264">V3</text>
    <g v-for="(k, i) in v3" :key="`a${i}`">
      <rect class="gpu-m" :x="barX + i * PER" y="246" :width="PER - 2" height="26" />
      <text class="gpu-tk gpu-mid" :x="barX + i * PER + PER / 2" y="264">{{ k }} 64 MiB</text>
    </g>
    <text class="gpu-t" :x="barX + 4 * PER + 6" y="264">256 MiB</text>
    <text class="gpu-t" x="16" y="298">V4</text>
    <g v-for="(k, i) in v4" :key="`b${i}`">
      <rect class="gpu-m" :x="barX + i * PER" y="280" :width="PER - 2" height="26" />
      <text class="gpu-tk gpu-mid" :x="barX + i * PER + PER / 2" y="298">{{ k }} 64 MiB</text>
    </g>
    <text class="gpu-t" :x="barX + 3 * PER + 6" y="298">192 MiB，流量之比 4/3 ≈ 1.33</text>

    <!-- 下标 -->
    <text class="gpu-tt" x="16" y="340">smem 下标：线程 128 的 16 个位置（N = 4096，Block = 256）</text>
    <g v-for="i in segs" :key="`s${i}`">
      <rect class="gpu-n" :x="16 + i * SEG" y="352" :width="SEG" height="22" />
      <rect class="gpu-m" :x="16 + i * SEG + SEG / 2 - 2" y="352" width="4" height="22" />
      <text class="gpu-ts" :x="16 + i * SEG" y="390">{{ i * 256 }}</text>
    </g>
    <text class="gpu-ts" x="16" y="407">每格 256 个 float，最后一格的末尾是 4095，正好是 N − 1，不越界。</text>
  </svg>
</template>
