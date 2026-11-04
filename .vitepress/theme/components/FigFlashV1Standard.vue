<script setup>
// 数字与 flashv1_verify.py 的 F 节一致：N = 4096，d = 128，全部按 FP16
const N = 4096
const d = 128
const total = 4 * N * N + 4 * N * d
const mib = ((total * 2) / 2 ** 20).toFixed(0)
const chipY = 150
const chipH = 34
const boxY = 40
const boxH = 44
const chips = [
  { t: 'Q', x: 50, w: 44, note: '读 Nd' },
  { t: 'K', x: 104, w: 44, note: '读 Nd' },
  { t: 'S', x: 270, w: 60, note: '写 N²，读 N²' },
  { t: 'P', x: 510, w: 60, note: '写 N²，读 N²' },
  { t: 'V', x: 600, w: 44, note: '读 Nd' },
  { t: 'O', x: 690, w: 44, note: '写 Nd' }
]
const boxes = [
  { t: 'S = QKᵀ / √d', x: 90 },
  { t: 'P = softmax(S)', x: 330 },
  { t: 'O = PV', x: 570 }
]
const arrows = [
  [72, chipY, 130, boxY + boxH + 2],
  [126, chipY, 165, boxY + boxH + 2],
  [220, boxY + boxH, 285, chipY - 2],
  [315, chipY, 380, boxY + boxH + 2],
  [460, boxY + boxH, 525, chipY - 2],
  [555, chipY, 610, boxY + boxH + 2],
  [622, chipY, 660, boxY + boxH + 2],
  [705, boxY + boxH, 712, chipY - 2]
]
const lowChips = [
  { t: 'K', x: 90, w: 44 },
  { t: 'V', x: 144, w: 44 },
  { t: 'Q', x: 300, w: 44 },
  { t: 'O', x: 440, w: 44 },
  { t: 'm', x: 590, w: 44 },
  { t: 'l', x: 644, w: 44 }
]
const lowNotes = [
  { t: '只读 1 遍，2Nd', x: 139 },
  { t: '读 T_c 遍', x: 322 },
  { t: '读写各 T_c 遍', x: 462 },
  { t: '读写各 T_c 遍，每遍 2N', x: 639 }
]
const up = [
  [112, 380, 112, 326],
  [166, 380, 166, 326],
  [322, 380, 322, 326],
  [452, 380, 452, 326],
  [612, 380, 612, 326],
  [666, 380, 666, 326]
]
const down = [
  [472, 324, 472, 378],
  [622, 324, 622, 378],
  [676, 324, 676, 378]
]
</script>

<template>
  <svg class="w680" viewBox="0 0 800 464" role="img" aria-label="上图：标准实现分三步，S 和 P 各写入 HBM 再读回，共 4N² 加 4Nd 个元素。下图：FlashAttention 在一个 Kernel 内完成，只读写 Q、K、V、O、m、l。">
    <defs>
      <marker id="fv1-std" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor" /></marker>
    </defs>
    <text class="gpu-tt" x="16" y="22">标准实现：三步分开，S 和 P 都落到 HBM</text>
    <g v-for="b in boxes" :key="b.t">
      <rect class="gpu-c" :x="b.x" :y="boxY" width="160" :height="boxH" />
      <text class="gpu-t gpu-mid" :x="b.x + 80" :y="boxY + 27">{{ b.t }}</text>
    </g>
    <g v-for="c in chips" :key="c.t">
      <rect class="gpu-m" :x="c.x" :y="chipY" :width="c.w" :height="chipH" />
      <text class="gpu-t gpu-mid" :x="c.x + c.w / 2" :y="chipY + 22">{{ c.t }}</text>
      <text class="gpu-ts gpu-mid" :x="c.x + c.w / 2" :y="chipY + chipH + 18">{{ c.note }}</text>
    </g>
    <line v-for="(a, i) in arrows" :key="i" class="gpu-ln" :x1="a[0]" :y1="a[1]" :x2="a[2]" :y2="a[3]" marker-end="url(#fv1-std)" />
    <text class="gpu-t" x="16" y="232">合计 4N² + 4Nd 个元素。N = 4096、d = 128、FP16 时，单个 head 约 {{ mib }} MiB。</text>

    <line class="gpu-grid" x1="16" x2="784" y1="248" y2="248" />

    <text class="gpu-tt" x="16" y="274">FlashAttention：一个 Kernel 做完，中间结果留在片上</text>
    <rect class="gpu-c" x="90" y="282" width="620" height="42" />
    <text class="gpu-t gpu-mid" x="400" y="308">循环 T_c × T_r 次：Sᵢⱼ → P̃ᵢⱼ → 更新 mᵢ、lᵢ、Oᵢ，全部在 SRAM 内</text>
    <g v-for="c in lowChips" :key="c.t">
      <rect class="gpu-m" :x="c.x" y="380" :width="c.w" height="34" />
      <text class="gpu-t gpu-mid" :x="c.x + c.w / 2" y="402">{{ c.t }}</text>
    </g>
    <text v-for="n in lowNotes" :key="n.x" class="gpu-ts gpu-mid" :x="n.x" y="432">{{ n.t }}</text>
    <line v-for="(a, i) in up" :key="'u' + i" class="gpu-ln" :x1="a[0]" :y1="a[1]" :x2="a[2]" :y2="a[3]" marker-end="url(#fv1-std)" />
    <line v-for="(a, i) in down" :key="'d' + i" class="gpu-ln" :x1="a[0]" :y1="a[1]" :x2="a[2]" :y2="a[3]" marker-end="url(#fv1-std)" />
    <text class="gpu-t" x="16" y="456">合计 T_c(3Nd + 4N) + 2Nd 个元素，HBM 里没有任何 N × N 的张量。</text>
  </svg>
</template>
