<script setup>
import { computed, ref, useId } from 'vue'

const BLOCKS = [2, 3, 4, 5, 7, 8, 16, 32]
const PRESETS = [
  { name: '方块对齐', N: 64, br: 16, bc: 16 },
  { name: 'Br > Bc', N: 128, br: 32, bc: 16 },
  { name: '末块不完整', N: 100, br: 16, bc: 16 },
  { name: '都不整除', N: 50, br: 7, bc: 5 }
]

const N = ref(64)
const brIdx = ref(BLOCKS.indexOf(16))
const bcIdx = ref(BLOCKS.indexOf(16))
const nId = useId()
const brId = useId()
const bcId = useId()

const Br = computed(() => BLOCKS[brIdx.value])
const Bc = computed(() => BLOCKS[bcIdx.value])

function apply(p) {
  N.value = p.N
  brIdx.value = BLOCKS.indexOf(p.br)
  bcIdx.value = BLOCKS.indexOf(p.bc)
}

const SIZE = 480
const O = 26
const scale = computed(() => SIZE / N.value)

// 用块的真实行列范围判断，末块不完整时也准确
const tiles = computed(() => {
  const out = []
  for (let r = 0; r < N.value; r += Br.value) {
    for (let c = 0; c < N.value; c += Bc.value) {
      const rEnd = Math.min(N.value, r + Br.value) - 1
      const cEnd = Math.min(N.value, c + Bc.value) - 1
      const kind = r >= cEnd ? 'full' : rEnd < c ? 'skip' : 'part'
      const nominalFull = r >= c + Bc.value - 1
      out.push({
        key: `${r}-${c}`,
        r,
        c,
        rEnd,
        cEnd,
        kind,
        tailOnly: kind === 'full' && !nominalFull,
        elems: (rEnd - r + 1) * (cEnd - c + 1)
      })
    }
  }
  return out
})

const NAMES = { full: '完整', part: '部分', skip: '跳过' }
const count = (k) => tiles.value.filter((t) => t.kind === k).length
const elemsOf = (k) => tiles.value.filter((t) => t.kind === k).reduce((a, t) => a + t.elems, 0)
const full = computed(() => count('full'))
const part = computed(() => count('part'))
const skip = computed(() => count('skip'))
const total = computed(() => tiles.value.length)
const Tr = computed(() => Math.ceil(N.value / Br.value))
const Tc = computed(() => Math.ceil(N.value / Bc.value))
const tailOnly = computed(() => tiles.value.filter((t) => t.tailOnly).length)
const maxPart = computed(() => {
  const per = {}
  for (const t of tiles.value) if (t.kind === 'part') per[t.r] = (per[t.r] || 0) + 1
  return Math.max(0, ...Object.values(per))
})
const visible = computed(() => (N.value * (N.value + 1)) / 2)
const processedElems = computed(() => elemsOf('full') + elemsOf('part'))
const pct = (a, b) => ((a / b) * 100).toFixed(1)
</script>

<template>
  <figure class="gpu-fig gpu-lab flv2-causal">
    <div class="gpu-tools">
      <button
        v-for="p in PRESETS"
        :key="p.name"
        type="button"
        :class="{ active: N === p.N && Br === p.br && Bc === p.bc }"
        @click="apply(p)"
      >
        {{ p.name }}
      </button>
    </div>
    <div class="gpu-lab-row">
      <label :for="nId">序列长度 N = <b>{{ N }}</b></label>
      <input :id="nId" v-model.number="N" type="range" min="8" max="128" />
    </div>
    <div class="gpu-lab-row">
      <label :for="brId">Q 块大小 B<sub>r</sub> = <b>{{ Br }}</b></label>
      <input :id="brId" v-model.number="brIdx" type="range" min="0" :max="BLOCKS.length - 1" />
    </div>
    <div class="gpu-lab-row">
      <label :for="bcId">K/V 块大小 B<sub>c</sub> = <b>{{ Bc }}</b></label>
      <input :id="bcId" v-model.number="bcIdx" type="range" min="0" :max="BLOCKS.length - 1" />
    </div>
    <div class="gpu-read" aria-live="polite">
      <span><b>{{ skip }}</b> 跳过</span>
      <span><b>{{ part }}</b> 部分</span>
      <span><b>{{ full }}</b> 完整</span>
      <span>共 {{ Tr }} × {{ Tc }} = {{ total }} 个 Tile，处理 {{ pct(full + part, total) }}%，可见元素 {{ pct(visible, N * N) }}%。</span>
    </div>
    <div class="gpu-legend gpu-chart-legend">
      <span><i class="flv2-t-full"></i>完整：不用 mask</span>
      <span><i class="flv2-t-part"></i>部分：要逐元素 mask</span>
      <span><i class="flv2-t-skip"></i>跳过：不读 K、V</span>
    </div>
    <div class="gpu-scroll" tabindex="0">
      <svg
        class="flv2-grid"
        :viewBox="`0 0 ${SIZE + O + 8} ${SIZE + O + 8}`"
        role="img"
        :aria-label="`N = ${N}，Br = ${Br}，Bc = ${Bc}：跳过 ${skip} 个 Tile，部分 ${part} 个，完整 ${full} 个。`"
      >
        <defs>
          <pattern id="flv2-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" class="flv2-hatch-bg" />
            <rect width="3" height="6" class="flv2-hatch-fg" />
          </pattern>
        </defs>
        <text class="flv2-axis" :x="O" y="17">列（K/V）→</text>
        <text class="flv2-axis" x="17" :y="O + 92" :transform="`rotate(-90 17 ${O + 92})`">← 行（Q）</text>
        <rect
          v-for="t in tiles"
          :key="t.key"
          :class="`flv2-c-${t.kind}`"
          :x="O + t.c * scale"
          :y="O + t.r * scale"
          :width="(t.cEnd - t.c + 1) * scale"
          :height="(t.rEnd - t.r + 1) * scale"
        >
          <title>行 {{ t.r }}~{{ t.rEnd }}，列 {{ t.c }}~{{ t.cEnd }}：{{ NAMES[t.kind] }}</title>
        </rect>
        <line class="gpu-ln flv2-diag" :x1="O" :y1="O" :x2="O + SIZE" :y2="O + SIZE" />
      </svg>
    </div>
    <p class="flv2-note">
      每个 Q 块最多 {{ maxPart }} 个部分 Tile，B<sub>r</sub> / B<sub>c</sub> 上取整是 {{ Math.ceil(Br / Bc) }}。
      部分和完整 Tile 合计覆盖 {{ pct(processedElems, N * N) }}% 的元素，多出可见元素的那部分就是部分 Tile 里被 mask 的格子。
      <template v-if="tailOnly">有 {{ tailOnly }} 个 Tile 按名义块大小算会被当成部分，按实际范围其实是完整，多做一次 mask 也不影响结果。</template>
    </p>
    <figcaption>
      矩形大小按实际行列数绘制，所以不完整的末块更窄或更矮。黑色斜线是对角线，线上（行 = 列）的元素可见。把鼠标停在 Tile 上可以看它的行列范围。分类按块的实际行列范围计算，和逐元素判断一致。
    </figcaption>
  </figure>
</template>

<style>
.flv2-causal sub {
  font-size: 0.7em;
}
.flv2-grid {
  display: block;
  max-width: 520px;
  margin: 0 auto;
}
.flv2-c-full,
.flv2-c-part,
.flv2-c-skip {
  stroke-width: 0.8;
}
.flv2-c-full {
  fill: var(--gpu-c-fill);
  stroke: var(--gpu-c);
}
.flv2-c-part {
  fill: url(#flv2-hatch);
  stroke: var(--gpu-c);
}
.flv2-c-skip {
  fill: var(--gpu-n-fill);
  stroke: var(--vp-c-divider);
}
.flv2-hatch-bg {
  fill: var(--gpu-c-fill);
}
.flv2-hatch-fg {
  fill: var(--gpu-c);
  opacity: 0.55;
}
.flv2-diag {
  stroke-width: 1.4;
}
.flv2-axis {
  font-size: 15px;
  fill: var(--vp-c-text-2);
}
.gpu-legend i.flv2-t-full {
  background: var(--gpu-c-fill);
  border-color: var(--gpu-c);
}
.gpu-legend i.flv2-t-part {
  background: repeating-linear-gradient(135deg, var(--gpu-c) 0 2px, var(--gpu-c-fill) 2px 5px);
  border-color: var(--gpu-c);
}
.gpu-legend i.flv2-t-skip {
  background: var(--gpu-n-fill);
  border-color: var(--vp-c-divider);
}
</style>
