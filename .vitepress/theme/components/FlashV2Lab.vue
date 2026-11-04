<script setup>
import { computed, ref, useId } from 'vue'

const N = 8192
const D = 128
const ELEM = 2 // Q、K、V、O 按 FP16
const STAT = 4 // m、l、L 按 FP32
const STEPS = [16, 32, 64, 128, 256]
const SRAM_KB = 164
const BW = 2039e9
const PEAK = 312e12

const version = ref('v1')
const causal = ref(false)
const trIdx = ref(2)
const tcIdx = ref(2)
const trId = useId()
const tcId = useId()

const Tr = computed(() => STEPS[trIdx.value])
const Tc = computed(() => STEPS[tcIdx.value])
const Br = computed(() => N / Tr.value)
const Bc = computed(() => N / Tc.value)

// 实际处理的 Tile 数：causal 时跳过全在对角线上方的块，逐块数，不用近似式
const visits = computed(() => {
  if (!causal.value) return Tr.value * Tc.value
  let n = 0
  for (let i = 0; i < Tr.value; i++) {
    n += Math.min(Tc.value, Math.ceil(((i + 1) * Br.value) / Bc.value))
  }
  return n
})

const stats = computed(() => {
  const v = visits.value
  const q = Br.value * D * ELEM
  const kv = Bc.value * D * ELEM
  const s = Br.value * STAT
  return {
    v1: {
      rows: [
        { key: 'Q', op: '读', count: v, bytes: v * q, kind: 'r' },
        { key: 'K', op: '读', count: Tc.value, bytes: Tc.value * kv, kind: 'r' },
        { key: 'V', op: '读', count: Tc.value, bytes: Tc.value * kv, kind: 'r' },
        { key: 'O', op: '读', count: v, bytes: v * q, kind: 'r' },
        { key: 'O', op: '写', count: v, bytes: v * q, kind: 'w' },
        { key: 'm、l', op: '读', count: 2 * v, bytes: 2 * v * s, kind: 'r' },
        { key: 'm、l', op: '写', count: 2 * v, bytes: 2 * v * s, kind: 'w' }
      ],
      sram: (2 * Br.value + 2 * Bc.value) * D * ELEM
    },
    v2: {
      rows: [
        { key: 'Q', op: '读', count: Tr.value, bytes: Tr.value * q, kind: 'r' },
        { key: 'K', op: '读', count: v, bytes: v * kv, kind: 'r' },
        { key: 'V', op: '读', count: v, bytes: v * kv, kind: 'r' },
        { key: 'O', op: '读', count: 0, bytes: 0, kind: 'r' },
        { key: 'O', op: '写', count: Tr.value, bytes: Tr.value * q, kind: 'w' },
        { key: 'L', op: '读', count: 0, bytes: 0, kind: 'r' },
        { key: 'L', op: '写', count: Tr.value, bytes: Tr.value * s, kind: 'w' }
      ],
      sram: (Br.value + 2 * Bc.value) * D * ELEM
    }
  }
})

const total = (rows) => rows.reduce((a, r) => a + r.bytes, 0)
const now = computed(() => stats.value[version.value])
const other = computed(() => stats.value[version.value === 'v1' ? 'v2' : 'v1'])
const otherName = computed(() => (version.value === 'v1' ? 'V2' : 'V1'))
const nowTotal = computed(() => total(now.value.rows))
const otherTotal = computed(() => total(other.value.rows))
const scale = computed(() =>
  Math.max(...stats.value.v1.rows.map((r) => r.bytes), ...stats.value.v2.rows.map((r) => r.bytes))
)
const rows = computed(() =>
  now.value.rows.map((r, i) => ({
    ...r,
    pct: (r.bytes / scale.value) * 100,
    ghost: (other.value.rows[i].bytes / scale.value) * 100
  }))
)

const mb = (b) => (b / 1e6).toFixed(b >= 1e8 ? 0 : b > 0 && b < 1e5 ? 2 : 1)
const us = (b) => (b / BW) * 1e6
const flopUs = computed(() => (((4 * N * N * D) / (causal.value ? 2 : 1)) / PEAK) * 1e6)
const sramKb = computed(() => now.value.sram / 1024)
const over = computed(() => sramKb.value > SRAM_KB)
</script>

<template>
  <figure class="gpu-fig gpu-lab flv2-lab">
    <div class="gpu-tools">
      <button type="button" :class="{ active: version === 'v1' }" :aria-pressed="version === 'v1'" @click="version = 'v1'">V1：外循环 K/V</button>
      <button type="button" :class="{ active: version === 'v2' }" :aria-pressed="version === 'v2'" @click="version = 'v2'">V2：外循环 Q</button>
      <button type="button" :class="{ active: causal }" :aria-pressed="causal" @click="causal = !causal">Causal Mask：{{ causal ? '开' : '关' }}</button>
    </div>
    <div class="gpu-lab-row">
      <label :for="trId">Q 分 T<sub>r</sub> = <b>{{ Tr }}</b> 块，B<sub>r</sub> = {{ Br }}</label>
      <input :id="trId" v-model.number="trIdx" type="range" min="0" :max="STEPS.length - 1" />
    </div>
    <div class="gpu-lab-row">
      <label :for="tcId">K/V 分 T<sub>c</sub> = <b>{{ Tc }}</b> 块，B<sub>c</sub> = {{ Bc }}</label>
      <input :id="tcId" v-model.number="tcIdx" type="range" min="0" :max="STEPS.length - 1" />
    </div>
    <div class="gpu-read" aria-live="polite">
      <span><b>{{ mb(nowTotal) }}</b> MB 总搬运量</span>
      <span>{{ otherName }} 是 {{ mb(otherTotal) }} MB，比值 {{ (nowTotal / otherTotal).toFixed(2) }}</span>
    </div>
    <div class="gpu-legend gpu-chart-legend">
      <span><i class="flv2-key r"></i>读</span>
      <span><i class="flv2-key w"></i>写</span>
      <span><i class="flv2-key g"></i>{{ otherName }} 的同一项</span>
    </div>
    <div class="gpu-scroll" tabindex="0">
      <div class="flv2-table" role="table" :aria-label="`${version.toUpperCase()} 每种张量的 HBM 读写量`">
        <div class="flv2-row head" role="row">
          <span role="columnheader">张量</span>
          <span role="columnheader" class="num">块次数</span>
          <span role="columnheader">数据量</span>
          <span role="columnheader" class="num">MB</span>
        </div>
        <div v-for="(r, i) in rows" :key="i" class="flv2-row" role="row">
          <span role="cell">{{ r.key }} {{ r.op }}</span>
          <span role="cell" class="num">{{ r.count }}</span>
          <span role="cell" class="flv2-track">
            <i v-if="r.bytes" class="flv2-bar" :class="r.kind" :style="{ width: r.pct + '%' }"></i>
            <i v-if="other.rows[i].bytes" class="flv2-ghost" :style="{ left: `calc(${r.ghost}% - 1px)` }"></i>
          </span>
          <span role="cell" class="num">{{ mb(r.bytes) }}</span>
        </div>
      </div>
    </div>
    <p class="flv2-note">
      全部走 HBM 时，这些搬运在 2039 GB/s 下至少要 {{ us(nowTotal).toFixed(0) }} μs，矩阵乘 FLOP 在 312 TFLOPS 下至少要 {{ flopUs.toFixed(0) }} μs。
      片上需要放 {{ sramKb.toFixed(0) }} KB<template v-if="over">，<b class="flv2-warn">超过 A100 每个 SM 最多 {{ SRAM_KB }} KB 的共享内存，这组块大小跑不起来</b></template><template v-else>，没超过 A100 每个 SM 的 {{ SRAM_KB }} KB</template>。
    </p>
    <figcaption>
      按两版论文的算法逐项计数：N = 8192，d = 128，一个头一条序列。Q、K、V、O 按 FP16，m、l、L 按 FP32。这里统计的是搬进搬出 SRAM 的数据量，没有建模 L2 缓存，所以是上界，不是实测。片上用量只算论文算法里放进 SRAM 的块：V1 是 Q、O、K、V，V2 是 Q、K、V，没有计入对齐和其他开销。
    </figcaption>
  </figure>
</template>

<style>
.flv2-lab sub {
  font-size: 0.7em;
}
.flv2-key {
  border: 0 !important;
  border-radius: 2px;
  width: 14px !important;
  height: 14px !important;
}
.flv2-key.r,
.flv2-bar.r {
  background: var(--gpu-m-fill);
  box-shadow: inset 0 0 0 1.2px var(--gpu-m);
}
.flv2-key.w,
.flv2-bar.w {
  background: var(--gpu-m);
}
.flv2-key.g {
  width: 3px !important;
  background: var(--vp-c-text-1);
}
.flv2-table {
  min-width: 480px;
  font-size: 0.875rem;
}
.flv2-row {
  display: grid;
  grid-template-columns: 58px 56px minmax(0, 1fr) 52px;
  gap: 10px;
  align-items: center;
  min-height: 28px;
  font-variant-numeric: tabular-nums;
}
@media (max-width: 560px) {
  .flv2-table {
    min-width: 0;
    font-size: 0.8125rem;
  }
  .flv2-row {
    grid-template-columns: 50px 40px minmax(0, 1fr) 40px;
    gap: 6px;
  }
}
.flv2-row.head {
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
}
.flv2-row .num {
  text-align: right;
}
.flv2-track {
  position: relative;
  display: block;
  height: 18px;
  background: var(--gpu-n-fill);
  border-radius: 2px;
}
.flv2-bar {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  min-width: 2px;
  border-radius: 0 4px 4px 0;
  transition: width 0.15s ease;
}
.flv2-ghost {
  position: absolute;
  top: -3px;
  bottom: -3px;
  width: 2px;
  background: var(--vp-c-text-1);
  transition: left 0.15s ease;
}
@media (prefers-reduced-motion: reduce) {
  .flv2-bar,
  .flv2-ghost {
    transition: none;
  }
}
.vp-doc .flv2-note {
  margin: 4px 16px 12px;
  font-size: 0.875rem;
  line-height: 1.65;
  color: var(--vp-c-text-2);
}
.flv2-warn {
  color: var(--gpu-m);
}
</style>
