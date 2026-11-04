<script setup>
import { computed, ref, useId, watch } from 'vue'

const Ns = [512, 1024, 2048, 4096, 8192]
const ds = [32, 64, 128, 256]
const srams = [
  { kb: 96, note: '96 KB' },
  { kb: 128, note: '128 KB' },
  { kb: 163, note: '163 KB，CUDA 单 Block 上限' },
  { kb: 192, note: '192 KB，论文取值' }
]

const N = ref(4096)
const d = ref(128)
const kb = ref(192)
const t = ref(1)
const sliderId = useId()

// 公式取自论文 Algorithm 1，元素按 FP16 计，与 flashv1_verify.py 一致
const M = computed(() => (kb.value * 1024) / 2)
const Bc = computed(() => Math.ceil(M.value / (4 * d.value)))
const Br = computed(() => Math.min(Bc.value, d.value))
const Tc = computed(() => Math.ceil(N.value / Bc.value))
const Tr = computed(() => Math.ceil(N.value / Br.value))
const total = computed(() => Tc.value * Tr.value)

watch([Tc, Tr], (_, [oldC, oldR]) => {
  const frac = (t.value - 1) / Math.max(1, oldC * oldR - 1)
  t.value = Math.max(1, Math.round(frac * (total.value - 1)) + 1)
})

const j = computed(() => Math.floor((t.value - 1) / Tr.value))
const i = computed(() => (t.value - 1) % Tr.value)

function acc(jj, ii) {
  const rows = jj * N.value + Math.min((ii + 1) * Br.value, N.value)
  return {
    kv: 2 * d.value * Math.min((jj + 1) * Bc.value, N.value),
    q: rows * d.value,
    o: 2 * rows * d.value,
    ml: 4 * rows
  }
}
const sum = (o) => o.kv + o.q + o.o + o.ml
const now = computed(() => acc(j.value, i.value))
const full = computed(() => acc(Tc.value - 1, Tr.value - 1))
const flashNow = computed(() => sum(now.value))
const flashAll = computed(() => sum(full.value))
const stdAll = computed(() => 4 * N.value * N.value + 4 * N.value * d.value)
const scaleIo = computed(() => Math.max(flashAll.value, stdAll.value))
const pctIo = (v) => `${(100 * v) / scaleIo.value}%`
const mib = (v) => {
  const m = (v * 2) / 2 ** 20
  return m >= 100 ? m.toFixed(0) : m.toFixed(1)
}
const ratio = computed(() => stdAll.value / flashAll.value)

const sram = computed(() => {
  const parts = {
    kv: 2 * Bc.value * d.value,
    q: Br.value * d.value,
    o: Br.value * d.value,
    s: Br.value * Bc.value + 2 * Br.value
  }
  return { ...parts, total: parts.kv + parts.q + parts.o + parts.s }
})
const scaleSram = computed(() => Math.max(M.value, sram.value.total))
const pctSram = (v) => `${(100 * v) / scaleSram.value}%`
const fill = computed(() => sram.value.total / M.value)

// 矩阵按 N 等比绘制
const mx = 104
const my = 62
const mS = 340
const cx = (r) => mx + (Math.min(r, N.value) / N.value) * mS
const cy = (r) => my + (Math.min(r, N.value) / N.value) * mS
const vPath = computed(() => {
  if ((Bc.value / N.value) * mS < 5) return ''
  let p = ''
  for (let r = Bc.value; r < N.value; r += Bc.value) p += `M${cx(r).toFixed(1)} ${my} V${my + mS} `
  return p
})
const hPath = computed(() => {
  if ((Br.value / N.value) * mS < 5) return ''
  let p = ''
  for (let r = Br.value; r < N.value; r += Br.value) p += `M${mx} ${cy(r).toFixed(1)} H${mx + mS} `
  return p
})
const tile = computed(() => ({
  x: cx(j.value * Bc.value),
  y: cy(i.value * Br.value),
  w: Math.max(2, cx((j.value + 1) * Bc.value) - cx(j.value * Bc.value)),
  h: Math.max(2, cy((i.value + 1) * Br.value) - cy(i.value * Br.value))
}))
const doneCols = computed(() => cx(j.value * Bc.value) - mx)
const doneRows = computed(() => cy(i.value * Br.value) - my)
const curRows = computed(() => Math.min(Br.value, N.value - i.value * Br.value))
const curCols = computed(() => Math.min(Bc.value, N.value - j.value * Bc.value))
const firstInner = computed(() => i.value === 0)

function nextQ() {
  t.value = Math.min(total.value, t.value + 1)
}
function nextKV() {
  t.value = Math.min(total.value, (j.value + 1) * Tr.value + 1)
}
function finish() {
  t.value = total.value
}
function reset() {
  t.value = 1
}
</script>

<template>
  <figure class="gpu-fig gpu-lab fv1-lab">
    <div class="fv1-ctl">
      <span>序列长度 N</span>
      <button v-for="v in Ns" :key="v" type="button" :aria-pressed="v === N" :class="{ active: v === N }" @click="N = v">{{ v }}</button>
    </div>
    <div class="fv1-ctl">
      <span>头维度 d</span>
      <button v-for="v in ds" :key="v" type="button" :aria-pressed="v === d" :class="{ active: v === d }" @click="d = v">{{ v }}</button>
    </div>
    <div class="fv1-ctl">
      <span>SRAM</span>
      <button v-for="s in srams" :key="s.kb" type="button" :aria-pressed="s.kb === kb" :class="{ active: s.kb === kb }" @click="kb = s.kb">{{ s.note }}</button>
    </div>

    <div class="gpu-read fv1-top" aria-live="polite">
      <span><b>{{ Br }}</b> B_r</span>
      <span><b>{{ Bc }}</b> B_c</span>
      <span><b>{{ Tr }} × {{ Tc }}</b> 个 Tile</span>
      <span><b>{{ ratio.toFixed(2) }}</b> 倍，标准 ÷ FlashAttention</span>
    </div>

    <div class="gpu-lab-row">
      <label :for="sliderId">第 <b>{{ t }}</b> / {{ total }} 个 Tile：外循环 j = <b>{{ j + 1 }}</b> / {{ Tc }}，内循环 i = <b>{{ i + 1 }}</b> / {{ Tr }}</label>
      <input :id="sliderId" v-model.number="t" type="range" min="1" :max="total" />
    </div>
    <div class="fv1-ctl fv1-steps">
      <button type="button" :disabled="t >= total" @click="nextQ">下一个 Q 块</button>
      <button type="button" :disabled="j >= Tc - 1" @click="nextKV">下一个 K/V 块</button>
      <button type="button" :disabled="t >= total" @click="finish">走到底</button>
      <button type="button" :disabled="t <= 1" @click="reset">回到开头</button>
    </div>

    <div class="gpu-scroll" tabindex="0">
      <svg
        class="w680"
        viewBox="0 0 680 418"
        role="img"
        :aria-label="`共 ${Tr} 行 ${Tc} 列 Tile，当前位于第 ${j + 1} 个 K/V 块、第 ${i + 1} 个 Q 块。`"
      >
        <text class="gpu-ts" :x="mx" y="16">K、V（每块只读一次）</text>
        <rect class="gpu-m" :x="mx" y="24" :width="mS" height="24" />
        <rect :x="tile.x" y="24" :width="tile.w" height="24" fill="var(--gpu-m)" />

        <text class="gpu-ts gpu-end" x="60" :y="my + 14">Q、O</text>
        <text class="gpu-ts gpu-end" x="60" :y="my + 30">m、l</text>
        <text class="gpu-ts gpu-end" x="60" :y="my + 50">每遍都</text>
        <text class="gpu-ts gpu-end" x="60" :y="my + 66">重新读写</text>
        <rect class="gpu-m" x="68" :y="my" width="24" :height="mS" />
        <rect x="68" :y="tile.y" width="24" :height="tile.h" fill="var(--gpu-m)" />

        <rect class="gpu-c" :x="mx" :y="my" :width="doneCols" :height="mS" stroke="none" style="stroke: none" />
        <rect class="gpu-c" :x="tile.x" :y="my" :width="tile.w" :height="doneRows" style="stroke: none" />
        <path class="gpu-grid" :d="vPath + hPath" fill="none" />
        <rect class="gpu-o gpu-dash" :x="mx" :y="my" :width="mS" :height="mS" />
        <rect :x="tile.x" :y="tile.y" :width="tile.w" :height="tile.h" fill="var(--gpu-c)" />

        <text class="gpu-t gpu-tb" x="466" y="74">这一步</text>
        <text class="gpu-ts" x="466" y="100">从 HBM 读入</text>
        <text v-if="firstInner" class="gpu-tk" x="466" y="118">Kⱼ、Vⱼ：各 {{ curCols }} × {{ d }}</text>
        <text v-else class="gpu-tk" x="466" y="118">Kⱼ、Vⱼ 已在片上，不读</text>
        <text class="gpu-tk" x="466" y="136">Qᵢ、Oᵢ：各 {{ curRows }} × {{ d }}</text>
        <text class="gpu-tk" x="466" y="154">mᵢ、lᵢ：各 {{ curRows }}</text>
        <text class="gpu-ts" x="466" y="186">只在 SRAM 内</text>
        <text class="gpu-tk" x="466" y="204">Sᵢⱼ、P̃ᵢⱼ：{{ curRows }} × {{ curCols }}</text>
        <text class="gpu-ts" x="466" y="236">写回 HBM</text>
        <text class="gpu-tk" x="466" y="254">Oᵢ、mᵢ、lᵢ</text>
        <text class="gpu-ts" x="466" y="296">虚线框是从未存储的</text>
        <text class="gpu-ts" x="466" y="314">N × N 矩阵。青色是已算过</text>
        <text class="gpu-ts" x="466" y="332">的 Tile，琥珀色是 HBM</text>
        <text class="gpu-ts" x="466" y="350">中的张量，实心块是当前</text>
        <text class="gpu-ts" x="466" y="368">正在片上的部分。</text>
      </svg>
    </div>

    <div class="fv1-bars">
      <p class="fv1-title">
        HBM 累计读写：已用 <b>{{ mib(flashNow) }}</b> MiB，FlashAttention 全程 <b>{{ mib(flashAll) }}</b> MiB，标准实现全程 <b>{{ mib(stdAll) }}</b> MiB
      </p>
      <div class="fv1-bar">
        <span class="fv1-name">FlashAttention</span>
        <div class="fv1-track">
          <div class="gpu-stack">
            <span class="seg sa" :style="{ width: pctIo(now.kv) }"></span>
            <span class="seg sc" :style="{ width: pctIo(now.q) }"></span>
            <span class="seg sb" :style="{ width: pctIo(now.o) }"></span>
            <span class="seg fv1-seg-n" :style="{ width: pctIo(now.ml) }"></span>
          </div>
          <span class="gpu-cap" :style="{ left: pctIo(flashAll) }"></span>
        </div>
      </div>
      <div class="fv1-bar">
        <span class="fv1-name">标准实现</span>
        <div class="fv1-track">
          <div class="gpu-stack"><span class="seg fv1-seg-std" :style="{ width: pctIo(stdAll) }"></span></div>
        </div>
      </div>
      <div class="gpu-legend fv1-legend">
        <span><i class="swatch sa"></i>K、V 读 {{ mib(now.kv) }} MiB</span>
        <span><i class="swatch sc"></i>Q 读 {{ mib(now.q) }} MiB</span>
        <span><i class="swatch sb"></i>O 读写 {{ mib(now.o) }} MiB</span>
        <span><i class="swatch sn"></i>m、l 读写 {{ mib(now.ml) }} MiB</span>
        <span><i class="fv1-capkey"></i>FlashAttention 全程</span>
      </div>

      <p class="fv1-title">
        SRAM 占用：<b>{{ sram.total.toLocaleString('en-US') }}</b> 个元素，是 M = {{ M.toLocaleString('en-US') }} 的 <b>{{ (100 * fill).toFixed(1) }}%</b><template v-if="fill > 1">，超出部分说明公式只保证量级</template>
      </p>
      <div class="fv1-bar">
        <span class="fv1-name">一个 Tile</span>
        <div class="fv1-track">
          <div class="gpu-stack">
            <span class="seg sa" :style="{ width: pctSram(sram.kv) }"></span>
            <span class="seg sc" :style="{ width: pctSram(sram.q) }"></span>
            <span class="seg sb" :style="{ width: pctSram(sram.o) }"></span>
            <span class="seg fv1-seg-n" :style="{ width: pctSram(sram.s) }"></span>
          </div>
          <span class="gpu-cap" :style="{ left: pctSram(M) }"></span>
        </div>
      </div>
      <div class="gpu-legend fv1-legend">
        <span><i class="swatch sa"></i>Kⱼ、Vⱼ {{ sram.kv.toLocaleString('en-US') }}</span>
        <span><i class="swatch sc"></i>Qᵢ {{ sram.q.toLocaleString('en-US') }}</span>
        <span><i class="swatch sb"></i>Oᵢ {{ sram.o.toLocaleString('en-US') }}</span>
        <span><i class="swatch sn"></i>Sᵢⱼ、mᵢ、lᵢ {{ sram.s.toLocaleString('en-US') }}</span>
        <span><i class="fv1-capkey"></i>容量 M</span>
      </div>
    </div>

    <figcaption>
      MiB 指单个 (batch, head) 的前向读写量，所有元素按 FP16 计 2 字节。块大小用论文 Algorithm 1 的公式算出，读写量按算法逐步累加。标准实现的读写量不随 M 变化。模型不含 L2 缓存和压缩，只用来比较读写的结构。
    </figcaption>
  </figure>
</template>

<style>
.fv1-ctl {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 8px;
  padding: 10px 16px 0;
  font-size: 0.875rem;
}
.fv1-ctl > span {
  min-width: 76px;
  color: var(--vp-c-text-2);
}
.fv1-ctl button {
  padding: 3px 10px;
  border: 1px solid var(--vp-c-divider);
  font-size: 0.8125rem;
  color: var(--vp-c-text-1);
}
.fv1-ctl button:hover:not(:disabled) {
  border-color: var(--gpu-c);
}
.fv1-ctl button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.fv1-ctl button:focus-visible {
  outline: 2px solid var(--gpu-c);
  outline-offset: -2px;
}
.fv1-ctl button.active {
  border-color: var(--gpu-ca);
  background: var(--vp-c-brand-soft);
  font-weight: 600;
}
.fv1-steps {
  padding-top: 0;
}
.gpu-lab .fv1-top {
  padding: 12px 16px 0;
}
.fv1-bars {
  padding: 0 16px 14px;
}
.vp-doc .fv1-title {
  margin: 14px 0 8px;
  font-size: 0.875rem;
  line-height: 1.6;
}
.fv1-bar {
  display: grid;
  grid-template-columns: 104px minmax(0, 1fr);
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  font-size: 0.8125rem;
}
.fv1-name {
  color: var(--vp-c-text-2);
}
.fv1-track {
  position: relative;
}
.fv1-track .gpu-stack {
  height: 20px;
}
.fv1-track .gpu-stack {
  gap: 0;
}
.fv1-track .gpu-stack .seg {
  padding-left: 0;
  border-radius: 0;
}
.fv1-seg-n {
  background: var(--gpu-n-fill);
  box-shadow: inset 0 0 0 1px var(--vp-c-divider);
}
.fv1-seg-std {
  background: var(--vp-c-text-3);
}
.fv1-legend {
  margin: 6px 0 0;
  font-size: 0.8125rem;
  gap: 4px 16px;
}
.fv1-legend .swatch {
  width: 12px;
  height: 12px;
  border: 0;
  border-radius: 2px;
}
.fv1-legend .swatch.sn {
  background: var(--gpu-n-fill);
  box-shadow: inset 0 0 0 1px var(--vp-c-divider);
}
.fv1-legend i.fv1-capkey {
  width: 2px;
  height: 14px;
  border: 0;
  background: var(--vp-c-text-1);
}
</style>
