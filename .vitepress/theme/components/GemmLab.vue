<script setup>
import { computed, ref, useId } from 'vue'

// A100 80GB SXM4（compute capability 8.0）
const PEAK = 19.5e12
const BW = 2039e9
const SMEM_SM = 164 * 1024
const STATIC_LIMIT = 48 * 1024
const REGS_SM = 65536
const WARPS_SM = 64
const REG_MAX = 255
const OTHER_REGS = 30
const M = 4096
const N = 4096
const K = 4096

const dims = [16, 32, 64, 128, 256, 512]
const bks = [4, 8, 16, 32, 64, 128]
const tts = [4, 8, 16]

const bmIdx = ref(3)
const bnIdx = ref(3)
const bkIdx = ref(1)
const ttIdx = ref(1)
const stages = ref(2)
const blocks = ref(2)
const ids = { bm: useId(), bn: useId(), bk: useId(), tt: useId(), st: useId(), bl: useId() }

const BM = computed(() => dims[bmIdx.value])
const BN = computed(() => dims[bnIdx.value])
const BK = computed(() => bks[bkIdx.value])
const TT = computed(() => tts[ttIdx.value])

const fmt = (b) => {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`
  return `${(b / 1024).toFixed(1)} KB`
}

const r = computed(() => {
  const bm = BM.value
  const bn = BN.value
  const bk = BK.value
  const tt = TT.value
  const s = stages.value
  const nb = blocks.value
  const ai = (bm * bn) / (2 * (bm + bn))
  const gm = Math.ceil(M / bm)
  const gn = Math.ceil(N / bn)
  const reads = 4 * (gn * M * K + gm * N * K)
  const writes = 4 * M * N
  const naive = 8 * M * N * K
  const flop = 2 * M * N * K
  const aiAll = flop / (reads + writes)
  const limit = Math.min(PEAK, aiAll * BW)
  const smemBlock = s * (bm + bn) * bk * 4
  const smemSm = smemBlock * nb
  const threads = (bm / tt) * (bn / tt)
  const regsThread = tt * tt + 2 * tt + OTHER_REGS
  const regsSm = regsThread * threads * nb
  const warpsSm = Math.ceil(threads / 32) * nb
  const problems = []
  if (smemBlock > SMEM_SM - 1024) problems.push('单个 Block 的共享内存超过 163 KB 上限')
  else if (smemSm > SMEM_SM) problems.push(`${nb} 个 Block 放不进一个 SM 的 164 KB`)
  if (regsThread > REG_MAX) problems.push(`每线程约需 ${regsThread} 个寄存器，超过 255 的上限，会溢出到栈上`)
  else if (regsSm > REGS_SM) problems.push(`一个 SM 需要 ${regsSm} 个寄存器，超过 65536`)
  if (threads > 1024) problems.push(`Block 需要 ${threads} 个线程，超过 1024 的上限`)
  else if (threads < 32) problems.push(`Block 只有 ${threads} 个线程，不足一个 Warp`)
  if (warpsSm > WARPS_SM) problems.push(`需要 ${warpsSm} 个 Warp 驻留，超过每 SM 64 个`)
  const notes = []
  if (smemBlock > STATIC_LIMIT && !problems.length) notes.push('单 Block 超过 48 KB，需要动态共享内存并显式开启')
  return {
    ai,
    aiAll,
    reads,
    writes,
    factor: naive / reads,
    limit,
    pct: (limit / PEAK) * 100,
    smemBlock,
    smemSm,
    smemPct: Math.min(100, (smemSm / SMEM_SM) * 100),
    threads,
    regsThread,
    regsSm,
    warpsSm,
    problems,
    notes
  }
})
</script>

<template>
  <figure class="gpu-fig gpu-lab">
    <div class="gpu-lab-row gemm-lab-row">
      <label :for="ids.bm">BM = <b>{{ BM }}</b></label>
      <input :id="ids.bm" v-model.number="bmIdx" type="range" min="0" :max="dims.length - 1" />
      <label :for="ids.bn">BN = <b>{{ BN }}</b></label>
      <input :id="ids.bn" v-model.number="bnIdx" type="range" min="0" :max="dims.length - 1" />
      <label :for="ids.bk">BK = <b>{{ BK }}</b></label>
      <input :id="ids.bk" v-model.number="bkIdx" type="range" min="0" :max="bks.length - 1" />
      <label :for="ids.tt">TM = TN = <b>{{ TT }}</b></label>
      <input :id="ids.tt" v-model.number="ttIdx" type="range" min="0" :max="tts.length - 1" />
      <label :for="ids.st">缓冲级数 <b>{{ stages }}</b></label>
      <input :id="ids.st" v-model.number="stages" type="range" min="1" max="4" />
      <label :for="ids.bl">每 SM <b>{{ blocks }}</b> 个 Block</label>
      <input :id="ids.bl" v-model.number="blocks" type="range" min="1" max="4" />
    </div>
    <div class="gpu-read" aria-live="polite">
      <span><b>{{ r.ai.toFixed(1) }}</b> FLOP/byte，tile 算术强度</span>
      <span><b>{{ fmt(r.reads) }}</b> 显存读取，朴素写法的 1/{{ Math.round(r.factor) }}</span>
      <span><b>{{ (r.smemSm / 1024).toFixed(0) }} KB</b> 每 SM 的共享内存</span>
    </div>
    <div class="gemm-meter">
      <div class="gemm-meter-h">
        <span>共享内存占用（每 SM，上限 164 KB）</span>
        <span>每 Block {{ (r.smemBlock / 1024).toFixed(1) }} KB</span>
      </div>
      <div class="gemm-bar"><i :class="{ over: r.smemSm > 164 * 1024 }" :style="{ width: r.smemPct + '%' }"></i></div>
      <div class="gemm-meter-h">
        <span>带宽允许的峰值算力占比</span>
        <span>{{ (r.limit / 1e12).toFixed(1) }} TFLOPS，{{ r.pct.toFixed(0) }}%</span>
      </div>
      <div class="gemm-bar"><i class="c" :style="{ width: r.pct + '%' }"></i></div>
    </div>
    <p class="gemm-lab-verdict">
      Block {{ r.threads }} 个线程，每线程约 {{ r.regsThread }} 个寄存器，全矩阵算术强度 {{ r.aiAll.toFixed(1) }} FLOP/byte（含写回 C）。
      <template v-if="r.problems.length">
        <strong>放不下：</strong>{{ r.problems.join('；') }}。
      </template>
      <template v-else>
        这组参数放得下。<template v-if="r.notes.length">{{ r.notes.join('；') }}。</template>
      </template>
    </p>
    <figcaption>
      按 A100 和 M = N = K = 4096 计算：峰值 19.5 TFLOPS、显存 2039 GB/s，所以平衡点约 9.6 FLOP/byte。显存读取量按“每个 A 元素被每列 Block 读一次、每个 B 元素被每行 Block 读一次”精确统计。每线程寄存器按 TM×TN + TM + TN + 30 估算，30 是原稿给的索引和循环变量开销。
    </figcaption>
  </figure>
</template>

<style>
.gemm-bars {
  min-width: 620px;
}
.gemm-bars .gpu-bars {
  margin: 0;
}
.gpu-lab .gemm-lab-row {
  display: grid;
  grid-template-columns: max-content minmax(0, 1fr) max-content minmax(0, 1fr);
  gap: 8px 14px;
}
.gpu-lab .gemm-lab-row input {
  width: 100%;
  min-width: 60px;
}
@media (max-width: 560px) {
  .gpu-lab .gemm-lab-row {
    grid-template-columns: max-content minmax(0, 1fr);
  }
}
.gemm-meter {
  padding: 4px 16px 8px;
  font-size: 0.8125rem;
  color: var(--vp-c-text-2);
}
.gemm-meter-h {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 2px 16px;
  margin-top: 8px;
}
.gemm-bar {
  height: 12px;
  margin-top: 4px;
  background: var(--gpu-n-fill);
  border: 1px solid var(--vp-c-divider);
}
.gemm-bar i {
  display: block;
  height: 100%;
  background: var(--gpu-m);
}
.gemm-bar i.c {
  background: var(--gpu-c);
}
.gemm-bar i.over {
  background: repeating-linear-gradient(135deg, var(--gpu-m) 0 5px, transparent 5px 8px);
}
.vp-doc .gemm-lab-verdict {
  margin: 8px 16px 12px;
  font-size: 0.875rem;
  line-height: 1.6;
}
</style>
