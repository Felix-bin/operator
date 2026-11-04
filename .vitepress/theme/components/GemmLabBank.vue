<script setup>
import { computed, ref, useId } from 'vue'

const WARP = 32
const options = [
  { id: 'row8', name: '行主序，8 × 8', step: 8, map: (l) => [l >> 3, l & 7] },
  { id: '2x16', name: '2 × 16，8 × 8', step: 8, map: (l) => [l >> 4, l & 15] },
  { id: 'col8', name: '列主序，8 × 8', step: 8, map: (l) => [l & 3, l >> 2] },
  { id: 'row4', name: '行主序，拆成 4 × 4', step: 4, map: (l) => [l >> 3, l & 7] },
  { id: 'z8', name: 'Z-order，8 × 8', step: 8, map: (l) => [(l & 1) + ((l >> 4) << 1), (l & 15) >> 1] },
  { id: 'z4', name: 'Z-order，拆成 4 × 4', step: 4, map: (l) => [(l & 1) + ((l >> 4) << 1), (l & 15) >> 1] }
]
const layoutId = ref('row8')
const qw = ref(0)
const sliderId = useId()

const layout = computed(() => options.find((o) => o.id === layoutId.value))

// 一组车道读 float4 时，单个 Bank 上落了多少个不同地址
function ways(addrs) {
  const m = new Map()
  for (const a of new Set(addrs)) for (let e = 0; e < 4; e++) m.set((a + e) % 32, (m.get((a + e) % 32) || 0) + 1)
  return Math.max(...m.values())
}
function analyze(addr) {
  const condA = addr.every((a, l) => a === addr[l ^ 1])
  const condB = addr.every((a, l) => a === addr[l ^ 2])
  const merged = condA || condB
  const groups = merged ? [[0, 16], [16, 32]] : [[0, 8], [8, 16], [16, 24], [24, 32]]
  const perGroup = groups.map(([a, b]) => ways(addr.slice(a, b)))
  return {
    addr,
    merged,
    cond: condA ? 'lane i 与 lane i⊕1' : condB ? 'lane i 与 lane i⊕2' : '',
    groups,
    perGroup,
    total: perGroup.reduce((s, v) => s + v, 0)
  }
}
const addrs = computed(() => {
  const a = []
  const b = []
  for (let l = 0; l < WARP; l++) {
    const [r, c] = layout.value.map(l)
    a.push(r * layout.value.step)
    b.push(c * layout.value.step)
  }
  return { a: analyze(a), b: analyze(b) }
})

function view(res, name) {
  const idx = res.merged ? (qw.value >> 1) : qw.value
  const [lo, hi] = res.groups[idx]
  const unique = [...new Set(res.addr.slice(lo, hi))]
  const count = Array(32).fill(0)
  const hits = new Set()
  for (const a of unique) for (let e = 0; e < 4; e++) {
    count[(a + e) % 32]++
    hits.add(a + e)
  }
  const rows = Array.from({ length: 4 }, (_, r) => ({
    label: `第 ${r} 层`,
    cells: Array.from({ length: 32 }, (_, bank) => {
      const addr = r * 32 + bank
      if (!hits.has(addr)) return ''
      return count[bank] > 1 ? 'rd-clash' : 'rd-hit'
    })
  }))
  return {
    name,
    lanes: `车道 ${lo}~${hi - 1}`,
    unique: unique.length,
    ways: res.perGroup[idx],
    merged: res.merged,
    cond: res.cond,
    total: res.total,
    rows,
    count
  }
}
const views = computed(() => [view(addrs.value.a, 'tileA'), view(addrs.value.b, 'tileB')])
const total = computed(() => addrs.value.a.total + addrs.value.b.total)
const lanes = computed(() => {
  const grid = []
  for (let l = 0; l < WARP; l++) {
    const [r, c] = layout.value.map(l)
    grid.push({ l, r, c, q: l >> 3 })
  }
  return grid
})
</script>

<template>
  <figure class="gpu-fig gpu-lab">
    <div class="gpu-tools">
      <button
        v-for="o in options"
        :key="o.id"
        type="button"
        :class="{ active: layoutId === o.id }"
        :aria-pressed="layoutId === o.id"
        @click="layoutId = o.id"
      >
        {{ o.name }}
      </button>
    </div>
    <div class="gpu-lab-row">
      <label :for="sliderId">看第 <b>{{ qw }}</b> 个 quarter-warp</label>
      <input :id="sliderId" v-model.number="qw" type="range" min="0" max="3" />
    </div>
    <div class="gpu-read" aria-live="polite">
      <span><b>{{ addrs.a.total }}</b> 个 wavefront，读 tileA</span>
      <span><b>{{ addrs.b.total }}</b> 个 wavefront，读 tileB</span>
      <span><b>{{ total }}</b> 合计</span>
    </div>
    <div class="gemm-lanes-wrap">
      <div class="gemm-lanes" role="img" :aria-label="`${layout.name}：32 条车道在 4 乘 8 网格中的位置`">
        <i
          v-for="c in lanes"
          :key="c.l"
          :class="{ on: c.q === qw }"
          :style="{ gridRow: c.r + 1, gridColumn: c.c + 1 }"
        >{{ c.l }}</i>
      </div>
    </div>
    <div v-for="v in views" :key="v.name" class="gemm-bank-block">
      <p class="gemm-bank-title">
        <strong>{{ v.name }}</strong>：{{ v.lanes }}，{{ v.unique }} 个不同的 float4，这组需要 {{ v.ways }} 个 wavefront。<template v-if="v.merged">所有 {{ v.cond }} 地址相同，返回寄存器可以打包，按 16 车道一组判断冲突。</template><template v-else>不满足打包条件，每个 quarter-warp 独立。</template>
      </p>
      <div class="gpu-scroll" tabindex="0">
        <div class="gpu-rd-grid gpu-rd-bank" role="img" :aria-label="`${v.name} 的 Bank 占用`">
          <span class="head"></span>
          <span v-for="l in [0, 8, 16, 24]" :key="l" class="head span8">Bank {{ l }}</span>
          <span class="head"></span>
          <template v-for="row in v.rows" :key="row.label">
            <span class="label">{{ row.label }}</span>
            <i v-for="(c, i) in row.cells" :key="i" :class="c || 'rd-blank'"></i>
            <span class="tail"></span>
          </template>
          <span class="label">地址数</span>
          <span v-for="(n, i) in v.count" :key="i" class="count" :class="{ clash: n > 1 }">{{ n || '' }}</span>
          <span class="tail"></span>
        </div>
      </div>
    </div>
    <figcaption>
      上方是 32 条车道在 4 × 8 网格中的位置，高亮的是当前 quarter-warp。下方每列是一个 Bank，每行是一层，实心格表示同一 Bank 落了多个不同地址，要排队。统计的是一条 LDS.128，一次 p 步要读 A、B 各两次。打包和按 16 车道判冲突的规则是用实测反推的，下文说明。
    </figcaption>
  </figure>
</template>

<style>
.gemm-lanes {
  display: grid;
  grid-template-columns: repeat(8, 30px);
  grid-auto-rows: 24px;
  gap: 2px;
  justify-content: center;
  margin-bottom: 8px;
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
}
.gemm-lanes i {
  display: flex;
  align-items: center;
  justify-content: center;
  font-style: normal;
  background: var(--gpu-n-fill);
  color: var(--vp-c-text-2);
}
.gemm-lanes i.on {
  background: var(--gpu-c-fill);
  box-shadow: inset 0 0 0 1.2px var(--gpu-c);
  color: var(--vp-c-text-1);
  font-weight: 600;
}
.gemm-lanes-wrap {
  padding: 8px 16px 0;
}
.gemm-bank-block {
  margin-top: 8px;
}
.gemm-bank-block .gpu-scroll {
  padding-top: 0;
}
.vp-doc .gemm-bank-title {
  margin: 0;
  padding: 0 16px 4px;
  font-size: 0.8125rem;
  line-height: 1.6;
}
.gemm-bank-block .gpu-rd-bank {
  margin-top: 4px;
}
</style>
