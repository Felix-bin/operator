<script setup>
const CELL = 21
const X0 = 70
const list = (n) => Array.from({ length: n }, (_, i) => i)

// 第一个 quarter-warp 的 8 条车道各读一个 float4，起始下标 = lane × step
function panel(step, y, title) {
  const owner = new Map()
  for (let lane = 0; lane < 8; lane++) for (let e = 0; e < 4; e++) owner.set(lane * step + e, lane)
  const bankUse = Array(32).fill(0)
  for (const addr of owner.keys()) bankUse[addr % 32]++
  const ways = Math.max(...bankUse)
  const layers = ways === 1 ? 1 : 2
  const rows = list(layers).map((l) => ({
    l,
    cells: list(32).map((b) => {
      const addr = l * 32 + b
      const hit = owner.has(addr)
      return { b, hit, clash: hit && bankUse[b] > 1, lane: hit && addr % 4 === 0 ? owner.get(addr) : null }
    })
  }))
  return { y, title, rows, ways }
}
const panels = [
  panel(8, 44, '车道间距 8：两层都要用，同一 Bank 落了两个地址'),
  panel(4, 152, '车道间距 4：8 条车道正好铺满一层')
]
</script>

<template>
  <svg class="w680" viewBox="0 0 760 188" role="img" aria-label="共享内存有 32 个 Bank，地址除以 32 的余数决定 Bank。8 条车道间隔 8 个 float 读取时，车道 0 和车道 4 落在同样的 Bank，形成 2 路冲突。间隔 4 个 float 时正好铺满一层，没有冲突。">
    <g v-for="p in panels" :key="p.y">
      <text class="gpu-t gpu-tb" :x="X0" :y="p.y - 22">{{ p.title }}</text>
      <text
        v-for="b in [0, 8, 16, 24]"
        :key="b"
        class="gpu-ts"
        :x="X0 + b * CELL"
        :y="p.y - 4"
      >Bank {{ b }}</text>
      <g v-for="row in p.rows" :key="row.l">
        <text class="gpu-ts" :x="X0 - 8" :y="p.y + row.l * CELL + 15" text-anchor="end">第 {{ row.l }} 层</text>
        <g v-for="c in row.cells" :key="c.b">
          <rect
            :class="c.clash ? 'gemm-clash' : c.hit ? 'gpu-c' : 'gpu-n'"
            :x="X0 + c.b * CELL"
            :y="p.y + row.l * CELL"
            :width="CELL"
            :height="CELL"
          />
          <text
            v-if="c.lane !== null"
            :class="c.clash ? 'gemm-clash-t' : 'gpu-tk'"
            :x="X0 + c.b * CELL + 7"
            :y="p.y + row.l * CELL + 15"
          >{{ c.lane }}</text>
        </g>
      </g>
    </g>
  </svg>
</template>

<style>
.gemm-clash {
  fill: var(--gpu-m);
  stroke: var(--gpu-m);
  stroke-width: 1.2;
}
.gemm-clash-t {
  font-size: 11.5px;
  font-weight: 700;
  fill: var(--vp-c-bg);
}
</style>
