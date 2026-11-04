<script setup>
const SZ = 13
const G = 16
const hit = (m, n) => 1 - (m + n) / (2 * m * n)
const panels = [
  { x0: 10, m: 1, n: 16 },
  { x0: 280, m: 2, n: 8 },
  { x0: 550, m: 4, n: 4 }
].map((p) => {
  const cells = []
  for (let r = 0; r < p.m; r++) for (let c = 0; c < p.n; c++) cells.push({ r, c })
  return { ...p, cells, hit: (hit(p.m, p.n) * 100).toFixed(1), loads: p.m + p.n, reqs: 2 * p.m * p.n }
})
const list = (n) => Array.from({ length: n }, (_, i) => i)
</script>

<template>
  <svg class="w680" viewBox="0 0 800 300" role="img" aria-label="同样 16 个 Block，排成一行时需要加载 17 个不同的 tile，命中率约 47 百分比；排成 4 乘 4 方块时只需 8 个，命中率 75 百分比。">
    <g v-for="p in panels" :key="p.x0">
      <text class="gpu-tt" :x="p.x0" y="22">{{ p.m }} 行 × {{ p.n }} 列</text>
      <g v-for="r in list(G / 2)" :key="r">
        <rect
          v-for="c in list(G)"
          :key="c"
          class="gpu-n"
          :x="p.x0 + 14 + c * SZ"
          :y="50 + r * SZ"
          :width="SZ"
          :height="SZ"
        />
      </g>
      <rect
        v-for="(c, i) in p.cells"
        :key="i"
        class="gpu-c"
        :x="p.x0 + 14 + c.c * SZ"
        :y="50 + c.r * SZ"
        :width="SZ"
        :height="SZ"
      />
      <rect v-for="r in p.m" :key="`a${r}`" class="gpu-m" :x="p.x0" :y="50 + (r - 1) * SZ" width="10" :height="SZ" />
      <rect v-for="c in p.n" :key="`b${c}`" class="gpu-m" :x="p.x0 + 14 + (c - 1) * SZ" y="36" :width="SZ" height="10" />
      <text class="gpu-t" :x="p.x0" y="178">不同的 tile：{{ p.m }} + {{ p.n }} = {{ p.loads }} 个</text>
      <text class="gpu-t" :x="p.x0" y="200">总请求：{{ p.reqs }} 个 tile</text>
      <text class="gpu-t gpu-tb" :x="p.x0" y="222">L2 命中率 {{ p.hit }}%</text>
    </g>
    <text class="gpu-ts" x="10" y="256">青色是同一波里运行的 Block，琥珀色是它们要读的 A 块（左侧）和 B 块（上方）。</text>
    <text class="gpu-ts" x="10" y="278">同一个 A 块被同一行的 Block 共用，同一个 B 块被同一列的 Block 共用，只有第一次读会缺失。</text>
  </svg>
</template>
