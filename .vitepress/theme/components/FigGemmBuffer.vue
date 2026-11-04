<script setup>
const X0 = 96
// 示意时长（不按比例）
const UNIT = { ldg: 62, sts: 28, bar: 14, fma: 170 }
function single(y) {
  const out = []
  let x = X0
  for (let t = 0; t < 2; t++) {
    out.push({ x, w: UNIT.ldg, y, cls: 'gpu-wait', label: 'LDG 等待' })
    x += UNIT.ldg
    out.push({ x, w: UNIT.sts, y, cls: 'gpu-m', label: 'STS' })
    x += UNIT.sts
    out.push({ x, w: UNIT.bar, y, cls: 'gpu-n', label: '' })
    x += UNIT.bar
    out.push({ x, w: UNIT.fma, y, cls: 'gpu-c', label: 'FMA' })
    x += UNIT.fma
    out.push({ x, w: UNIT.bar, y, cls: 'gpu-n', label: '' })
    x += UNIT.bar
  }
  return { out, end: x }
}
const s = single(58)

const DB = 340
const dbl = [0, 1].map((t) => ({
  x: X0 + t * (DB - 6),
  fma: { x: X0 + t * (DB - 6), w: DB - 20, label: `FMA，读 buffer ${t}` },
  ldg: { x: X0 + t * (DB - 6) + 6, w: UNIT.ldg + 40, label: 'LDG 在途' },
  sts: { x: X0 + t * (DB - 6) + DB - 20 - UNIT.sts, w: UNIT.sts, label: 'STS' },
  bar: { x: X0 + t * (DB - 6) + DB - 20, w: UNIT.bar },
  target: `写 buffer ${1 - t}`
}))
</script>

<template>
  <svg class="w680" viewBox="0 0 800 264" role="img" aria-label="单缓冲时线程先等 LDG 和 STS，再做计算，每步两次屏障。双缓冲时 LDG 发射后计算立刻继续，读取 buffer 0 的同时把下一块写进 buffer 1，每步只需一次屏障。">
    <text class="gpu-tt" x="10" y="22">单缓冲：等数据、算、再等</text>
    <text class="gpu-ts" x="10" y="76">一个线程</text>
    <g v-for="(b, i) in s.out" :key="i">
      <rect :class="b.cls" :x="b.x" :y="b.y" :width="b.w" height="28" />
      <text v-if="b.label" class="gpu-tk gpu-mid" :x="b.x + b.w / 2" :y="b.y + 18">{{ b.label }}</text>
    </g>
    <text class="gpu-ts" :x="X0" y="106">灰色窄条是 __syncthreads()，每个 K 步两次。</text>

    <text class="gpu-tt" x="10" y="146">双缓冲：边算边运</text>
    <text class="gpu-ts" x="10" y="184">计算</text>
    <text class="gpu-ts" x="10" y="222">访存</text>
    <g v-for="(t, i) in dbl" :key="i">
      <rect class="gpu-c" :x="t.fma.x" y="166" :width="t.fma.w" height="28" />
      <text class="gpu-tk gpu-mid" :x="t.fma.x + t.fma.w / 2" y="184">{{ t.fma.label }}</text>
      <rect class="gpu-wait" :x="t.ldg.x" y="204" :width="t.ldg.w" height="28" />
      <text class="gpu-tk gpu-mid" :x="t.ldg.x + t.ldg.w / 2" y="222">{{ t.ldg.label }}</text>
      <rect class="gpu-m" :x="t.sts.x" y="204" :width="t.sts.w" height="28" />
      <text class="gpu-tk gpu-mid" :x="t.sts.x + t.sts.w / 2" y="222">STS</text>
      <text class="gpu-ts gpu-end" :x="t.sts.x - 8" y="222">{{ t.target }}</text>
      <rect class="gpu-n" :x="t.bar.x" y="166" :width="t.bar.w" height="66" />
    </g>
    <text class="gpu-ts" :x="X0" y="252">灰色竖条是唯一的屏障：所有线程读完 buffer 0、写完 buffer 1，然后两者交换。</text>
  </svg>
</template>
