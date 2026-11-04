<script setup>
const S = 3
const X0 = 96
const W = 200
const H = 34
const Y0 = 52
const iters = [0, 1, 2, 3].map((t) => ({
  t,
  slots: Array.from({ length: S }, (_, s) => {
    if (s === t % S) return { cls: 'gpu-c', label: `正在计算 tile ${t}` }
    if (s === (t + 1) % S) return { cls: 'gpu-n', label: `tile ${t + 1}：已提交，到了就算` }
    return { cls: 'gpu-wait', label: `tile ${t + 2}：cp.async 写入` }
  })
}))
</script>

<template>
  <svg class="w680" viewBox="0 0 720 242" role="img" aria-label="三级流水线：每一轮有一个槽位在计算，一个槽位已经提交等待就绪，还有一个槽位正被异步写入。每轮结束后三个槽位轮换。">
    <text v-for="s in S" :key="s" class="gpu-t gpu-tb gpu-mid" :x="X0 + (s - 0.5) * W" y="36">smem 槽位 {{ s - 1 }}</text>
    <g v-for="(it, r) in iters" :key="it.t">
      <text class="gpu-ts" x="10" :y="Y0 + r * (H + 8) + 22">第 {{ it.t }} 轮</text>
      <g v-for="(c, s) in it.slots" :key="s">
        <rect :class="c.cls" :x="X0 + s * W + 2" :y="Y0 + r * (H + 8)" :width="W - 4" :height="H" />
        <text class="gpu-tk gpu-mid" :x="X0 + (s + 0.5) * W" :y="Y0 + r * (H + 8) + 22">{{ c.label }}</text>
      </g>
    </g>
    <text class="gpu-ts" x="10" y="232">这一轮写入的槽位，正是上一轮刚算完的那个；屏障保证所有线程都已经读完它。</text>
  </svg>
</template>
