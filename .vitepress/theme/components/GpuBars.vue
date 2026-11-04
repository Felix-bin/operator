<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  rows: { type: Array, required: true },
  scale: { type: String, default: 'linear' },
  ticks: { type: Array, required: true },
  unit: { type: String, default: '' },
  label: { type: String, required: true }
})

const lo = computed(() => props.ticks[0])
const hi = computed(() => props.ticks[props.ticks.length - 1])
const pos = (v) =>
  props.scale === 'log'
    ? (Math.log10(v / lo.value) / Math.log10(hi.value / lo.value)) * 100
    : ((v - lo.value) / (hi.value - lo.value)) * 100

const tip = ref(null)
function show(row, e) {
  const host = e.currentTarget.closest('.gpu-bars').getBoundingClientRect()
  const box = e.currentTarget.getBoundingClientRect()
  tip.value = { row, x: e.clientX ? e.clientX - host.left : box.left - host.left, y: box.bottom - host.top + 6 }
}
</script>

<template>
  <div class="gpu-bars" role="group" :aria-label="label" @pointerleave="tip = null">
    <div
      v-for="r in rows"
      :key="r.label"
      class="row"
      tabindex="0"
      @pointermove="show(r, $event)"
      @focus="show(r, $event)"
      @blur="tip = null"
    >
      <span class="name">{{ r.label }}</span>
      <span class="track">
        <i v-for="t in ticks" :key="t" class="tick" :style="{ left: pos(t) + '%' }"></i>
        <b class="bar" :style="{ width: pos(r.value) + '%' }"></b>
        <span class="val" :style="{ left: pos(r.value) + '%' }">{{ r.value.toLocaleString('en-US') }} {{ unit }}</span>
      </span>
    </div>
    <div class="axis">
      <span></span>
      <span class="track">
        <span v-for="t in ticks" :key="t" class="tl" :style="{ left: pos(t) + '%' }">{{ t.toLocaleString('en-US') }}</span>
      </span>
    </div>
    <div v-if="tip && tip.row.note" class="tip" :style="{ left: Math.max(0, tip.x - 80) + 'px', top: tip.y + 'px' }">
      <strong>{{ tip.row.value.toLocaleString('en-US') }} {{ unit }}</strong> {{ tip.row.note }}
    </div>
  </div>
</template>
