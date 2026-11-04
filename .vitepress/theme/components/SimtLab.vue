<script setup>
import { computed, ref } from 'vue'

const presets = {
  mix: [3, -1, 5, 2, -4, 7, -2, 6],
  all: [3, 1, 5, 2, 4, 7, 2, 6],
  one: [-3, -1, -5, -2, -4, 7, -2, -6]
}
const steps = [
  { code: 'load x', when: 'all', val: (x) => x },
  { code: 'cmp x > 0', when: 'all', val: (x) => (x > 0 ? '真' : '假') },
  { code: 'if: y = x * x', when: 'then', val: (x) => x * x },
  { code: 'if: y = y + 1', when: 'then', val: (x) => x * x + 1 },
  { code: 'if: y = y * 2', when: 'then', val: (x) => (x * x + 1) * 2 },
  { code: 'else: y = 0', when: 'else', val: () => 0 },
  { code: 'z = y + x', when: 'all', val: (x) => (x > 0 ? (x * x + 1) * 2 : 0) + x }
]

const xs = ref(presets.mix.slice())
const pos = computed(() => xs.value.filter((x) => x > 0).length)
const neg = computed(() => xs.value.length - pos.value)

const rows = computed(() =>
  steps.map((step) => ({
    code: step.code,
    skipped: (step.when === 'then' && pos.value === 0) || (step.when === 'else' && neg.value === 0),
    cells: xs.value.map((x) => {
      const on = step.when === 'all' || (step.when === 'then' ? x > 0 : x <= 0)
      return { on, text: on ? String(step.val(x)) : '' }
    })
  }))
)
const cycles = computed(() => rows.value.filter((r) => !r.skipped).length)
const active = computed(() =>
  rows.value
    .filter((r) => !r.skipped)
    .reduce((sum, r) => sum + r.cells.filter((c) => c.on).length, 0)
)
const util = computed(() => Math.round((active.value / (cycles.value * xs.value.length)) * 100))
const note = computed(() =>
  pos.value === 0 || neg.value === 0
    ? '所有车道走同一路径，没有车道闲置。'
    : `${pos.value} 条车道走长分支，${neg.value} 条走短分支。走长分支的车道越少，闲置越多。`
)

function flip(i) {
  xs.value[i] = -xs.value[i]
}
function load(name) {
  xs.value = presets[name].slice()
}
</script>

<template>
  <figure class="gpu-fig">
    <div class="gpu-tools">
      <button type="button" @click="load('mix')">混合输入</button>
      <button type="button" @click="load('all')">全部为正</button>
      <button type="button" @click="load('one')">只有一条为正</button>
    </div>
    <div class="gpu-scroll" tabindex="0">
      <table class="gpu-simt" aria-label="8 条车道的 SIMT 执行过程">
        <thead>
          <tr>
            <th class="code" scope="col">指令</th>
            <th v-for="(_, i) in xs" :key="i" scope="col">车道 {{ i }}</th>
          </tr>
          <tr>
            <th class="code" scope="row">输入 x（可点击）</th>
            <td v-for="(x, i) in xs" :key="i">
              <button
                type="button"
                :class="{ neg: x < 0 }"
                :aria-label="`翻转车道 ${i} 的输入，当前为 ${x}`"
                @click="flip(i)"
              >
                {{ x }}
              </button>
            </td>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="row.code" :class="{ skip: row.skipped }">
            <td class="code">{{ row.code }}</td>
            <td v-if="row.skipped" class="skipnote" :colspan="xs.length">无车道需要执行，整段跳过</td>
            <template v-else>
              <td v-for="(cell, i) in row.cells" :key="i" :class="cell.on ? 'on' : 'off'">{{ cell.text }}</td>
            </template>
          </tr>
        </tbody>
      </table>
    </div>
    <div class="gpu-read" aria-live="polite">
      <span><b>{{ util }}%</b> 车道利用率</span>
      <span><b>{{ cycles }}</b> 周期</span>
      <span>{{ note }}</span>
    </div>
    <figcaption>
      每行是一条指令，从上到下依次执行，每列是一条车道。青色格表示该车道在执行，斜线格表示被掩码关闭。若某条分支没有车道需要执行，整段都会被跳过。
    </figcaption>
  </figure>
</template>
