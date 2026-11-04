<script setup>
const S = [1.0, 0.2, 2.0, 3.5, 0.5, 1.5, 2.5, 4.0, -1.0]
const V = [1, -2, 0.5, 3, -1, 2, 0.5, -0.5, 1]
const BS = 3

function directPrefix(n) {
  const s = S.slice(0, n)
  const m = Math.max(...s)
  const d = s.reduce((a, v) => a + Math.exp(v - m), 0)
  return s.reduce((a, v, j) => a + (Math.exp(v - m) / d) * V[j], 0)
}

let m = -Infinity
let d = 0
let O = 0
const steps = []
for (let b = 0; b < S.length; b += BS) {
  const sb = S.slice(b, b + BS)
  const vb = V.slice(b, b + BS)
  const mBlock = Math.max(...sb)
  const mNew = Math.max(m, mBlock)
  const factor = Math.exp(m - mNew)
  const P = sb.map((s) => Math.exp(s - mBlock))
  const dBlock = P.reduce((a, p) => a + p, 0)
  const dNew = d * factor + dBlock * Math.exp(mBlock - mNew)
  const PV = P.reduce((a, p, j) => a + p * vb[j], 0)
  const ONew = (O * d * factor + Math.exp(mBlock - mNew) * PV) / dNew
  steps.push({ b: b / BS, sb, mBlock, mNew, factor, dNew, ONew, direct: directPrefix(b + BS) })
  m = mNew
  d = dNew
  O = ONew
}
const f = (v) => v.toFixed(4)
const sn = (v) => String(v).replace('-', '−')
</script>

<template>
  <div>
    <div class="gpu-scroll" tabindex="0">
      <svg class="w680" viewBox="0 0 764 172" role="img" aria-label="FlashAttention 每次把一块 K 和 V 从显存读到片上 SRAM，算出这一块的分数，更新 m、d、O，再写回显存；不保存整个注意力矩阵。">
        <defs>
          <marker id="online-flash-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="currentColor" /></marker>
        </defs>
        <text class="gpu-ts" x="0" y="14">显存（HBM）</text>
        <g v-for="i in 3" :key="i">
          <rect :class="i === 2 ? 'gpu-m' : 'gpu-n'" :x="(i - 1) * 62" y="24" width="58" height="52" />
          <text class="gpu-tk gpu-mid" :x="(i - 1) * 62 + 29" y="46">K{{ i }}</text>
          <text class="gpu-tk gpu-mid" :x="(i - 1) * 62 + 29" y="64">V{{ i }}</text>
        </g>
        <text class="gpu-ts" x="0" y="96">一次只读一块</text>
        <path class="gpu-ln" d="M186 50 L214 50" marker-end="url(#online-flash-arrow)" />
        <text class="gpu-ts" x="216" y="14">片上 SRAM</text>
        <rect class="gpu-m" x="216" y="24" width="146" height="52" />
        <text class="gpu-tk gpu-mid" x="289" y="46">当前块 K2、V2</text>
        <text class="gpu-tk gpu-mid" x="289" y="64">算 S = Q K2ᵀ，用完即弃</text>
        <path class="gpu-ln" d="M362 50 L390 50" marker-end="url(#online-flash-arrow)" />
        <rect class="gpu-c" x="392" y="24" width="168" height="52" />
        <text class="gpu-tk gpu-mid" x="476" y="46">算 m_block、d_block</text>
        <text class="gpu-tk gpu-mid" x="476" y="64">P = exp(S − m_block)</text>
        <path class="gpu-ln" d="M560 50 L588 50" marker-end="url(#online-flash-arrow)" />
        <rect class="gpu-c" x="590" y="24" width="170" height="52" />
        <text class="gpu-tk gpu-mid" x="675" y="46">更新 m、d</text>
        <text class="gpu-tk gpu-mid" x="675" y="64">修正旧 O，加上新块</text>
        <rect class="gpu-m" x="590" y="108" width="170" height="52" />
        <text class="gpu-tk gpu-mid" x="675" y="130">写回 m、d、O</text>
        <text class="gpu-tk gpu-mid" x="675" y="148">大小只和 N 成正比</text>
        <path class="gpu-ln" d="M675 76 L675 106" marker-end="url(#online-flash-arrow)" />
        <path class="gpu-ln gpu-dash" d="M590 134 L153 134 L153 78" marker-end="url(#online-flash-arrow)" />
        <text class="gpu-ts" x="166" y="152">下一块：同样的递推，只是一次吃进一块</text>
      </svg>
    </div>
    <div class="gpu-scroll" tabindex="0">
      <table class="online-flash-t">
        <thead>
          <tr>
            <th>块</th>
            <th>这一块的 S</th>
            <th>m_block</th>
            <th>m_new</th>
            <th>修正因子</th>
            <th>d_new</th>
            <th>O_new</th>
            <th>直接算到这一块</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="s in steps" :key="s.b">
            <td>{{ s.b + 1 }}</td>
            <td>{{ s.sb.map(sn).join(', ') }}</td>
            <td>{{ f(s.mBlock) }}</td>
            <td>{{ f(s.mNew) }}</td>
            <td>{{ f(s.factor) }}</td>
            <td>{{ f(s.dNew) }}</td>
            <td><b>{{ f(s.ONew) }}</b></td>
            <td><b>{{ f(s.direct) }}</b></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style>
.gpu-page .vp-doc .online-flash-t {
  display: table;
  width: 100%;
  min-width: 640px;
  margin: 0;
  font-size: 0.8125rem;
  font-variant-numeric: tabular-nums;
}
.gpu-page .vp-doc .online-flash-t th,
.gpu-page .vp-doc .online-flash-t td {
  padding: 5px 8px;
  white-space: nowrap;
}
</style>
