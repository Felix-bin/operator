<script setup>
const SUB = ['₀', '₁', '₂', '₃']
const T = 4
const rows = [0, 1, 2, 3]

// V1：外循环 K/V，一次一列
const v1 = { x: 46, y: 86, w: 72, h: 44, px: 76, py: 48 }
// V2：外循环 Q，每行一个 Thread Block
const v2 = { x: 80, y: 86, w: 52, h: 44, px: 56, py: 48 }
</script>

<template>
  <div class="flv2-pair">
    <svg viewBox="0 0 370 346" role="img" aria-label="V1 的 4 乘 4 Tile 网格。按列走，每列开头读一次 K 和 V，每一格都要读 Q、O、m、l 并写回 O、m、l。">
      <text class="gpu-tt" x="10" y="22">V1：外循环 K/V，内循环 Q</text>
      <text class="gpu-ts" x="10" y="42">序号是一个 Thread Block 处理 Tile 的先后</text>
      <g v-for="c in rows" :key="`k${c}`">
        <rect class="gpu-m" :x="v1.x + c * v1.px" y="54" :width="v1.w" height="24" />
        <text class="gpu-tk gpu-mid" :x="v1.x + c * v1.px + v1.w / 2" y="70">读 K{{ SUB[c] }} V{{ SUB[c] }}</text>
      </g>
      <g v-for="r in rows" :key="`r${r}`">
        <text class="gpu-t" x="14" :y="v1.y + r * v1.py + 27">Q{{ SUB[r] }}</text>
        <g v-for="c in rows" :key="c">
          <rect class="gpu-m" :x="v1.x + c * v1.px" :y="v1.y + r * v1.py" :width="v1.w" :height="v1.h" />
          <text class="gpu-tk gpu-mid gpu-tb" :x="v1.x + c * v1.px + v1.w / 2" :y="v1.y + r * v1.py + 13">{{ c * T + r + 1 }}</text>
          <text class="gpu-tk gpu-mid" :x="v1.x + c * v1.px + v1.w / 2" :y="v1.y + r * v1.py + 26">读 Q O m l</text>
          <text class="gpu-tk gpu-mid" :x="v1.x + c * v1.px + v1.w / 2" :y="v1.y + r * v1.py + 39">写 O m l</text>
        </g>
      </g>
      <text class="gpu-t" x="10" y="298">每个 K/V 块读 1 次</text>
      <text class="gpu-t" x="10" y="318">每个 Q 块读 Tc = 4 次</text>
      <text class="gpu-t" x="10" y="338">每个 O 块读写 2Tc = 8 次，m、l 同样</text>
    </svg>

    <svg viewBox="0 0 370 346" role="img" aria-label="V2 的 4 乘 4 Tile 网格。按行走，每行开头读一次 Q，每一格读 K 和 V，每行结束写一次 O 和 L，各行互不依赖，分给不同的 Thread Block。">
      <text class="gpu-tt" x="10" y="22">V2：外循环 Q，内循环 K/V</text>
      <text class="gpu-ts" x="10" y="42">序号是块内的先后，不同行同时进行</text>
      <text class="gpu-ts gpu-mid" x="41" y="72">每行开始</text>
      <text v-for="c in rows" :key="`h${c}`" class="gpu-ts gpu-mid" :x="v2.x + c * v2.px + v2.w / 2" y="72">K{{ SUB[c] }} V{{ SUB[c] }}</text>
      <text class="gpu-ts gpu-mid" x="333" y="72">每行结束</text>
      <g v-for="r in rows" :key="`v2r${r}`">
        <rect class="gpu-m" x="10" :y="v2.y + r * v2.py" width="62" :height="v2.h" />
        <text class="gpu-tk gpu-mid gpu-tb" x="41" :y="v2.y + r * v2.py + 18">Block {{ r }}</text>
        <text class="gpu-tk gpu-mid" x="41" :y="v2.y + r * v2.py + 34">读 Q{{ SUB[r] }}</text>
        <g v-for="c in rows" :key="c">
          <rect class="gpu-m" :x="v2.x + c * v2.px" :y="v2.y + r * v2.py" :width="v2.w" :height="v2.h" />
          <text class="gpu-tk gpu-mid gpu-tb" :x="v2.x + c * v2.px + v2.w / 2" :y="v2.y + r * v2.py + 18">{{ c + 1 }}</text>
          <text class="gpu-tk gpu-mid" :x="v2.x + c * v2.px + v2.w / 2" :y="v2.y + r * v2.py + 34">读 K V</text>
        </g>
        <rect class="gpu-m" x="304" :y="v2.y + r * v2.py" width="58" :height="v2.h" />
        <text class="gpu-tk gpu-mid" x="333" :y="v2.y + r * v2.py + 18">写 O{{ SUB[r] }}</text>
        <text class="gpu-tk gpu-mid" x="333" :y="v2.y + r * v2.py + 34">写 L{{ SUB[r] }}</text>
      </g>
      <text class="gpu-t" x="10" y="298">每个 Q 块读 1 次，每个 O 块写 1 次，L 同样</text>
      <text class="gpu-t" x="10" y="318">每个 K/V 块读 Tr = 4 次，比 V1 多</text>
      <text class="gpu-t" x="10" y="338">O 不用读回，中间结果留在芯片上</text>
    </svg>
  </div>
</template>

<style>
.flv2-pair {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 16px 24px;
}
.gpu-fig .flv2-pair svg {
  flex: 1 1 360px;
  min-width: 0;
  max-width: 440px;
}
</style>
