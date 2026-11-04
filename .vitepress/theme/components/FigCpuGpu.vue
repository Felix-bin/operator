<template>
  <svg class="w680" viewBox="0 0 880 365" role="img" aria-label="CPU 与 GPU 芯片面积分配对比：CPU 有少数几个带乱序调度和分支预测的大核心以及大容量缓存，GPU 由大量小型计算单元占据大部分面积。">
    <defs>
      <g id="f1-core">
        <rect class="gpu-n" width="162" height="65"/>
        <rect class="gpu-f" x="8" y="8" width="98" height="49"/>
        <text class="gpu-tk gpu-mid" x="57" y="29">乱序调度</text>
        <text class="gpu-tk gpu-mid" x="57" y="46">分支预测</text>
        <rect class="gpu-c" x="112" y="8" width="42" height="49"/>
        <text class="gpu-tk gpu-mid" x="133" y="37">ALU</text>
      </g>
      <g id="f1-row">
        <rect class="gpu-c" x="0" width="10" height="10"/><rect class="gpu-c" x="13" width="10" height="10"/><rect class="gpu-c" x="26" width="10" height="10"/><rect class="gpu-c" x="39" width="10" height="10"/><rect class="gpu-c" x="52" width="10" height="10"/><rect class="gpu-c" x="65" width="10" height="10"/>
      </g>
      <g id="f1-cl">
        <use href="#f1-row" y="0"/><use href="#f1-row" y="13"/><use href="#f1-row" y="26"/><use href="#f1-row" y="39"/><use href="#f1-row" y="52"/>
      </g>
    </defs>
    <text class="gpu-tt" x="40" y="36">CPU：少数大核心</text>
    <rect class="gpu-o" x="40" y="60" width="360" height="240"/>
    <use href="#f1-core" x="52" y="72"/><use href="#f1-core" x="226" y="72"/>
    <use href="#f1-core" x="52" y="147"/><use href="#f1-core" x="226" y="147"/>
    <rect class="gpu-m" x="52" y="222" width="336" height="66"/>
    <text class="gpu-t gpu-mid" x="220" y="260">大容量 L3 缓存</text>
    <text class="gpu-t" x="40" y="328">目标：让一条指令流尽快跑完</text>
    <text class="gpu-ts" x="40" y="350">代价：大量面积花在预测和调度上</text>

    <text class="gpu-tt" x="480" y="36">GPU：大量小型计算单元</text>
    <rect class="gpu-o" x="480" y="60" width="360" height="240"/>
    <rect class="gpu-f" x="492" y="72" width="336" height="26"/>
    <text class="gpu-tk gpu-mid" x="660" y="90">命令处理与任务调度</text>
    <use href="#f1-cl" x="492" y="106"/><use href="#f1-cl" x="579" y="106"/><use href="#f1-cl" x="666" y="106"/><use href="#f1-cl" x="753" y="106"/>
    <use href="#f1-cl" x="492" y="176"/><use href="#f1-cl" x="579" y="176"/><use href="#f1-cl" x="666" y="176"/><use href="#f1-cl" x="753" y="176"/>
    <rect class="gpu-m" x="492" y="248" width="336" height="40"/>
    <text class="gpu-t gpu-mid" x="660" y="273">L2 缓存与显存控制器</text>
    <text class="gpu-t" x="480" y="328">目标：单位时间完成的任务总量最大</text>
    <text class="gpu-ts" x="480" y="350">代价：单线程慢，分支和随机访存开销大</text>
  </svg>
</template>
