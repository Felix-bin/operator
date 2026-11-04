<template>
  <svg class="w680" viewBox="0 0 820 340" role="img" aria-label="GPU 存储层次：寄存器文件、L1 缓存与共享内存位于计算单元内，L2 缓存位于芯片内，显存位于显卡上，系统内存要跨过 PCIe。越往下容量越大、速度越慢、共享范围越广。">
    <defs>
      <marker id="a6" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <line class="gpu-ln" x1="50" y1="30" x2="50" y2="320" marker-end="url(#a6)"/>
    <text class="gpu-ts gpu-mid" transform="translate(36 175) rotate(-90)">容量更大 · 速度更慢 · 共享范围更广</text>

    <rect class="gpu-m" x="90" y="30" width="130" height="44"/><text class="gpu-t gpu-mid" x="155" y="57">寄存器文件</text>
    <text class="gpu-t" x="420" y="57">每条车道私有，RTX 4090 每个 SM 共 256 KB</text>

    <rect class="gpu-m" x="90" y="92" width="190" height="44"/><text class="gpu-t gpu-mid" x="185" y="119">L1 缓存 · 共享内存</text>
    <text class="gpu-t" x="420" y="119">每个 SM 一份，RTX 4090 为 128 KB</text>

    <line class="gpu-ln gpu-dash" x1="90" y1="145" x2="800" y2="145"/>
    <text class="gpu-ts gpu-end" x="800" y="141">计算单元边界</text>

    <rect class="gpu-m" x="90" y="154" width="250" height="44"/><text class="gpu-t gpu-mid" x="215" y="181">L2 缓存</text>
    <text class="gpu-t" x="420" y="181">全芯片共享，RTX 4090 为 72 MB</text>

    <line class="gpu-ln gpu-dash" x1="90" y1="207" x2="800" y2="207"/>
    <text class="gpu-ts gpu-end" x="800" y="203">芯片边界</text>

    <rect class="gpu-m" x="90" y="216" width="310" height="44"/><text class="gpu-t gpu-mid" x="245" y="243">显存（GDDR / HBM）</text>
    <text class="gpu-t" x="420" y="243">整卡共享，消费级常见 8–24 GB</text>

    <line class="gpu-ln gpu-dash" x1="90" y1="269" x2="800" y2="269"/>
    <text class="gpu-ts gpu-end" x="800" y="265">显卡边界（PCIe）</text>

    <rect class="gpu-m" x="90" y="278" width="310" height="44"/><text class="gpu-t gpu-mid" x="245" y="305">系统内存</text>
    <text class="gpu-t" x="420" y="305">带宽远低于显存，应尽量少走</text>
  </svg>
</template>
