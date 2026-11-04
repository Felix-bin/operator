<template>
  <svg class="w680" viewBox="0 0 850 345" role="img" aria-label="CUDA 执行层级与硬件的对应关系：Grid 分布到整块 GPU，每个 Block 运行在一个 SM 上，Warp 是调度器发指令的单位，每个 Thread 在发射时占用一条执行车道。">
    <defs>
      <marker id="a8" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
      <g id="f8-row">
        <rect class="gpu-c" x="0" width="22" height="20"/><rect class="gpu-c" x="27" width="22" height="20"/><rect class="gpu-c" x="54" width="22" height="20"/><rect class="gpu-c" x="81" width="22" height="20"/><rect class="gpu-c" x="108" width="22" height="20"/><rect class="gpu-c" x="135" width="22" height="20"/><rect class="gpu-c" x="162" width="22" height="20"/><rect class="gpu-c" x="189" width="22" height="20"/>
      </g>
    </defs>
    <text class="gpu-t gpu-tb" x="20" y="26">程序的层级</text>
    <text class="gpu-t gpu-tb" x="510" y="26">对应的硬件</text>

    <rect class="gpu-n" x="20" y="40" width="320" height="290"/>
    <text class="gpu-t gpu-tb" x="34" y="66">Grid：4096 个 Block</text>
    <rect class="gpu-o" x="40" y="80" width="280" height="230"/>
    <text class="gpu-t gpu-tb" x="54" y="106">Block：256 个 Thread，8 个 Warp</text>
    <rect class="gpu-f" x="60" y="120" width="240" height="170"/>
    <text class="gpu-t gpu-tb" x="74" y="146">Warp：32 个 Thread 同行</text>
    <use href="#f8-row" x="74" y="156"/><use href="#f8-row" x="74" y="181"/><use href="#f8-row" x="74" y="206"/><use href="#f8-row" x="74" y="231"/>
    <rect x="74" y="231" width="22" height="20" fill="none" stroke="currentColor" stroke-width="2"/>
    <line class="gpu-ln" x1="85" y1="251" x2="85" y2="261"/>
    <text class="gpu-tk gpu-tb" x="74" y="276">Thread：处理 1 个像素</text>

    <line class="gpu-ln" x1="340" y1="62" x2="510" y2="62" marker-end="url(#a8)"/>
    <line class="gpu-ln" x1="320" y1="102" x2="510" y2="102" marker-end="url(#a8)"/>
    <line class="gpu-ln" x1="300" y1="142" x2="510" y2="142" marker-end="url(#a8)"/>
    <line class="gpu-ln" x1="300" y1="272" x2="510" y2="272" marker-end="url(#a8)"/>
    <text class="gpu-ts gpu-mid" x="425" y="57">分布到</text>
    <text class="gpu-ts gpu-mid" x="425" y="97">运行在</text>
    <text class="gpu-ts gpu-mid" x="425" y="137">调度单位</text>
    <text class="gpu-ts gpu-mid" x="425" y="267">发射时占用</text>

    <rect class="gpu-n" x="510" y="44" width="320" height="36"/><text class="gpu-t gpu-mid" x="670" y="67">整块 GPU：各 GPC 的 SM 共同承接</text>
    <rect class="gpu-c" x="510" y="84" width="320" height="36"/><text class="gpu-t gpu-mid" x="670" y="107">一个 SM：Block 中途不迁移</text>
    <rect class="gpu-f" x="510" y="124" width="320" height="36"/><text class="gpu-t gpu-mid" x="670" y="147">Warp 调度器：每次选一个 Warp 发指令</text>
    <rect class="gpu-c" x="510" y="254" width="320" height="36"/><text class="gpu-t gpu-mid" x="670" y="277">一条车道，外加一份私有寄存器</text>

    <text class="gpu-ts" x="510" y="192">由程序员指定：Grid 和 Block 的大小</text>
    <text class="gpu-ts" x="510" y="214">由硬件固定：Warp 宽度为 32</text>
  </svg>
</template>
