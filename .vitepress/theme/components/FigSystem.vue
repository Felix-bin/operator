<template>
  <svg class="w680" viewBox="0 0 800 600" role="img" aria-label="显卡结构图：CPU 通过很细的 PCIe 链路向 GPU 发送命令；命令处理器和调度器把任务分给多个 GPC；GPC 通过片上互连访问 L2 分片和内存控制器；内存控制器通过很粗的链路连接显存。">
    <defs>
      <marker id="a2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
      <g id="f2-gpc">
        <rect class="gpu-n" width="122" height="130"/>
        <rect class="gpu-f" x="8" y="28" width="106" height="24"/>
        <text class="gpu-tk gpu-mid" x="61" y="44">光栅引擎 · ROP</text>
        <rect class="gpu-c" x="8" y="58" width="49" height="30"/><text class="gpu-tk gpu-mid" x="32.5" y="77">SM</text>
        <rect class="gpu-c" x="65" y="58" width="49" height="30"/><text class="gpu-tk gpu-mid" x="89.5" y="77">SM</text>
        <rect class="gpu-c" x="8" y="92" width="49" height="30"/><text class="gpu-tk gpu-mid" x="32.5" y="111">SM</text>
        <rect class="gpu-c" x="65" y="92" width="49" height="30"/><text class="gpu-tk gpu-mid" x="89.5" y="111">SM</text>
      </g>
      <g id="f2-part">
        <rect class="gpu-n" width="122" height="82"/>
        <rect class="gpu-m" x="8" y="8" width="106" height="30"/><text class="gpu-tk gpu-mid" x="61" y="27">L2 缓存分片</text>
        <rect class="gpu-m" x="8" y="44" width="106" height="30"/><text class="gpu-tk gpu-mid" x="61" y="63">内存控制器</text>
      </g>
    </defs>

    <!-- 主机 -->
    <rect class="gpu-f" x="20" y="60" width="130" height="48"/><text class="gpu-t gpu-mid" x="85" y="89">CPU</text>
    <rect class="gpu-m" x="20" y="150" width="130" height="48"/><text class="gpu-t gpu-mid" x="85" y="179">系统内存</text>
    <line class="gpu-ln" x1="85" y1="108" x2="85" y2="150" marker-start="url(#a2)" marker-end="url(#a2)"/>
    <text class="gpu-ts" x="93" y="133">写入命令</text>

    <!-- 芯片外框 -->
    <rect class="gpu-o" x="220" y="30" width="560" height="458"/>
    <text class="gpu-t gpu-tb" x="236" y="50">GPU 芯片</text>

    <!-- PCIe -->
    <line class="gpu-thin" x1="150" y1="84" x2="240" y2="84"/>
    <text class="gpu-ts gpu-mid" x="185" y="62">PCIe 4.0</text>
    <text class="gpu-ts gpu-mid" x="185" y="77">≈32 GB/s</text>

    <!-- 前端 -->
    <rect class="gpu-f" x="240" y="60" width="130" height="48"/><text class="gpu-t gpu-mid" x="305" y="89">PCIe 接口与复制引擎</text>
    <rect class="gpu-f" x="400" y="60" width="150" height="48"/><text class="gpu-t gpu-mid" x="475" y="89">命令处理器</text>
    <rect class="gpu-f" x="580" y="60" width="184" height="48"/><text class="gpu-t gpu-mid" x="672" y="89">全局调度器</text>
    <line class="gpu-ln" x1="370" y1="84" x2="400" y2="84" marker-end="url(#a2)"/>
    <line class="gpu-ln" x1="550" y1="84" x2="580" y2="84" marker-end="url(#a2)"/>

    <!-- 派发总线 -->
    <path class="gpu-ln" d="M672 108 V130"/>
    <path class="gpu-ln" d="M301 130 H703"/>
    <line class="gpu-ln" x1="301" y1="130" x2="301" y2="150" marker-end="url(#a2)"/>
    <line class="gpu-ln" x1="435" y1="130" x2="435" y2="150" marker-end="url(#a2)"/>
    <line class="gpu-ln" x1="569" y1="130" x2="569" y2="150" marker-end="url(#a2)"/>
    <line class="gpu-ln" x1="703" y1="130" x2="703" y2="150" marker-end="url(#a2)"/>
    <text class="gpu-ts gpu-mid" x="502" y="124">派发工作</text>

    <!-- GPC -->
    <use href="#f2-gpc" x="240" y="150"/><use href="#f2-gpc" x="374" y="150"/><use href="#f2-gpc" x="508" y="150"/><use href="#f2-gpc" x="642" y="150"/>
    <text class="gpu-t gpu-tb gpu-mid" x="301" y="170">GPC 1</text><text class="gpu-t gpu-tb gpu-mid" x="435" y="170">GPC 2</text><text class="gpu-t gpu-tb gpu-mid" x="569" y="170">GPC 3</text><text class="gpu-t gpu-tb gpu-mid" x="703" y="170">GPC N</text>

    <!-- GPC 到互连 -->
    <line class="gpu-ln" x1="301" y1="280" x2="301" y2="320" marker-start="url(#a2)" marker-end="url(#a2)"/>
    <line class="gpu-ln" x1="435" y1="280" x2="435" y2="320" marker-start="url(#a2)" marker-end="url(#a2)"/>
    <line class="gpu-ln" x1="569" y1="280" x2="569" y2="320" marker-start="url(#a2)" marker-end="url(#a2)"/>
    <line class="gpu-ln" x1="703" y1="280" x2="703" y2="320" marker-start="url(#a2)" marker-end="url(#a2)"/>
    <text class="gpu-ts gpu-mid" x="502" y="305">读写请求</text>

    <rect class="gpu-n" x="240" y="320" width="524" height="34"/>
    <text class="gpu-t gpu-mid" x="502" y="342">片上互连</text>

    <line class="gpu-ln" x1="301" y1="354" x2="301" y2="392" marker-start="url(#a2)" marker-end="url(#a2)"/>
    <line class="gpu-ln" x1="435" y1="354" x2="435" y2="392" marker-start="url(#a2)" marker-end="url(#a2)"/>
    <line class="gpu-ln" x1="569" y1="354" x2="569" y2="392" marker-start="url(#a2)" marker-end="url(#a2)"/>
    <line class="gpu-ln" x1="703" y1="354" x2="703" y2="392" marker-start="url(#a2)" marker-end="url(#a2)"/>
    <text class="gpu-ts gpu-mid" x="502" y="377">按地址分发</text>

    <!-- 存储分区 -->
    <use href="#f2-part" x="240" y="392"/><use href="#f2-part" x="374" y="392"/><use href="#f2-part" x="508" y="392"/><use href="#f2-part" x="642" y="392"/>

    <!-- 显存链路，线宽与带宽成比例 -->
    <line class="gpu-thick" x1="301" y1="466" x2="301" y2="540"/>
    <line class="gpu-thick" x1="435" y1="466" x2="435" y2="540"/>
    <line class="gpu-thick" x1="569" y1="466" x2="569" y2="540"/>
    <line class="gpu-thick" x1="703" y1="466" x2="703" y2="540"/>
    <text class="gpu-ts gpu-mid" x="502" y="516">合计 ≈1008 GB/s</text>

    <rect class="gpu-m" x="240" y="540" width="122" height="44"/><text class="gpu-t gpu-mid" x="301" y="567">显存</text>
    <rect class="gpu-m" x="374" y="540" width="122" height="44"/><text class="gpu-t gpu-mid" x="435" y="567">显存</text>
    <rect class="gpu-m" x="508" y="540" width="122" height="44"/><text class="gpu-t gpu-mid" x="569" y="567">显存</text>
    <rect class="gpu-m" x="642" y="540" width="122" height="44"/><text class="gpu-t gpu-mid" x="703" y="567">显存</text>

    <text class="gpu-ts" x="20" y="530">链路线宽与带宽成比例</text>
    <text class="gpu-ts" x="20" y="549">PCIe 1 px ≈ 32 GB/s</text>
    <text class="gpu-ts" x="20" y="568">每组显存 8 px ≈ 252 GB/s</text>

    <!-- 步骤编号 -->
    <circle class="gpu-dot" cx="20" cy="60" r="10"/><text class="gpu-dot-t" x="20" y="64">1</text>
    <circle class="gpu-dot" cx="400" cy="60" r="10"/><text class="gpu-dot-t" x="400" y="64">2</text>
    <circle class="gpu-dot" cx="580" cy="60" r="10"/><text class="gpu-dot-t" x="580" y="64">3</text>
    <circle class="gpu-dot" cx="240" cy="150" r="10"/><text class="gpu-dot-t" x="240" y="154">4</text>
    <circle class="gpu-dot" cx="240" cy="320" r="10"/><text class="gpu-dot-t" x="240" y="324">5</text>
    <circle class="gpu-dot" cx="240" cy="392" r="10"/><text class="gpu-dot-t" x="240" y="396">6</text>
  </svg>
</template>
