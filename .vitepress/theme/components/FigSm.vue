<template>
  <svg class="w680" viewBox="0 0 850 480" role="img" aria-label="SM 内部结构：调度器选择线程组，取指译码后把同一条指令广播给所有车道；车道读写各自的寄存器，通过加载存储单元访问 L1 缓存和共享内存，L1 未命中时才走片上互连。内存返回后线程组重新就绪。">
    <defs>
      <marker id="a3" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <!-- 反馈 -->
    <path class="gpu-ln gpu-dash" d="M40 310 H18 V32 H105 V50" marker-end="url(#a3)"/>
    <text class="gpu-ts" x="115" y="28">内存数据返回，该线程组重新就绪</text>

    <!-- 控制 -->
    <rect class="gpu-f" x="40" y="50" width="130" height="48"/><text class="gpu-t gpu-mid" x="105" y="79">线程组调度器</text>
    <rect class="gpu-f" x="270" y="50" width="110" height="48"/><text class="gpu-t gpu-mid" x="325" y="79">指令缓存</text>
    <rect class="gpu-f" x="450" y="50" width="100" height="48"/><text class="gpu-t gpu-mid" x="500" y="79">译码</text>
    <line class="gpu-ln" x1="170" y1="74" x2="270" y2="74" marker-end="url(#a3)"/>
    <text class="gpu-ts gpu-mid" x="220" y="66">选就绪线程组</text>
    <line class="gpu-ln" x1="380" y1="74" x2="450" y2="74" marker-end="url(#a3)"/>
    <text class="gpu-ts gpu-mid" x="415" y="66">取指</text>

    <!-- 广播 -->
    <path class="gpu-ln" d="M500 98 V130"/>
    <path class="gpu-ln" d="M76 130 H765"/>
    <text class="gpu-ts" x="510" y="120">同一条指令广播给所有车道</text>
    <line class="gpu-ln" x1="76" y1="130" x2="76" y2="150" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="156" y1="130" x2="156" y2="150" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="236" y1="130" x2="236" y2="150" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="316" y1="130" x2="316" y2="150" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="396" y1="130" x2="396" y2="150" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="476" y1="130" x2="476" y2="150" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="556" y1="130" x2="556" y2="150" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="636" y1="130" x2="636" y2="150" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="765" y1="130" x2="765" y2="150" marker-end="url(#a3)"/>

    <!-- 车道 -->
    <rect class="gpu-c" x="40" y="150" width="72" height="56"/><text class="gpu-tk gpu-mid" x="76" y="174">车道 0</text><text class="gpu-ts gpu-mid" x="76" y="192">ALU</text>
    <rect class="gpu-c" x="120" y="150" width="72" height="56"/><text class="gpu-tk gpu-mid" x="156" y="174">车道 1</text><text class="gpu-ts gpu-mid" x="156" y="192">ALU</text>
    <rect class="gpu-c" x="200" y="150" width="72" height="56"/><text class="gpu-tk gpu-mid" x="236" y="174">车道 2</text><text class="gpu-ts gpu-mid" x="236" y="192">ALU</text>
    <rect class="gpu-c" x="280" y="150" width="72" height="56"/><text class="gpu-tk gpu-mid" x="316" y="174">车道 3</text><text class="gpu-ts gpu-mid" x="316" y="192">ALU</text>
    <rect class="gpu-c" x="360" y="150" width="72" height="56"/><text class="gpu-tk gpu-mid" x="396" y="174">车道 4</text><text class="gpu-ts gpu-mid" x="396" y="192">ALU</text>
    <rect class="gpu-c" x="440" y="150" width="72" height="56"/><text class="gpu-tk gpu-mid" x="476" y="174">车道 5</text><text class="gpu-ts gpu-mid" x="476" y="192">ALU</text>
    <rect class="gpu-c" x="520" y="150" width="72" height="56"/><text class="gpu-tk gpu-mid" x="556" y="174">车道 6</text><text class="gpu-ts gpu-mid" x="556" y="192">ALU</text>
    <rect class="gpu-c" x="600" y="150" width="72" height="56"/><text class="gpu-tk gpu-mid" x="636" y="174">车道 7</text><text class="gpu-ts gpu-mid" x="636" y="192">ALU</text>

    <!-- 专用单元 -->
    <rect class="gpu-f" x="700" y="150" width="130" height="116"/>
    <text class="gpu-t gpu-tb gpu-mid" x="765" y="176">专用单元</text>
    <text class="gpu-tk gpu-mid" x="765" y="200">纹理采样</text>
    <text class="gpu-tk gpu-mid" x="765" y="220">光线求交</text>
    <text class="gpu-tk gpu-mid" x="765" y="240">矩阵运算</text>

    <!-- 寄存器 -->
    <line class="gpu-ln" x1="76" y1="206" x2="76" y2="230" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="156" y1="206" x2="156" y2="230" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="236" y1="206" x2="236" y2="230" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="316" y1="206" x2="316" y2="230" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="396" y1="206" x2="396" y2="230" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="476" y1="206" x2="476" y2="230" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="556" y1="206" x2="556" y2="230" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="636" y1="206" x2="636" y2="230" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <rect class="gpu-m" x="40" y="230" width="632" height="36"/>
    <text class="gpu-t gpu-mid" x="356" y="253">寄存器文件：每条车道一份，读操作数、写结果</text>
    <line class="gpu-ln" x1="672" y1="248" x2="700" y2="248" marker-start="url(#a3)" marker-end="url(#a3)"/>

    <!-- 加载存储 -->
    <line class="gpu-ln" x1="356" y1="266" x2="356" y2="292" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <text class="gpu-ts" x="366" y="283">地址与数据</text>
    <rect class="gpu-f" x="40" y="292" width="632" height="36"/>
    <text class="gpu-t gpu-mid" x="356" y="315">加载 / 存储单元</text>

    <!-- 本地存储 -->
    <line class="gpu-ln" x1="190" y1="328" x2="190" y2="360" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <line class="gpu-ln" x1="522" y1="328" x2="522" y2="360" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <rect class="gpu-m" x="40" y="360" width="300" height="40"/><text class="gpu-t gpu-mid" x="190" y="385">L1 数据缓存</text>
    <rect class="gpu-m" x="372" y="360" width="300" height="40"/><text class="gpu-t gpu-mid" x="522" y="385">共享内存：同组线程协作，不经过 L2</text>

    <!-- 外部 -->
    <line class="gpu-ln" x1="190" y1="400" x2="190" y2="432" marker-start="url(#a3)" marker-end="url(#a3)"/>
    <text class="gpu-ts" x="200" y="421">未命中</text>
    <rect class="gpu-n" x="40" y="432" width="300" height="32"/>
    <text class="gpu-t gpu-mid" x="190" y="453">片上互连 → L2 → 显存</text>
  </svg>
</template>
