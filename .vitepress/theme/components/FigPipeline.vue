<template>
  <svg class="w680" viewBox="0 0 920 335" role="img" aria-label="图形路径依次经过输入装配、顶点着色器、图元装配、光栅化、片元着色器、ROP 和帧缓冲；计算路径从任务派发直接到计算着色器再到读写缓冲区。顶点、片元和计算着色器都运行在同一批统一着色器核心上。">
    <defs>
      <marker id="a7" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <text class="gpu-t gpu-tb" x="22" y="30">图形路径</text>

    <rect class="gpu-f" x="22" y="60" width="96" height="56"/><text class="gpu-t gpu-mid" x="70" y="84">输入装配</text><text class="gpu-ts gpu-mid" x="70" y="103">读顶点缓冲</text>
    <rect class="gpu-c" x="152" y="60" width="96" height="56"/><text class="gpu-t gpu-mid" x="200" y="84">顶点着色器</text><text class="gpu-ts gpu-mid" x="200" y="103">坐标变换</text>
    <rect class="gpu-f" x="282" y="60" width="96" height="56"/><text class="gpu-t gpu-mid" x="330" y="84">图元装配</text><text class="gpu-ts gpu-mid" x="330" y="103">裁剪与剔除</text>
    <rect class="gpu-f" x="412" y="60" width="96" height="56"/><text class="gpu-t gpu-mid" x="460" y="84">光栅化</text><text class="gpu-ts gpu-mid" x="460" y="103">三角形转片元</text>
    <rect class="gpu-c" x="542" y="60" width="96" height="56"/><text class="gpu-t gpu-mid" x="590" y="84">片元着色器</text><text class="gpu-ts gpu-mid" x="590" y="103">计算颜色</text>
    <rect class="gpu-f" x="672" y="60" width="96" height="56"/><text class="gpu-t gpu-mid" x="720" y="84">ROP</text><text class="gpu-ts gpu-mid" x="720" y="103">深度与混合</text>
    <rect class="gpu-m" x="802" y="60" width="96" height="56"/><text class="gpu-t gpu-mid" x="850" y="84">帧缓冲</text><text class="gpu-ts gpu-mid" x="850" y="103">最终图像</text>

    <line class="gpu-ln" x1="118" y1="88" x2="152" y2="88" marker-end="url(#a7)"/><text class="gpu-ts gpu-mid" x="135" y="52">顶点</text>
    <line class="gpu-ln" x1="248" y1="88" x2="282" y2="88" marker-end="url(#a7)"/><text class="gpu-ts gpu-mid" x="265" y="52">变换后顶点</text>
    <line class="gpu-ln" x1="378" y1="88" x2="412" y2="88" marker-end="url(#a7)"/><text class="gpu-ts gpu-mid" x="395" y="52">三角形</text>
    <line class="gpu-ln" x1="508" y1="88" x2="542" y2="88" marker-end="url(#a7)"/><text class="gpu-ts gpu-mid" x="525" y="52">片元</text>
    <line class="gpu-ln" x1="638" y1="88" x2="672" y2="88" marker-end="url(#a7)"/><text class="gpu-ts gpu-mid" x="655" y="52">着色片元</text>
    <line class="gpu-ln" x1="768" y1="88" x2="802" y2="88" marker-end="url(#a7)"/><text class="gpu-ts gpu-mid" x="785" y="52">像素</text>

    <line class="gpu-ln gpu-dash" x1="200" y1="116" x2="200" y2="170" marker-end="url(#a7)"/><text class="gpu-ts" x="210" y="148">运行于</text>
    <line class="gpu-ln gpu-dash" x1="590" y1="116" x2="590" y2="170" marker-end="url(#a7)"/><text class="gpu-ts" x="600" y="148">运行于</text>

    <rect class="gpu-c" x="22" y="170" width="876" height="50"/>
    <text class="gpu-t gpu-tb gpu-mid" x="460" y="200">统一着色器核心：同一批 SM，按任务动态分配</text>

    <line class="gpu-ln gpu-dash" x1="200" y1="270" x2="200" y2="220" marker-end="url(#a7)"/><text class="gpu-ts" x="210" y="250">运行于</text>

    <text class="gpu-t gpu-tb" x="22" y="258">计算路径</text>
    <rect class="gpu-f" x="22" y="270" width="96" height="50"/><text class="gpu-t gpu-mid" x="70" y="300">任务派发</text>
    <rect class="gpu-c" x="152" y="270" width="96" height="50"/><text class="gpu-t gpu-mid" x="200" y="300">计算着色器</text>
    <rect class="gpu-m" x="282" y="270" width="110" height="50"/><text class="gpu-t gpu-mid" x="337" y="300">读写缓冲区</text>
    <line class="gpu-ln" x1="118" y1="295" x2="152" y2="295" marker-end="url(#a7)"/>
    <line class="gpu-ln" x1="248" y1="295" x2="282" y2="295" marker-start="url(#a7)" marker-end="url(#a7)"/>
    <text class="gpu-ts" x="420" y="290">计算路径跳过图元装配、光栅化和 ROP，</text>
    <text class="gpu-ts" x="420" y="309">直接读写缓冲区。AI 训练和推理主要走这条路径。</text>
  </svg>
</template>
