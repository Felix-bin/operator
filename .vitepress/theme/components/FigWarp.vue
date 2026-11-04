<template>
  <svg class="w680" viewBox="0 0 800 200" role="img" aria-label="线程组调度时序：四个线程组轮流发指令，一组等待内存时另一组接替，执行单元每个周期都有指令可发。">
    <defs>
      <marker id="a4" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <text class="gpu-t" x="20" y="37">线程组 A</text>
    <text class="gpu-t" x="20" y="69">线程组 B</text>
    <text class="gpu-t" x="20" y="101">线程组 C</text>
    <text class="gpu-t" x="20" y="133">线程组 D</text>

    <rect class="gpu-c" x="110" y="20" width="112" height="24"/><text class="gpu-tk gpu-mid" x="166" y="37">发指令</text>
    <rect class="gpu-wait" x="222" y="20" width="280" height="24"/><text class="gpu-tk gpu-mid" x="362" y="37">等内存</text>
    <rect class="gpu-c" x="502" y="20" width="84" height="24"/>
    <rect class="gpu-wait" x="586" y="20" width="196" height="24"/>

    <rect class="gpu-c" x="222" y="52" width="112" height="24"/>
    <rect class="gpu-wait" x="334" y="52" width="252" height="24"/>
    <rect class="gpu-c" x="586" y="52" width="84" height="24"/>

    <rect class="gpu-c" x="334" y="84" width="84" height="24"/>
    <rect class="gpu-wait" x="418" y="84" width="252" height="24"/>
    <rect class="gpu-c" x="670" y="84" width="84" height="24"/>

    <rect class="gpu-c" x="418" y="116" width="84" height="24"/>
    <rect class="gpu-wait" x="502" y="116" width="252" height="24"/>
    <rect class="gpu-c" x="754" y="116" width="28" height="24"/>

    <line class="gpu-ln" x1="110" y1="160" x2="782" y2="160" marker-end="url(#a4)"/>
    <text class="gpu-ts" x="110" y="180">时间（时钟周期）</text>
    <text class="gpu-ts gpu-end" x="782" y="180">每个周期都有一组线程在发指令</text>
  </svg>
</template>
