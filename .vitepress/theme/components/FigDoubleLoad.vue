<template>
  <svg class="w680" viewBox="0 0 800 276" role="img" aria-label="V2 的一个 Block 读取 256 个元素，每个线程一个。V3 的一个 Block 读取 512 个元素，线程 tid 同时读取第 tid 和第 tid 加 256 个元素，相加后写入共享内存。">
    <defs>
      <marker id="rd-load" markerWidth="8" markerHeight="8" refX="7" refY="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>

    <text class="gpu-tt" x="16" y="20">V2：每线程 1 个元素</text>
    <text class="gpu-ts" x="16" y="52">input</text>
    <rect class="gpu-m" x="90" y="34" width="320" height="28"/>
    <text class="gpu-tk gpu-mid" x="250" y="53">元素 0 ~ 255</text>
    <text class="gpu-ts" x="16" y="106">smem</text>
    <rect class="gpu-c" x="90" y="88" width="320" height="28"/>
    <text class="gpu-tk gpu-mid" x="250" y="107">256 个线程，256 个值</text>
    <line class="gpu-ln" x1="93" y1="62" x2="93" y2="87" marker-end="url(#rd-load)"/>
    <line class="gpu-ln" x1="250" y1="62" x2="250" y2="87" marker-end="url(#rd-load)"/>
    <line class="gpu-ln" x1="407" y1="62" x2="407" y2="87" marker-end="url(#rd-load)"/>
    <text class="gpu-t" x="440" y="54">一个 Block 处理 256 个元素</text>
    <text class="gpu-t" x="440" y="76">需要 524,288 个 Block</text>
    <text class="gpu-t" x="440" y="98">第 1 轮之后，后 128 个线程不再工作</text>
    <text class="gpu-t" x="440" y="120">它们只做了一次搬运</text>

    <line class="gpu-grid" x1="16" y1="138" x2="784" y2="138"/>

    <text class="gpu-tt" x="16" y="166">V3：每线程 2 个元素</text>
    <text class="gpu-ts" x="16" y="198">input</text>
    <rect class="gpu-m" x="90" y="180" width="320" height="28"/>
    <text class="gpu-tk gpu-mid" x="250" y="199">元素 0 ~ 255</text>
    <rect class="gpu-m" x="410" y="180" width="320" height="28"/>
    <text class="gpu-tk gpu-mid" x="570" y="199">元素 256 ~ 511</text>
    <text class="gpu-ts" x="16" y="252">smem</text>
    <rect class="gpu-c" x="90" y="234" width="320" height="28"/>
    <text class="gpu-tk gpu-mid" x="250" y="253">256 个线程，每个值是两个元素之和</text>

    <rect class="gpu-dot" x="123" y="180" width="4" height="28"/>
    <rect class="gpu-dot" x="443" y="180" width="4" height="28"/>
    <path class="gpu-ln" d="M125 208 L125 233" marker-end="url(#rd-load)"/>
    <path class="gpu-ln" d="M445 208 C445 226 200 214 130 233" marker-end="url(#rd-load)"/>
    <text class="gpu-t" x="440" y="232">tid 同时读 input[gid] 和 input[gid + 256]</text>
    <text class="gpu-t" x="440" y="254">一个 Block 处理 512 个元素，需要 262,144 个 Block</text>
  </svg>
</template>
