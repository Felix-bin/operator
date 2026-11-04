<template>
  <svg class="w680" viewBox="0 0 720 320" role="img" aria-label="跨 SM 交接时序：生产者写数据，用 fence 等待 L2 确认后再写标志；消费者轮询标志，读到新值后才读数据。">
    <defs>
      <marker id="a10" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <rect class="gpu-c" x="50" y="12" width="120" height="32"/><text class="gpu-t gpu-mid" x="110" y="33">生产者 SM</text>
    <rect class="gpu-m" x="300" y="12" width="120" height="32"/><text class="gpu-t gpu-mid" x="360" y="33">L2 分片</text>
    <rect class="gpu-c" x="550" y="12" width="120" height="32"/><text class="gpu-t gpu-mid" x="610" y="33">消费者 SM</text>
    <line class="gpu-ln gpu-dash" x1="110" y1="44" x2="110" y2="308"/>
    <line class="gpu-ln gpu-dash" x1="360" y1="44" x2="360" y2="308"/>
    <line class="gpu-ln gpu-dash" x1="610" y1="44" x2="610" y2="308"/>

    <line class="gpu-ln" x1="110" y1="74" x2="360" y2="74" marker-end="url(#a10)"/><text class="gpu-ts gpu-mid" x="235" y="68">① 写数据</text>
    <rect class="gpu-f" x="100" y="78" width="20" height="46"/>
    <text class="gpu-tk gpu-end" x="94" y="105">fence 等待</text>
    <line class="gpu-ln" x1="360" y1="118" x2="120" y2="118" marker-end="url(#a10)"/><text class="gpu-ts gpu-mid" x="235" y="112">② 确认写入完成</text>
    <line class="gpu-ln" x1="110" y1="158" x2="360" y2="158" marker-end="url(#a10)"/><text class="gpu-ts gpu-mid" x="235" y="152">③ 写标志</text>

    <line class="gpu-ln" x1="610" y1="96" x2="360" y2="96" marker-end="url(#a10)"/><text class="gpu-ts gpu-mid" x="485" y="90">读标志</text>
    <line class="gpu-ln" x1="360" y1="128" x2="610" y2="128" marker-end="url(#a10)"/><text class="gpu-ts gpu-mid" x="485" y="122">旧值，继续轮询</text>
    <line class="gpu-ln" x1="610" y1="182" x2="360" y2="182" marker-end="url(#a10)"/><text class="gpu-ts gpu-mid" x="485" y="176">读标志</text>
    <line class="gpu-ln" x1="360" y1="214" x2="610" y2="214" marker-end="url(#a10)"/><text class="gpu-ts gpu-mid" x="485" y="208">④ 新值</text>
    <line class="gpu-ln" x1="610" y1="254" x2="360" y2="254" marker-end="url(#a10)"/><text class="gpu-ts gpu-mid" x="485" y="248">⑤ 读数据</text>
    <line class="gpu-ln" x1="360" y1="288" x2="610" y2="288" marker-end="url(#a10)"/><text class="gpu-ts gpu-mid" x="485" y="282">数据</text>
  </svg>
</template>
