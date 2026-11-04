<template>
  <svg class="w680" viewBox="0 0 770 225" role="img" aria-label="一次矩阵乘法经过的硬件：主机提交任务，权重和输入从显存经 L2 进入共享内存，由 Tensor Core 计算，部分和留在累加存储，完整结果交给向量单元处理后写回显存。">
    <defs>
      <marker id="a9" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0 0 L8 4 L0 8 z" fill="currentColor"/></marker>
    </defs>
    <rect class="gpu-f" x="20" y="60" width="110" height="54"/><text class="gpu-t gpu-mid" x="75" y="92">主机 CPU</text>
    <rect class="gpu-m" x="174" y="60" width="110" height="54"/><text class="gpu-t gpu-mid" x="229" y="92">显存</text>
    <rect class="gpu-m" x="328" y="60" width="110" height="54"/><text class="gpu-t gpu-mid" x="383" y="92">L2 缓存</text>
    <rect class="gpu-m" x="482" y="60" width="110" height="54"/><text class="gpu-t gpu-mid" x="537" y="84">共享内存</text><text class="gpu-ts gpu-mid" x="537" y="102">当前 tile</text>
    <rect class="gpu-c" x="636" y="60" width="110" height="54"/><text class="gpu-t gpu-mid" x="691" y="92">Tensor Core</text>

    <line class="gpu-ln" x1="130" y1="87" x2="174" y2="87" marker-end="url(#a9)"/><text class="gpu-ts gpu-mid" x="152" y="46">上传数据</text>
    <line class="gpu-ln" x1="284" y1="87" x2="328" y2="87" marker-end="url(#a9)"/><text class="gpu-ts gpu-mid" x="306" y="46">未命中才读</text>
    <line class="gpu-ln" x1="438" y1="87" x2="482" y2="87" marker-end="url(#a9)"/><text class="gpu-ts gpu-mid" x="460" y="46">搬入片上</text>
    <line class="gpu-ln" x1="592" y1="87" x2="636" y2="87" marker-end="url(#a9)"/><text class="gpu-ts gpu-mid" x="614" y="46">操作数</text>

    <rect class="gpu-m" x="636" y="160" width="110" height="50"/><text class="gpu-t gpu-mid" x="691" y="182">累加存储</text><text class="gpu-ts gpu-mid" x="691" y="199">部分和留在片上</text>
    <line class="gpu-ln" x1="691" y1="114" x2="691" y2="160" marker-start="url(#a9)" marker-end="url(#a9)"/>

    <rect class="gpu-c" x="482" y="160" width="110" height="50"/><text class="gpu-t gpu-mid" x="537" y="182">向量单元</text><text class="gpu-ts gpu-mid" x="537" y="199">Softmax、激活</text>
    <line class="gpu-ln" x1="636" y1="185" x2="592" y2="185" marker-end="url(#a9)"/><text class="gpu-ts gpu-mid" x="614" y="152">完整结果</text>

    <path class="gpu-ln" d="M482 185 H229 V114" marker-end="url(#a9)"/>
    <text class="gpu-ts gpu-mid" x="355" y="177">经 L2 写回</text>
  </svg>
</template>
