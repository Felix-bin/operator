---
title: Reduce 算子优化图解
description: 从最朴素的树形规约出发，逐版消除 Warp 分化、Bank 冲突、多余同步和访存浪费，用图和可交互实验讲清每一步为什么更快。
pageClass: gpu-page
---

# Reduce 算子优化图解

<p class="gpu-lede">在 A100 上给 1.3 亿个 float 求和，总共 512 MiB。显存带宽 2039 GB/s，理论上 263 μs 就能全部读完。按原稿记录的吞吐估算，最朴素的写法要约 1.75 ms，慢约 6.7 倍。每个元素只做一次加法，这些时间花到哪里去了？</p>

本文把同一个求和 Kernel 改写八次，从 V0 到 V7。每一版只解决一个问题，原稿记录的带宽利用率从约 15% 升到约 85%。阅读前最好先看过[GPU 架构图解](../about_gpu/gpu)，尤其是[分支如何影响车道利用率](../about_gpu/gpu#lane)。

## Reduce 是什么 {#basics}

Reduce（规约）把多个输入合并成一个输出，常见的有求和、求最大值和点积。本文以求和为例，其他操作的优化思路相同。

串行求和要一个接一个地加。并行的做法是两两配对，每轮让数据量减半。下图用 8 个数对比这两种做法。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigReduceTree /></div>
  <figcaption>树形规约的轮数是 log₂N，但加法总数和串行一样，都是 N − 1 次。并行省下的是轮数，不是加法。</figcaption>
</figure>

### 瓶颈不在加法 {#bound}

先算一笔账。每读一个 4 字节的 float，只做 1 次加法，所以算术强度只有 $1/4 = 0.25$ FLOP/byte。显存满负荷时，A100 每秒只需要做约 510 G 次加法，而它的单精度峰值算力是 19.5 TFLOPS。

<div class="gpu-chain">
  <div><b>0.25</b><span>FLOP/byte，算术强度</span></div>
  <div><b>510 G</b><span>满带宽时每秒要做的加法</span></div>
  <div><b>19.5 T</b><span>A100 单精度峰值 FLOP/s</span></div>
  <div><b>2.6%</b><span>加法需要动用的算力</span></div>
</div>

用[Roofline](../about_gpu/ai-compute#roofline)的语言说，Reduce 紧贴带宽那条斜线，是典型的访存受限任务。所以优化目标不是把加法变快，而是让 GPU 尽量多发读请求。线程每做一件和读数据无关的事，内存通道就少一份请求，利用率也就上不去。

### 一个 Block 里都在做什么 {#map}

下图是 Kernel 的完整流程，也是全文的地图。规约这一段要重复 8 轮，所以它的每个环节都值得优化。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigReduceMap /></div>
  <figcaption>示意图：Block 为 256 个线程。图中标出了每个环节由哪个版本优化。</figcaption>
</figure>

### 测试条件 {#setup}

| 项目 | 数值 |
| --- | --- |
| GPU | A100 80GB SXM4（compute capability 8.0） |
| 理论显存带宽 | 2,039 GB/s |
| SM 数量 | 108 |
| 每个 SM 的共享内存 | 最多 164 KB |
| 数据规模 | $N = 2^{27}$ 个 float，共 512 MiB |
| Block 大小 | 256 个线程，即 8 个 Warp |

<p class="gpu-note">下文标为“原稿数据”的利用率和吞吐都来自本文的原始整理稿，测量方法和完整数据未知，我们也没有重新测量。由它们换算的耗时和倍数只是带假设的估计，不等于验证过的实测结果。</p>

## V0：朴素的树形规约 {#v0}

最直接的写法是：先把数据搬进共享内存（smem），然后让步长 `step` 从 1 开始逐轮翻倍。每轮只有编号是 `2*step` 倍数的线程工作，把偏移 `step` 处的值加到自己身上。

```cpp
// V0: 朴素的树形规约（步长从小到大）
__global__ void reduce_v0(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局内存数据加载到共享内存
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 树形规约：步长从 1 开始逐步翻倍
    for (int step = 1; step < blockDim.x; step *= 2) {
        if (tid % (2 * step) == 0) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    // 每个 Block 的结果写回全局内存
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

注意结果写到了 `output[blockIdx.x]`，也就是每个 Block 只产出一个部分和。$2^{27}$ 个元素会得到 524,288 个部分和，还要对 `output` 再做规约才能得到最终结果。后面各版本都是如此。

### 问题：Warp 分化 {#divergence}

问题出在 `if (tid % (2 * step) == 0)`。同一个 Warp 的 32 条车道共用一条指令，只要有一条车道要执行 `if` 体，整组就要占用一次发射，其余车道被掩码关闭。这种情况叫 Warp 分化。

拖动滑块，看看 8 轮下来车道是怎么被浪费的：

<ReduceLab :versions="['v0']" />

第 1 轮时，每个 Warp 都只有一半车道在工作。到了第 5 轮，8 个 Warp 都还要发射，每个 Warp 却只有 1 条车道有活干。第 6 轮起开始有 Warp 被整组跳过，分化的 Warp 变少了，但没被跳过的 Warp 里依然只有 1 条车道工作。

把 8 轮加起来，`if` 体要发射 47 次 Warp，而 255 次加法如果每次发射都能占满 32 条车道，8 次就够了。原稿数据：V0 带宽利用率约 15%，约 306 GB/s。

## V1：让 Warp 整组进出 {#v1}

活跃线程散落在各个 Warp 里，是因为我们按 `tid` 的间隔来挑线程。换个映射方式：让活跃线程总是编号最小的那一批，再用 `tid` 算出它该操作的地址。

```cpp
// V1: strided index 方式，减少 Warp 分化
__global__ void reduce_v1(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 步长从 1 开始逐步翻倍，但用 strided index 映射活跃线程
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = threadIdx.x * 2 * s;
        if (index < blockDim.x) {
            smem[index] += smem[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

切换版本对比同一轮的差异：

<ReduceLab :versions="['v0', 'v1']" initial="v1" />

第 1 轮只有前 4 个 Warp 工作，而且都是整组工作，后 4 个 Warp 直接跳过。发射次数从 47 次降到 12 次。活跃线程不足 32 个时才会分化，也就是最后 5 轮，这时只涉及 Warp 0。

原稿数据：V1 带宽利用率约 32%，约 653 GB/s，吞吐约为 V0 的 2.1 倍。

## V2：反转步长，解决 Bank 冲突 {#v2}

### Shared Memory 的 Bank {#bank}

共享内存分成 32 个 Bank，每个 Bank 宽 4 字节。连续地址依次落在 Bank 0、1、…、31，然后回到 Bank 0。也就是说，地址除以 32 的余数决定了它在哪个 Bank。一个 Bank 同一时刻只能服务一个地址。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigBankConflict /></div>
  <figcaption>示意图：为了清晰，每侧只画了 4 条车道。多个线程读同一个地址不算冲突，读同一个 Bank 的不同地址才要排队。</figcaption>
</figure>

### V1 带来的新问题 {#v1-conflict}

V1 的地址是 `tid * 2 * s`，间隔越拉越大。当 $s = 2$ 时，tid 0、8、16、24 分别访问地址 0、32、64、96，它们除以 32 的余数都是 0。

V0 反而没有这个问题，因为它的每条车道都访问 `smem[tid]`，32 条车道正好对应 32 个 Bank。所以 V1 减少了分化，却引入了 V0 没有的 Bank 冲突。

### 把规约方向反过来 {#flip}

解决办法是让步长从 `blockDim.x / 2` 开始逐轮减半。对比两种顺序：

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigReduceOrder /></div>
  <figcaption>示意图：青色槽位保存结果，琥珀色槽位是被读取的操作数。步长从小到大时，结果写在分散的槽位上，需要越跳越远。步长从大到小时，结果始终写在连续的低位槽位上。</figcaption>
</figure>

```cpp
// V2: 步长从大到小，消除 Bank 冲突
__global__ void reduce_v2(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 步长从 blockDim.x/2 开始，每轮减半
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

现在每条活跃车道访问 `smem[tid]` 和 `smem[tid + s]`，同一条指令里的地址是连续的，不会落到同一个 Bank。下面的实验在线程图下方加上了地址图，默认停在 V1 的 $s = 2$：

<ReduceLab :versions="['v1', 'v2']" initial="v1" :round="2" bank />

把三个版本的 8 轮合计放在一起：

| 版本 | `if` 体发射的 Warp 数 | 其中分化 | 最大冲突 | 读取折算拍数 |
| --- | --- | --- | --- | --- |
| V0 | 47 | 47 | 无 | 47 |
| V1 | 12 | 5 | 8 路 | 47 |
| V2 | 12 | 5 | 无 | 12 |

表中有两点值得注意。第一，V1 的读取折算拍数和 V0 一样，都是 47，冲突把少发射的那部分读取又排队排了回来。原稿数据里 V1 仍然更快。一种解释是，冲突只拖慢访存指令，分支、加法这些指令的发射次数确实减少了。这个模型本身不能证明这一点。

第二，V2 和 V1 的分化次数相同。当 $s \le 16$ 时，Warp 0 里只有部分车道活跃，所以 V2 并没有彻底消除分化，它消除的是 Bank 冲突。好在这 5 次分化都只发生在 Warp 0，开销很小，而下一步的 V4 会进一步处理这几轮。

原稿数据：V2 带宽利用率约 40%，约 816 GB/s，吞吐约为 V1 的 1.25 倍。

## V3：让每个线程多干一份活 {#v3}

前面每个线程只负责 1 个元素。第 1 轮之后，后 128 个线程就再也不工作了，它们的全部贡献就是搬一次数据。不如让每个线程在加载时就读 2 个元素并求和。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigDoubleLoad /></div>
  <figcaption>示意图：线程数和规约轮数都没有变，但每个 Block 处理的数据翻倍，需要的 Block 数减半。两条黑色竖线表示同一个线程读取的两个元素，它们在各自段内的位置相同。</figcaption>
</figure>

```cpp
// V3: 每线程处理 2 个元素，减少空闲线程
__global__ void reduce_v3(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 每个线程加载 2 个相距 blockDim.x 的元素并求和
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    // 步长从大到小的规约（同 V2）
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

每个 Block 现在处理 `blockDim.x * 2` 个元素，所以启动 Kernel 时，Block 数要设为 `(n + blockDim.x * 2 - 1) / (blockDim.x * 2)`。

规约一次的开销没变，能摊到的数据却多了一倍，这是预期 V3 变快的原因。原稿数据：V3 带宽利用率约 45%，约 918 GB/s，吞吐约为 V2 的 1.12 倍。

## V4：展开最后一个 Warp {#v4}

当 $s \le 32$ 时，只有 Warp 0 还在工作，这时用 `__syncthreads()` 同步整个 Block 就太重了。经典做法是把这 6 轮展开成直线代码，不再用屏障。

```cpp
// 辅助函数：展开最后 32 个线程的规约
__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid +  8];
    smem[tid] += smem[tid +  4];
    smem[tid] += smem[tid +  2];
    smem[tid] += smem[tid +  1];
}

// V4: 展开最后一个 Warp（在 V3 基础上）
__global__ void reduce_v4(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 每线程处理 2 个元素（继承自 V3）
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    // 规约循环仅执行到 s > 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // 最后一个 Warp 内的规约，不用 __syncthreads()
    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigUnroll /></div>
  <figcaption>V4 把 s ≤ 32 的 6 轮展开，规约阶段的屏障从 8 次减到 2 次。下一节的 V5 再去掉剩下的循环判断。</figcaption>
</figure>

原稿数据：V4 带宽利用率约 52%，约 1060 GB/s，吞吐约为 V3 的 1.15 倍。

### 这段代码在 A100 上并不安全 {#its}

`warpReduce` 有一个隐含前提：32 条车道严格同步，每条语句都是所有车道先一起读、再一起写。车道 16 既要往 `smem[16]` 写，车道 0 又要从 `smem[16]` 读，只有读在写前，结果才对。

这个前提在早期 GPU 上成立。但从 compute capability 7.0（Volta）开始，GPU 引入了独立线程调度（Independent Thread Scheduling），同一 Warp 的线程不再保证逐条指令同步。A100 是 8.0，也在此列。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigLockstep /></div>
  <figcaption>两图都是语句 <code>smem[tid] += smem[tid + 16]</code>。把 smem 的前 64 个位置设为 1 到 64 做模拟，正确结果是 2080。只要车道 16 在这一步抢先，结果就变成 2113，多出来的 33 正是 smem[32] 的初值。</figcaption>
</figure>

原代码里的 `volatile` 只能阻止编译器把值缓存在寄存器里，并不能决定不同车道的先后顺序。这种写法在实测中可能一直正确，但官方文档已明确不再保证。NVIDIA 的要求是用 `__syncwarp()` 显式同步。一种安全的写法如下，它用 `tid < s` 避免写入别的车道还要读的位置：

```cpp
// 不依赖锁步：只有下一步要用到结果的车道才写入
if (tid < 32) {
    for (unsigned int s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncwarp();
    }
}
```

这样写比 `__syncthreads()` 轻，但比原来的直线代码多了判断和同步。所以在新架构上，这几轮更值得直接换成下面 V6 的 Warp Shuffle。

## V5：把循环交给编译器 {#v5}

V4 前面的 $s = 128$ 和 $s = 64$ 两轮仍在运行时循环，每次都要判断边界、跳转、更新循环变量。如果 Block 大小在编译期已知，这些开销都可以省掉。办法是把它做成模板参数：

```cpp
// V5: 完全循环展开（模板参数）
template <int BLOCK_SIZE>
__global__ void reduce_v5(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

    // 每线程处理 2 个元素
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + BLOCK_SIZE < n) val += input[gid + BLOCK_SIZE];
    smem[tid] = val;
    __syncthreads();

    // 编译期展开：BLOCK_SIZE 已知，不满足的分支会被编译器直接删除
    if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads(); }

    // 最后 Warp 内展开（同样依赖锁步，见上一节）
    if (tid < 32) {
        volatile float* vsmem = smem;
        if (BLOCK_SIZE >= 64) vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) output[blockIdx.x] = smem[0];
}
```

调用时按 Block 大小选择实例：

```cpp
switch (block_size) {
    case 512: reduce_v5<512><<<grid, 512, 512*sizeof(float)>>>(d_in, d_out, n); break;
    case 256: reduce_v5<256><<<grid, 256, 256*sizeof(float)>>>(d_in, d_out, n); break;
    case 128: reduce_v5<128><<<grid, 128, 128*sizeof(float)>>>(d_in, d_out, n); break;
}
```

以 `BLOCK_SIZE = 256` 为例，`BLOCK_SIZE >= 512` 在编译期为假，整段都不生成指令。循环没有了，但 `tid < 128` 这样的线程判断还在，因为它取决于运行时的线程编号。

原稿数据：V5 带宽利用率约 62%，约 1265 GB/s，吞吐约为 V4 的 1.19 倍。

## V6：用 Warp Shuffle 绕开共享内存 {#v6}

前面每一版都在优化共享内存的访问。如果 Warp 内部根本不用它呢？`__shfl_down_sync` 能让车道直接读取另一条车道的寄存器：

```cpp
// 每条车道读取 lane_id + delta 号车道的 var
float val = __shfl_down_sync(0xffffffff, var, delta);
```

第一个参数是掩码，`0xffffffff` 表示 32 条车道全部参与。它不用计算地址，也没有 Bank 冲突。数据不经过共享内存，V4 的读写竞争也就不存在了。`_sync` 后缀表示调用方用掩码指明参与的车道，这些车道会在指令处汇合。但它[不是内存屏障](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions)，不保证内存顺序，共享内存的读写仍然要靠 `__syncthreads()` 这样的同步。

下图用 8 条车道演示，实际的 32 条车道只是多两轮。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigShuffle /></div>
  <figcaption>图中数值是真实的计算结果。青色车道的值会继续流向车道 0。灰色车道的值不再被用到，其中源车道越界时，车道读到的是自己的值，例如第一轮车道 4 的 5 + 5 = 10。</figcaption>
</figure>

### 两级规约 {#two-level}

Shuffle 只能在一个 Warp 内进行，所以 256 个线程要分两级：先在每个 Warp 内规约，再通过共享内存把 8 个结果交给 Warp 0，由它做第二次规约。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigTwoLevel /></div>
  <figcaption>示意图：共享内存只用了 8 个位置，而且每个 Warp 只写一次，只有这一步需要 <code>__syncthreads()</code>。</figcaption>
</figure>

```cpp
// Warp 内规约辅助函数
__device__ float warpReduceSum(float val) {
    // 每次将右半边的值加到左半边
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // lane 0 持有最终结果
}

// V6: Warp Shuffle + 两级规约
__global__ void reduce_v6(float* input, float* output, int n) {
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32;      // 线程在 Warp 内的编号（0~31）
    int wid  = tid / 32;      // 该线程属于哪个 Warp

    // 每线程处理 2 个元素
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];

    // 第一级：Warp 内规约
    val = warpReduceSum(val);

    // 将每个 Warp 的结果（仅 lane 0 有效）存入 Shared Memory
    __shared__ float warp_results[32];  // 最多 32 个 Warp（1024/32）
    if (lane == 0) {
        warp_results[wid] = val;
    }
    __syncthreads();

    // 第二级：Warp 间规约（用 Warp 0 处理）
    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    if (tid == 0) output[blockIdx.x] = val;
}
```

原稿数据：V6 带宽利用率约 72%，约 1468 GB/s，吞吐约为 V5 的 1.16 倍。

## V7：向量化加载与 Grid Stride Loop {#v7}

到 V6 为止，规约已经很轻了，但还有一个结构性的浪费：每处理 512 个元素，就要启动一个 Block 并规约一次。V7 同时改变了读法和结构。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigStride /></div>
  <figcaption>示意图：上方琥珀色格是 Block 1 中 2 号线程处理的元素，青色格属于 Block 1 的其他线程。线程走完数组后才规约，因此整个任务只规约 Grid 大小那么多次。</figcaption>
</figure>

**向量化加载**：把 `float` 换成 `float4`，同样一条加载指令取回 4 倍的数据。指令少了，访存流水线更容易被填满。使用时要保证地址按 16 字节对齐，`cudaMalloc` 返回的内存可以满足。

**Grid Stride Loop**：固定 Grid 大小，让每个线程循环处理多段数据，每次跨过 `gridDim.x * blockDim.x` 个元素。

按 A100 算一下：Grid 设为 SM 数的 4 倍，即 432 个 Block，共 110,592 个线程。$2^{25}$ 个 float4 平摊下来，每个线程循环约 304 次才规约一次。V3 到 V6 需要 262,144 个 Block，这里只要 432 个，规约的次数约为原来的 1/607。

```cpp
// V7: float4 向量化加载 + Grid Stride Loop + Warp Shuffle
__global__ void reduce_v7(float* input, float* output, int n) {
    int tid  = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    // float4 加载：每线程每次处理 4 个 float
    float4* input4 = reinterpret_cast<float4*>(input);
    int n4 = n / 4;  // float4 的元素数量

    float val = 0.0f;

    // Grid Stride Loop：每个线程以 gridDim.x * blockDim.x 为步长迭代
    for (int idx = blockIdx.x * blockDim.x + tid;
         idx < n4;
         idx += gridDim.x * blockDim.x)
    {
        float4 data = input4[idx];
        val += data.x + data.y + data.z + data.w;
    }

    // 处理 n 不是 4 的倍数时的尾部元素
    int tail_start = n4 * 4;
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid;
         idx < n;
         idx += gridDim.x * blockDim.x)
    {
        val += input[idx];
    }

    // Warp 内规约
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    __shared__ float warp_results[32];
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    if (tid == 0) output[blockIdx.x] = val;
}
```

调用方式：

```cpp
// 固定 Grid 大小为 SM 数 × 4
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
int grid_size  = num_sms * 4;   // 432 for A100
int block_size = 256;

reduce_v7<<<grid_size, block_size>>>(d_input, d_partial, n);
```

运行结束后，`d_partial` 里还有 432 个部分和，再求一次和就能得到最终结果。

原稿数据：V7 带宽利用率约 85%，约 1733 GB/s，吞吐约为 V6 的 1.18 倍。

## 八个版本放在一起 {#summary}

<GpuBars
  label="V0 到 V7 的原稿显存吞吐数据"
  unit="GB/s"
  :ticks="[0, 500, 1000, 1500, 2000]"
  :rows="[
    { label: 'V0 朴素树形', value: 306, note: '利用率约 15%，估算读完 512 MiB 约 1754 μs' },
    { label: 'V1 strided index', value: 653, note: '利用率约 32%，估算约 822 μs' },
    { label: 'V2 步长反转', value: 816, note: '利用率约 40%，估算约 658 μs' },
    { label: 'V3 每线程两元素', value: 918, note: '利用率约 45%，估算约 585 μs' },
    { label: 'V4 展开末 Warp', value: 1060, note: '利用率约 52%，估算约 506 μs' },
    { label: 'V5 模板展开', value: 1265, note: '利用率约 62%，估算约 424 μs' },
    { label: 'V6 Warp Shuffle', value: 1468, note: '利用率约 72%，估算约 366 μs' },
    { label: 'V7 float4 + Stride', value: 1733, note: '利用率约 85%，估算约 310 μs' }
  ]"
/>

<p class="gpu-note">吞吐和利用率来自本文的原始整理稿，没有重新测量，也没有说明对部分和的二次规约是否计入。耗时是由 512 MiB 除以吞吐算出的估计，理论峰值为 2039 GB/s，对应 263 μs。各版本的增益在不同 GPU 和 Block 大小下会有差异。</p>

每一版对应一个具体的瓶颈：

| 瓶颈 | 表现 | 解决版本 | 做法 |
| --- | --- | --- | --- |
| Warp 分化 | 车道被掩码关闭，发射次数多 | V1 | 让活跃线程集中在低编号 |
| Bank 冲突 | 同一条读指令要排队 | V2 | 反转步长，地址保持连续 |
| 空闲线程 | 线程搬完数据就闲置 | V3 | 加载时每线程读两个元素 |
| 多余同步 | 单 Warp 也用全 Block 屏障 | V4 | 展开最后 6 轮 |
| 循环开销 | 判断和跳转 | V5 | 模板参数，编译期展开 |
| 共享内存访问 | 读写都要经过 smem | V6 | Warp Shuffle 走寄存器 |
| 规约过于频繁 | 每 512 个元素规约一次 | V7 | float4 与 Grid Stride Loop |

### 实际该选哪个 {#choose}

| 场景 | 建议 |
| --- | --- |
| 学习原理 | V2，逻辑清晰，能看到 Warp 分化和 Bank 冲突两个问题 |
| 自己写 Kernel | V6 或 V7，用 `__shfl_down_sync`，不依赖锁步 |
| 数组特别大 | V7，Grid 大小固定，不受 Block 数量影响 |
| 生产环境 | 优先用 [CUB](https://github.com/NVIDIA/cccl/tree/main/cub) 的 `cub::DeviceReduce`，由官方针对各代 GPU 维护 |

如果要在自己的卡上验证，可以用 Nsight Compute 查看显存吞吐，并关注共享内存的 Bank 冲突计数。这些思路也不限于 Reduce，大部分访存受限的 Kernel 都能用上。

## 检验一下 {#check}

::: details 思考：Block 改成 512 个线程，V1 和 V2 各有几轮分化？
活跃线程数不足 32 时才可能分化。V1 每轮活跃 $512 / (2s)$ 个线程，$s = 16, 32, 64, 128, 256$ 时不足 32，共 5 轮。V2 每轮活跃 $s$ 个线程，$s = 16, 8, 4, 2, 1$ 时不足 32，也是 5 轮。

Block 变大只会增加整组工作的轮数，分化轮数不变，因为分化只发生在最后一个 Warp 内部。
:::

::: details 思考：把求和改成求最大值，代码要改哪些地方？
除了把 `+=` 换成 `max`，还要改填充值。代码里越界线程填的是 `0.0f`，它是加法的单位元，却不是 `max` 的单位元。如果所有输入都是负数，结果会错误地变成 0。应该改用 `-INFINITY`，V6、V7 第二级规约里的 `0.0f` 也一样。
:::

::: details 思考：Shuffle 第一轮，高编号车道读到了自己的值，相当于翻倍。为什么不影响结果？
因为这些值再也不会流向车道 0。以 32 条车道为例，第一轮后只有车道 0~15 的值还有用。第二轮只有车道 0~7 的结果有用，它们读取的车道 8~15 都是有效值。车道 16 及以上的值虽然变了，但不在通往车道 0 的路径上。对应上文的 8 车道图，也就是灰色格子。
:::

::: details 思考：V1 的 Bank 冲突最多是 8 路，为什么不是 32 路？
冲突路数取决于同一 Warp 中有多少线程落在同一个 Bank。$s = 4$ 时地址间隔是 8，32 个线程平均落在 4 个 Bank 上，每个 8 路。$s = 8$ 时间隔是 16，但只有 16 个线程活跃，落在 2 个 Bank 上，还是 8 路。$s = 16$ 时间隔是 32，全落在 Bank 0，但只有 8 个线程活跃。步长再变大，活跃线程继续减半，路数也随之下降。
:::

<p class="gpu-note">本文各版本的完整代码在仓库的 <code>code/reduce/reduce.cu</code>，每个版本都会校验输出。运行方法见 <code>code/README.md</code>。</p>

<p class="gpu-note">本文以 <a href="https://github.com/caomaolufei/AIInfraGuide" target="_blank" rel="noreferrer">AI Infra Guide</a> 的<a href="https://github.com/caomaolufei/AIInfraGuide/blob/main/docs/guides/%E6%A8%A1%E5%9D%97%E4%BA%8C-CUDA%E7%BC%96%E7%A8%8B%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/3.1-CUDA%20Reduce%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96.md" target="_blank" rel="noreferrer">《3.1 CUDA Reduce 算子优化》</a>为蓝本重写，文中的“原稿”和“原始整理稿”都指这篇文章。感谢原作者的整理。</p>

## 参考资料 {#refs}

- [Optimizing Parallel Reduction in CUDA - Mark Harris, NVIDIA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [CUDA Programming Guide - Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)：独立线程调度和 warp-synchronous 代码
- [CUDA Programming Guide - C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)：`__syncwarp` 和 Warp Shuffle 函数
- [Volta Tuning Guide](https://docs.nvidia.com/cuda/archive/11.4.3/volta-tuning-guide/index.html)
- [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [深入浅出 GPU 优化系列：reduce 优化](https://zhuanlan.zhihu.com/p/426978026)
- [CUDA 性能优化篇：reduce 算子](https://zhuanlan.zhihu.com/p/17996548596)
- [BBuf 的 CUDA 笔记三：reduce 优化入门学习笔记](https://zhuanlan.zhihu.com/p/596012674)
- [NVIDIA CUB Library](https://github.com/NVIDIA/cccl/tree/main/cub)

<p class="gpu-note">本页自绘的结构图为教学简化。Bank 划分按默认的 32 位模式讨论，不同代际的细节有所不同。</p>
