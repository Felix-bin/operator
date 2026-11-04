---
title: Softmax 优化图解
description: 先用减最大值解决数值溢出，再逐版消除线程不足、访存分散、同步过多和重复读取，用图和可交互实验讲清每一步为什么更快。
pageClass: gpu-page
---

# Softmax 优化图解

<p class="gpu-lede">在 A100 上逐行计算 4096×4096 矩阵的 Softmax。朴素公式只要有一个分数超过 88.72，这一行就会出现 NaN。修好数值问题之后，256 MiB 的读写流量在 2039 GB/s 下理论上只要 132 μs，按原稿数据估算，最直接的写法要约 2.6 ms，慢约 20 倍。公式只有一行，这些问题都出在哪里？</p>

本文先解决“算得对”，再解决“算得快”。同一个 Kernel 改写五次，从 V0 到 V4，每一版对应一个具体瓶颈。规约的基础技巧在 [Reduce 算子优化图解](../reduce/reduce) 中已经讲过，这里只讲 Softmax 带来的不同之处。下一篇 [Online Softmax 图解](./online-softmax) 会从算法上继续减少读取。

## 公式和它的数值陷阱 {#basics}

Softmax 把一组任意实数变成一个概率分布：每个分数越大占比越高，所有占比之和为 1。

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=0}^{N-1} e^{x_j}}
$$

在 Self-Attention 中，它作用于 $QK^T$ 矩阵的每一行，把注意力分数变成归一化权重。每行长度 $N$ 就是序列长度，常见的有 2048、4096，甚至更长。

### 指数能装下多大的数 {#range}

问题出在 $e^{x}$。指数函数增长极快，浮点格式能表示的 $x$ 范围其实很窄：

| 格式 | 最大有限值 | $e^x$ 变成 inf | 开始变成次正规数 | $e^x$ 舍入成 0 |
| --- | --- | --- | --- | --- |
| FP32 | $3.40 \times 10^{38}$ | $x > 88.72$ | $x < -87.34$ | $x < -103.97$ |
| FP16 | 65504 | $x > 11.09$ | $x < -9.70$ | $x < -17.33$ |

超界之后，结果也不只是一个 inf。朴素公式有两种出错方式，触发条件不同：

- **上溢**：只要一项超界，分母就是 inf。超界的那一项是 inf ÷ inf，得到 NaN；其余项是有限值除以 inf，全变成 0。
- **下溢**：必须所有项都舍入成 0，分母才是 0，这时 0 ÷ 0 全部是 NaN。只要有一项还没下溢，分母就不是 0。比如 FP32 中的 $e^{-100} \approx 3.8 \times 10^{-44}$ 还是一个次正规数，不等于 0，只是精度很差。

### 减最大值 {#safe}

给每个 $x_i$ 减去同一个常数 $c$，结果不变，因为 $e^{-c}$ 在分子和分母中相消：

$$
\frac{e^{x_i - c}}{\sum_{j} e^{x_j - c}} = \frac{e^{x_i} \cdot e^{-c}}{\sum_{j} e^{x_j} \cdot e^{-c}} = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

取 $c = m = \max_j x_j$，就得到 Safe Softmax。它带来两个保证：

1. 所有指数都不大于 0，每一项都落在 $[0, 1]$ 内，$e^{x}$ 不再上溢。
2. 最大的那一项恰好是 $e^0 = 1$，所以分母不小于 1。**减最大值之后，下溢不再能产生 NaN。** 差距过大的项会舍入成 0，这也没有问题，因为它们的真实占比本来就小到无法表示。

改变下面的分数，看 NaN 何时出现。数轴的上一行是朴素公式的指数，下一行是减去最大值之后的指数：

<SoftmaxLab />

试试“整行掩码”：两种写法都是 NaN，这是 Safe Softmax 的例外。Attention 常用 -inf 屏蔽位置，部分屏蔽时一切正常，因为 $-\infty - m = -\infty$，指数为 0。但如果一整行都是 -inf，那么 $m = -\infty$，而 $-\infty - (-\infty)$ 是 NaN。输入含 +inf 或 NaN 时同样如此。所以减最大值保证的是有限输入下的结果，这几种情况要在上层处理。

分母的上溢也要留意。分母最多等于 $N$，这对 FP32 不成问题。FP16 的最大值只有 65504，树形规约时，$N = 65536$ 个相等的输入会让分母变成 inf，结果全部是 0。按顺序累加还有另一个问题：FP16 的和到 2048 之后，再加 1 就会被舍入掉。这个和不再增长，后面的项都丢了。下面的代码都用 FP32。

## 三遍扫描：瓶颈在哪 {#passes}

Safe Softmax 自然分成三遍：先求最大值，再用它求分母，最后归一化。每一遍都依赖上一遍的结果，也都要读一次整行。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigSoftmaxPasses /></div>
  <figcaption>m 和 d 是每行一个的标量，它们不占全局内存流量。第 3 遍除了 d 也用到 m，图中省略了这条线。</figcaption>
</figure>

这里要注意写入。每个元素读 3 次、写 1 次，每次 4 字节，共 16 字节流量。按[Roofline](../about_gpu/ai-compute#roofline)的思路算一笔账：

<div class="gpu-chain">
  <div><b>16</b><span>每个元素的流量，字节</span></div>
  <div><b>127 G</b><span>满带宽时每秒处理的元素</span></div>
  <div><b>约 77</b><span>算力允许每个元素用的指令槽</span></div>
  <div><b>2 次</b><span>每个元素要算的 expf</span></div>
</div>

A100 单精度峰值 19.5 TFLOPS 把一次 FMA 算作 2 个 FLOP，相当于每秒约 9.75 T 个指令槽。除以每秒 127 G 个元素，每个元素约有 77 个。这是理想上限，地址计算、分支和规约也要占用。`expf` 也不能当普通 FLOP 算：用 nvcc 按 sm_80 编译，一次 `expf` 约 10 条指令，其中一条 `MUFU.EX2` 走 SFU。

所以这笔账只说明：如果每次访问都到达显存，算力看起来还有余量，Softmax 在这种规模和实现下可能先受带宽限制。它不能证明这一点。行不长时，重复读取可能命中缓存，到达显存的流量没有 16 字节，[下一篇的实测](./online-softmax#cache)就是这样；规约开销大时，瓶颈也可能不在带宽。要确认，只能在目标 GPU 上测量。下文的优化方向有两个：让 GPU 尽量发满读写请求，以及减少要搬的字节。它们的收益都取决于瓶颈确实在这一侧。

### 测试条件 {#setup}

| 项目 | 数值 |
| --- | --- |
| GPU | A100 80GB SXM4（compute capability 8.0） |
| 理论显存带宽 | 2,039 GB/s |
| SM 数量 | 108 |
| 每个 SM 的共享内存 | 最多 164 KB，每个 Block 最多 163 KB |
| 矩阵 | $(M, N) = (4096, 4096)$，一遍 64 MiB |
| Block 大小 | 256 个线程，即 8 个 Warp（V1 起） |

<p class="gpu-note">下文标为“原稿数据”的利用率和吞吐都来自本文的原始整理稿，测量方法和完整数据未知，我们也没有重新测量。由它们换算的耗时和倍数只是带假设的估计，不等于验证过的实测结果。</p>

## V0：一个线程一行 {#v0}

最直接的并行方式：每个线程独立处理一行，行与行之间互不相干。

```cpp
// V0: 朴素实现，一个线程处理一行
__global__ void softmax_v0(float* input, float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float* x = input  + row * N;
    float* y = output + row * N;

    // Pass 1: 求最大值
    float max_val = -INFINITY;
    for (int i = 0; i < N; i++) {
        max_val = fmaxf(max_val, x[i]);
    }

    // Pass 2: 指数求和
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += expf(x[i] - max_val);
    }

    // Pass 3: 归一化
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; i++) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}
```

结果是对的，但很慢，原因有三个。

**线程太少。** 4096 行只对应 4096 个线程，也就是 128 个 Warp，平摊到 108 个 SM 上，每个 SM 只有约 1.2 个 Warp。[有效带宽受在途请求数限制](../about_gpu/memory-interconnect#little)，这么少的线程发不出足够的请求。

**地址分散。** 同一个 Warp 的 32 个线程分别处理 32 行，同一时刻读的地址相差一整行，也就是 $N \times 4 = 16$ KiB。全局内存以 32 字节的事务访问，每条车道只能各发一次。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigSoftmaxAccess /></div>
  <figcaption>示意图：只画出 8 条车道和 16 列，数字是车道号。琥珀色格是一条读指令真正要的元素，粗框是对应的 32 字节事务。V0 框内其余 7 个元素要到后面几轮循环才会用到，能否省下事务要看缓存。事务数按官方文档的 32 字节粒度算出。</figcaption>
</figure>

**重复计算。** `expf(x[i] - max_val)` 在 Pass 2 和 Pass 3 中各算一次，值完全相同。这一点要等到 V4 才解决。

原稿数据：V0 带宽利用率约 5%，约 102 GB/s。 原稿没有给出 V0 的 Block 大小，所以上面的 Warp 数只说明线程总量，不代表具体的调度方式。

## V1：一个 Block 一行 {#v1}

让一个 Block 的线程协作处理同一行，每个线程负责其中一段。这样一来，线程总数从 4096 变成 $4096 \times 256$，排队等待的 Block 也足够把每个 SM 填满。每个线程只需处理 $4096 / 256 = 16$ 个元素。

相邻线程读相邻元素，32 条车道正好合成 4 次事务，上图右侧就是这种情况。这也是官方文档说的完美合并访问。

代价是 max 和 sum 分散在各个线程手里，要做两次规约。于是 Softmax 被拆成“max 规约 + sum 规约 + 逐元素归一化”：

```cpp
// V1: 一个 Block 处理一行，Block 内并行规约
__global__ void softmax_v1(float* input, float* output, int M, int N) {
    extern __shared__ float smem[];  // 用于规约

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input  + row * N;
    float* y = output + row * N;

    // Pass 1: 并行求最大值
    float max_val = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    smem[tid] = max_val;
    __syncthreads();

    // Shared Memory 规约求全局最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Pass 2: 并行求指数和
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += expf(x[i] - max_val);
    }
    smem[tid] = sum;
    __syncthreads();

    // Shared Memory 规约求总和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    sum = smem[0];
    __syncthreads();

    // Pass 3: 归一化写出
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}
```

启动时用第三个参数分配 `smem`：

```cpp
int block_size = 256;
int smem_size = block_size * sizeof(float);
softmax_v1<<<M, block_size, smem_size>>>(d_input, d_output, M, N);
```

步长从大到小的顺序就是 [Reduce V2 的写法](../reduce/reduce#flip)，每个 Warp 的 `smem[tid]` 和 `smem[tid + s]` 地址都连续，所以这里没有 Bank 冲突。另外，`blockDim.x` 必须是 2 的幂。用脚本模拟 96 个线程，这种折半方式会漏掉部分线程的值，结果不对。

原稿数据：V1 带宽利用率约 35%，约 714 GB/s，吞吐约为 V0 的 7 倍。

## V2：两级规约 {#v2}

### 规约为什么在这里更贵 {#reduce-cost}

Reduce 可以用 [Grid Stride Loop](../reduce/reduce#v7) 把规约次数压到很少，Softmax 却不行：每一行都要有自己的 $m$ 和 $d$，有多少行就至少要规约多少次，而且每行两次。一行只有 4096 个元素时，规约的开销摊不薄。

V1 每次规约要写一次初值、走 8 轮、再读一次结果，每一步都有屏障。两次规约合计 20 次 `__syncthreads()`，这些都和读数据无关。

### 改成 Warp 内 + Warp 间 {#two-level}

用 [Warp Shuffle](../reduce/reduce#v6) 在寄存器之间交换数据，先在每个 Warp 内规约，再用共享内存跨 Warp。和 Reduce 有一个不同：Reduce 只要线程 0 拿到结果，而这里所有线程都要用 $m$ 和 $d$，所以最后还要广播一次。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigSoftmaxTwoLevel /></div>
  <figcaption>示意图：以求最大值为例。所有线程都会读 warp_max，但只有 Warp 0 的结果继续往下走。虚线是屏障：一次保证 warp_max 写完才读，另一次保证 block_result 写完才读。</figcaption>
</figure>

```cpp
// Warp 内规约：Max
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp 内规约：Sum
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 两级规约：Block 内 Max
__device__ float blockReduceMaxShuffle(float val) {
    __shared__ float warp_max[32];

    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val = warpReduceMax(val);

    if (lane == 0) warp_max[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    val = (lane < num_warps) ? warp_max[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);

    // 广播结果给所有线程
    __shared__ float block_result;
    if (threadIdx.x == 0) block_result = val;
    __syncthreads();
    return block_result;
}

// 两级规约：Block 内 Sum
__device__ float blockReduceSumShuffle(float val) {
    __shared__ float warp_sum[32];

    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0) warp_sum[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    val = (lane < num_warps) ? warp_sum[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);

    __shared__ float block_result;
    if (threadIdx.x == 0) block_result = val;
    __syncthreads();
    return block_result;
}

// V2: Warp Shuffle 两级规约
__global__ void softmax_v2(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input  + row * N;
    float* y = output + row * N;

    // Pass 1: 求行最大值
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float max_val = blockReduceMaxShuffle(local_max);

    // Pass 2: 求指数和
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }
    float sum = blockReduceSumShuffle(local_sum);

    // Pass 3: 归一化
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}
```

`warp_max` 和 `block_result` 是静态大小的共享内存，由编译器统计，所以 V2 不再需要启动参数：`softmax_v2<<<M, 256>>>(d_input, d_output, M, N)`。掩码 `0xffffffff` 要求 Warp 的 32 条车道都参与，所以 `blockDim.x` 必须是 32 的倍数，但不必是 2 的幂。

### 省下了什么 {#v2-savings}

下表是 Block 为 256 时，处理一行所需的规约开销，由脚本按代码逐条计数。

| 每行开销 | V1 | V2 |
| --- | --- | --- |
| `__syncthreads()` 次数 | 20 | 4 |
| 共享内存访问（两次规约，线程级） | 2554 | 658 |
| Shuffle 调用（两次规约，Warp 级） | 0 | 90 |

V2 把 smem 访问次数降到了约 1/4，屏障降到了 1/5。代价是多了一些 Shuffle，它们走寄存器，不用计算地址，也不和其他访存争抢。

这里要澄清两个容易混淆的地方：

- **`_sync` 不等于天然同步。** `_sync` 后缀的语义是调用方用 `mask` 指明参与的车道，让它们在硬件执行这条指令前[正确汇合](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-sync-intrinsic-constraints)。这是这个函数提供的汇合，不是 Warp 的固有性质，用错掩码还是未定义行为。它也[不是内存屏障](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions)，不保证内存顺序。
- **V2 也没有摆脱 `__syncthreads()`。** Shuffle 只能在一个 Warp 内传值，跨 Warp 还是要经过共享内存，所以每次规约仍有 2 次屏障。

原稿数据：V2 带宽利用率约 52%，约 1060 GB/s，吞吐约为 V1 的 1.5 倍。

## V3：float4 向量化访存 {#v3}

规约变轻以后，瓶颈回到了全局内存。每个线程每轮循环只读 4 字节，循环变量、地址计算和分支这些开销都要摊到这 4 字节上。把 `float` 换成 `float4`，同样一条读指令带回 16 字节：

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigSoftmaxFloat4 /></div>
  <figcaption>示意图：字节数和事务数按相邻线程读相邻元素算出。流量没有变，变的是每个字节要付出的指令开销。</figcaption>
</figure>

循环从每线程 16 次减到 4 次。事务大小还是 32 字节，但每条在途的读指令能带回的字节变成了 4 倍。如果同时在途的读指令数受限，这就等于多了在途字节，对应 [Little 定律](../about_gpu/memory-interconnect#little) 里的并发项。

```cpp
// V3: float4 向量化加载
__global__ void softmax_v3(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input  + row * N;
    float* y = output + row * N;

    // 向量化指针
    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int N4 = N / 4;  // float4 元素数

    // Pass 1: 向量化求最大值
    float local_max = -INFINITY;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(data.x, data.y),
                                            fmaxf(data.z, data.w)));
    }
    // 处理尾部元素（N 不是 4 的倍数时）
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float max_val = blockReduceMaxShuffle(local_max);

    // Pass 2: 向量化求指数和
    float local_sum = 0.0f;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        local_sum += expf(data.x - max_val) + expf(data.y - max_val)
                   + expf(data.z - max_val) + expf(data.w - max_val);
    }
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }
    float sum = blockReduceSumShuffle(local_sum);

    // Pass 3: 向量化归一化
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        float4 result;
        result.x = expf(data.x - max_val) * inv_sum;
        result.y = expf(data.y - max_val) * inv_sum;
        result.z = expf(data.z - max_val) * inv_sum;
        result.w = expf(data.w - max_val) * inv_sum;
        y4[i] = result;
    }
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}
```

原稿数据：V3 带宽利用率约 65%，约 1325 GB/s，吞吐约为 V2 的 1.25 倍。

### 尾部循环解决不了对齐 {#align}

`float4` 要求[16 字节对齐](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)。`cudaMalloc` 的返回值满足这一点，但这只保证了第一行。每行的起点是 `input + row * N`，字节偏移是 $4 \cdot row \cdot N$。只有 $N$ 是 4 的倍数时，每一行才都对齐：

| $N$ | 行 0 到行 4 的偏移（除以 16 的余数，字节） |
| --- | --- |
| 4096 | 0、0、0、0、0 |
| 4098 | 0、8、0、8、0 |
| 4097 | 0、4、8、12、0 |

代码里的尾部循环只处理行尾不足 4 个的元素，它无法改变行首的地址。

这种读取不能用，但具体表现要看环境。CUDA 12.4 的文档写的是[读取未自然对齐的 16 字节数据会得到错误的结果](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#device-memory-accesses)。我们在 RTX 4060 Laptop（compute capability 8.9）上直接运行这段 V3，$N = 4096$ 正常，$N = 4094$、$4097$、$4098$ 时 Kernel 都报错 `misaligned address`，并不是默默算错。这只是这张卡的结果，A100 上没有验证。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigSoftmaxAlign /></div>
  <figcaption>示意图：每格是一个 float，竖线和上方数字标出 16 字节边界，粗框是行 1 第一次 float4 读取的范围，琥珀色是行 1，灰色是行 0 和行 2，虚线格是填充。为了画得下，这里用 N = 6、8 代替真实大小。</figcaption>
</figure>

修法有两种。第一种不改数据布局：$N$ 不是 4 的倍数时不走 `float4`，让所有元素都走原来的标量循环。这一行改动在 RTX 4060 上跑过，上面三个 $N$ 的结果都正确。V4 第 1 段的 `x4` 也要这样改。

```cpp
// N 不是 4 的倍数时 N4 = 0，float4 循环不执行，全部交给标量循环
int N4 = (N % 4 == 0) ? N / 4 : 0;
```

第二种保留 `float4`：让每行占用的位置（pitch）是 4 的倍数，行首就都对齐了。它需要输入和输出都按这个布局存放，这段没有实际运行过：

```cpp
// 行距取 4 的倍数，行首才能都对齐
int pitch = (N + 3) & ~3;
float* x = input  + row * pitch;
float* y = output + row * pitch;
```

## V4：省掉第三次读取 {#v4}

### 能省的只有最后一次 {#v4-idea}

V1 到 V3 把输入读了三遍。第 1 遍求 $m$，第 2 遍要用 $m$ 算指数，这两遍是先后依赖的。第 3 遍用到的 $e^{x_i - m}$，第 2 遍其实已经算过一次。只是当时还不知道 $d$，没法立刻归一化：$d$ 要等整行都累加完才确定。

要去掉第三次读取，就必须把这 $N$ 个值存下来。全局内存太慢，于是放进共享内存：

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigSoftmaxSmem /></div>
  <figcaption>V4 全局内存读两遍，但代码上还是三段循环。第三段从 smem 读回指数值，不再读 input，也不再算 expf。流量按 64 MiB 一遍算出。</figcaption>
</figure>

```cpp
// V4: 两遍 Kernel（Pass 1 求 max，Pass 2 融合 exp-sum + normalize）
// 需要额外 Shared Memory 暂存 exp 值
__global__ void softmax_v4(float* input, float* output, int M, int N) {
    extern __shared__ float smem[];  // 大小 = N 个 float（暂存 exp 值）

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input  + row * N;
    float* y = output + row * N;

    float4* x4 = reinterpret_cast<float4*>(x);
    int N4 = N / 4;

    // Pass 1: 求行最大值（向量化 + Warp Shuffle）
    float local_max = -INFINITY;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(data.x, data.y),
                                            fmaxf(data.z, data.w)));
    }
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float max_val = blockReduceMaxShuffle(local_max);

    // Pass 2: 计算 exp 并暂存，同时累加求 sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float exp_val = expf(x[i] - max_val);
        smem[i] = exp_val;  // 暂存到 Shared Memory
        local_sum += exp_val;
    }
    float sum = blockReduceSumShuffle(local_sum);

    // 直接从 Shared Memory 读取 exp 值并归一化写出
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = smem[i] * inv_sum;
    }
}
```

这段代码有三点值得确认：

- **下标不越界。** $i$ 最大是 $N - 1$，所以 $N = 4096$、Block 为 256 时下标到 4095，正好在 `smem` 的 $N$ 个 float 之内。脚本遍历不同的 $N$ 和 Block，最大下标都是 $N - 1$。但如果照抄 V1 的 `smem_size = block_size * sizeof(float)`，`smem` 只有 256 个 float，这里就越界了。在 RTX 4060 上这样启动，会报错 `an illegal memory access was encountered`。
- **没有 Bank 冲突。** 同一个 Warp 的 32 条车道访问的 $i$ 连续，正好落在 32 个不同的 Bank 上。
- **第 2 段还是标量读取。** 原稿只把第 1 段写成了 `float4`，第 2 段的 `x[i]` 和第 3 段的 `y[i]` 都是逐个 float 访问，这两段还有继续向量化的空间。

### 启动参数怎么算 {#v4-launch}

V4 同时用了两种共享内存：`extern __shared__ float smem[]` 的大小在启动时决定，`warp_max`、`warp_sum` 和 `block_result` 是编译时确定的静态变量。按[官方定义](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#shared-memory)，启动参数只表示“除静态部分之外”额外分配的字节，静态部分由编译器统计，不用算进去。

```cpp
size_t smem_bytes = N * sizeof(float);  // N = 4096 时是 16384 字节

// 静态和动态加起来超过 48 KB 时必须声明，否则启动失败
cudaFuncSetAttribute(softmax_v4, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
softmax_v4<<<M, 256, smem_bytes>>>(d_input, d_output, M, N);
```

但容量账要把两部分都算上。静态变量是 $2 \times (32 \times 4 + 4) = 264$ 字节。编译出的 PTX 里，`smem` 按 16 字节对齐，所以它从第 272 字节开始，$N = 4096$ 时每个 Block 需要 16,656 字节。这 272 字节也能实测出来：在 RTX 4060 上不声明时，动态部分 48,880 字节能启动，48,884 字节就报 `invalid argument`，而 $48880 + 272$ 正好是 48 KB。

### 容量上限 {#v4-capacity}

A100 有三个数字都要看：单个 Block 超过 48 KB 就要[显式声明](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#configuring-l1-shared-memory-balance)，单个 Block [最多 163 KB](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)，每个 SM 最多 164 KB。

<SoftmaxCapacityLab />

算下来，不声明时 $N$ 最大约 12,220，声明之后最大约 41,660。$N = 4096$ 和 $N = 32768$ 都放得下，所以“N 超过 4096 就不够”并不是容量限制。

但放得下不等于跑得快。这个实验真正要看的是驻留的 Block 数：[调度器不会把共享内存超额的 Block 放到 SM 上](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#kernel-launch-and-occupancy)。按实验的估算，A100 上 $N = 4096$ 时每个 SM 最多放 8 个 Block：共享内存够放 9 个，但 [2048 个线程的上限](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)只够 8 个。$N = 32768$ 时只能放 1 个，也就是 256 个线程，在途请求会大幅减少。这些都是上限：2048 个线程平摊 64K 个寄存器，每个线程只有 32 个，Kernel 用得更多时实际驻留还会更少。RTX 4060 的共享内存只有 100 KB，影响来得更早：占用率查询显示，V3 每个 SM 驻留 6 个 Block（受线程上限 1536 限制），V4 在 $N = 4096$ 时是 5 个，$N = 8192$ 时只剩 2 个。而且[共享内存和 L1 共用同一块物理空间](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)（A100 合计 192 KB），它越大，L1 就越小。所以大 $N$ 时带宽利用率会先于容量撞墙，具体从哪个 $N$ 开始不划算，只能实测，原稿也只测了 $N = 4096$。

### 能快多少 {#v4-speedup}

三次读变成两次，读流量少 1/3，看上去可以快 1.5 倍。但 Softmax 还有一次写入，写也占带宽。把写算进去，每个元素的流量从 16 字节变成 12 字节，理论比值是 $4/3 \approx 1.33$。4096×4096 时的流量是：

| | 读 | 写 | 合计 | 满带宽耗时 |
| --- | --- | --- | --- | --- |
| V1 到 V3 | 192 MiB（201.3 MB） | 64 MiB | 256 MiB | 132 μs |
| V4 | 128 MiB（134.2 MB） | 64 MiB | 192 MiB | 99 μs |

这 1.33 也只是流量受限时的估计。同一行在几遍之间可能命中 L1 或 L2，实际到达显存的流量未必这么多，而 V4 又占用了一部分 L1。

这一点在实机上很明显。我们在 RTX 4060 笔记本上运行 4096×4096，V0 约 3.4 ms，V1 到 V4 都在 0.6~0.8 ms，V4 相对 V3 没有可见的提升。这张卡不是 A100，缓存和带宽都不同，机器上也还有别的负载，所以这组数字不能替代下面的 A100 数据。但它说明 V4 的收益取决于重复读取是否真的到达显存，换一张卡要重新测。

原稿数据：V4 带宽利用率约 75%，约 1529 GB/s（$N = 4096$），吞吐约为 V3 的 1.15 倍。 注意这是吞吐的比值，不是速度的比值：V4 要搬的字节也更少了，见下一节。

## 五个版本放在一起 {#summary}

<GpuBars
  label="V0 到 V4 的原稿吞吐数据"
  unit="GB/s"
  :ticks="[0, 500, 1000, 1500, 2000]"
  :rows="[
    { label: 'V0 一线程一行', value: 102, note: '利用率约 5%' },
    { label: 'V1 一 Block 一行', value: 714, note: '利用率约 35%' },
    { label: 'V2 两级规约', value: 1060, note: '利用率约 52%' },
    { label: 'V3 float4', value: 1325, note: '利用率约 65%' },
    { label: 'V4 暂存 exp', value: 1529, note: '利用率约 75%，N = 4096' }
  ]"
/>

吞吐的比值只有在搬的字节相同时才等于速度的比值。V0 到 V3 都是 256 MiB，V4 是 192 MiB，所以要先换算成耗时：

| 版本 | 利用率 | 吞吐 | 流量 | 估算耗时 | 估算相对 V0 |
| --- | --- | --- | --- | --- | --- |
| V0 | 约 5% | 102 GB/s | 256 MiB | 约 2633 μs | 1.0 倍 |
| V1 | 约 35% | 714 GB/s | 256 MiB | 约 376 μs | 7.0 倍 |
| V2 | 约 52% | 1060 GB/s | 256 MiB | 约 253 μs | 10.4 倍 |
| V3 | 约 65% | 1325 GB/s | 256 MiB | 约 203 μs | 13.0 倍 |
| V4 | 约 75% | 1529 GB/s | 192 MiB | 约 132 μs | 20.0 倍 |

<p class="gpu-note">利用率和吞吐来自本文的原始整理稿，没有重新测量，原稿也没有说明吞吐是否包含写入。耗时和相对速度是由流量除以吞吐算出的估计，假设吞吐包含读和写。按这个假设，V4 相对 V3 约快 1.54 倍，即流量减少带来的 1.33 倍乘以吞吐提升的 1.15 倍。</p>

每一版对应一个瓶颈：

| 瓶颈 | 表现 | 解决版本 | 做法 |
| --- | --- | --- | --- |
| 数值溢出 | 超界时出现 inf 和 NaN | V0 起 | 减最大值 |
| 线程少、地址散 | 每个 SM 只有约 1 个 Warp，32 条车道 32 次事务 | V1 | 一个 Block 处理一行 |
| 规约太重 | 每行 20 次屏障 | V2 | Warp Shuffle 两级规约 |
| 指令开销 | 每条读指令只带回 4 字节 | V3 | float4，注意每行都要对齐 |
| 重复读取 | 同一行读 3 次，expf 算 2 次 | V4 | 把 exp 暂存到共享内存 |

### 实际该选哪个 {#choose}

| 场景 | 建议 |
| --- | --- |
| 学习原理 | V1 或 V2，逻辑清晰 |
| $N$ 不大，且每个 SM 还能驻留足够多的 Block | V4 |
| $N$ 很大（共享内存放不下，或驻留 Block 太少） | V3 或 [Online Softmax](./online-softmax) |
| Softmax 是 Attention 的一部分 | 优先用融合实现，如 [FlashAttention](../flash-attention/v1)，避免中间结果落到全局内存 |

V4 只是用共享内存的容量换掉了一次读取，$N$ 越大越不划算。[下一篇](./online-softmax)换一个思路：在同一遍扫描中同时维护最大值和分母，不暂存整行也能省掉一次读取。

## 检验一下 {#check}

::: details 思考：最大值减完之后，某些项还是舍入成了 0，结果可信吗？
可信。FP32 中只有 $x_i - m < -103.97$ 的项才会变成 0，这种项的真实占比不到分母 $d \ge 1$ 的 $7 \times 10^{-46}$，本来就不可表示。而最大的一项始终是 1，不会丢失。开启 `-ftz=true` 时，阈值会提前到约 $-87.34$，但结论不变。
:::

::: details 思考：一行全是相等的值，N = 4096，用 FP16 累加分母，会发生什么？
每项都是 1。按顺序累加时，和到 2048 之后，FP16 相邻数的间隔变成 2，再加 1 刚好是舍入的中点，按“向偶数舍入”回到 2048，于是分母停在这里，每个输出是 $1/2048$，比真值 $1/4096$ 大一倍。树形规约两边大小相近，能正确得到 4096。这也是分母通常用 FP32 累加的原因之一。
:::

::: details 思考：Reduce 的 Grid Stride Loop 把规约次数压到了几百次，Softmax 能这样做吗？
不能。Reduce 只要一个总和，线程可以跨任意数据累加，最后才规约。Softmax 每行都要有自己的 $m$ 和 $d$，线程不能跨行混在一起，所以 4096 行至少要规约 $2 \times 4096$ 次。也正因如此，V2 优化单次规约才这么值得。
:::

::: details 思考：Block 从 256 改成 512，N = 4096 时 V4 的启动参数要变吗？
不用变。`smem` 按元素下标存放，大小只和 $N$ 有关，还是 16384 字节。变的是每个线程的工作量（从 16 个元素降到 8 个）和驻留方式：每个 SM 按共享内存仍够放约 9 个 Block，但 2048 个线程的上限只够 4 个，这时限制驻留量的是线程数。`warp_max[32]` 能支持的 Block 最大是 1024 个线程。
:::

::: details 思考：V4 的 smem[i] 只有写入它的线程自己会读，为什么还要放在共享内存里？
这里确实没有线程间通信，共享内存只是当作一块片上暂存。之所以不用线程局部数组，是因为局部内存放在片外，访问要走全局内存路径。如果 $N / \text{blockDim}$ 足够小，而且下标在编译期确定，也可以暂存在寄存器里（每个线程最多 255 个 32 位寄存器），不占共享内存。[Online Softmax 的 V3](./online-softmax#local) 详细比较了这几种暂存方式。
:::

<p class="gpu-note">本文各版本的完整代码在仓库的 <code>code/softmax/softmax.cu</code>，每个版本都会校验输出。运行方法见 <code>code/README.md</code>。</p>

<p class="gpu-note">本文以 <a href="https://github.com/caomaolufei/AIInfraGuide" target="_blank" rel="noreferrer">AI Infra Guide</a> 的<a href="https://github.com/caomaolufei/AIInfraGuide/blob/main/docs/guides/%E6%A8%A1%E5%9D%97%E4%BA%8C-CUDA%E7%BC%96%E7%A8%8B%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/5.1-CUDA%20Softmax%E6%9C%B4%E7%B4%A0%E5%AE%9E%E7%8E%B0%E4%BC%98%E5%8C%96.md" target="_blank" rel="noreferrer">《5.1 CUDA Softmax 朴素实现优化》</a>为蓝本重写，文中的“原稿”和“原始整理稿”都指这篇文章。感谢原作者的整理。</p>

## 参考资料 {#refs}

- [CUDA Programming Guide - Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)：32 字节事务与合并访问、共享内存、占用率
- [CUDA Programming Guide - C/C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)：`__shared__`、启动参数、Warp Shuffle、`float4` 对齐
- [CUDA Programming Guide - Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)：A100 每 SM 164 KB、每 Block 163 KB、48 KB 声明门槛
- [CUDA Programming Guide - Configuring L1/Shared Memory Balance](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#configuring-l1-shared-memory-balance)
- [CUDA 12.4 C++ Programming Guide - Device Memory Accesses](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#device-memory-accesses)：未对齐读取的后果
- [Online normalizer calculation for softmax - Milakov & Gimelshein, 2018](https://arxiv.org/abs/1805.02867)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [深入浅出 GPU 优化系列：softmax 优化](https://zhuanlan.zhihu.com/p/341059988)

<p class="gpu-note">本页自绘的结构图为教学简化。容量按 1 KB = 1024 字节估算，实际值以设备查询为准。文中标注 RTX 4060 的结果是用原稿代码在 RTX 4060 Laptop（compute capability 8.9）上实际运行得到的，A100 上的行为没有验证。</p>
