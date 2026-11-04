---
title: Online Softmax 图解
description: 用一个修正因子把三遍扫描变成两遍，逐元素走通递推，查清并修复规约中的 NaN，再讲清它如何通向 FlashAttention。
pageClass: gpu-page
---

# Online Softmax 图解

<p class="gpu-lede">逐行计算 Softmax 时，三遍扫描让每个元素访问 4 次全局内存。Online Softmax 用一个修正因子把前两遍合成一遍，访问次数降到 3 次，论文在 V100 上测到最高 1.3 倍的提速。可是照着常见写法实现，Block 为 256 时输出全是 NaN。这个递推怎么省下一遍读取，NaN 又是从哪里来的？</p>

本文接着[上一篇 Softmax 优化图解](./softmax)，从递推公式出发，把 Kernel 改写五次，从 V0 到 V4。版本号是本文自己的，和上一篇无关。规约的基础技巧见 [Reduce 算子优化图解](../reduce/reduce)。最后一节讲清它如何通向 [FlashAttention](../flash-attention/v1)。

## 省下的是哪一次访问 {#passes}

Safe Softmax 要先知道最大值 $m$，才能算分母 $d$，然后才能输出。三个量依次依赖，所以要读三遍。Online Softmax 把前两遍合并，V3 再尝试去掉一次读取。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigOnlinePasses /></div>
  <figcaption>虚线格是 V3 的暂存数组，它不是对 input 或 output 的访问，但也可能走到显存，V3 一节再细说。</figcaption>
</figure>

这里有一个容易算错的账。只数读取，3 次变 2 次，流量少 1/3，看上去能快 1.5 倍。但输出也要写，写同样占显存带宽。[Online Softmax 论文](https://arxiv.org/abs/1805.02867)的算法正是这么数的：4 次访问变 3 次，比值 $4/3 \approx 1.33$。它在 V100 上测到的“最高 1.3 倍”也和这个比值吻合。

所以“少读一遍”和“快 33%”可以同时成立，但不是从 3 次变 2 次推出来的。而且两遍方案每个元素还要访问 3 次，并不是访问次数的下限。真正的下限是读一次、写一次，也就是上图的第三行。

### 算力还够吗 {#budget}

合并的代价是多算一次 `expf`。Safe 每个元素算 2 次，Online 算 3 次。按 A100 的 2039 GB/s 和 19.5 TFLOPS 算一笔账：

<div class="gpu-chain">
  <div><b>12</b><span>每个元素的流量，字节</span></div>
  <div><b>170 G</b><span>满带宽时每秒处理的元素</span></div>
  <div><b>约 57</b><span>算力允许每个元素用的指令槽</span></div>
  <div><b>3 次</b><span>每个元素要算的 expf</span></div>
</div>

19.5 TFLOPS 把一次 FMA 算作 2 个 FLOP，所以每个元素约有 57 个指令槽。这是理想上限。按 sm_80 编译，一次 `expf` 约 10 条指令，其中一条 `MUFU.EX2` 走 SFU，3 次就占去约一半，还没算地址、分支和合并规约。

所以只能说，在每次访问都到达显存的假设下，带宽可能先到头，而且余量比 Reduce 小得多。行不长时重复读取还可能命中缓存，[后面的实测](#cache)会看到。“额外计算会被内存延迟完全隐藏”说得太绝对，瓶颈在哪里要在目标 GPU 上测量。[Roofline](../about_gpu/ai-compute#roofline) 上，它离拐点比 Reduce 近。

## 递推：用修正代替重算 {#recurrence}

设已经处理了前 $k$ 个元素，维护两个量：

$$
m_k = \max(x_0, \ldots, x_{k-1}), \qquad d_k = \sum_{j=0}^{k-1} e^{x_j - m_k}
$$

新元素 $x_k$ 到来时，最大值可能变大，变大之后，$d_k$ 里的每一项都用错了基准。好消息是它们错得一样多：

$$
e^{x_j - m_{k+1}} = e^{x_j - m_k} \cdot e^{m_k - m_{k+1}}
$$

$e^{m_k - m_{k+1}}$ 和 $j$ 无关，可以提到求和号外面。于是不必重新计算旧的元素，只要把旧的和整体乘一个因子，再加上新元素：

$$
m_{k+1} = \max(m_k, x_k), \qquad d_{k+1} = d_k \cdot e^{m_k - m_{k+1}} + e^{x_k - m_{k+1}}
$$

点“下一步”走一遍。遇到新的最大值时，注意修正前后的条形：旧元素按同一比例缩小。

<OnlineLab />

论文的算法写成伪代码是这样，第一遍就同时得到了 $m$ 和 $d$：

```python
# 第 1 遍：同时计算 m 和 d
m, d = -inf, 0
for x_j in x:
    m_new = max(m, x_j)
    d = d * exp(m - m_new) + exp(x_j - m_new)
    m = m_new

# 第 2 遍：归一化输出
for j in range(N):
    y[j] = exp(x[j] - m) / d
```

有两点值得注意：

- 没遇到新最大值时，修正因子是 $e^0 = 1$，公式退化成累加。但代码没有分支，`expf(0)` 照样要算。公式上没有额外项，不等于运行时没有开销。
- 不溢出的保证和 Safe 相同：$x_j - m \le 0$，修正因子不超过 1，而且[论文证明了](https://arxiv.org/abs/1805.02867) $1 \le d_k \le k$，float 足够存放任何实际长度的 $d$。但这不等于舍入误差完全一致，下面的 [V1](#associativity) 有实测数字。

测试条件和上一篇相同：A100 80GB SXM4，$(M, N) = (4096, 4096)$，Block 为 256 个线程。下文标为“原稿数据”的带宽数字都来自原始整理稿，测量方法和完整数据未知，我们也没有重新测量。由它们换算的耗时和倍数只是带假设的估计，不等于验证过的实测结果。

## V0：一个线程一行 {#v0}

先写最直接的版本，确认算法正确：

```cpp
// V0: 一个线程处理一行
__global__ void online_softmax_v0(float* input, float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float* x = input  + row * N;
    float* y = output + row * N;

    // 第 1 遍：同时计算 max 和 sum
    float m = -FLT_MAX;   // 不用 -INFINITY，原因见 V1
    float d = 0.0f;

    for (int i = 0; i < N; i++) {
        float xi = x[i];
        float m_new = fmaxf(m, xi);
        d = d * expf(m - m_new) + expf(xi - m_new);
        m = m_new;
    }

    // 第 2 遍：归一化
    float inv_d = 1.0f / d;
    for (int i = 0; i < N; i++) {
        y[i] = expf(x[i] - m) * inv_d;
    }
}
```

问题和[上一篇的 V0](./softmax#v0) 相同：同一 Warp 的 32 个线程分别读 32 行，地址相差一整行，请求无法合并。原稿数据：V0 带宽利用率约 7%，约 143 GB/s。

## V1：局部递推，再合并 {#v1}

递推是串行的，每一步都依赖上一步，看起来无法并行。关键在于：两段数据各自算出 $(m, d)$ 之后，可以合并。

$$
m = \max(m_1, m_2), \qquad d = d_1 \cdot e^{m_1 - m} + d_2 \cdot e^{m_2 - m}
$$

这和递推是同一个公式。一个元素 $x$ 就相当于状态 $(x, 1)$，递推只是每次合并一个元素的特例。

### 合并顺序为什么无所谓 {#associativity}

注意 $d \cdot e^{m}$ 这个量。合并后：

$$
d \cdot e^{m} = d_1 e^{m_1} + d_2 e^{m_2}
$$

也就是说，$d \cdot e^{m}$ 始终等于这段元素的 $\sum e^{x_j}$，只是为了不溢出才拆成两部分存。求和与取最大值都满足结合律和交换律，所以合并也满足。[论文](https://arxiv.org/abs/1805.02867)同样给出了这个结论。于是可以用树形规约：

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigOnlineMerge /></div>
  <figcaption>图中数值由单精度实时算出。琥珀色是从内存读入的元素，青色是合并结果。虚线表示这一侧的 m 较小，它的 d 要乘修正因子缩小；实线表示这一侧的 m 就是新的最大值，因子为 1。</figcaption>
</figure>

以上是实数上的结论。浮点数还有舍入，所以用单精度实际算了一遍。随机取 20 万组三元组比较 $(a \oplus b) \oplus c$ 和 $a \oplus (b \oplus c)$：$m$ 始终相同，$d$ 有 91.5% 完全相同，最大相对差 $2.0 \times 10^{-7}$，约 1.7 个机器精度。把合并扩展到整行，和双精度参考值比较，结果是树形反而更准：

| 输入 | 串行递推 | 树形合并 | 三遍 Safe |
| --- | --- | --- | --- |
| $[-8, 8]$ 均匀分布，$N = 4096$ | $5.7 \times 10^{-6}$ | $8.7 \times 10^{-8}$ | $5.7 \times 10^{-6}$ |
| 升序，步长 0.001，$N = 4096$ | $2.2 \times 10^{-5}$ | $2.6 \times 10^{-8}$ | $9.1 \times 10^{-7}$ |
| 升序，步长 $10^{-5}$，$N = 65536$ | $1.2 \times 10^{-4}$ | $5.6 \times 10^{-8}$ | $1.4 \times 10^{-7}$ |
| $[-1, 1]$ 均匀分布，$N = 2^{20}$ | $8.0 \times 10^{-5}$ | $9.1 \times 10^{-8}$ | $8.0 \times 10^{-5}$ |

<p class="gpu-note">d 的最大相对误差，每种输入取 5 次。模拟里 expf 按正确舍入计算，真实的 expf 误差会略大。</p>

串行累加的误差随长度增长，树形合并的舍入路径只有 $\log_2 N$ 层，所以更小。第二行、第三行是串行递推的最坏情况：每一步都要修正，误差比 Safe 明显更大。不过实际的 V1 只在每个线程内串行十几个元素，$N = 4096$、Block 为 256 时，两级规约的最大误差是 $1.1 \times 10^{-7}$。

### 两级规约 {#v1-flow}

沿用上一篇的[两级结构](./softmax#two-level)：Warp 内用 Shuffle，Warp 间经过共享内存。和普通规约的区别是要同时传 $m$ 和 $d$。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigOnlineFlow /></div>
  <figcaption>示意图，没有按比例。青色是寄存器上的计算，琥珀色是内存访问。一次合并规约同时得到 m 和 d，Safe 则要为 max 和 sum 各做一次。</figcaption>
</figure>

| 对比项 | 三遍 Safe | 两遍 Online |
| --- | --- | --- |
| 每个元素的读、写 | 3 次、1 次 | 2 次、1 次 |
| 两级规约 | 2 次 | 1 次 |
| 屏障 | 4 道 | 2 道 |
| 每个元素的 `expf` | 2 次 | 3 次 |
| 规约每轮、每条车道的 `expf` | 0 次 | 2 次 |

### 照搬 -INFINITY，输出全是 NaN {#nan}

按论文的初值，$m_0 = -\infty$、$d_0 = 0$，合并函数和第二级规约可以写成这样：

```cpp
// 有问题的写法
__device__ void warpReduceOnline(float& m, float& d) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);

        float m_new = fmaxf(m, m2);
        d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }
}

// 第二级：Warp 0 做最终合并
if (wid == 0) {
    local_m = (lane < num_warps) ? warp_m[lane] : -INFINITY;
    local_d = (lane < num_warps) ? warp_d[lane] : 0.0f;
    warpReduceOnline(local_m, local_d);
}
```

看起来 $(-\infty, 0)$ 是合并的单位元：$d = 0$ 乘什么都该是 0。但如果两边都是空车道，`m - m_new` 就是 $-\infty - (-\infty)$，按 IEEE 754 得 NaN，`expf(NaN)` 还是 NaN，而 `0 * NaN` 依然是 NaN。Block 为 256 时，第二级只有 8 条车道有数据，车道 8~31 全是这个初值。下面的实验逐轮算出了每条车道的状态：

<OnlineLabNaN />

默认情况下，路径是这样的：第 1 轮车道 8 读车道 24，两边都空，车道 8 变成 NaN。第 2 轮车道 0 读车道 8，`d2 * expf(m2 - m_new)` 是 `NaN * 0`，NaN 就进了车道 0。这条路径上的读取都没有越界，所以[越界时保持原值](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)的规则不是原因。把越界行为换成其他值再模拟一次，结果还是 NaN。

切换 Block 大小可以看到触发条件：`num_warps` 小于 16 时一定出错，也就是 Block 小于 512。Block 为 512 时，车道 16~31 也会变成 NaN，但车道 0 读它们的时候它们还没变，之后也不会再读。这恰恰是这种错误容易漏测的原因。

同样的问题也出现在第一级。线程没有分到元素时（$N$ 小于 Block），它的状态就是初值：一个 Warp 里有数据的车道少于 16 条，或者有整个 Warp 都是空的，结果就是 NaN。输入含有掩码时还有第三个入口：如果一段数据的第一个元素就是 `-INFINITY`（Attention 里常用来屏蔽位置），串行递推的第一步也会算出 NaN。

### 修复：把初值换成 -FLT_MAX {#fix}

只要让初值是有限数，两个空状态相减就得 0，`0 * expf(0)` 还是 0，空状态成了真正的单位元。它和真实元素合并时，`expf` 的参数是一个极大的负数，结果是 0，也不受影响。代码不用加任何分支：

```cpp
#include <cfloat>   // FLT_MAX

// warpReduceOnline 本身不用改，只改初值
local_m = (lane < num_warps) ? warp_m[lane] : -FLT_MAX;
local_d = (lane < num_warps) ? warp_d[lane] : 0.0f;
```

输入里的 `-INFINITY` 也能处理：`fmaxf` 仍然取 `-FLT_MAX`，`expf(-INFINITY - m)` 是 0，这个元素的贡献为 0。输入里有 NaN 时，结果仍然是 NaN，不会被悄悄吞掉。

这个问题不只是纸面推演。我们把两种初值的 Kernel 编译后，放到本机的 RTX 4060 Laptop 上实际运行（nvcc 13.4，$M = 4$，输入取 $[-8, 8]$ 的均匀分布），并和双精度参考值比较：

| 场景 | `-INFINITY` 的 NaN 数 | `-FLT_MAX` 的 NaN 数 |
| --- | --- | --- |
| Block 32、64、128、256，$N = 4096$ | 16384 / 16384（全部） | 0 |
| Block 512、1024，$N = 4096$ | 0 | 0 |
| Block 1024，$N = 100$ | 200 / 200（$M = 2$） | 0 |
| Block 1024，$N = 1000$ | 2000 / 2000（$M = 2$） | 0 |
| Block 1024，$N = 1008$、$1020$ | 0 | 0 |
| V0，偶数位置为 `-INFINITY` | 8192 / 8192（$M = 2$） | 0 |
| V1，Block 1024，偶数位置为 `-INFINITY` | 8192 / 8192（$M = 2$） | 0 |

实测结果和 JavaScript 模拟的结论一致。修复后，$N = 4096$ 的输出和参考值的绝对误差最大约 $9 \times 10^{-10}$。这里没有在 A100 上运行，但 NaN 只取决于 IEEE 754 的运算规则，和具体 GPU 无关。

这个修复的适用范围要说清楚：它假设有效输入都是有限值，掩码只用 `-INFINITY`。下面两种输入它不处理，要由调用方决定策略：

- **整行都被屏蔽。** $d = 0$，输出是 NaN，和 Safe 的 0/0 一样。可以在写出前判断 $d > 0$，并约定这一行输出全 0。[FlashAttention V1 的 `l_safe`](../flash-attention/v1#edge) 是同样的思路。
- **输入含 `+INFINITY`。** 递推会算 $+\infty - (+\infty)$，$d$ 立刻变成 NaN，Safe 也一样。它通常说明上游已经溢出，建议在进入 Kernel 前校验并报错，不要静默钳位。

修复后的 V1：

```cpp
// Warp 内 Online 合并规约
__device__ void warpReduceOnline(float& m, float& d) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);

        float m_new = fmaxf(m, m2);
        d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }
}

// V1: 一个 Block 一行 + 两级合并规约
__global__ void online_softmax_v1(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    float* x = input  + row * N;
    float* y = output + row * N;

    // 第 1 遍：每个线程处理一段
    float local_m = -FLT_MAX;
    float local_d = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(local_m, xi);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }

    // 第一级：Warp 内合并
    warpReduceOnline(local_m, local_d);

    // Warp 间通过 Shared Memory 交换
    __shared__ float warp_m[32];
    __shared__ float warp_d[32];

    if (lane == 0) {
        warp_m[wid] = local_m;
        warp_d[wid] = local_d;
    }
    __syncthreads();

    // 第二级：Warp 0 做最终合并
    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        local_m = (lane < num_warps) ? warp_m[lane] : -FLT_MAX;
        local_d = (lane < num_warps) ? warp_d[lane] : 0.0f;
        warpReduceOnline(local_m, local_d);
    }

    // 广播最终结果
    __shared__ float final_m, final_d;
    if (tid == 0) {
        final_m = local_m;
        final_d = local_d;
    }
    __syncthreads();

    float row_max = final_m;
    float inv_sum = 1.0f / final_d;

    // 第 2 遍：归一化输出
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - row_max) * inv_sum;
    }
}
```

原稿数据：V1 带宽利用率约 58%，约 1183 GB/s，吞吐约为 V0 的 8.3 倍。原稿没有说明测量时是否校验过输出。带宽计数器看不出 NaN，所以这个问题不会反映在带宽数字上，这也是这组数字不能当确定结论的原因之一。

## V2：float4 读取 {#v2}

和[上一篇的 V3](./softmax#v3) 思路相同：每条读指令取回 16 字节而不是 4 字节，读同样多的数据，发出的读指令和循环判断都只有原来的 1/4。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigOnlineLoad /></div>
  <figcaption>示意图：为了画得下，这里用 4 个线程，第三排用 N = 6。t1 表示线程 1 负责的元素，用琥珀色标出；第三排的琥珀色是第 1 行。红框是行 1 第一次 float4 读取的范围。</figcaption>
</figure>

float4 只减少读指令。递推还是逐元素做的，`expf` 的次数一点没少，所以 V2 的收益全部来自访存这一侧。

```cpp
// V2: float4 读取 + Online 递推
__global__ void online_softmax_v2(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    float* x = input  + row * N;
    float* y = output + row * N;

    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    // N 不是 4 的倍数时，行首不一定对齐，全部退回逐元素读取
    int N4 = (N % 4 == 0) ? N / 4 : 0;

    // 第 1 遍：向量化读取，逐元素递推
    float local_m = -FLT_MAX;
    float local_d = 0.0f;

    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];

        // 对 4 个元素依次更新 (m, d)
        float vals[4] = {data.x, data.y, data.z, data.w};
        for (int k = 0; k < 4; k++) {
            float m_new = fmaxf(local_m, vals[k]);
            local_d = local_d * expf(local_m - m_new) + expf(vals[k] - m_new);
            local_m = m_new;
        }
    }
    // N4 == 0 时，这个循环处理整行
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(local_m, xi);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }

    // 两级合并（与 V1 相同）
    warpReduceOnline(local_m, local_d);

    __shared__ float warp_m[32];
    __shared__ float warp_d[32];
    if (lane == 0) {
        warp_m[wid] = local_m;
        warp_d[wid] = local_d;
    }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        local_m = (lane < num_warps) ? warp_m[lane] : -FLT_MAX;
        local_d = (lane < num_warps) ? warp_d[lane] : 0.0f;
        warpReduceOnline(local_m, local_d);
    }

    __shared__ float final_m, final_d;
    if (tid == 0) {
        final_m = local_m;
        final_d = local_d;
    }
    __syncthreads();

    float row_max = final_m;
    float inv_sum = 1.0f / final_d;

    // 第 2 遍：向量化归一化
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        float4 result;
        result.x = expf(data.x - row_max) * inv_sum;
        result.y = expf(data.y - row_max) * inv_sum;
        result.z = expf(data.z - row_max) * inv_sum;
        result.w = expf(data.w - row_max) * inv_sum;
        y4[i] = result;
    }
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - row_max) * inv_sum;
    }
}
```

代码里的 `N4` 改过。如果照常写 `N4 = N / 4`，只处理行尾的尾部循环解决不了[行首不对齐的问题](./softmax#align)。上图第三排就是这种情况。在本机上用 $N = 4094$ 运行，Kernel 直接报错 `misaligned address`，改成上面的写法后输出正确。也可以按上一篇的做法，让行距取 4 的倍数。

`vals[4]` 不会带来额外开销。它的下标是循环变量，而循环次数在编译期已知，编译器能全部展开成常量下标。用 ptxas 看（目标 sm_80），这个 Kernel 的栈帧是 0 字节。这一点在下一节会变得很重要。

原稿数据：V2 带宽利用率约 70%，约 1427 GB/s，吞吐约为 V1 的 1.21 倍。

## V3：只读一次 {#v3}

两遍方案把 `input` 读了两次，第二次只是因为第一遍读到的值没有留下来。如果每个线程把自己读到的元素暂存起来，等 $m$ 和 $d$ 算好后直接用，就省掉了第二次读取。

```cpp
// V3: 暂存到线程局部数组，只读一次 input
// 调用前必须保证 N <= blockDim.x * MAX_ELEMS_PER_THREAD，否则数组越界
#define MAX_ELEMS_PER_THREAD 32

__global__ void online_softmax_v3(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    float* x = input  + row * N;
    float* y = output + row * N;

    // 线程局部数组。下标在运行时才知道，它在局部内存里，不在寄存器里
    float cache[MAX_ELEMS_PER_THREAD];

    // 唯一一遍全局读取：读数据 + 递推
    float local_m = -FLT_MAX;
    float local_d = 0.0f;
    int count = 0;

    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x[i];
        cache[count] = xi;
        count++;

        float m_new = fmaxf(local_m, xi);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }

    // 两级合并
    warpReduceOnline(local_m, local_d);

    __shared__ float warp_m[32];
    __shared__ float warp_d[32];
    if (lane == 0) {
        warp_m[wid] = local_m;
        warp_d[wid] = local_d;
    }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        local_m = (lane < num_warps) ? warp_m[lane] : -FLT_MAX;
        local_d = (lane < num_warps) ? warp_d[lane] : 0.0f;
        warpReduceOnline(local_m, local_d);
    }

    __shared__ float final_m, final_d;
    if (tid == 0) {
        final_m = local_m;
        final_d = local_d;
    }
    __syncthreads();

    float row_max = final_m;
    float inv_sum = 1.0f / final_d;

    // 从暂存读取并写出，不再读 input
    int idx = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(cache[idx] - row_max) * inv_sum;
        idx++;
    }
}
```

### 数组其实不在寄存器里 {#local}

这个方案常被叫作“寄存器缓存”，但这段代码并不是。[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf) 第 10.2.4 节写得很明确：编译器判断可能被动态索引的数组，很可能放进局部内存（Local Memory）；局部内存在片外，访问和全局内存一样贵，这个名字说的是作用域，不是位置。这里的 `count` 是运行时变量，正好属于这种情况。

可以直接验证。把 Kernel 编译成 PTX，这个数组声明成 `.local`，读写是 `st.local` 和 `ld.local`。ptxas（目标 sm_80）报告的栈帧恰好是 128 字节，也就是 32 个 float。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigOnlineCache /></div>
  <figcaption>三种暂存方式。局部内存在这些架构上默认走 L1 和 L2，所以未必每次都到显存，但它不像寄存器和共享内存那样保证留在片上。</figcaption>
</figure>

想让数据真的留在寄存器里，下标必须在编译期确定。比如把每个线程的元素数做成模板参数 `K`，循环展开后下标就成了常量。这要求 $N$ 正好等于 $\text{blockDim} \times K$：

```cpp
// 每线程元素数编译期已知，N 必须等于 blockDim.x * K
float cache[K];
for (int k = 0; k < K; k++) {
    float xi = x[tid + k * blockDim.x];
    cache[k] = xi;
    // ... 递推同上
}
// ... 两级合并同上
for (int k = 0; k < K; k++) {
    y[tid + k * blockDim.x] = expf(cache[k] - row_max) * inv_sum;
}
```

用 ptxas 查看不同写法实际用了多少寄存器（目标 sm_80，全部用 `-O3` 编译）：

| 写法 | 寄存器 | 栈帧 | 每个 SM 最多驻留线程（估算） |
| --- | --- | --- | --- |
| V2（不暂存） | 31 | 0 | 2048 |
| V3，运行时下标，`[32]` | 23 | 128 字节 | 2048 |
| 编译期 $K = 8$ | 40 | 0 | 约 1600 |
| 编译期 $K = 16$ | 64 | 0 | 1024 |
| 编译期 $K = 32$ | 128 | 0 | 512 |
| 编译期 $K = 64$ | 32 | 256 字节 | 2048 |

<p class="gpu-note">驻留线程数按 A100 每个 SM 64K 个 32 位寄存器平分估算，并以 2048 为上限，实际分配有粒度。这两个数字和每线程 255 个寄存器的上限都见官方的 <a href="https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html">Compute Capabilities</a>。</p>

这张表说明了寄存器方案真正的代价。寄存器数大约是每个元素 4 个，不止存放值本身的 1 个，多出来的应该是循环展开后的地址和中间值。所以 $K = 32$ 虽然没有超过每线程 255 个的上限，却要占 128 个寄存器，每个 SM 只能驻留 512 个线程，也就是 2 个 256 线程的 Block，而不暂存时是 8 个。$K = 64$ 则换了一种结果：PTX 里这个数组已经是 `.local`，说明循环没有完全展开，下标不再是常量。驻留的线程少了，在途的读写请求也少了。[官方文档](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)对此的说法是，占用率低一定会影响隐藏内存延迟的能力。

还有第三个选择：把一行放进共享内存。它留在片上，读写地址连续，上一篇的 [V4](./softmax#v4) 就用共享内存暂存过 exp 的值，容量和驻留 Block 数的分析也完全适用。这个写法在本机上跑过，$N = 4096$ 时输出正确。

### “N ≤ 8192”从哪来 {#v3-limit}

这个数字来自 $256 \times 32$，也就是 Block 大小乘以代码里选的数组长度。它是代码的容量上限，不是性能拐点。原稿只测了 $N = 4096$，$N = 8192$ 没有数据。把数组开大或者把 Block 开到 1024，上限都会变。

反过来，超过这个上限也不是“效果变差”，而是写出数组边界，属于未定义行为。原代码没有任何检查，所以调用前必须判断。

原稿数据：V3 带宽利用率约 82%，约 1672 GB/s（$N = 4096$，Block 为 256）。 这个数字不能直接和 V2 比，流量不同，下面的[汇总](#summary)会换算。

## V4：固定 Block 数 {#v4}

前面的版本都是一个 Block 处理一行，$M$ 有多大就启动多少个 Block。V4 让固定数量的 Block 循环处理多行，算法和 V2 完全相同，变的只是 Block 和行的映射。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigOnlineMapping /></div>
  <figcaption>示意图：12 行、4 个 Block。琥珀色是 Block 1 处理的行。按 A100 的实际参数，Grid 为 432（SM 数的 4 倍），M = 4096 时每个 Block 处理约 9.5 行。</figcaption>
</figure>

```cpp
// V4: Grid Stride 多行处理
__global__ void online_softmax_v4(float* input, float* output, int M, int N) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    // Grid Stride 遍历所有行
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        float* x = input  + row * N;
        float* y = output + row * N;

        float4* x4 = reinterpret_cast<float4*>(x);
        float4* y4 = reinterpret_cast<float4*>(y);
        int N4 = (N % 4 == 0) ? N / 4 : 0;

        // 第 1 遍：向量化读取，逐元素递推
        float local_m = -FLT_MAX;
        float local_d = 0.0f;

        for (int i = tid; i < N4; i += blockDim.x) {
            float4 data = x4[i];
            float vals[4] = {data.x, data.y, data.z, data.w};
            for (int k = 0; k < 4; k++) {
                float m_new = fmaxf(local_m, vals[k]);
                local_d = local_d * expf(local_m - m_new) + expf(vals[k] - m_new);
                local_m = m_new;
            }
        }
        for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
            float xi = x[i];
            float m_new = fmaxf(local_m, xi);
            local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
            local_m = m_new;
        }

        // 两级合并
        warpReduceOnline(local_m, local_d);

        __shared__ float warp_m[32];
        __shared__ float warp_d[32];
        if (lane == 0) {
            warp_m[wid] = local_m;
            warp_d[wid] = local_d;
        }
        __syncthreads();

        int num_warps = blockDim.x / 32;
        if (wid == 0) {
            local_m = (lane < num_warps) ? warp_m[lane] : -FLT_MAX;
            local_d = (lane < num_warps) ? warp_d[lane] : 0.0f;
            warpReduceOnline(local_m, local_d);
        }

        __shared__ float final_m, final_d;
        if (tid == 0) {
            final_m = local_m;
            final_d = local_d;
        }
        __syncthreads();

        float row_max = final_m;
        float inv_sum = 1.0f / final_d;

        // 第 2 遍：向量化归一化
        for (int i = tid; i < N4; i += blockDim.x) {
            float4 data = x4[i];
            float4 result;
            result.x = expf(data.x - row_max) * inv_sum;
            result.y = expf(data.y - row_max) * inv_sum;
            result.z = expf(data.z - row_max) * inv_sum;
            result.w = expf(data.w - row_max) * inv_sum;
            y4[i] = result;
        }
        for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
            y[i] = expf(x[i] - row_max) * inv_sum;
        }
    }
}
```

调用方式：

```cpp
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
int grid_size = min(M, num_sms * 4);
int block_size = 256;
online_softmax_v4<<<grid_size, block_size>>>(d_input, d_output, M, N);
```

这段代码比原稿少了循环末尾的一个 `__syncthreads()`。它的注释说是为了“重置 Shared Memory”，但屏障不会重置任何内存。检查一下时序：下一行写 `warp_m` 时，上一行对它的读取都在第二道屏障之前；`final_m` 要等下一行的第一道屏障之后才会被改写，而那时所有线程都已经读完了。所以这道屏障可以去掉，去掉后在本机上输出也正确。

Grid Stride 省下的是启动 Block 的开销，不是规约。[Reduce 的 V7](../reduce/reduce#v7) 用它把规约次数减少了几百倍，是因为整个数组只需要一个和；Softmax 每行都要一个自己的 $(m, d)$，$M$ 行就要规约 $M$ 次，一次也省不掉。所以不要期望它有同样的收益。每一行也都会重新初始化 $m$ 和 $d$，并不存在跨行复用的状态。

原稿数据：V4 带宽利用率约 72%，约 1468 GB/s，和 V2 基本持平。 两者流量相同，2 个百分点的差距不足以说明问题，原稿也说收益主要在 $M$ 极大时。它同样解决不了 $M$ 很小而 $N$ 很大的情况：$M = 1$ 时只有一个 Block 在工作，那需要多个 Block 协作处理同一行。

## 五个版本放在一起 {#summary}

<div class="gpu-scroll online-bars-wrap" tabindex="0">
<GpuBars
  label="V0 到 V4 的原稿吞吐数据"
  unit="GB/s"
  :ticks="[0, 500, 1000, 1500, 2000]"
  :rows="[
    { label: 'V0 一线程一行', value: 143, note: '利用率约 7%' },
    { label: 'V1 合并规约', value: 1183, note: '利用率约 58%' },
    { label: 'V2 float4', value: 1427, note: '利用率约 70%' },
    { label: 'V3 只读一次', value: 1672, note: '利用率约 82%，N = 4096' },
    { label: 'V4 Grid Stride', value: 1468, note: '利用率约 72%' }
  ]"
/>
</div>

吞吐的比值只有在流量相同时才等于速度的比值。V3 少一次读取，要先换算成耗时：

| 版本 | 利用率 | 吞吐 | 流量 | 估算耗时 | 估算相对 V0 |
| --- | --- | --- | --- | --- | --- |
| V0 | 约 7% | 143 GB/s | 192 MiB | 约 1408 μs | 1.0 倍 |
| V1 | 约 58% | 1183 GB/s | 192 MiB | 约 170 μs | 8.3 倍 |
| V2 | 约 70% | 1427 GB/s | 192 MiB | 约 141 μs | 10.0 倍 |
| V3 | 约 82% | 1672 GB/s | 128 MiB | 约 80 μs | 17.5 倍 |
| V4 | 约 72% | 1468 GB/s | 192 MiB | 约 137 μs | 10.3 倍 |

<p class="gpu-note">利用率和吞吐来自原始整理稿，没有重新测量，原稿也没有说明吞吐是否包含写入和局部内存的流量。耗时和相对速度是由流量除以吞吐算出的估计，假设吞吐包含读和写，且 input、output 的每次访问都到达显存；下一节会看到，后一个假设并不总是成立。原稿写的“V3 是 V2 的 1.17 倍”只是吞吐的比值。</p>

每一版对应一个瓶颈：

| 瓶颈 | 表现 | 解决版本 | 做法 |
| --- | --- | --- | --- |
| 重复读取 | $m$、$d$ 要分两遍求 | 全部 | Online 递推，一遍同时得到 |
| 线程少、地址散 | 32 条车道读 32 行 | V1 | 一个 Block 一行，合并规约 |
| 读指令多 | 每条指令只带回 4 字节 | V2 | float4，注意每行都要对齐 |
| 还剩一次重复读取 | `input` 仍读两次 | V3 | 暂存一行，付出容量或驻留线程的代价 |
| Block 启动 | $M$ 很大时启动次数多 | V4 | 固定 Grid，循环处理多行 |

### 读几遍不等于访问显存几遍 {#cache}

前面的账都假设每次访问都到达显存。但同一行的第二次读取紧跟在第一次之后，可能直接命中缓存。为了验证，我们在本机 RTX 4060 Laptop（24 个 SM，显存带宽约 256 GB/s）上固定总数据量为 256 MiB，只改变行长 $N$：

<figure class="gpu-fig">
  <FigOnlineSweep />
  <figcaption>本机实测，不是 A100。M × N = 2 的 26 次方个 float，Block 为 256，取两次运行的平均，每次 9 次的中位数。把指针移到图上可以看某个 N 的数值。虚线是假设流量为 2N、3N、4N 且带宽 230 GB/s 时的理论耗时。V3 只画到它的容量上限 N = 8192。</figcaption>
</figure>

结果和预期不一样，也更有意思：

- 行不太长时（本机 $N \le 16384$），三遍、两遍和只读一次**耗时相同**，都在 2.4 ms 左右，刚好是 $2N$ 流量的理论耗时。这说明重复的读取几乎都命中了缓存，显存只看到了一次读和一次写。此时 Online 在显存流量上没有优势。
- 行足够长时（$N \ge 65536$），缓存装不下同时在处理的所有行。两条线分别稳定在 $4N$ 和 $3N$ 的位置，比值约 $4.6 / 3.5 = 1.31$，和论文在 V100 上的 1.3 倍几乎一样。论文也观察到了同样的现象：向量较短时几种算法性能相近，缓存开始颠簸后才拉开差距。

判断一下 A100 落在哪一边。Block 为 256 时每个 SM 最多驻留 8 个 Block，108 个 SM 同时处理 864 行，$N = 4096$ 时共约 13.8 MiB，而 [A100 的 L2 有 40 MB](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)。只看容量，第二遍读取有机会大量命中，原稿在 $N = 4096$ 下测得的差距不一定来自显存流量。这只是粗略估算，我们无法在 A100 上验证，要确认只能用 Nsight Compute 对比实际的显存流量。

::: details 本机实测数据
| $N$ | $M$ | 三遍 Safe | Online V2 | V3 局部数组 | V4 |
| --- | --- | --- | --- | --- | --- |
| 256 | 262144 | 2.38 | 2.83 | 3.19 | 3.71 |
| 512 | 131072 | 2.40 | 2.40 | 2.35 | 2.31 |
| 1024 | 65536 | 2.42 | 2.39 | 2.34 | 2.41 |
| 4096 | 16384 | 2.38 | 2.40 | 2.32 | 2.36 |
| 8192 | 8192 | 2.36 | 2.36 | 2.32 | 2.40 |
| 16384 | 4096 | 2.40 | 2.40 | | 2.40 |
| 32768 | 2048 | 3.44 | 3.09 | | 2.63 |
| 65536 | 1024 | 4.41 | 3.48 | | 3.28 |
| 131072 | 512 | 4.59 | 3.49 | | 3.41 |
| 262144 | 256 | 4.60 | 3.51 | | 3.50 |
| 1048576 | 64 | 4.62 | 3.52 | | 3.51 |

单位是毫秒。V4 在 $N = 256$ 时明显慢于 V2，$N = 32768$ 时明显快于 V2。后者可能是因为 Grid 只有 96 个，同时在途的行变少了，缓存更不容易颠簸，这只是推测。总之在这台机器上，看不出 V4 在 $M$ 极大时有收益。
:::

所以 V3 值不值得用，取决于重复读取有没有命中缓存：命中时它省不下什么，而且付出了容量限制；不命中时收益才明显。但不命中往往意味着 $N$ 很大，这时 V3 的数组又装不下了。这是 V3 适用范围窄的真正原因。

### 实际该选哪个 {#choose}

| 场景 | 建议 |
| --- | --- |
| 学习原理 | V1，逻辑清晰，能看到合并规约的全过程 |
| 独立的 Softmax，$N$ 较大 | V2。不需要暂存，$N$ 越大优势越明显 |
| $N$ 较小且固定 | 先用 Nsight Compute 看第二遍读取是否已经命中缓存，再决定要不要 V3，或者编译期 $K$ 的版本 |
| $M$ 很大 | V2 或 V4，实测对比后再选 |
| Attention 中的 Softmax | 不要单独做 Softmax，用 FlashAttention 的融合实现 |

## 通向 FlashAttention {#flash}

Self-Attention 的计算是：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

传统做法先算出整个 $N \times N$ 的分数矩阵，再做 Softmax，再乘 $V$。序列长度为 128K（131072）、单头、每个分数 4 字节时，这个矩阵就要 64 GiB，放不进片上 SRAM，只能反复读写显存。

Online Softmax 恰好提供了需要的工具：**一行分数不必同时存在，可以一段一段地合并进状态。** 这正是[合并公式](#v1)允许的事。

这里还有一个关键的不同。独立的 Softmax 要输出每个 $y_i$，而每个 $y_i$ 都要除以最终的 $d$，所以必须再读一遍。Attention 不需要 $y_i$ 本身，只需要 $y$ 和 $V$ 的乘积 $O$。旧的 $O$ 是之前各项的加权和，基准变化时每一项要乘同一个因子，所以修正 $O$ 只要一次乘法，**连第二遍都省了**。

<figure class="gpu-fig">
  <FigOnlineFlash />
  <figcaption>上图是一行分数的处理流程。下表用 9 个分数、每 3 个一块算了一遍，所有数值由组件实时算出。每一行的 O_new 都和“直接算到这一块”相等，不必等到最后才正确。</figcaption>
</figure>

更新公式是这样的。设这一块的 $\tilde m = \text{rowmax}(S)$，$\tilde P = \exp(S - \tilde m)$，$\tilde d$ 是 $\tilde P$ 的行和：

$$
m_{\text{new}} = \max(m_{\text{old}}, \tilde m), \qquad
d_{\text{new}} = d_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \tilde d \cdot e^{\tilde m - m_{\text{new}}}
$$

$$
O_{\text{new}} = O_{\text{old}} \cdot \frac{d_{\text{old}}}{d_{\text{new}}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \frac{e^{\tilde m - m_{\text{new}}}}{d_{\text{new}}} \cdot \tilde P V
$$

这和 [FlashAttention 论文](https://arxiv.org/abs/2205.14135)算法 1 的第 10~12 行一致。要特别注意 $\tilde P$ 的定义：它只减去了这一块的最大值，**没有除以这一块的和**。如果把 $\tilde P$ 理解成这一块的 Softmax，结果就错了。用脚本验证（64 个分数，每 16 个一块，$V$ 为 4 维）：

| $\tilde P$ 的取法 | 和直接计算的最大误差（双精度） |
| --- | --- |
| $\exp(S - \tilde m)$，未归一化 | $4.2 \times 10^{-17}$ |
| 这一块的 Softmax，已归一化 | $7.8 \times 10^{-2}$ |

把这些量和前文对应起来，就是下面这张表。

| Online Softmax | FlashAttention |
| --- | --- |
| 元素 $x_j$，状态 $(x_j, 1)$ | 一块分数，状态 $(\tilde m, \tilde d)$ |
| 合并公式 | 第 11 行，更新 $m$ 和 $\ell$（即本文的 $d$） |
| 修正因子 $e^{m_{\text{old}} - m_{\text{new}}}$ | 同一个因子，还要作用在 $O$ 上 |
| 第二遍读取，算 $y_i$ | 不需要，$O$ 一路修正 |
| 寄存器里的 $(m, d)$ | 每行一份，存放在显存，用到时再读入 |

FlashAttention-2 把 $O$ 的除法也推迟到最后：先累积 $\tilde O = e^{m_{\text{old}} - m_{\text{new}}} \tilde O + e^{\tilde m - m_{\text{new}}} \tilde P V$，结束时再除以 $d$。同样的脚本验证过，误差也在 $10^{-17}$ 量级。分块大小、SRAM 用量和 IO 复杂度留给 [FlashAttention V1 图解](../flash-attention/v1) 和 [V2 图解](../flash-attention/v2)。

## 检验一下 {#check}

::: details 思考：把 Block 改成 512，第二级规约就不会出 NaN 了。还需要修复吗？
需要。第二级确实不再出错，但这只是路径上的巧合。$N$ 小于 Block 时第一级仍有 NaN，比如 Block 为 1024、$N = 1000$。输入的第一个元素是 `-INFINITY` 时，串行递推的第一步也会出 NaN。Block 大小也不应该决定结果对不对。
:::

::: details 思考：为什么不在合并时加一个判断，比如 d 为 0 就跳过？
可以，但要改的地方更多。只改 `warpReduceOnline` 还不够，模拟显示，输入含 `-INFINITY` 时递推本身就会先产生 NaN，递推循环也要加判断。而换成 `-FLT_MAX` 只改初值，不增加任何比较，也不会产生新的分支。
:::

::: details 思考：d 为什么不会下溢到 0，让 1/d 溢出？
只要这段里有一个有效元素，最大的那个元素贡献的就是 $e^0 = 1$，所以 $d \ge 1$。修正时 $d$ 虽然可能被乘得很小，但新的最大值马上加进来一个 1。其他元素的贡献下溢成 0 也没关系，它们本来就比 1 小得多。
:::

::: details 思考：既然 N 不大时几种方案耗时相同，Online 还有什么用？
第一，它少一次规约，屏障从 4 道减到 2 道，$N$ 很小时规约开销占比更大。第二，$N$ 较大、缓存装不下时，访存节省才会兑现，这时它又不需要暂存整行。第三，它提供了分段合并的能力，这是 FlashAttention 能够成立的前提，而这一点和节省读取无关。
:::

::: details 思考：FlashAttention 第一块整块都被掩码时，会遇到本文的 NaN 吗？
如果初值是 $m = -\infty$，而这一块的最大值也是 $-\infty$，那 $e^{m_{\text{old}} - m_{\text{new}}}$ 同样是 NaN，$\exp(S - \tilde m)$ 也一样。Causal Mask 下每行的第 0 个位置都有效，第一块不会整块被掩码，所以通常碰不到。但左填充的 Padding Mask 可能让一行的前几块全被屏蔽。写实现时同样要留意初值和这种全掩码的块。
:::

::: details 思考：怎么确认自己写的数组留在了寄存器里？
编译时加 `--ptxas-options=-v`，看每个 Kernel 报告的 stack frame，或者用 `nvcc -ptx` 查看有没有 `.local` 声明以及 `ld.local`、`st.local` 指令。运行时可以用 `cudaFuncGetAttributes` 读取 `localSizeBytes` 和 `numRegs`。本文表格里的数字就是这样得到的。
:::

<p class="gpu-note">本文各版本的完整代码在仓库的 <code>code/softmax/online_softmax.cu</code>，每个版本都会校验输出。运行方法见 <code>code/README.md</code>。</p>

<p class="gpu-note">本文以 <a href="https://github.com/caomaolufei/AIInfraGuide" target="_blank" rel="noreferrer">AI Infra Guide</a> 的<a href="https://github.com/caomaolufei/AIInfraGuide/blob/main/docs/guides/%E6%A8%A1%E5%9D%97%E4%BA%8C-CUDA%E7%BC%96%E7%A8%8B%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/5.2-CUDA%20Online%20Softmax%E5%AE%9E%E7%8E%B0.md" target="_blank" rel="noreferrer">《5.2 CUDA Online Softmax 实现优化》</a>为蓝本重写，文中的“原稿”和“原始整理稿”都指这篇文章。感谢原作者的整理。</p>

## 参考资料 {#refs}

- [Online normalizer calculation for softmax - Milakov & Gimelshein, 2018](https://arxiv.org/abs/1805.02867)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [From Online Softmax to FlashAttention - UW CSE 599M](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [CUDA Programming Guide - C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)：Warp Shuffle 函数和越界行为
- [CUDA Programming Guide - Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)：寄存器和驻留线程上限
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)：第 10.2.4 节局部内存，第 11.1 节占用率
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)：A100 的 L1 和 L2 容量
- [深入浅出 GPU 优化系列：softmax 优化](https://zhuanlan.zhihu.com/p/341059988)
- [FlashAttention 核心原理详解](https://zhuanlan.zhihu.com/p/669926191)

<p class="gpu-note">A100 的带宽数字来自原始整理稿，没有重新测量。RTX 4060 Laptop 上的数字是本文实测，换一块 GPU 拐点会变，不能直接推到 A100 上。</p>
