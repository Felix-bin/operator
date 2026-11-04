---
title: GEMM 算子优化图解
description: 从朴素的矩阵乘法出发，逐级做 Block、Thread、Warp 分块，再消除 Bank 冲突、隐藏访存延迟，用图、可交互实验和真实 GPU 上的测量讲清每一步为什么更快。
pageClass: gpu-page
---

# GEMM 算子优化图解

<p class="gpu-lede">A100 的单精度峰值是 19.5 TFLOPS，可是朴素的矩阵乘法每读 1 字节只做 0.25 次浮点运算。即使 2039 GB/s 的显存全部跑满，也只能支撑 510 GFLOPS，只有峰值的 2.6%。乘法器大部分时间都在等数据。</p>

本文把同一个矩阵乘法 Kernel 一步步改写。主线只有一条：让每个字节的数据参与尽可能多的计算，然后让数据按时到达。每一步都对应一个具体瓶颈，并写清它在什么条件下成立。阅读前最好先看过[GPU 架构图解](../about_gpu/gpu)和[Roofline](../about_gpu/ai-compute#roofline)。本文也是[Reduce](../reduce/reduce)的延续，那一篇已经介绍过 Warp 分化和 Bank。

## GEMM 要算什么 {#basics}

GEMM（General Matrix Multiply）计算 $C = A \times B$，其中 $A$ 是 $M \times K$，$B$ 是 $K \times N$，$C$ 是 $M \times N$。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmDef /></div>
  <figcaption>每个输出元素是 A 的一行和 B 的一列的内积。A 的同一行要参与 N 个输出，B 的同一列要参与 M 个输出，这种复用是后面所有优化的基础。</figcaption>
</figure>

每个输出元素做 $K$ 次乘加，总运算量约为 $2MNK$ FLOPs，所以衡量 Kernel 的指标是

$$
\text{FLOP/s} = \frac{2MNK}{\text{Kernel 耗时}}
$$

全连接层、注意力里的 $QK^T$ 和 $\text{Attention} \cdot V$、通过 im2col 改写的卷积，底层都是矩阵乘法，所以 GEMM 的优化方法在深度学习里到处都能用上。

本文讨论 FP32 的 FMA 路径，也就是 A100 这 19.5 TFLOPS 对应的通用计算单元。[Tensor Core](../about_gpu/ai-compute#tensor-core)是另一条路径，算力和平衡点都不同，但分块和缓冲的思路一样。

### 瓶颈在搬运 {#bound}

判断瓶颈用算术强度，也就是每搬运 1 字节能做多少次浮点运算。把峰值算力除以峰值带宽，得到平衡点。在这个模型下，算术强度低于它，上限由带宽决定；高于它，带宽才不再是首要限制，能否真正跑满算力还要看其他瓶颈。

<div class="gpu-chain">
  <div><b>0.25</b><span>FLOP/byte，朴素写法的算术强度</span></div>
  <div><b>9.6</b><span>FLOP/byte，A100 的平衡点</span></div>
  <div><b>510 G</b><span>满带宽时能支撑的 FLOP/s</span></div>
  <div><b>2.6%</b><span>占 19.5 TFLOPS 的比例</span></div>
</div>

这个上限没算缓存，实际会高一些。但量级说明了问题：要让乘法器忙起来，算术强度要提高几十倍。

GPU 的数据要过两道关。显存到共享内存（smem）这一关，平衡点约 9.6。共享内存到寄存器这一关，Bank 决定了 smem 每拍最多给出 $32 \times 4 = 128$ 字节，而一个 A100 的 SM 每拍能做 64 次 FMA，也就是 128 FLOPs，所以平衡点约 1 FLOP/byte。下图是全文的地图，后面三节依次处理这条链上的每一步。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmChain /></div>
  <figcaption>三级搬运，数字按 BM = BN = 128、BK = 8、TM = TN = 8 算出。每一级都沿 K 方向走一小步，把搬来的数据用够了再搬下一批。</figcaption>
</figure>

### 测试条件 {#setup}

算账统一用 A100，真实测量用手头能拿到的另一块 GPU。两者的具体数字不同，本文会明确区分。

| 项目 | 算账用 | 实测用 |
| --- | --- | --- |
| GPU | A100 80GB SXM4 | GeForce RTX 4060 Laptop |
| Compute capability | 8.0 | 8.9 |
| SM 数量 | 108 | 24 |
| FP32 峰值 | 19.5 TFLOPS | 未标定 |
| 显存带宽 | 2039 GB/s | 未标定 |
| 每 SM 的共享内存 | 164 KB，单 Block 最多 163 KB | 未涉及 |
| 工具链 | 不适用 | CUDA 13.4，`nvcc -O2 -arch=sm_89` |

## 朴素实现 {#naive}

每个线程计算 $C$ 的一个元素，沿 $K$ 做内积。

```cpp
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

算一个元素做 $2K$ 次运算，读 $A$ 的一行和 $B$ 的一列共 $8K$ 字节，所以算术强度是 $2K / 8K = 0.25$。问题不在代码写得不好，而在于它完全没有利用复用：$A$ 的同一行被 $N$ 个线程各读一遍。0.25 是按每次读取都到达显存算的，缓存命中时实际的显存流量会更低。

## 第一级：Block Tiling {#block}

把 $C$ 切成 $BM \times BN$ 的小块，每个 Block 负责一块。计算这一块只需要 $A$ 的 $BM$ 行和 $B$ 的 $BN$ 列。不再让每个线程各读各的，而是先把它们搬进共享内存，整个 Block 共用。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmTiling /></div>
  <figcaption>示意图：琥珀色实心格是当前 K 步搬运的小段，浅色是同一行、同一列的其他 K 步。一次搬运 BM × BK 和 BK × BN，服务这个 Block 的全部 BM × BN 个输出。</figcaption>
</figure>

### 算术强度 {#block-ai}

每个 K 步做 $2 \cdot BM \cdot BN \cdot BK$ 次运算，搬运 $4 \cdot BK \cdot (BM + BN)$ 字节：

$$
\text{算术强度} = \frac{2 \cdot BM \cdot BN \cdot BK}{4 \cdot BK \cdot (BM + BN)} = \frac{BM \cdot BN}{2(BM + BN)}
$$

取 $BM = BN = 128$，得到 32 FLOP/byte，超过 A100 的平衡点。这说明它有机会不受显存带宽限制：按这个比例，满算力约需 609 GB/s，不到总带宽的三分之一。

这是访问计数模型的估算，不是实测。它只数了 $A$、$B$ 的读取，没算缓存命中、边界块和 $C$ 的写入。写入量约为读取量的 $64/K$，$K = 2048$ 时约 3%，$K = 64$ 时和读取一样多。它也不代表 Kernel 真能跑满算力，后面还有共享内存这一关。

$BK$ 被约掉了，这一点值得多看一眼。一个元素被读的次数只取决于 Block 怎么排：$A$ 的每个元素被每一列 Block 读一次，$B$ 的每个元素被每一行 Block 读一次。所以显存读取总量是

$$
4\left(\left\lceil \tfrac{N}{BN} \right\rceil MK + \left\lceil \tfrac{M}{BM} \right\rceil NK\right) \text{ 字节}
$$

和朴素写法的 $8MNK$ 相比，$BM = BN = 128$ 时正好是 1/128。降低访问量的是 $BM$ 和 $BN$，和 $BK$ 无关。

### 为什么还要沿 K 切 {#why-k}

既然 $BK$ 不影响访问量，为什么不一次把 $BM \times K$ 和 $K \times BN$ 全搬进来，乘完了事？因为放不下。$K = 10000$、$BM = BN = 128$ 时，这两块合计 10.24 MB，而 A100 一个 SM 的共享内存只有 164 KB。沿 K 切是为了放得下，代价是每个 K 步要多一次搬运和同步。

下面的实验用真实的算式替你算账。把 $BM$ 和 $BN$ 调得不一样，算术强度会下降，因为 $BM + BN$ 固定时，两者相等时 $\frac{1}{BM} + \frac{1}{BN}$ 最小。把 $BK$ 调大，读取量不变，只有共享内存占用在涨。再把 $BM$、$BN$ 调到 256，看看什么先不够用。

<GemmLab />

### 线程怎么排 {#load}

一个 Block 256 个线程，要把 1024 个 float 的 tileA 和 1024 个 float 的 tileB 搬进来，每个线程各搬 4 个。线程排布的目标是让一个 Warp 内的相邻线程读连续地址，这样请求容易合并。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmLoad /></div>
  <figcaption>由下方代码的循环直接算出：BM = BN = 128、BK = 8、Block 256 个线程。线程在三个阶段使用三种不同的坐标，tid 本身不变。tileC 里的小方块为了看清而放大了，实际每个元素只占其中一格。</figcaption>
</figure>

```cpp
template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_block_tiling(float* A, float* B, float* C,
                                   int M, int K, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;
    int tid = threadIdx.x;

    // 加载 tileA 时的线程重排
    constexpr int A_BLOCK_X = BK;  // = 8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;  // = 32
    int a_thread_x = tid % A_BLOCK_X;
    int a_thread_y = tid / A_BLOCK_X;

    // 加载 tileB 时的线程重排
    constexpr int B_BLOCK_X = 32;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;  // = 8
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;

    // 计算 tileC 时的线程排布（16×16）
    constexpr int C_BLOCK_X = 16;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;  // = 16
    int c_thread_x = tid % C_BLOCK_X;
    int c_thread_y = tid / C_BLOCK_X;

    // 每个线程负责 Tm×Tn 个输出元素
    constexpr int Tm = BM / C_BLOCK_Y;  // = 8
    constexpr int Tn = BN / C_BLOCK_X;  // = 8
    float Ct[Tm][Tn] = {0.0f};

    // K-Loop
    for (int k = 0; k < K; k += BK) {
        // 协作加载 tileA（使用跨步循环覆盖 BM 行）
        for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) {
            int r = r0 + i, c = k + a_thread_x;
            As[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }

        // 协作加载 tileB（使用跨步循环覆盖 BN 列）
        for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {
            int r = k + b_thread_y, c = c0 + j;
            Bs[b_thread_y][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }

        __syncthreads();

        // 外积方式计算 As × Bs
        #pragma unroll
        for (int p = 0; p < BK; p++) {
            for (int i = 0; i < Tm; i++) {
                int row = c_thread_y + i * C_BLOCK_Y;
                for (int j = 0; j < Tn; j++) {
                    int col = c_thread_x + j * C_BLOCK_X;
                    Ct[i][j] += As[row][p] * Bs[p][col];
                }
            }
        }

        __syncthreads();
    }

    // 写回结果
    for (int i = 0; i < Tm; i++) {
        int r = r0 + c_thread_y + i * C_BLOCK_Y;
        for (int j = 0; j < Tn; j++) {
            int c = c0 + c_thread_x + j * C_BLOCK_X;
            if (r < M && c < N) C[r * N + c] = Ct[i][j];
        }
    }
}
```

启动时 Grid 取 $\lceil N/BN \rceil \times \lceil M/BM \rceil$。越界的位置填 0，写回时跳过，所以 $M$、$N$、$K$ 不必整除分块大小。

两次 `__syncthreads()` 都不能省。第一次保证所有线程加载完才开始读，第二次保证所有线程读完才开始覆盖下一步的数据。这里只依赖屏障，不依赖同一 Warp 的车道同步前进，所以[Volta 之后的独立线程调度](../reduce/reduce#its)不影响它的正确性。后面的版本也都如此。

用这组参数和这段代码在 RTX 4060 上跑，结果和 CPU 参考值逐元素一致（含 $M$、$N$、$K$ 不整除的情况），吞吐从朴素版的 955 GFLOP/s 升到 4788 GFLOP/s。

### Block 怎么排：L2 命中率 {#wave}

同一时刻运行的 Block 集合称为一波（Wave）。A100 每 SM 驻留 2 个 Block 时，一波有 216 个 Block。它们读的 tile 有重叠，第一个读到的 Block 把数据带进 L2，后面的就能命中。

假设一波 Block 处于同一个 K 步，在 M、N 方向分别跨 $W_m$、$W_n$ 个 tile，且 L2 放得下。此时共有 $W_m W_n$ 对读取，不同的 A 块只有 $W_m$ 个，不同的 B 块只有 $W_n$ 个，所以

$$
\text{L2 命中率} \approx 1 - \frac{W_m \cdot BM + W_n \cdot BN}{W_m W_n (BM + BN)}
$$

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmWave /></div>
  <figcaption>数字由上式算出，和逐块统计的结果一致。A100 的 216 个 Block 排成 1 × 216 时命中率约 50%，2 × 108 时约 75%，12 × 18 时约 93%。</figcaption>
</figure>

可以看到波越接近方形，命中率越高。这是一个只算首次缺失的简化模型，没有考虑 Block 实际的启动时序和 L2 替换。CUDA 也不承诺 Block 的启动顺序，你能控制的是 `blockIdx` 到 tile 坐标的映射。

## 第二级：Thread Tiling {#thread}

Block Tiling 解决了显存这一关，下一个瓶颈是共享内存到寄存器。如果线程还是逐个元素累加，每做一次 FMA 就要读 $A$、$B$ 各一个 float，这一层的算术强度仍然是 0.25，低于约 1 FLOP/byte 的平衡点。也就是说，满算力时共享内存供不上数据，表现是大量的 MIO Throttle Stall。

解决方法和上一节同构：让每个线程负责一小块输出，把读出来的数据放在寄存器里反复用。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmOuter /></div>
  <figcaption>两者每个 p 步都是把 A 的一个切片和 B 的一个切片相乘，差别只在一个线程一次拿多少。琥珀色是从共享内存读的 float，青色是寄存器里的累加结果。</figcaption>
</figure>

算式和 Block 那一级形状完全相同，因为这就是把分块下沉到了寄存器层：

$$
\text{算术强度} = \frac{2 \cdot TM \cdot TN}{4(TM + TN)} = \frac{TM \cdot TN}{2(TM + TN)}
$$

$TM = TN = 8$ 时是 2 FLOP/byte，内积是 0.25，提升 8 倍。

### 8 × 8 是寄存器决定的 {#regs}

| 项目 | 寄存器数 |
| --- | --- |
| `c_frag` | 64 |
| `a_frag` + `b_frag` | 16 |
| 索引、循环变量等 | 约 30 |
| 合计 | 约 110，留余量按 128 算 |

按每个 SM 65536 个寄存器算，128 × 256 × 2 = 65536，正好放下 2 个 Block，也就是 16 个 Warp，占用率（occupancy）25%。这看起来很低，但 GEMM 每个线程有 64 条独立的累加链，指令级并行足够多，不需要靠更多 Warp 来隐藏延迟。实际编译下面这类 Kernel 时加上 `__launch_bounds__(256, 2)`，nvcc 用到的最高寄存器编号约 124，符合这个预算。

如果取 $TM = TN = 16$，`c_frag` 就要 256 个寄存器，超过每线程 255 的上限，只能溢出到栈上。

```cpp
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_thread_tiling(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 每个线程在 C 中负责 TM×TN 的子块
    int tid = threadIdx.x;
    int thread_row = (tid / (BN / TN)) * TM;
    int thread_col = (tid % (BN / TN)) * TN;

    // 寄存器存储
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};

    int by = blockIdx.y, bx = blockIdx.x;

    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 A、B 到 Shared Memory（含越界填 0，同上一节）
        load_tile_A(A, As, by, bk, tid, M, K);
        load_tile_B(B, Bs, bx, bk, tid, K, N);
        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++) {
            // 从 Shared Memory 加载到寄存器
            for (int i = 0; i < TM; i++)
                a_frag[i] = As[thread_row + i][k];
            for (int j = 0; j < TN; j++)
                b_frag[j] = Bs[k][thread_col + j];

            // 寄存器上做外积
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    c_frag[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    // 写回 Global Memory：边界检查不能省
    for (int i = 0; i < TM; i++) {
        int r = by * BM + thread_row + i;
        for (int j = 0; j < TN; j++) {
            int c = bx * BN + thread_col + j;
            if (r < M && c < N) C[r * N + c] = c_frag[i][j];
        }
    }
}
```

写回的边界检查很容易漏。$N$ 不整除 $BN$ 时，`c` 超过 $N$ 的元素会绕到下一行，把别的元素覆盖掉，这是静默的错误结果。用脚本模拟 $M = 128$、$N = 100$，不加检查时 12800 个结果错了 3136 个。

在 RTX 4060 上，这一步把吞吐从 4788 提到了 7078 GFLOP/s。

## 第三级：Warp Tiling {#warp}

再往上一层看：一个 Warp 的 32 个线程在 $M$、$N$ 方向怎么排？同一行的线程读同样的 $A$，同一列的线程读同样的 $B$。如果硬件能把同一个地址的读取合并（下一节的广播），读的不同元素就少了。

设 Warp 在 $M$ 方向有 $x$ 个线程、$N$ 方向有 $y$ 个线程，$x \cdot y = 32$。一个 p 步读 $x \cdot TM$ 个不同的 A 和 $y \cdot TN$ 个不同的 B，32 个线程共做 $32 \cdot TM \cdot TN$ 次 FMA，取 $TM = TN$ 时：

$$
\frac{32 \cdot TM^2 \cdot 2}{(x + y) \cdot TM \cdot 4} = \frac{16 \cdot TM}{x + y}
$$

$x + y$ 越小越好，整数解里 $4 \times 8$ 和 $8 \times 4$ 最优，$TM = 8$ 时约 10.7 FLOP/byte，而 $1 \times 32$ 只有 3.9。CUTLASS 通用配置的 `WarpShape<32, 64, 8>` 也是这个形状：每个 Warp 负责 32 × 64 的子块。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmWarp /></div>
  <figcaption>由下方代码算出。琥珀色格是 tid = 18，它在 Warp 0 的第 2 行第 2 列。</figcaption>
</figure>

```cpp
int warp_id = tid >> 5;           // tid / 32
int lane_id = tid & 31;           // tid % 32

// warp 在 Block 中的位置（4×2 排列）
int warp_row = warp_id >> 1;      // M 方向：0~3
int warp_col = warp_id & 1;       // N 方向：0~1

// lane 在 warp 内的位置（4×8 排列，行主序）
int lane_row = lane_id >> 3;      // M 方向：0~3
int lane_col = lane_id & 7;       // N 方向：0~7

// 线程负责的 TM×TN 子块在 Block Tile 中的起始行列
int row_c = (warp_row * 4 + lane_row) * TM;
int col_c = (warp_col * 8 + lane_col) * TN;
```

不过别急着下结论。这个算式只数了不同元素的个数，暗含一个前提：硬件会按 Warp 把相同地址全部合并。实测并不支持这个前提。只把排布从 $2 \times 16$（上一节的 16 × 16 线程网格，每个 Warp 恰好是 2 行 16 列）换成行主序的 $4 \times 8$，RTX 4060 上的吞吐反而从 7078 降到 6615 GFLOP/s。主要原因应该是此时 tileA 还没转置，$4 \times 8$ 的 4 行分别读 `As[row][k]`、`As[row + 8][k]`、`As[row + 16][k]`、`As[row + 24][k]`，地址相差 64 个 float，全部落在同一个 Bank，形成 4 路冲突。我们单独测过这条读取：换成每行 9 个 float 的间距后，耗时从 30.1 ns 降到 15.8 ns。下面两节就来补上这个前提。

## 向量化访存：float4 {#float4}

GPU 一条指令可以读写 128 位。从显存搬数据时，一个线程一次取 4 个 float，加载指令就少了 3/4。

```cpp
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

// 一次加载 4 个连续 float
float4 tmp = FLOAT4(A[row * K + col]);
```

### 对齐是硬性前提 {#align}

这个宏把 `float*` 强转成 `float4*`，编译器会认为地址按 16 字节对齐并生成 128 位访问。如果地址不对齐，就是运行时错误。我们在 RTX 4060 上试过，从 float 下标 1 处读 `float4`，`cudaDeviceSynchronize()` 直接返回 `misaligned address`。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmAlign /></div>
  <figcaption>每个方格是一个 float，竖线是 16 字节边界。第 1 行的起点是 K，所以 K 不是 4 的倍数时，很多行都不对齐。</figcaption>
</figure>

需要同时满足这些条件：

| 对象 | 条件 |
| --- | --- |
| `A[r * K + c]` | `K` 是 4 的倍数，`c` 是 4 的倍数 |
| `B[r * N + c]` | `N` 是 4 的倍数，`c` 是 4 的倍数 |
| `As`、`Bs` | 声明时加 `alignas(16)`，读取的下标是 4 的倍数 |
| `BK` | 是 4 的倍数 |
| 矩阵本身 | `cudaMalloc` 的基地址足够对齐，自己做偏移要小心 |

`K` 或 `N` 不是 4 的倍数时，可以按 4 的倍数的行距分配矩阵并在行尾补零，或者这一维退回逐个 float 加载。边界检查的粒度也要变成 float4，否则读取会越过矩阵末尾。

### 转置 A {#transpose}

$B$ 按行存，每个线程要读 `Bs[k][col..col+7]`，地址连续，可以直接用 float4。$A$ 不行：线程要的是一列，地址相隔 $BK$ 个 float。把 $A$ 转置后存进共享内存，就成了连续地址。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmTranspose /></div>
  <figcaption>每个方格是一个 float，琥珀色是线程要读的元素。转置不改变内容，只改变存放顺序，读取也就从 8 条指令变成 2 条。</figcaption>
</figure>

```cpp
// 从 Global Memory 加载 A 的一行（连续），转置存入 Shared Memory
float4 a_val = FLOAT4(A[global_row * K + bk + load_col]);
As[load_col + 0][local_row] = a_val.x;
As[load_col + 1][local_row] = a_val.y;
As[load_col + 2][local_row] = a_val.z;
As[load_col + 3][local_row] = a_val.w;

// 从 Shared Memory 读取时可以用 float4（连续地址）
float4 a_frag_vec = FLOAT4(As[k][thread_row]);
```

注意这里的代价：读显存是 1 条 128 位指令，写进 `As` 却要 4 条 32 位指令，因为这 4 个值落在 4 层。转置让读取省了指令，让加载多了指令，后面会看到这也带来了写冲突。

这一步在 RTX 4060 上把吞吐提到 8081 GFLOP/s。

## 消除 Bank 冲突 {#bank}

### 从 Bank 到 wavefront {#wavefront}

共享内存分成 32 个 Bank，每个宽 4 字节，地址（以 float 为单位）除以 32 的余数决定 Bank。在一次访问里，同一个 Bank 只能给出一个地址；不同线程读同一个地址不算冲突，读同一个 Bank 的不同地址才要排队。Reduce 页已经[讲过这部分](../reduce/reduce#bank)。

float4 又多了一层规则。一条 LDS.128 要返回 $32 \times 16 = 512$ 字节，而单次最多 128 字节，所以它被拆成 4 份，NVIDIA 现在把每一份叫 wavefront（旧称 transaction）。车道 0~7 一份，8~15 一份，依此类推，每份 8 条车道，所以叫 quarter-warp。Bank 冲突是在单个 wavefront 内部判定的：有冲突就要重放，多产生 wavefront。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmBankMap /></div>
  <figcaption>第一个 quarter-warp 的 8 条车道读 tileB，每条车道读一个 float4，方格内的数字是车道号。实心格是同一 Bank 落了两个地址的位置，上图因此需要 2 个 wavefront。</figcaption>
</figure>

还有一种情况能少发 wavefront。共享内存每拍返回 128 字节，也就是 1 个寄存器。NVIDIA 工程师在[论坛](https://forums.developer.nvidia.com/t/unexpected-shared-memory-bank-conflict/77228/2)上说明，如果所有活跃车道都满足 lane $i$ 与 lane $i \oplus 1$ 同地址，或都满足 lane $i$ 与 lane $i \oplus 2$ 同地址，返回值可以打包，LDS.128 的请求数从 4 降到 2。

这条说明明确写的是 compute capability 7.0 到 7.5。官方文档没有给 8.0 的规则，所以下面的结论在 A100 上只能算合理推断。为了不只停留在推断上，我们在 RTX 4060（8.9）上做了测量。

### 动手试一试 {#bank-lab}

先看默认的行主序 8 × 8：tileA 因为 4 个 quarter-warp 内都同地址，打包后只要 2 个 wavefront；tileB 每个 quarter-warp 都有 2 路冲突，要 8 个。再切到“拆成 4 × 4”和 Z-order，看看冲突和打包条件怎么变。

<GemmLabBank />

### 实测结果 {#bank-measure}

上面实验的规则不只来自论坛。我们按每个 SM 2 个 Block（16 个 Warp）启动，让它们反复执行 LDS.128，测量平均每轮耗时。结果和模型给出的 wavefront 数高度吻合，大约是 $7 \times \text{wavefront} + 2.6$ ns。摊到 16 个 Warp，一个 wavefront 约 0.44 ns。我们没有记录当时的频率，按 2~3 GHz 算是一两拍。

| 线程排布 | A 模型 | B 模型 | 合计 | A 实测 | B 实测 |
| --- | --- | --- | --- | --- | --- |
| 行主序，8 × 8 | 2 | 8 | 10 | 17.8 ns | 58.7 ns |
| 列主序，8 × 8 | 4 | 2 | 6 | 30.7 ns | 17.8 ns |
| 行主序，拆成 4 × 4 | 2 | 4 | 6 | 17.8 ns | 30.7 ns |
| Z-order，8 × 8 | 2 | 4 | 6 | 17.8 ns | 30.2 ns |
| Z-order，拆成 4 × 4 | 2 | 2 | 4 | 17.7 ns | 17.8 ns |

<p class="gpu-note">测量环境是 RTX 4060 Laptop，没有使用 Nsight Compute 的硬件计数器，所以 wavefront 数是由耗时反推的，不是计数器读数。一共测了 17 种组合，这里列出主要的 5 组，其余的结果和模型也一致。</p>

有一个细节只靠推理得不到。Z-order 8 × 8 的 tileB，每个 quarter-warp 内都没有冲突，也满足打包条件，按理应该只要 2 个 wavefront，可实测是 4 个。把规则改成“打包后两个 quarter-warp 合成一组，按 16 条车道判断冲突”，17 种组合才全部对得上。这组里车道 0~15 的起始下标是 0、8、…、56，跨了两层。这也解释了为什么 Z-order 必须配合 4 × 4 拆分才能降到 4 个。

### 办法一：把 8 × 8 拆成 4 × 4 {#split}

不要让相邻车道的 tileB 起点相差 8，而是相差 4，这样 quarter-warp 的 8 个 float4 正好铺满一层 Bank。代价是一个线程负责的不再是连续的 8 × 8，而是分散的 4 个 4 × 4。第二个 float4 跳到 64 个 float 之后，也就是 2 层之后，所以同样不冲突。写回坐标相应变为：

```cpp
// 每个线程的 8 行分成两组，每组 4 行连续，组间相差 64
for (int i = 0; i < Tm; i++) {
    int r = r0 + 4 * C_THREAD_Y + (i / 4) * 4 * C_BLOCK_Y + (i % 4);
    for (int j = 0; j < Tn; j++) {
        int c = c0 + 4 * C_THREAD_X + (j / 4) * 4 * C_BLOCK_X + (j % 4);
        if (r < M && c < N) C[r * N + c] = Ct[i][j];
    }
}
```

用脚本验证过，256 个线程的 64 个输出恰好不重不漏地覆盖 128 × 128。这时合计从 10 降到 6，RTX 4060 上吞吐到 8919 GFLOP/s。

### 办法二：Z-order 线程映射 {#zorder}

行主序下，quarter-warp 占一整行，tileA 上全是同地址（打包过于充分），tileB 上全不同（打不了包）。换一种映射，让 quarter-warp 变成 2 行 4 列的形状，两边就都能打包。maxas 的作者早年在 Maxwell 上就发现过这种[锯齿形排布](https://github.com/NervanaSystems/maxas/wiki/SGEMM)能消除“神秘的 Bank 冲突”。

```cpp
int lane_id = threadIdx.x % 32;
int lane_row = lane_id % 2 + (lane_id / 16) * 2;    // 4 行
int lane_col = (lane_id % 16) / 2;                  // 8 列
```

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmLayout /></div>
  <figcaption>Z-order 下，翻转 bit 0 只改变行，所以 lane i 与 lane i⊕1 读同一个 B 地址。翻转 bit 1 只改变列，所以 lane i 与 lane i⊕2 读同一个 A 地址。两边各满足一个打包条件。</figcaption>
</figure>

配合 4 × 4 拆分，tileA 和 tileB 都只要 2 个 wavefront，合计 4 个，比行主序的 10 个少 60%。RTX 4060 上吞吐到 9068 GFLOP/s。

### 还有一处冲突：写 tileA {#write-conflict}

读的问题解决了，再看加载阶段。转置之后，`As[k][row]` 的 Bank 是 `(k * 128 + row) % 32 = row % 32`，和 $k$ 无关。加载线程按 `row = tid / 2`、`k = (tid % 2) * 4` 排布时，同一行的两个线程写不同的层、同一个 Bank，这就是写冲突。

Padding 和 XOR 都是为这里准备的，它们的作用是让不同的层落到不同的 Bank。

```cpp
// 方案一：每层多留 4 个 float
__shared__ float As[BK][BM + 4];

// 方案二：不占额外空间，用 k 打乱 row
As[k][row ^ (k * 4)] = value;                  // 写入
float val = As[k][target_row ^ (k * 4)];       // 读取用同样的规则
```

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmSwizzle /></div>
  <figcaption>统计 Warp 0 执行第一条 As 写入时，k = 0 和 k = 4 两组线程用到的 Bank，实心格表示两组撞到同一个 Bank。由地址公式直接算出。</figcaption>
</figure>

两种方案都保持了 float4 对齐：$BM + 4 = 132$ 是 4 的倍数，每层起点仍然按 16 字节对齐；`k * 4` 不碰低 2 位，`row` 的低 2 位不变，连续 4 个 float 仍然连续。

它们对读取没有帮助，也没有害处。读取时所有线程的 $k$ 相同：Padding 只是整层平移；XOR 的掩码也相同，而 $r_1 \equiv r_2 \pmod{32}$ 当且仅当 $(r_1 \oplus m) \equiv (r_2 \oplus m) \pmod{32}$，所以哪些线程撞到同一个 Bank 完全不变。

实测情况要说清楚。在 RTX 4060 上，把场景放大成 4 个线程写同一个 Bank 的 4 路冲突，每轮从 29.6 ns 降到 15.6 ns，Padding 和 XOR 都有效，说明这个机制是真实的。但实际的 2 路场景测不出差异（15.8 ns 对 15.7~15.9 ns），因为 2 路冲突的两拍还没有超过这条写路径本身的开销。放到完整 Kernel 里，加了 XOR 的 Z-order 版本反而慢了约 2%（9068 对 8852 GFLOP/s）。所以这是一个有条件的优化，要在目标 GPU 上测过再决定。

::: details 思考：这些分析是否依赖 Warp 内严格同步？Volta 之后还成立吗？
要分开看正确性和性能。

正确性上，本文的 Kernel 只靠 `__syncthreads()` 保证顺序，不依赖车道的先后，A100 上不受独立线程调度影响，这一点和 [Reduce 的 V4](../reduce/reduce#its) 不同。

性能上，广播和打包都要求同一条指令的车道一起执行。只要 Warp 没有分化，实际上它们就是一起发射的，上面的测量也证实了这一点。但边界 Tile 里的 `(r < M && c < K) ? ... : 0.0f` 会造成分化，分化之后的车道不再保证一起发射，这时 wavefront 数可能变多。所以这些数字只描述没有越界的内部 Tile。
:::

## 隐藏访存延迟：双缓冲 {#buffer}

到这里，算术强度和带宽都够了，但时序上还有浪费：线程要等数据就绪才能开始计算。显存读取要几百拍，共享内存读取要几十拍，这段时间乘法器是空闲的。

算一笔账：A100 上一个 SM 每拍 64 次 FMA，驻留 2 个 Block，一个 K 步有 $128 \times 128 \times 8 = 131072$ 次 FMA，大约 4096 拍。如果这中间还要等几百拍的显存加两次屏障，就有约 10% 的时间在空转。

### 共享内存到寄存器：p-Loop 双缓冲 {#p-loop}

用两组寄存器，一组在算的同时，另一组在读下一步的数据：

```cpp
float regA[2][Tm], regB[2][Tn];

#pragma unroll
for (int p = 0; p < BK + 1; p++) {
    // 计算阶段：使用上一步读取的数据
    if (p > 0) {
        for (int i = 0; i < Tm; i++)
            for (int j = 0; j < Tn; j++)
                Ct[i][j] += regA[(p - 1) & 1][i] * regB[(p - 1) & 1][j];
    }

    // 预取阶段：读取本步数据到另一组寄存器
    if (p < BK) {
        for (int i = 0; i < Tm / 4; i++)
            FLOAT4(regA[p & 1][i * 4]) = FLOAT4(As[p][...]);
        for (int j = 0; j < Tn / 4; j++)
            FLOAT4(regB[p & 1][j * 4]) = FLOAT4(Bs[p][...]);
    }
}
```

一个 Warp 的 LDS 发射后不必等结果，只要后面的 FFMA 不依赖这个结果，就可以继续发射。所以这段代码让 FFMA 把 LDS 的延迟盖住了。

### 显存到共享内存：K-Loop 双缓冲 {#k-loop}

同样的道理，共享内存也放两份。先发射 LDG，LDG 在途时继续用当前 buffer 计算，算完把数据写进另一个 buffer，然后交换。

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmBuffer /></div>
  <figcaption>示意图，不按比例。虚线框是 LDG 已发射、数据还没回来的时间。双缓冲下这段时间线程在做 FMA，而且每个 K 步只需要一次屏障。</figcaption>
</figure>

```cpp
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_double_buffer(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    alignas(16) __shared__ float As[2][BK][BM];
    alignas(16) __shared__ float Bs[2][BK][BN];

    float a_frag[2][TM], b_frag[2][TN];
    float c_frag[TM][TN] = {0.0f};
    float ldg_a[4], ldg_b[4];  // 暂存 Global Memory 加载结果

    int buf_idx = 0;

    // Prologue: 加载第一个 tile
    load_global_to_shared(A, B, As[0], Bs[0], /*tile_k=*/0);
    __syncthreads();

    // 预取第一组寄存器数据
    load_shared_to_reg(As[0], Bs[0], a_frag[0], b_frag[0], /*k=*/0);

    for (int tile_k = BK; tile_k < K + BK; tile_k += BK) {
        int next_buf = 1 - buf_idx;

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // 预取下一个 k 步的寄存器数据
            if (k + 1 < BK) {
                load_shared_to_reg(As[buf_idx], Bs[buf_idx],
                                   a_frag[1 - k % 2], b_frag[1 - k % 2], k + 1);
            }

            // 在第一个 k 步时，发射下一 tile 的 Global Memory 加载
            if (k == 0 && tile_k < K) {
                issue_ldg(A, B, ldg_a, ldg_b, tile_k);
            }

            // 执行 TM×TN 的外积计算
            outer_product(a_frag[k % 2], b_frag[k % 2], c_frag);
        }

        // 将 LDG 结果写入下一 buffer
        if (tile_k < K) {
            store_reg_to_shared(ldg_a, ldg_b, As[next_buf], Bs[next_buf]);
        }
        __syncthreads();

        buf_idx = next_buf;
        load_shared_to_reg(As[buf_idx], Bs[buf_idx], a_frag[0], b_frag[0], 0);
    }

    // 写回结果
    store_c(c_frag, C);
}
```

用脚本走过这段循环的下标，在 $K = 8$、40、64 等情况下，每个 $k$ 都恰好按顺序算到一次。$K$ 不是 $BK$ 的倍数时，多出来的步数读到的是加载时填的 0。这里每线程暂存 4 个 float，正好是 1024 个元素除以 256 个线程，两份加起来是 8 个寄存器。共享内存的占用也翻倍：每 Block 16 KB。

一次屏障就够，是因为线程写入 `next_buf` 时，没有人在读它；线程读 `buf_idx` 时，也没有人在写它。屏障之后两者才交换。

把上面的结构补全后在 RTX 4060 上测（同时带 XOR），吞吐从 8852 升到 9061 GFLOP/s，只提升约 2%。这比原稿给出的约 10% 小得多，两边的 GPU 不同，我们也没有进一步分析原因。这一步的收益需要在目标 GPU 上单独确认。

## Ampere 的异步拷贝 {#async}

K-Loop 双缓冲还有一处不够好：数据要经过寄存器中转，每个线程占 8 个寄存器，还多了 STS 指令。Compute capability 8.0 引入了 `cp.async`，直接把显存异步拷到共享内存：

```text
cp.async.cg.shared.global [smem_ptr], [gmem_ptr], 16;
cp.async.commit_group;
cp.async.wait_group N;   // 最多还有 N 组未完成
```

拷贝大小只能是 4、8、16 字节，源和目的都要按大小对齐，CUDA 头文件 `cuda_pipeline_primitives.h` 里的断言就是这么写的。`wait_group` 只等当前线程自己的拷贝，所以后面还要跟一次 `__syncthreads()`，才能保证其他线程的拷贝也完成了。

拷贝不经过寄存器，也就不能在途中转置。解决办法是把 $A$ 改成列主序存放，这时 `A[k * M + r]` 连续，直接拷进 `As[k][r]` 就是转置后的布局。加载也不再需要 STS，一个 Warp 写的是同一层的连续地址，上一节的写冲突也就不存在了。代价是调用方要提供列主序的 $A$，并且 $M$ 是 4 的倍数。

多份缓冲可以轮换使用，这就是流水线：

<figure class="gpu-fig">
  <div class="gpu-scroll" tabindex="0"><FigGemmPipe /></div>
  <figcaption>3 级流水线前 4 轮的槽位状态，由下方代码的下标算出。级数越多，能隐藏的延迟越长，占用的共享内存也越多。</figcaption>
</figure>

下面是完整的 Kernel，这段代码原样在 RTX 4060 上和 CPU 结果逐元素比对过，包括 $M$、$N$、$K$ 不整除分块大小的情况。

```cpp
#include <cuda_pipeline.h>

#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

// A 列主序（M 是 4 的倍数），B 行主序（N 是 4 的倍数），Block 为 256 个线程
template <int S>
__global__ __launch_bounds__(256, 2)
void sgemm_async(const float* A, const float* B, float* C, int M, int N, int K) {
    alignas(16) __shared__ float As[S][BK][BM];
    alignas(16) __shared__ float Bs[S][BK][BN];

    const int tid = threadIdx.x;
    const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;

    // Warp 按 4×2 排列，车道按 Z-order 排列
    const int warp = tid >> 5, lane = tid & 31;
    const int ty = (warp >> 1) * 4 + (lane & 1) + ((lane >> 4) << 1);
    const int tx = (warp & 1) * 8 + ((lane & 15) >> 1);
    const int row0 = ty * 4, col0 = tx * 4;      // 拆成 4 个 4×4

    // 加载：每个线程拷 A、B 各一个 float4
    const int ck = tid >> 5, ce = (tid & 31) * 4;
    const int n_tile = (K + BK - 1) / BK;

    auto issue = [&](int t) {
        if (t >= n_tile) return;
        const int buf = t % S, k = t * BK + ck;
        const int r = r0 + ce, c = c0 + ce;
        if (k < K && r < M) __pipeline_memcpy_async(&As[buf][ck][ce], &A[k * M + r], 16);
        else FLOAT4(As[buf][ck][ce]) = make_float4(0, 0, 0, 0);
        if (k < K && c < N) __pipeline_memcpy_async(&Bs[buf][ck][ce], &B[k * N + c], 16);
        else FLOAT4(Bs[buf][ck][ce]) = make_float4(0, 0, 0, 0);
    };

    float c_frag[TM][TN] = {0.0f};

    // 提前发射 S - 1 组
    for (int s = 0; s < S - 1; s++) {
        issue(s);
        __pipeline_commit();
    }

    for (int t = 0; t < n_tile; t++) {
        // 等第 t 组到齐；屏障同时保证上一轮的槽位没有人再读
        __pipeline_wait_prior(S - 2);
        __syncthreads();

        issue(t + S - 1);        // 写入的正是上一轮算完的槽位
        __pipeline_commit();

        const int buf = t % S;
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float4 a0 = FLOAT4(As[buf][k][row0]), a1 = FLOAT4(As[buf][k][row0 + 64]);
            float4 b0 = FLOAT4(Bs[buf][k][col0]), b1 = FLOAT4(Bs[buf][k][col0 + 64]);
            float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    c_frag[i][j] += a[i] * b[j];
        }
    }

    // 写回
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int r = r0 + row0 + (i >> 2) * 64 + (i & 3);
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int c = c0 + col0 + (j >> 2) * 64 + (j & 3);
            if (r < M && c < N) C[r * N + c] = c_frag[i][j];
        }
    }
}
```

屏障的位置很关键。如果写成“先发射下一组、再等待和屏障”，写入的槽位可能还有慢线程在读，就是数据竞争。这里先过屏障再发射，所以每轮一次屏障就够了。

这个版本在 RTX 4060 上是最快的：2 级 9808、3 级 9883、4 级 10109 GFLOP/s。级数之间的差别已经接近测量波动，不能说明级数越多越好。

### 高层接口 {#pipeline-api}

CUDA 11 起提供了 C++ 封装：

```cpp
#include <cuda/pipeline>
#include <cooperative_groups.h>

auto group = cooperative_groups::this_thread_block();
__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;
auto pipe = cuda::make_pipeline(group, &pipe_state);

// 发射异步拷贝
pipe.producer_acquire();
cuda::memcpy_async(group, smem_ptr, gmem_ptr, sizeof(float) * BK * BM, pipe);
pipe.producer_commit();

// 等待数据就绪
pipe.consumer_wait();
// ... 计算 ...
pipe.consumer_release();
```

这段代码能编译，但要注意 `memcpy_async` 拷的是一段连续内存。矩阵的一个 tile 在显存里是分散的（行距是 $K$ 或 $N$），不能直接这样拷；要么这一维本身连续，要么像上面那样逐个 float4 拷。

### CUTLASS 与 Hopper {#cutlass}

NVIDIA 的 [CUTLASS](https://github.com/NVIDIA/cutlass) 把这些手法做成了模板。它的[默认配置](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/device/default_gemm_configuration.h)可以和本文对照：通用 SIMT 路径的 Block 形状是 `128×128×8`、Warp 形状是 `32×64×8`、缓冲 2 级，和本文的参数完全一致。Ampere 的 Tensor Core 路径是 `128×256×64`、缓冲 3 级，按 FP16 算，一个 Block 的 smem 约 147 KB。

Hopper（compute capability 9.0）又往前走了一步：TMA 硬件负责多维张量的地址计算和搬运，`wgmma.async` 让 Warp Group 直接对共享内存做矩阵乘。数据搬运和计算进一步解耦，但目标没变。

## 手写 SASS：寄存器 Bank {#sass}

再往下是指令级优化，这一层基本都来自逆向。maxas 作者在 Maxwell 上发现，寄存器文件也分 Bank，映射是[寄存器编号模 4](https://github.com/NervanaSystems/maxas/wiki/SGEMM)。一条 FFMA 的多个源操作数落在同一个 Bank 时会多等一拍，所以手写时要安排寄存器号，还可以用 `.reuse` 标记把操作数留在缓存里复用。NVIDIA 没有公开这些细节，Maxwell 之后的代际是否完全相同，本文没有查到依据。

```text
// 目标：FFMA R_c, R_a, R_b, R_c
// 确保 R_a, R_b, R_c 的 Bank 尽量不同
// 策略：按照 c_frag 的访问顺序分配寄存器号
```

另一个手法是把 LDS 和 FFMA 交错排列，用 FFMA 盖住 LDS 的延迟，这和 p-Loop 双缓冲的原理相同：

```text
LDS R0, [R_addr]        // 发射 smem 读取
FFMA R10, R4, R5, R10   // 等待时做不依赖 R0 的计算
FFMA R11, R4, R6, R11
FFMA R12, R4, R7, R12
...                     // 足够多的独立 FFMA 才能盖住 LDS 的延迟
FFMA R14, R0, R5, R14   // 此时 R0 已就绪
```

maxas 在 Maxwell 上用这些手法把效率做到了峰值的 96%~98%。工具上可以用 maxas 或 [CuAssembler](https://github.com/cloudcores/CuAssembler)。除非在追求最后几个百分点，否则不建议从这里开始。

## 调参与写法 {#tuning}

### 推荐参数 {#params}

| 参数 | 推荐值 | 理由 |
| --- | --- | --- |
| BM, BN | 128 | 算术强度 32，按模型有机会不受显存带宽限制 |
| BK | 8 | 2 级缓冲每 Block 16 KB，每 SM 32 KB，放得下 |
| TM, TN | 8 | 每线程约 110 个寄存器，放得下 2 个 Block |
| Block 大小 | 256 | (128/8) × (128/8) = 16 × 16 |
| Warp 形状 | 4 × 8，Z-order | 配合 4 × 4 拆分，每次 4 个 wavefront |
| 寄存器 | `__launch_bounds__(256, 2)` | 让编译器按每 SM 2 个 Block 分配 |

### 写法上的坑 {#pitfalls}

下面这些是用 nvcc 13.4 编译后看 SASS 确认的，不是推测。

除以 2 的幂的常量不用手动改位运算。`threadIdx.x / 8` 和 `threadIdx.x >> 3` 生成的指令数相同（都是 11 条）。真正昂贵的是运行时才知道的除数，`threadIdx.x / d` 生成 29 条指令。所以要把分块参数做成模板参数或 `constexpr`，这才是省指令的关键：

```cpp
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // BM/TM, BN/TN 等在编译期求值
    constexpr int THREAD_X = BN / TN;
    constexpr int THREAD_Y = BM / TM;
    ...
}
```

`#pragma unroll` 要求循环次数是常量。`for (int i = start; i < start + TM; i++)` 没有问题，起点是运行时的值也能展开，生成的代码和从 0 开始的写法同样长。真正展不开的是次数依赖运行时值的循环，比如本文第一个 Kernel 的 `for (int i = a_thread_y; i < BM; i += 32)`：虽然实际总是 4 次，但编译器算不出来，生成的代码里留着分支。改成计数循环就好：

```cpp
#pragma unroll
for (int it = 0; it < BM / A_BLOCK_Y; it++) {
    int i = a_thread_y + it * A_BLOCK_Y;
    ...
}
```

只跑 2 次的内层循环可以手动展开，`BK = 8` 时用 float4 加载一行正好 2 次：

```cpp
tileA[...] = ldg_ptr[0];
tileA[...] = ldg_ptr[1];
```

还有几条经验：

- 用 [Nsight Compute](https://docs.nvidia.com/nsight-compute/) 定位瓶颈：`Stall Long Scoreboard` 对应显存延迟没藏住，`Stall MIO Throttle` 对应共享内存指令压力过大，`Stall Short Scoreboard` 对应共享内存延迟，`Stall Barrier` 对应同步等待。
- 不要追求高占用率，GEMM 靠的是指令级并行。但 Block 数不要少于 SM 数，否则有 SM 闲着。
- 记得处理 $K$ 不整除 $BK$ 的边界。

## 全部放在一起 {#summary}

下图是本页实际跑出来的结果：M = N = K = 2048，每种取 9 次中最快的一次，重复两轮的差异都在 2.5% 以内。

<div class="gpu-scroll" tabindex="0"><div class="gemm-bars">

<GpuBars
  label="各阶段在 RTX 4060 Laptop 上的吞吐"
  unit="GFLOP/s"
  :ticks="[0, 3000, 6000, 9000, 12000]"
  :rows="[
    { label: '朴素', value: 955, note: '每线程一个元素' },
    { label: 'Block Tiling', value: 4788, note: 'smem 复用' },
    { label: 'Thread Tiling', value: 7078, note: '寄存器外积' },
    { label: 'Warp 4×8 行主序', value: 6615, note: '未转置，读 A 时 4 路冲突' },
    { label: 'float4 + 转置 A', value: 8081, note: '10 个 wavefront' },
    { label: '拆成 4×4', value: 8919, note: '6 个 wavefront' },
    { label: 'Z-order + 4×4', value: 9068, note: '4 个 wavefront' },
    { label: '再加 XOR', value: 8852, note: '2 路写冲突，这里测不出收益' },
    { label: '再加双缓冲', value: 9061, note: 'LDG + STS，单步一次屏障' },
    { label: 'cp.async 2 级', value: 9808, note: '不经过寄存器中转' },
    { label: 'cp.async 4 级', value: 10109, note: '每 Block 32 KB' }
  ]"
/>

</div></div>

<p class="gpu-note">测量平台是笔记本 GPU，受功耗和频率影响，只能看相对关系，不代表 A100 的数字。这组 Kernel 没有做到手写 SASS 的程度，也没有和 cuBLAS 比较。</p>

原稿给出了另一组相对 cuBLAS 的性能，我们没有重新测量，也查不到出处，仅作参考：朴素 6%~11%，Shared Memory Tiling 25%~40%，Thread Tiling 50%~65%，float4 60%~75%，消除 Bank 冲突 70%~80%，Warp Tiling 80%~87%，双缓冲 90%~97%，手写 SASS 95%~99%。和我们的结果相比，最明显的差别在于双缓冲的收益。

每一步对应一个具体的瓶颈：

| 瓶颈 | 表现 | 做法 | 效果 |
| --- | --- | --- | --- |
| 显存带宽 | 算术强度 0.25 | Block Tiling | 提到 32，显存读取减为 1/128 |
| 共享内存吞吐 | 每 FMA 读两次 | Thread Tiling | 单线程提到 2 FLOP/byte |
| 指令数 | 逐个 float 搬运 | float4，转置 A | 读取指令减为 1/4，有对齐要求 |
| 读 Bank 冲突 | wavefront 变多 | 4 × 4 拆分，Z-order | 从 10 个到 4 个 |
| 写 Bank 冲突 | 加载阶段排队 | Padding 或 XOR | 机制确实存在，收益看平台 |
| 访存延迟 | 等数据、两次屏障 | 双缓冲，cp.async | 边算边运 |

这些思路不限于 GEMM。后面的 [Softmax](../softmax/softmax) 和 [FlashAttention](../flash-attention/v1) 同样会遇到分块、共享内存和同步的问题。先算清楚算术强度，再逐级把数据留在更近的地方，这个套路都能用上。

## 检验一下 {#check}

::: details 思考：把 BK 从 8 改成 16，算术强度和显存读取量怎么变？共享内存呢？
算术强度和显存读取量都不变，$BK$ 已经从算式里约掉了。变的是共享内存占用翻倍，每个 Block 从 8 KB 变成 16 KB（按 1 级缓冲算），以及屏障次数减半。所以 $BK$ 是用共享内存换同步次数。在上面的实验里调一下就能看到。
:::

::: details 思考：朴素写法的算术强度是 0.25，Thread Tiling 也是从 0.25 提到 2。它们是一回事吗？
不是。数字相同只是因为都是“读 2 个 float，做 1 次 FMA”，但它们描述的是两层。朴素写法的 0.25 对的是显存，平衡点约 9.6。Block Tiling 把显存这层提到 32 之后，共享内存这层还是读 2 个 float 做 1 次 FMA，算术强度 0.25，平衡点约 1。Thread Tiling 解决的是这一层。分层来看，每一层都有自己的算术强度。
:::

::: details 思考：Warp 用 2 × 16 还是 4 × 8？能只看 x + y 吗？
不能。$x + y$ 只数了不同元素的个数。硬件的代价按 wavefront 算，一条 LDS.128 是 4 个，只有同一组车道内才可能广播，打包也只对 lane i⊕1 或 i⊕2 生效。上面的表里，行主序的 $4 \times 8$ 和 $2 \times 16$ 合计都是 10 个 wavefront。真正把数字降下来的是 Z-order 和 4 × 4 拆分，这说明车道的编号方式和形状同样重要。
:::

::: details 思考：BM 和 BN 都取 256，TM = TN = 8，A100 上放得下吗？
放不下。Block 需要 $32 \times 32 = 1024$ 个线程，每线程约 110 个寄存器，一个 Block 就要 112640 个，超过每 SM 65536 个。共享内存倒是还够。这种 Tile 只有增大 $TM$、$TN$ 才行，但 $TM = TN = 16$ 时每线程又要约 318 个寄存器，超过 255 的上限。这就是 128 这个数字的由来之一。
:::

::: details 思考：把 A 换成列主序存放，float4 的对齐条件变成什么？
地址变成 `A[k * M + r]`，所以要求 $M$ 是 4 的倍数、`r` 是 4 的倍数。这正好和 $K$ 不再有关系。如果调用方能提供这种布局，$K$ 就可以是任意值，多出来的步数加载时填 0 即可。
:::

<p class="gpu-note">本文各阶段的完整代码在仓库的 <code>code/gemm/gemm.cu</code>，每个版本都会校验输出。运行方法见 <code>code/README.md</code>。</p>

<p class="gpu-note">本文以 <a href="https://github.com/caomaolufei/AIInfraGuide" target="_blank" rel="noreferrer">AI Infra Guide</a> 的<a href="https://github.com/caomaolufei/AIInfraGuide/blob/main/docs/guides/%E6%A8%A1%E5%9D%97%E4%BA%8C-CUDA%E7%BC%96%E7%A8%8B%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/4.1-CUDA%20GEMM%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96.md" target="_blank" rel="noreferrer">《4.1 CUDA GEMM 算子性能优化》</a>为蓝本重写，文中的“原稿”和“原始整理稿”都指这篇文章。感谢原作者的整理。</p>

## 参考资料 {#refs}

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance - Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM)
- [maxas Wiki: SGEMM - Scott Gray](https://github.com/NervanaSystems/maxas/wiki/SGEMM)：Maxwell 上的手写 SGEMM，寄存器 Bank 和锯齿形排布
- [NVIDIA 论坛：Unexpected shared memory bank conflict](https://forums.developer.nvidia.com/t/unexpected-shared-memory-bank-conflict/77228/2)：CC 7.0~7.5 上 LDS.128 的打包条件
- [NVIDIA 论坛：How to understand the bank conflict of shared memory](https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900)：32 个 Bank、按 wavefront 判定、冲突导致重放
- [CUDA Programming Guide - Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)：独立线程调度、异步拷贝、超过 48 KB 的共享内存
- [CUDA Programming Guide - Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)：共享内存、寄存器和 Warp 的上限
- [CUTLASS](https://github.com/NVIDIA/cutlass) 和 [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [Hands-on-GEMM](https://github.com/AyakaGEMM/Hands-on-GEMM)、[Optimizing-SGEMM-on-NVIDIA-Turing-GPUs](https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs)：可运行的教学实现
- [CUDA Matrix Multiply Optimization - Lei Mao](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- [Volta Architecture and Performance Optimization (GTC 2018)](https://on-demand.gputechconf.com/gtc/2018/presentation/s81006-volta-architecture-and-performance-optimization.pdf)
- [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/)

<p class="gpu-note">本页的 RTX 4060 数字来自为这一页编写的测试程序，由耗时而非硬件计数器得出。Bank 按默认的 32 位模式讨论，打包规则在 A100（8.0）上没有官方依据，也没有实测，属于推断。</p>
