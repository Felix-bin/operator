// GEMM 算子各阶段，讲解见 gemm/gemm.md。计算 C = A * B，A 是 M×K，B 是 K×N，C 是 M×N，都是行主序。
// async 版本例外：A 用列主序，行距为 M，所以要求 M 是 4 的倍数。
#include "harness.h"

#include <cuda_pipeline.h>

#define FLOAT4(p) (reinterpret_cast<float4*>(&(p))[0])
#define CFLOAT4(p) (reinterpret_cast<const float4*>(&(p))[0])

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8, BS = 256;

// 朴素实现：一个线程算一个元素
__global__ void sgemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// 第一级：Block Tiling。先把 tileA、tileB 搬进共享内存，整个 Block 共用
__global__ void sgemm_block(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN, tid = threadIdx.x;

    // 加载 tileA、tileB 和计算 C 时，线程各用一种排布
    constexpr int A_BLOCK_X = BK, A_BLOCK_Y = BS / A_BLOCK_X;
    int a_thread_x = tid % A_BLOCK_X, a_thread_y = tid / A_BLOCK_X;
    constexpr int B_BLOCK_X = 32;
    int b_thread_x = tid % B_BLOCK_X, b_thread_y = tid / B_BLOCK_X;
    constexpr int C_BLOCK_X = 16, C_BLOCK_Y = BS / C_BLOCK_X;
    int c_thread_x = tid % C_BLOCK_X, c_thread_y = tid / C_BLOCK_X;
    constexpr int Tm = BM / C_BLOCK_Y, Tn = BN / C_BLOCK_X;

    float Ct[Tm][Tn] = {0.0f};
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) {
            int r = r0 + i, c = k + a_thread_x;
            As[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
        #pragma unroll
        for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {
            int r = k + b_thread_y, c = c0 + j;
            Bs[b_thread_y][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int p = 0; p < BK; p++)
            for (int i = 0; i < Tm; i++) {
                int row = c_thread_y + i * C_BLOCK_Y;
                for (int j = 0; j < Tn; j++) {
                    int col = c_thread_x + j * C_BLOCK_X;
                    Ct[i][j] += As[row][p] * Bs[p][col];
                }
            }
        __syncthreads();
    }

    // 写回时要做边界检查：N 不能被 BN 整除时，最后一块的列会越界
    for (int i = 0; i < Tm; i++) {
        int r = r0 + c_thread_y + i * C_BLOCK_Y;
        for (int j = 0; j < Tn; j++) {
            int c = c0 + c_thread_x + j * C_BLOCK_X;
            if (r < M && c < N) C[r * N + c] = Ct[i][j];
        }
    }
}

// 第二级 Thread Tiling（WARP = 0）和第三级 Warp Tiling（WARP = 1，一个 Warp 占 4×8 个线程）
template <int WARP>
__global__ __launch_bounds__(256, 2) void sgemm_thread(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.x, by = blockIdx.y, bx = blockIdx.x;
    int thread_row, thread_col;
    if (WARP) {
        int w = tid >> 5, l = tid & 31;
        thread_row = ((w >> 1) * 4 + (l >> 3)) * TM;
        thread_col = ((w & 1) * 8 + (l & 7)) * TN;
    } else {
        thread_row = (tid / (BN / TN)) * TM;
        thread_col = (tid % (BN / TN)) * TN;
    }

    float a_frag[TM], b_frag[TN];
    float c_frag[TM][TN] = {0.0f};
    int ax = tid % BK, ay = tid / BK, bbx = tid % 32, bby = tid / 32;

    for (int bk = 0; bk < K; bk += BK) {
        #pragma unroll
        for (int it = 0; it < BM * BK / BS; it++) {
            int i = ay + it * (BS / BK), r = by * BM + i, c = bk + ax;
            As[i][ax] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
        #pragma unroll
        for (int it = 0; it < BN / 32; it++) {
            int j = bbx + it * 32, r = bk + bby, c = bx * BN + j;
            Bs[bby][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) a_frag[i] = As[thread_row + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b_frag[j] = Bs[k][thread_col + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) c_frag[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int r = by * BM + thread_row + i, c = bx * BN + thread_col + j;
            if (r < M && c < N) C[r * N + c] = c_frag[i][j];
        }
}

// float4 系列，要求 K、N 都是 4 的倍数。
// Z：0 为行主序车道，1 为 Z-order 车道。STEP：8 为连续 8×8，4 为拆成 4 个 4×4。
// SW：共享内存里的 tileA 做 XOR Swizzle。DB：K-Loop 双缓冲。
template <int Z, int STEP, int SW, int DB>
__global__ __launch_bounds__(256, 2) void sgemm_f4(const float* A, const float* B, float* C, int M, int N, int K) {
    alignas(16) __shared__ float As[1 + DB][BK][BM];
    alignas(16) __shared__ float Bs[1 + DB][BK][BN];

    const int tid = threadIdx.x;
    const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
    const int warp = tid >> 5, lane = tid & 31;
    const int lane_row = Z ? (lane & 1) + ((lane >> 4) << 1) : lane >> 3;
    const int lane_col = Z ? (lane & 15) >> 1 : lane & 7;
    const int ty = (warp >> 1) * 4 + lane_row;
    const int tx = (warp & 1) * 8 + lane_col;
    constexpr int d = STEP == 8 ? 4 : 64;
    const int row0 = ty * STEP, col0 = tx * STEP;

    // 加载时每个线程搬 tileA、tileB 各一个 float4
    const int a_r = tid >> 1, a_c = (tid & 1) * 4;
    const int b_r = tid >> 5, b_c = (tid & 31) * 4;

    auto swz = [](int row, int k) { return SW ? (row ^ (k * 4)) : row; };

    auto issue = [&](int bk, float4& la, float4& lb) {
        int r = r0 + a_r, c = bk + a_c;
        la = (r < M && c < K) ? CFLOAT4(A[r * K + c]) : make_float4(0, 0, 0, 0);
        int rb = bk + b_r, cb = c0 + b_c;
        lb = (rb < K && cb < N) ? CFLOAT4(B[rb * N + cb]) : make_float4(0, 0, 0, 0);
    };
    auto store = [&](int buf, const float4& la, const float4& lb) {
        As[buf][a_c + 0][swz(a_r, a_c + 0)] = la.x;
        As[buf][a_c + 1][swz(a_r, a_c + 1)] = la.y;
        As[buf][a_c + 2][swz(a_r, a_c + 2)] = la.z;
        As[buf][a_c + 3][swz(a_r, a_c + 3)] = la.w;
        FLOAT4(Bs[buf][b_r][b_c]) = lb;
    };

    float c_frag[TM][TN] = {0.0f};
    float4 ra[2][2], rb[2][2];
    auto fetch = [&](int buf, int k, int slot) {
        ra[slot][0] = FLOAT4(As[buf][k][swz(row0, k)]);
        ra[slot][1] = FLOAT4(As[buf][k][swz(row0 + d, k)]);
        rb[slot][0] = FLOAT4(Bs[buf][k][col0]);
        rb[slot][1] = FLOAT4(Bs[buf][k][col0 + d]);
    };
    auto outer = [&](int slot) {
        float a[8] = {ra[slot][0].x, ra[slot][0].y, ra[slot][0].z, ra[slot][0].w,
                      ra[slot][1].x, ra[slot][1].y, ra[slot][1].z, ra[slot][1].w};
        float b[8] = {rb[slot][0].x, rb[slot][0].y, rb[slot][0].z, rb[slot][0].w,
                      rb[slot][1].x, rb[slot][1].y, rb[slot][1].z, rb[slot][1].w};
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++) c_frag[i][j] += a[i] * b[j];
    };

    if (!DB) {
        for (int bk = 0; bk < K; bk += BK) {
            float4 la, lb;
            issue(bk, la, lb);
            store(0, la, lb);
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                fetch(0, k, 0);
                outer(0);
            }
            __syncthreads();
        }
    } else {
        // 下一块的读请求先发出去，计算当前块的同时等数据回来
        float4 la, lb;
        issue(0, la, lb);
        store(0, la, lb);
        __syncthreads();
        int buf = 0;
        for (int bk = BK; bk < K + BK; bk += BK) {
            const int nxt = DB - buf;
            const bool more = bk < K;
            if (more) issue(bk, la, lb);
            fetch(buf, 0, 0);
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                if (k + 1 < BK) fetch(buf, k + 1, (k + 1) & 1);
                outer(k & 1);
            }
            if (more) store(nxt, la, lb);
            __syncthreads();
            buf = nxt;
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int r = r0 + row0 + (i >> 2) * d + (i & 3);
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int c = c0 + col0 + (j >> 2) * d + (j & 3);
            if (r < M && c < N) C[r * N + c] = c_frag[i][j];
        }
    }
}

// Ampere 异步拷贝：S 级循环缓冲。A 用列主序、行距 M，B 用行主序，所以 M 和 N 都要是 4 的倍数
template <int S>
__global__ __launch_bounds__(256, 2) void sgemm_async(const float* A, const float* B, float* C, int M, int N, int K) {
    alignas(16) __shared__ float As[S][BK][BM];
    alignas(16) __shared__ float Bs[S][BK][BN];

    const int tid = threadIdx.x;
    const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;

    // Z-order 线程映射，配合 4×4 拆分
    const int warp = tid >> 5, lane = tid & 31;
    const int ty = (warp >> 1) * 4 + (lane & 1) + ((lane >> 4) << 1);
    const int tx = (warp & 1) * 8 + ((lane & 15) >> 1);
    const int row0 = ty * 4, col0 = tx * 4;

    // 每个线程负责拷贝 tileA、tileB 各一个 float4
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

    // 预先发出前 S - 1 块的拷贝
    for (int s = 0; s < S - 1; s++) {
        issue(s);
        __pipeline_commit();
    }

    for (int t = 0; t < n_tile; t++) {
        // 等第 t 块拷贝完成
        __pipeline_wait_prior(S - 2);
        __syncthreads();

        issue(t + S - 1);
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

using Kernel = void (*)(const float*, const float*, float*, int, int, int);
enum class Kind { Naive, Tile, Float4, Async };

struct Version {
    const char* name;
    Kernel kernel;
    Kind kind;
};

static const Version VERSIONS[] = {
    {"朴素实现", sgemm_naive, Kind::Naive},
    {"Block Tiling", sgemm_block, Kind::Tile},
    {"Thread Tiling", sgemm_thread<0>, Kind::Tile},
    {"Warp Tiling 4x8", sgemm_thread<1>, Kind::Tile},
    {"float4 行主序 8x8", sgemm_f4<0, 8, 0, 0>, Kind::Float4},
    {"float4 拆成 4x4", sgemm_f4<0, 4, 0, 0>, Kind::Float4},
    {"float4 Z-order 8x8", sgemm_f4<1, 8, 0, 0>, Kind::Float4},
    {"float4 Z-order 4x4", sgemm_f4<1, 4, 0, 0>, Kind::Float4},
    {"Z-order 4x4 + XOR", sgemm_f4<1, 4, 1, 0>, Kind::Float4},
    {"Z-order + 双缓冲", sgemm_f4<1, 4, 1, 1>, Kind::Float4},
    {"异步拷贝 2 级", sgemm_async<2>, Kind::Async},
    {"异步拷贝 3 级", sgemm_async<3>, Kind::Async},
    {"异步拷贝 4 级", sgemm_async<4>, Kind::Async},
};

static bool supports(const Version& v, int M, int N, int K) {
    if (v.kind == Kind::Float4) return N % 4 == 0 && K % 4 == 0;
    if (v.kind == Kind::Async) return M % 4 == 0 && N % 4 == 0;
    return true;
}

static void launch(const Version& v, const float* A, const float* B, float* C, int M, int N, int K) {
    if (v.kind == Kind::Naive) {
        v.kernel<<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(A, B, C, M, N, K);
    } else {
        v.kernel<<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM), BS>>>(A, B, C, M, N, K);
    }
}

// async 版本的 A 要转成列主序
static std::vector<float> layout_a(const Version& v, const std::vector<float>& a, int M, int K) {
    if (v.kind != Kind::Async) return a;
    std::vector<float> column(a.size());
    for (int i = 0; i < M; i++)
        for (int p = 0; p < K; p++) column[p * M + i] = a[i * K + p];
    return column;
}

// 输入是小整数，浮点加法没有舍入，结果必须和参考值完全相等
static bool run_case(const Version& v, int M, int N, int K, char* detail, size_t detail_size) {
    std::vector<float> a(M * K), b(K * N), expect(M * N);
    std::srand(M * 131 + N * 17 + K);
    for (float& x : a) x = (float)(std::rand() % 7 - 3);
    for (float& x : b) x = (float)(std::rand() % 7 - 3);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double s = 0;
            for (int p = 0; p < K; p++) s += (double)a[i * K + p] * b[p * N + j];
            expect[i * N + j] = (float)s;
        }

    DeviceArray<float> da(M * K), db(K * N), dc(M * N);
    da.upload(layout_a(v, a, M, K));
    db.upload(b);
    check_cuda(cudaMemset(dc.get(), 0, M * N * sizeof(float)), "清零");
    launch(v, da.get(), db.get(), dc.get(), M, N, K);
    check_cuda(cudaDeviceSynchronize(), v.name);

    std::vector<float> got = dc.download();
    int wrong = 0;
    for (int i = 0; i < M * N; i++) wrong += got[i] != expect[i];
    std::snprintf(detail, detail_size, "%d 个元素不一致", wrong);
    return wrong == 0;
}

int main(int argc, char** argv) {
    print_device();

    const int cases[][3] = {
        {128, 128, 8}, {256, 128, 64}, {128, 256, 16}, {200, 300, 36}, {132, 124, 8},
        {4, 4, 4}, {260, 508, 44}, {512, 512, 128}, {129, 127, 9}, {5, 3, 2}, {200, 300, 37},
    };

    Summary summary;
    for (const Version& v : VERSIONS) {
        for (const auto& c : cases) {
            if (!supports(v, c[0], c[1], c[2])) continue;
            char label[48], detail[64];
            std::snprintf(label, sizeof(label), "M=%d N=%d K=%d", c[0], c[1], c[2]);
            bool ok = run_case(v, c[0], c[1], c[2], detail, sizeof(detail));
            summary.record(v.name, label, ok, detail);
        }
        summary.report(v.name);
    }

    if (has_flag(argc, argv, "--perf")) {
        const int n = 2048;
        std::vector<float> a(n * n), b(n * n);
        for (float& x : a) x = (float)std::rand() / RAND_MAX;
        for (float& x : b) x = (float)std::rand() / RAND_MAX;
        std::printf("吞吐：M = N = K = 2048，取 5 次中最快的一次\n");
        for (const Version& v : VERSIONS) {
            DeviceArray<float> da(n * n), db(n * n), dc(n * n);
            da.upload(layout_a(v, a, n, n));
            db.upload(b);
            double sec = best_seconds([&] { launch(v, da.get(), db.get(), dc.get(), n, n, n); });
            std::printf("  %-24s %8.2f ms  %8.1f GFLOP/s\n", v.name, sec * 1e3, 2.0 * n * n * n / sec / 1e9);
        }
    }
    return summary.finish();
}
