// FlashAttention V1、V2 的前向 Kernel，讲解见 flash-attention/v1.md 和 v2.md。
// Q、K、V、O 的形状都是 [B, N, d]，B 是 batch × head。这里用 FP32 和 CUDA Core 演示循环结构，
// 没有用 Tensor Core 和 FP16，也没有实现反向传播，所以吞吐不能和官方实现比较。
//
// 两个版本都让一个线程负责 Q 的一行，也就是 V2 的“沿 Q 切分”。V1 官方实现沿 K/V 切分 Warp，这里没有实现。
// 两者真正的差别在循环顺序：
//   V1：一个 Block 处理一个 (batch, head)，外循环 K/V，O、m、l 每个 Tile 都要读写显存
//   V2：一个 Block 处理一个 Q 块，外循环 Q，O、m、l 留在片上，最后只写一次
#include "harness.h"

#include <cfloat>
#include <cmath>
#include <random>

constexpr int BR = 32;  // Q 块行数，也是 Block 的线程数
constexpr int BC = 32;  // K/V 块行数

// 共享内存布局。S 和 V2 的 O 只有所属线程访问，放在这里是为了和页面上的 SRAM 规划一致
struct Smem {
    float* K;  // [BC][d]
    float* V;  // [BC][d]
    float* Q;  // [BR][d]
    float* S;  // [BR][BC]
    float* O;  // [BR][d]，只有 V2 使用
};

__host__ __device__ inline size_t smem_floats(int d, bool with_o) {
    return BC * d * 2 + BR * d + BR * BC + (with_o ? BR * d : 0);
}

__device__ inline Smem layout_smem(float* base, int d) {
    Smem s;
    s.K = base;
    s.V = s.K + BC * d;
    s.Q = s.V + BC * d;
    s.S = s.Q + BR * d;
    s.O = s.S + BR * BC;
    return s;
}

// 一个 Block 的线程合作搬一个 Tile：first 行起共 rows 行，越界的行补 0
__device__ inline void load_tile(float* dst, const float* src, int first, int rows, int N, int d) {
    for (int idx = threadIdx.x; idx < rows * d; idx += blockDim.x) {
        int row = first + idx / d;
        dst[idx] = row < N ? src[row * d + idx % d] : 0.0f;
    }
}

// causal：只能看到列号不大于行号的位置。pad：前 pad 列被屏蔽，相当于左填充
__device__ inline bool visible(int row, int col, int N, bool causal, int pad) {
    return col < N && col >= pad && (!causal || col <= row);
}

// S_ij = Q_i * K_j^T / sqrt(d)，屏蔽的位置写 -INFINITY。返回这一块的行最大值
__device__ inline float compute_scores(const Smem& s, int row, int j0, int N, int d, bool causal, int pad, float scale) {
    const int tid = threadIdx.x;
    float block_max = -INFINITY;
    for (int j = 0; j < BC; j++) {
        float score = -INFINITY;
        if (visible(row, j0 + j, N, causal, pad)) {
            float dot = 0.0f;
            for (int c = 0; c < d; c++) dot += s.Q[tid * d + c] * s.K[j * d + c];
            score = dot * scale;
        }
        s.S[tid * BC + j] = score;
        block_max = fmaxf(block_max, score);
    }
    return block_max;
}

// V1：外循环遍历 K/V 块，内循环遍历 Q 块。grid = B
__global__ void flash_attention_v1(const float* Q, const float* K, const float* V, float* O, float* M, float* L,
                                   int N, int d, bool causal, int pad) {
    extern __shared__ float smem[];
    const Smem s = layout_smem(smem, d);
    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    const float scale = 1.0f / sqrtf((float)d);

    const float* Qb = Q + b * N * d;
    const float* Kb = K + b * N * d;
    const float* Vb = V + b * N * d;
    float* Ob = O + b * N * d;
    float* Mb = M + b * N;
    float* Lb = L + b * N;

    // ============ 外循环：遍历 K/V 块 ============
    for (int j0 = 0; j0 < N; j0 += BC) {
        // 每个 K/V 块只从显存加载一次，被所有 Q 块复用
        load_tile(s.K, Kb, j0, BC, N, d);
        load_tile(s.V, Vb, j0, BC, N, d);
        __syncthreads();

        // ============ 内循环：遍历 Q 块 ============
        for (int i0 = 0; i0 < N; i0 += BR) {
            // Causal 时，整块都在对角线上方的 Tile 不用算，也不用读写 O
            if (causal && j0 > i0 + BR - 1) continue;

            // 每个外循环都要从显存重新读 Q
            load_tile(s.Q, Qb, i0, BR, N, d);
            __syncthreads();

            const int row = i0 + tid;
            if (row < N) {
                float block_max = compute_scores(s, row, j0, N, d, causal, pad, scale);

                // m、l 也要从显存读
                float m = Mb[row];
                float l = Lb[row];
                float m_new = fmaxf(m, block_max);
                float alpha = expf(m - m_new);

                float sum_p = 0.0f;
                for (int j = 0; j < BC; j++) {
                    float p = expf(s.S[tid * BC + j] - m_new);
                    s.S[tid * BC + j] = p;
                    sum_p += p;
                }
                float l_new = alpha * l + sum_p;
                // 某行至今全被屏蔽时 l_new = 0，避免 0/0
                float l_safe = l_new > 0.0f ? l_new : 1.0f;

                // 每个 Tile 都要把 O 读回来，反归一化、更新、再归一化写回
                float old_coef = alpha * l / l_safe;
                for (int c = 0; c < d; c++) {
                    float acc = 0.0f;
                    for (int j = 0; j < BC; j++) acc += s.S[tid * BC + j] * s.V[j * d + c];
                    Ob[row * d + c] = Ob[row * d + c] * old_coef + acc / l_safe;
                }

                Mb[row] = m_new;
                Lb[row] = l_new;
            }
            __syncthreads();
        }
    }
}

// V2：外循环遍历 Q 块，内循环遍历 K/V 块。grid = (T_r, B)
__global__ void flash_attention_v2(const float* Q, const float* K, const float* V, float* O, float* L,
                                   int N, int d, bool causal, int pad) {
    extern __shared__ float smem[];
    const Smem s = layout_smem(smem, d);
    const int tid = threadIdx.x;
    const int b = blockIdx.y;
    const int i0 = blockIdx.x * BR;
    const int row = i0 + tid;
    const float scale = 1.0f / sqrtf((float)d);

    const float* Kb = K + b * N * d;
    const float* Vb = V + b * N * d;

    // Q 只加载一次，整个内循环都驻留在片上
    load_tile(s.Q, Q + b * N * d, i0, BR, N, d);
    __syncthreads();

    // 累积量也留在片上。初值用 -FLT_MAX：-INFINITY 遇到整块屏蔽的行会算出 NaN
    float m = -FLT_MAX;
    float l = 0.0f;
    for (int c = 0; c < d; c++) s.O[tid * d + c] = 0.0f;

    // Causal 时，对角线右边的 K/V 块可以提前跳过
    const int kv_end = causal ? min(N, i0 + BR) : N;
    for (int j0 = 0; j0 < kv_end; j0 += BC) {
        load_tile(s.K, Kb, j0, BC, N, d);
        load_tile(s.V, Vb, j0, BC, N, d);
        __syncthreads();

        if (row < N) {
            float block_max = compute_scores(s, row, j0, N, d, causal, pad, scale);
            float m_new = fmaxf(m, block_max);
            float alpha = expf(m - m_new);

            float sum_p = 0.0f;
            for (int j = 0; j < BC; j++) {
                float p = expf(s.S[tid * BC + j] - m_new);
                s.S[tid * BC + j] = p;
                sum_p += p;
            }
            l = alpha * l + sum_p;

            // 延迟归一化：内循环只做指数缩放，不做除法
            for (int c = 0; c < d; c++) {
                float acc = 0.0f;
                for (int j = 0; j < BC; j++) acc += s.S[tid * BC + j] * s.V[j * d + c];
                s.O[tid * d + c] = alpha * s.O[tid * d + c] + acc;
            }
            m = m_new;
        }
        __syncthreads();
    }

    // 最终归一化，O 只写回这一次。l_safe 只为整行都不可见的情况兜底
    if (row < N) {
        float l_safe = l > 0.0f ? l : 1.0f;
        float* Ob = O + b * N * d;
        for (int c = 0; c < d; c++) Ob[row * d + c] = s.O[tid * d + c] / l_safe;
        L[b * N + row] = m + logf(l_safe);
    }
}

struct Problem {
    int B, N, d;
    bool causal;
    int pad;
    std::vector<float> Q, K, V;
};

static Problem make_problem(int B, int N, int d, bool causal, int pad) {
    Problem p{B, N, d, causal, pad, {}, {}, {}};
    std::mt19937 rng(B * 1000003 + N * 1009 + d);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto* t : {&p.Q, &p.K, &p.V}) {
        t->resize((size_t)B * N * d);
        for (float& x : *t) x = dist(rng);
    }
    return p;
}

// 用双精度直接算 softmax(QK^T / sqrt(d)) V。一列都看不到的行输出 0
static void reference(const Problem& p, std::vector<double>& O, std::vector<double>& L) {
    const int B = p.B, N = p.N, d = p.d;
    O.assign((size_t)B * N * d, 0.0);
    L.assign((size_t)B * N, -HUGE_VAL);
    const double scale = 1.0 / std::sqrt((double)d);
    std::vector<double> s(N);
    for (int b = 0; b < B; b++) {
        const float* Q = p.Q.data() + (size_t)b * N * d;
        const float* K = p.K.data() + (size_t)b * N * d;
        const float* V = p.V.data() + (size_t)b * N * d;
        for (int r = 0; r < N; r++) {
            double m = -HUGE_VAL;
            for (int c = 0; c < N; c++) {
                s[c] = -HUGE_VAL;
                if (c >= p.pad && (!p.causal || c <= r)) {
                    double dot = 0;
                    for (int k = 0; k < d; k++) dot += (double)Q[r * d + k] * K[c * d + k];
                    s[c] = dot * scale;
                }
                m = std::max(m, s[c]);
            }
            if (m == -HUGE_VAL) continue;
            double sum = 0;
            for (int c = 0; c < N; c++) sum += std::exp(s[c] - m);
            for (int c = 0; c < N; c++) {
                double w = std::exp(s[c] - m) / sum;
                for (int k = 0; k < d; k++) O[(size_t)(b * N + r) * d + k] += w * V[c * d + k];
            }
            L[b * N + r] = m + std::log(sum);
        }
    }
}

static int max_block_smem() {
    int bytes;
    check_cuda(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0), "读取共享内存上限");
    return bytes;
}

struct Output {
    std::vector<float> O, L;
};

// 一个问题的全部显存数组。V2 不使用 M，V1 的 O、M、L 每次启动前都要重置
struct Buffers {
    DeviceArray<float> Q, K, V, O, M, L;
    size_t cells, count;

    explicit Buffers(const Problem& p)
        : Q((size_t)p.B * p.N * p.d), K((size_t)p.B * p.N * p.d), V((size_t)p.B * p.N * p.d),
          O((size_t)p.B * p.N * p.d), M((size_t)p.B * p.N), L((size_t)p.B * p.N),
          cells((size_t)p.B * p.N), count((size_t)p.B * p.N * p.d) {
        Q.upload(p.Q);
        K.upload(p.K);
        V.upload(p.V);
    }
    void reset_v1() {
        O.upload(std::vector<float>(count, 0.0f));
        M.upload(std::vector<float>(cells, -FLT_MAX));
        L.upload(std::vector<float>(cells, 0.0f));
    }
};

static void launch_v1(const Problem& p, Buffers& buf) {
    size_t bytes = smem_floats(p.d, false) * sizeof(float);
    check_cuda(cudaFuncSetAttribute(flash_attention_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes), "声明共享内存");
    flash_attention_v1<<<p.B, BR, bytes>>>(buf.Q.get(), buf.K.get(), buf.V.get(), buf.O.get(), buf.M.get(), buf.L.get(),
                                           p.N, p.d, p.causal, p.pad);
}

static void launch_v2(const Problem& p, Buffers& buf) {
    size_t bytes = smem_floats(p.d, true) * sizeof(float);
    check_cuda(cudaFuncSetAttribute(flash_attention_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes), "声明共享内存");
    dim3 grid((p.N + BR - 1) / BR, p.B);
    flash_attention_v2<<<grid, BR, bytes>>>(buf.Q.get(), buf.K.get(), buf.V.get(), buf.O.get(), buf.L.get(),
                                            p.N, p.d, p.causal, p.pad);
}

static Output run_v1(const Problem& p, Buffers& buf) {
    buf.reset_v1();
    launch_v1(p, buf);
    check_cuda(cudaDeviceSynchronize(), "V1");
    Output out;
    out.O = buf.O.download();
    std::vector<float> m = buf.M.download(), l = buf.L.download();
    out.L.resize(buf.cells);
    for (size_t i = 0; i < buf.cells; i++) out.L[i] = l[i] > 0 ? m[i] + std::log(l[i]) : -FLT_MAX;
    return out;
}

static Output run_v2(const Problem& p, Buffers& buf) {
    launch_v2(p, buf);
    check_cuda(cudaDeviceSynchronize(), "V2");
    Output out;
    out.O = buf.O.download();
    out.L = buf.L.download();
    return out;
}

// 对比 O 和 L。一列都看不到的行，O 应为 0，L 不比较
static bool compare(const Problem& p, const Output& out, char* detail, size_t detail_size) {
    std::vector<double> O, L;
    reference(p, O, L);
    double worst_o = 0, worst_l = 0;
    for (size_t i = 0; i < O.size(); i++) {
        double e = std::fabs(out.O[i] - O[i]);
        if (!(e <= worst_o)) worst_o = e;  // 这样写，NaN 也会被记下来
    }
    for (size_t i = 0; i < L.size(); i++) {
        if (L[i] == -HUGE_VAL) continue;
        double e = std::fabs(out.L[i] - L[i]);
        if (!(e <= worst_l)) worst_l = e;
    }
    std::snprintf(detail, detail_size, "O 最大误差 %.2e，L 最大误差 %.2e", worst_o, worst_l);
    return worst_o < 1e-3 && worst_l < 1e-3;
}

int main(int argc, char** argv) {
    print_device();
    const int smem_limit = max_block_smem();

    // pad 大于 BC 时，每行的第一块都被整块屏蔽，这是页面上 NaN 的入口。
    // pad 加 causal 时，前 pad 行什么都看不到，输出应为 0
    struct Case { int B, N, d; bool causal; int pad; };
    const Case cases[] = {
        {1, 32, 16, false, 0},  {2, 100, 16, false, 0}, {3, 97, 32, true, 0},  {2, 64, 128, false, 0},
        {1, 65, 8, true, 0},    {2, 200, 24, false, 70}, {2, 200, 24, true, 70}, {2, 257, 64, true, 0},
        {4, 512, 64, false, 0},
    };

    Summary summary;
    for (int version = 1; version <= 2; version++) {
        const char* name = version == 1 ? "FlashAttention V1" : "FlashAttention V2";
        for (const Case& c : cases) {
            char label[64], detail[96];
            std::snprintf(label, sizeof(label), "B=%d N=%d d=%d%s pad=%d", c.B, c.N, c.d, c.causal ? " causal" : "", c.pad);
            if (smem_floats(c.d, version == 2) * sizeof(float) > (size_t)smem_limit) {
                std::printf("  跳过 %-22s %s：共享内存超过这张卡的上限\n", name, label);
                continue;
            }
            Problem p = make_problem(c.B, c.N, c.d, c.causal, c.pad);
            Buffers buf(p);
            Output out = version == 1 ? run_v1(p, buf) : run_v2(p, buf);
            bool ok = compare(p, out, detail, sizeof(detail));
            summary.record(name, label, ok, detail);
        }
        summary.report(name);
    }

    if (has_flag(argc, argv, "--perf")) {
        Problem p = make_problem(8, 2048, 64, false, 0);
        Buffers buf(p);
        const double flops = 4.0 * p.B * p.N * p.N * p.d;  // 两次矩阵乘，一次 FMA 算 2 个 FLOP
        std::printf("耗时：B = 8，N = 2048，d = 64，取 3 次中最快的一次，只算两次矩阵乘的 FLOP。不能和官方实现比较\n");
        double v1 = 1e30, v2 = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            buf.reset_v1();  // 重置放在计时之外
            v1 = std::min(v1, best_seconds([&] { launch_v1(p, buf); }, 1));
            v2 = std::min(v2, best_seconds([&] { launch_v2(p, buf); }, 1));
        }
        std::printf("  FlashAttention V1  %9.2f ms  %8.1f GFLOP/s\n", v1 * 1e3, flops / v1 / 1e9);
        std::printf("  FlashAttention V2  %9.2f ms  %8.1f GFLOP/s\n", v2 * 1e3, flops / v2 / 1e9);
    }
    return summary.finish();
}
