// Online Softmax V0~V4，讲解见 softmax/online-softmax.md。
// 约定：有效输入都是有限值，掩码只用 -INFINITY。整行被屏蔽或含 +INFINITY 时输出是 NaN，由调用方处理。
#include "harness.h"

#include <cfloat>
#include <cmath>
#include <random>

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

// 两级合并并广播。empty 是空状态的 m，正确的取值是 -FLT_MAX
__device__ void blockReduceOnline(float local_m, float local_d, float empty, float& row_max, float& inv_sum) {
    int tid  = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

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
        local_m = (lane < num_warps) ? warp_m[lane] : empty;
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

    row_max = final_m;
    inv_sum = 1.0f / final_d;
}

// V1: 一个 Block 一行 + 两级合并规约。EMPTY_IS_INF 只用来复现页面上的 NaN
template <bool EMPTY_IS_INF>
__global__ void online_softmax_v1(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float empty = EMPTY_IS_INF ? -INFINITY : -FLT_MAX;

    float* x = input  + row * N;
    float* y = output + row * N;

    // 第 1 遍：每个线程处理一段
    float local_m = empty;
    float local_d = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(local_m, xi);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }

    float row_max, inv_sum;
    blockReduceOnline(local_m, local_d, empty, row_max, inv_sum);

    // 第 2 遍：归一化输出
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - row_max) * inv_sum;
    }
}

// V2: float4 读取 + Online 递推
__global__ void online_softmax_v2(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

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

    float row_max, inv_sum;
    blockReduceOnline(local_m, local_d, -FLT_MAX, row_max, inv_sum);

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

// V3: 暂存到线程局部数组，只读一次 input
// 调用前必须保证 N <= blockDim.x * MAX_ELEMS_PER_THREAD，否则数组越界
#define MAX_ELEMS_PER_THREAD 32

__global__ void online_softmax_v3(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

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

    float row_max, inv_sum;
    blockReduceOnline(local_m, local_d, -FLT_MAX, row_max, inv_sum);

    // 从暂存读取并写出，不再读 input
    int idx = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(cache[idx] - row_max) * inv_sum;
        idx++;
    }
}

// V4: Grid Stride 多行处理
__global__ void online_softmax_v4(float* input, float* output, int M, int N) {
    int tid = threadIdx.x;

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

        float row_max, inv_sum;
        blockReduceOnline(local_m, local_d, -FLT_MAX, row_max, inv_sum);

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

constexpr int BLOCK = 256;
constexpr int V0_BLOCK = 32;

enum class Launch { PerThread, PerRow, GridStride };

struct Version {
    const char* name;
    void (*kernel)(float*, float*, int, int);
    Launch launch;
    int max_n;           // 0 表示不限
    int bytes_per_elem;  // 每个元素的显存流量，用于换算吞吐
};

static const Version VERSIONS[] = {
    {"V0 一线程一行", online_softmax_v0, Launch::PerThread, 0, 12},
    {"V1 合并规约", online_softmax_v1<false>, Launch::PerRow, 0, 12},
    {"V2 float4", online_softmax_v2, Launch::PerRow, 0, 12},
    {"V3 只读一次", online_softmax_v3, Launch::PerRow, BLOCK * MAX_ELEMS_PER_THREAD, 8},
    {"V4 Grid Stride", online_softmax_v4, Launch::GridStride, 0, 12},
};

static void launch(const Version& v, float* input, float* output, int M, int N, int num_sms) {
    switch (v.launch) {
        case Launch::PerThread:
            v.kernel<<<(M + V0_BLOCK - 1) / V0_BLOCK, V0_BLOCK>>>(input, output, M, N);
            break;
        case Launch::PerRow:
            v.kernel<<<M, BLOCK>>>(input, output, M, N);
            break;
        case Launch::GridStride:
            v.kernel<<<std::min(M, num_sms * 4), BLOCK>>>(input, output, M, N);
            break;
    }
}

// 用双精度算参考值。掩码元素的输出是 0
static std::vector<double> reference(const std::vector<float>& data, int M, int N) {
    std::vector<double> out(M * N);
    for (int r = 0; r < M; r++) {
        double m = -HUGE_VAL, sum = 0;
        for (int i = 0; i < N; i++) m = std::max(m, (double)data[r * N + i]);
        for (int i = 0; i < N; i++) sum += std::exp(data[r * N + i] - m);
        for (int i = 0; i < N; i++) out[r * N + i] = std::exp(data[r * N + i] - m) / sum;
    }
    return out;
}

static std::vector<float> make_input(int M, int N, float offset, bool masked) {
    std::mt19937 rng(M * 31 + N);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> data(M * N);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dist(rng) + offset;
        // 偶数位置屏蔽：每段的第一个元素就是 -INFINITY，这是页面列出的第三个 NaN 入口
        if (masked && i % 2 == 0) data[i] = -INFINITY;
    }
    return data;
}

static int count_nan(const std::vector<float>& values) {
    int nans = 0;
    for (float x : values) nans += std::isnan(x);
    return nans;
}

int main(int argc, char** argv) {
    print_device();
    int num_sms;
    check_cuda(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0), "读取 SM 数");

    struct Case { int M, N; float offset; bool masked; };
    const Case cases[] = {
        {4, 4096, 0, false}, {8, 256, 0, false}, {3, 100, 0, false}, {5, 1000, 0, false}, {3, 4094, 0, false},
        {3, 4097, 0, false}, {2, 8192, 0, false}, {4, 4096, 100, false}, {4, 4096, 0, true}, {64, 512, 0, false},
    };

    Summary summary;
    for (const Version& v : VERSIONS) {
        for (const Case& c : cases) {
            if (v.max_n && c.N > v.max_n) continue;
            std::vector<float> data = make_input(c.M, c.N, c.offset, c.masked);
            std::vector<double> expect = reference(data, c.M, c.N);
            DeviceArray<float> input(c.M * c.N), output(c.M * c.N);
            input.upload(data);
            launch(v, input.get(), output.get(), c.M, c.N, num_sms);
            check_cuda(cudaDeviceSynchronize(), v.name);

            std::vector<float> got = output.download();
            double worst = 0;
            for (size_t i = 0; i < got.size(); i++) {
                double e = expect[i] == 0 ? std::fabs(got[i]) : std::fabs(got[i] - expect[i]) / expect[i];
                if (!(e <= worst)) worst = e;  // 这样写，NaN 也会被记下来
            }
            char label[48], detail[96];
            std::snprintf(label, sizeof(label), "M=%d N=%d%s", c.M, c.N, c.masked ? " 带掩码" : "");
            std::snprintf(detail, sizeof(detail), "最大相对误差 %.2e", worst);
            summary.record(v.name, label, worst < 1e-3, detail);
        }
        summary.report(v.name);
    }

    // 复现页面上的 NaN：Block 为 256 时，空状态用 -INFINITY 会让整行都变成 NaN
    {
        const int M = 4, N = 4096;
        std::vector<float> data = make_input(M, N, 0, false);
        DeviceArray<float> input(M * N), output(M * N);
        input.upload(data);
        online_softmax_v1<true><<<M, BLOCK>>>(input.get(), output.get(), M, N);
        check_cuda(cudaDeviceSynchronize(), "V1 -INFINITY");
        int nans = count_nan(output.download());
        online_softmax_v1<false><<<M, BLOCK>>>(input.get(), output.get(), M, N);
        check_cuda(cudaDeviceSynchronize(), "V1 -FLT_MAX");
        int fixed = count_nan(output.download());
        std::printf("空状态初值对比（Block 256，%d 个输出）：-INFINITY 有 %d 个 NaN，-FLT_MAX 有 %d 个\n", M * N, nans, fixed);
        summary.record("NaN 复现", "-INFINITY 必须出 NaN", nans == M * N);
        summary.record("NaN 复现", "-FLT_MAX 不能出 NaN", fixed == 0);
    }

    if (has_flag(argc, argv, "--perf")) {
        const int M = 4096, N = 4096;
        std::vector<float> data = make_input(M, N, 0, false);
        DeviceArray<float> input(M * N), output(M * N);
        input.upload(data);
        std::printf("吞吐：M = N = 4096，取 5 次中最快的一次。GB/s 按每次访问都到达显存换算\n");
        for (const Version& v : VERSIONS) {
            double sec = best_seconds([&] { launch(v, input.get(), output.get(), M, N, num_sms); });
            std::printf("  %-22s %8.1f us  %8.1f GB/s\n", v.name, sec * 1e6, (double)M * N * v.bytes_per_elem / sec / 1e9);
        }
    }
    return summary.finish();
}
