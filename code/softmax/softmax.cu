// Softmax 算子 V0~V4，讲解见 softmax/softmax.md。
#include "harness.h"

#include <cmath>
#include <random>

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

// V3: float4 向量化加载
__global__ void softmax_v3(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input  + row * N;
    float* y = output + row * N;

    // 向量化指针
    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    // N 不是 4 的倍数时 N4 = 0，float4 循环不执行，全部交给标量循环
    int N4 = (N % 4 == 0) ? N / 4 : 0;

    // Pass 1: 向量化求最大值
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

// V4: 两遍 Kernel（Pass 1 求 max，Pass 2 融合 exp-sum + normalize）
// 需要额外 Shared Memory 暂存 exp 值
__global__ void softmax_v4(float* input, float* output, int M, int N) {
    extern __shared__ float smem[];  // 大小 = N 个 float（暂存 exp 值）

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input  + row * N;
    float* y = output + row * N;

    float4* x4 = reinterpret_cast<float4*>(x);
    int N4 = (N % 4 == 0) ? N / 4 : 0;

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

constexpr int BLOCK = 256;
constexpr int V0_BLOCK = 32;

enum class Launch { PerThread, ReduceSmem, Static, ExpSmem };

struct Version {
    const char* name;
    void (*kernel)(float*, float*, int, int);
    Launch launch;
    int bytes_per_elem;  // 每个元素的显存流量，用于换算吞吐
};

static const Version VERSIONS[] = {
    {"V0 一线程一行", softmax_v0, Launch::PerThread, 16},
    {"V1 一 Block 一行", softmax_v1, Launch::ReduceSmem, 16},
    {"V2 两级规约", softmax_v2, Launch::Static, 16},
    {"V3 float4", softmax_v3, Launch::Static, 16},
    {"V4 暂存 exp", softmax_v4, Launch::ExpSmem, 12},
};

static void launch(const Version& v, float* input, float* output, int M, int N) {
    switch (v.launch) {
        case Launch::PerThread:
            v.kernel<<<(M + V0_BLOCK - 1) / V0_BLOCK, V0_BLOCK>>>(input, output, M, N);
            break;
        case Launch::ReduceSmem:
            v.kernel<<<M, BLOCK, BLOCK * sizeof(float)>>>(input, output, M, N);
            break;
        case Launch::Static:
            v.kernel<<<M, BLOCK>>>(input, output, M, N);
            break;
        case Launch::ExpSmem: {
            // 静态和动态加起来超过 48 KB 时必须声明，否则启动失败
            size_t smem_bytes = N * sizeof(float);
            check_cuda(cudaFuncSetAttribute(v.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes),
                       "声明共享内存");
            v.kernel<<<M, BLOCK, smem_bytes>>>(input, output, M, N);
            break;
        }
    }
}

static std::vector<float> make_input(int M, int N, float offset) {
    std::mt19937 rng(M * 31 + N);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> data(M * N);
    for (float& x : data) x = dist(rng) + offset;
    return data;
}

// 用双精度算参考值
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

int main(int argc, char** argv) {
    print_device();

    // N 不是 4 的倍数用来检查行首对齐，offset = 100 时朴素公式已经溢出
    struct Case { int M, N; float offset; };
    const Case cases[] = {
        {4, 4096, 0}, {8, 256, 0}, {3, 100, 0}, {5, 1000, 0}, {3, 4094, 0},
        {3, 4097, 0}, {2, 4098, 0}, {2, 8192, 0}, {4, 4096, 100},
    };

    Summary summary;
    for (const Version& v : VERSIONS) {
        for (const Case& c : cases) {
            std::vector<float> data = make_input(c.M, c.N, c.offset);
            std::vector<double> expect = reference(data, c.M, c.N);
            DeviceArray<float> input(c.M * c.N), output(c.M * c.N);
            input.upload(data);
            launch(v, input.get(), output.get(), c.M, c.N);
            check_cuda(cudaDeviceSynchronize(), v.name);

            char label[48], detail[96];
            std::snprintf(label, sizeof(label), "M=%d N=%d 偏移=%g", c.M, c.N, c.offset);
            std::vector<float> got = output.download();
            double worst = 0;
            for (size_t i = 0; i < got.size(); i++) {
                double e = std::fabs(got[i] - expect[i]) / std::max(expect[i], 1e-30);
                if (!(e <= worst)) worst = e;  // 这样写，NaN 也会被记下来
            }
            std::snprintf(detail, sizeof(detail), "最大相对误差 %.2e", worst);
            summary.record(v.name, label, worst < 1e-3, detail);
        }
        summary.report(v.name);
    }

    if (has_flag(argc, argv, "--perf")) {
        const int M = 4096, N = 4096;
        std::vector<float> data = make_input(M, N, 0);
        DeviceArray<float> input(M * N), output(M * N);
        input.upload(data);
        std::printf("吞吐：M = N = 4096，取 5 次中最快的一次。GB/s 按读写各一次都到达显存换算\n");
        for (const Version& v : VERSIONS) {
            double sec = best_seconds([&] { launch(v, input.get(), output.get(), M, N); });
            std::printf("  %-22s %8.1f us  %8.1f GB/s\n", v.name, sec * 1e6, (double)M * N * v.bytes_per_elem / sec / 1e9);
        }
    }
    return summary.finish();
}
