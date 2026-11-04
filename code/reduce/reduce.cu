// Reduce 算子 V0~V7，讲解见 reduce/reduce.md。
// 每个 Kernel 只产出部分和，写到 output，这里在主机上再求一次和。
#include "harness.h"

#include <cmath>
#include <random>

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

// 辅助函数：展开最后 32 个线程的规约。
// 依赖 Warp 内严格同步，compute capability 7.0 起官方不再保证，这里保留只为对照
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

// V4 的安全写法：不依赖锁步，只有下一步要用到结果的车道才写入
__global__ void reduce_v4_safe(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        for (unsigned int s = 32; s > 0; s >>= 1) {
            if (tid < s) {
                smem[tid] += smem[tid + s];
            }
            __syncwarp();
        }
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// V5: 完全循环展开（模板参数）。最后 Warp 内同样依赖锁步
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

    // 最后 Warp 内展开
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

constexpr int BLOCK = 256;
using Kernel = void (*)(float*, float*, int);

struct Version {
    const char* name;
    Kernel kernel;
    int elems_per_block;  // 一个 Block 处理的元素数；0 表示 Grid 固定为 SM 数的 4 倍
    bool uses_smem;
    bool lockstep;        // 依赖 Warp 锁步，结果没有保证
};

static const Version VERSIONS[] = {
    {"V0 朴素树形", reduce_v0, BLOCK, true, false},
    {"V1 strided index", reduce_v1, BLOCK, true, false},
    {"V2 步长反转", reduce_v2, BLOCK, true, false},
    {"V3 每线程两元素", reduce_v3, 2 * BLOCK, true, false},
    {"V4 展开末 Warp", reduce_v4, 2 * BLOCK, true, true},
    {"V4 安全写法", reduce_v4_safe, 2 * BLOCK, true, false},
    {"V5 模板展开", reduce_v5<BLOCK>, 2 * BLOCK, true, true},
    {"V6 Warp Shuffle", reduce_v6, 2 * BLOCK, false, false},
    {"V7 float4 + Stride", reduce_v7, 0, false, false},
};

static int grid_size(const Version& v, int n, int num_sms) {
    return v.elems_per_block ? (n + v.elems_per_block - 1) / v.elems_per_block : num_sms * 4;
}

static void launch(const Version& v, float* input, float* output, int n, int grid) {
    v.kernel<<<grid, BLOCK, v.uses_smem ? BLOCK * sizeof(float) : 0>>>(input, output, n);
}

// 跑一次并在主机上汇总部分和
static double run_sum(const Version& v, DeviceArray<float>& input, int n, int num_sms) {
    int grid = grid_size(v, n, num_sms);
    DeviceArray<float> partial(grid);
    check_cuda(cudaMemset(partial.get(), 0, grid * sizeof(float)), "清零");
    launch(v, input.get(), partial.get(), n, grid);
    check_cuda(cudaGetLastError(), v.name);
    double sum = 0;
    for (float x : partial.download()) sum += x;
    return sum;
}

int main(int argc, char** argv) {
    print_device();
    int num_sms;
    check_cuda(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0), "读取 SM 数");

    Summary summary;
    int lockstep_wrong = 0;
    const int sizes[] = {1, 31, 255, 256, 257, 512, 1000, 4099, 1 << 20};
    for (const Version& v : VERSIONS) {
        for (int n : sizes) {
            std::mt19937 rng(n);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            std::vector<float> data(n);
            double expect = 0;
            for (float& x : data) {
                x = dist(rng);
                expect += x;
            }
            DeviceArray<float> input(n);
            input.upload(data);
            double got = run_sum(v, input, n, num_sms);
            bool ok = std::fabs(got - expect) <= 1e-4 * expect;
            char label[32];
            std::snprintf(label, sizeof(label), "n=%d", n);
            // 依赖锁步的版本只报告，不算失败：这正是页面要说明的风险
            if (v.lockstep && !ok) {
                lockstep_wrong++;
                std::printf("  算错 %-22s %s（依赖锁步，官方不保证）\n", v.name, label);
                ok = true;
            }
            char detail[64];
            std::snprintf(detail, sizeof(detail), "结果 %.3f，应为 %.3f", got, expect);
            summary.record(v.name, label, ok, detail);
        }
        summary.report(v.name);
    }
    std::printf("依赖锁步的 V4、V5 这次算错 %d 次。没算错不代表安全，官方不再保证这种写法。\n", lockstep_wrong);

    if (has_flag(argc, argv, "--perf")) {
        const int n = 1 << 27;
        std::mt19937 rng(1);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> data(n);
        for (float& x : data) x = dist(rng);
        DeviceArray<float> input(n);
        input.upload(data);

        std::printf("吞吐：n = 2^27，512 MiB，取 5 次中最快的一次\n");
        for (const Version& v : VERSIONS) {
            int grid = grid_size(v, n, num_sms);
            DeviceArray<float> partial(grid);
            double sec = best_seconds([&] { launch(v, input.get(), partial.get(), n, grid); });
            std::printf("  %-22s %8.1f us  %8.1f GB/s\n", v.name, sec * 1e6, n * 4.0 / sec / 1e9);
        }
    }
    return summary.finish();
}
