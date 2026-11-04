// 各算子共用的主机端工具：显存缓冲、错误检查、计时和结果汇总
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

inline void check_cuda(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA 错误（%s）：%s\n", where, cudaGetErrorString(err));
        std::exit(2);
    }
}

// 显存里的一段数组，离开作用域时释放
template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(size_t count) : count_(count) {
        check_cuda(cudaMalloc(&data_, std::max<size_t>(count, 1) * sizeof(T)), "cudaMalloc");
    }
    ~DeviceArray() { cudaFree(data_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    T* get() const { return data_; }
    void upload(const std::vector<T>& src) {
        check_cuda(cudaMemcpy(data_, src.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "上传");
    }
    std::vector<T> download() const {
        std::vector<T> out(count_);
        check_cuda(cudaMemcpy(out.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "下载");
        return out;
    }

private:
    T* data_ = nullptr;
    size_t count_;
};

// 汇总校验结果，main 的返回值由它决定
class Summary {
public:
    void record(const char* name, const char* test_case, bool ok, const char* detail = "") {
        ok ? ++pass_ : ++fail_;
        if (!ok) std::printf("  失败 %-22s %-24s %s\n", name, test_case, detail);
    }
    // 打印一个版本的通过情况，并清空分项计数
    void report(const char* name) {
        std::printf("  %-22s %d/%d 通过\n", name, pass_ - version_pass_, pass_ + fail_ - version_total_);
        version_pass_ = pass_;
        version_total_ = pass_ + fail_;
    }
    int finish() const {
        std::printf("校验：%d 项通过，%d 项失败\n", pass_, fail_);
        return fail_ ? 1 : 0;
    }

private:
    int pass_ = 0, fail_ = 0;
    int version_pass_ = 0, version_total_ = 0;
};

// launch 启动 Kernel，这里等它跑完。取 reps 次中最快的一次，单位是秒
template <typename F>
double best_seconds(F launch, int reps = 5) {
    double best = 1e30;
    for (int i = 0; i < reps; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        launch();
        check_cuda(cudaDeviceSynchronize(), "等待 Kernel");
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

inline bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; i++)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

inline void print_device() {
    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, 0), "读取设备信息");
    std::printf("设备：%s，compute capability %d.%d，%d 个 SM\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
}
