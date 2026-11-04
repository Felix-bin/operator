# 算子代码

站点六篇算子图解页对应的可运行代码。每个程序都会把各版本的输出和主机上的参考值逐一比较，有版本算错时退出码非 0。

| 算子 | 代码 | 讲解 | 包含的版本 |
| --- | --- | --- | --- |
| Reduce | [reduce/reduce.cu](reduce/reduce.cu) | [reduce/reduce.md](../reduce/reduce.md) | V0~V7，以及 V4 的安全写法 |
| GEMM | [gemm/gemm.cu](gemm/gemm.cu) | [gemm/gemm.md](../gemm/gemm.md) | 朴素实现到 Warp Tiling、float4 各排布、双缓冲、异步拷贝 |
| Softmax | [softmax/softmax.cu](softmax/softmax.cu) | [softmax/softmax.md](../softmax/softmax.md) | V0~V4 |
| Online Softmax | [softmax/online_softmax.cu](softmax/online_softmax.cu) | [softmax/online-softmax.md](../softmax/online-softmax.md) | V0~V4，以及 `-INFINITY` 的 NaN 复现 |
| FlashAttention | [flash_attention/flash_attention.cu](flash_attention/flash_attention.cu) | [V1](../flash-attention/v1.md)、[V2](../flash-attention/v2.md) | V1、V2 的前向 |

## 运行

需要 Linux 或 WSL，以及 `nvcc`、`make` 和一块 NVIDIA GPU。

```bash
cd code
make test   # 编译并校验
make perf   # 校验之后再测吞吐
make clean
```

可以指定编译器和目标架构：

```bash
make NVCC=/usr/local/cuda/bin/nvcc test
make ARCH=sm_80 all   # 给 A100 编译
```

也可以只跑一个算子，比如 `make build/reduce && ./build/reduce --perf`。

## 和页面的差别

- **FlashAttention 是 CUDA 写的。** 页面上是伪代码和 Triton，这里写成 CUDA Kernel，这样不装 PyTorch 也能跑。它用 FP32 和 CUDA Core，只有前向，只演示循环结构，所以吞吐不能和官方实现比较。
- **FlashAttention 的两个版本都沿 Q 切分线程。** V1 官方实现沿 K/V 切分 Warp，这里没有实现。两个版本的差别在循环顺序和显存读写次数。
- **初值和屏蔽按修复后的写法。** FlashAttention 和 Online Softmax 都用 `-FLT_MAX`，FlashAttention 另外用 `l_safe`。
- **Online Softmax 把两级合并提成了函数。** 页面为了连贯，每个版本都写了一遍，逻辑相同。
- **GEMM 的 float4 系列合成了一个模板。** 用模板参数切换车道排布、4×4 拆分、XOR 和双缓冲。
- **没有收录的内容。** GEMM 手写 SASS，Softmax 的 pitch 写法，Online Softmax V3 的编译期常量下标写法，FlashAttention 的反向传播。

## 需要留意

- **Reduce V4、V5 依赖 Warp 锁步。** compute capability 7.0 起官方不再保证这种写法。它们算错时程序只提示，不算校验失败，没算错也不代表安全。
- **Online Softmax 的输入约定。** 有效输入都是有限值，掩码只用 `-INFINITY`。整行被屏蔽或含 `+INFINITY` 时输出是 NaN，由调用方处理。
- **吞吐数字只对当前这张卡有效。** 程序按每次访问都到达显存来换算 GB/s，缓存命中时这个数字可能超过显存带宽。
