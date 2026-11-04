# Operator

图解 GPU 基础和 AI 算子优化：每个优化都讲清楚它解决哪个瓶颈，配上示意图、可交互实验和能运行、会自我校验的代码。

这是一个 VitePress 站点，加上一组 CUDA 代码。

## 致敬 AI Infra Guide

这个项目的算子优化部分，以 [AI Infra Guide](https://github.com/caomaolufei/AIInfraGuide)（作者[草帽路飞](https://zhihu.com/people/zi-you-zi-zai-de-zhan-sheng)，[在线阅读](https://caomaolufei.github.io/AIInfraGuide/)）为蓝本。它系统梳理了从 GPU 硬件、CUDA 算子优化到分布式训练和推理优化的技术栈，是非常好的中文 AI Infra 入门资料。如果你想系统学习这个方向，请先去看它，也欢迎给它点个 Star。

我们在它的基础上做了这些事：

- 把文字讲解改写成带图示和交互实验的页面。
- 把算子代码放到真实 GPU 上运行，并对照论文和 NVIDIA 官方文档核对，修正了一些问题，比如 Online Softmax 的 `-INFINITY` 初值会让输出全部变成 NaN。这些修正记录在各页面中。

页面里说的“原稿”和“原始整理稿”，都指 AI Infra Guide 的对应文章。每篇页面末尾都有原文链接。

## 内容

| 分类 | 页面 | 代码 |
| --- | --- | --- |
| GPU 基础 | [GPU 架构图解](about_gpu/gpu.md) | |
| | [AI 计算与性能分析](about_gpu/ai-compute.md) | |
| | [显存、搬移与互联](about_gpu/memory-interconnect.md) | |
| 算子优化 | [Reduce](reduce/reduce.md) | [reduce.cu](code/reduce/reduce.cu) |
| | [GEMM](gemm/gemm.md) | [gemm.cu](code/gemm/gemm.cu) |
| | [Softmax](softmax/softmax.md) | [softmax.cu](code/softmax/softmax.cu) |
| | [Online Softmax](softmax/online-softmax.md) | [online_softmax.cu](code/softmax/online_softmax.cu) |
| FlashAttention | [V1](flash-attention/v1.md)、[V2](flash-attention/v2.md) | [flash_attention.cu](code/flash_attention/flash_attention.cu) |

## 本地预览站点

需要 Node.js。

```bash
npm install
npm run docs:dev
```

## 运行算子代码

需要 Linux 或 WSL，以及 `nvcc`、`make` 和一块 NVIDIA GPU。

```bash
cd code
make test   # 编译并校验每个版本的输出
make perf   # 校验之后再测吞吐
```

更多用法以及代码和页面的差别，见 [code/README.md](code/README.md)。

## 数据口径

- 页面上标为“原稿数据”的 A100 数字来自原稿，我们没有 A100，无法复测。
- 标明 RTX 4060 Laptop 的数字是我们自己测的，不能直接推到其他 GPU。
- 代码在 RTX 4060 Laptop（compute capability 8.9）上校验过，也能为 A100（`sm_80`）编译，但没有在 A100 上运行过。

## 版权

- 算子文章的蓝本来自 AI Infra Guide，它的 README 声明使用 MIT 许可证。
- `about_gpu/images/` 里标为“官方原图”的图片版权归 NVIDIA 和 AMD 所有，这里仅用于学习说明，页面上已标明出处。
- 本项目自身的内容暂未选择开源许可证。
