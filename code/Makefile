# 在 WSL 或 Linux 下运行：make test、make perf、make clean
# 换编译器：make NVCC=/usr/local/cuda/bin/nvcc
# 给 A100 编译：make ARCH=sm_80
NVCC ?= nvcc
ARCH ?= native
FLAGS := -O3 -std=c++17 -arch=$(ARCH) -Icommon

PROGRAMS := reduce gemm softmax online_softmax flash_attention
BINS := $(addprefix build/,$(PROGRAMS))

all: $(BINS)

build/reduce: reduce/reduce.cu common/harness.h
build/gemm: gemm/gemm.cu common/harness.h
build/softmax: softmax/softmax.cu common/harness.h
build/online_softmax: softmax/online_softmax.cu common/harness.h
build/flash_attention: flash_attention/flash_attention.cu common/harness.h

$(BINS):
	@mkdir -p build
	$(NVCC) $(FLAGS) $< -o $@

# 校验每个版本的输出。有版本算错时退出码非 0
test: all
	@status=0; for p in $(BINS); do echo "=== $$p"; ./$$p || status=1; done; exit $$status

# 校验之后再测吞吐。数字只对当前这张卡有效
perf: all
	@for p in $(BINS); do echo "=== $$p"; ./$$p --perf || exit 1; done

clean:
	rm -rf build

.PHONY: all test perf clean
