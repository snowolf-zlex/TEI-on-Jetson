# TEI on Jetson

[English](README.md) | **简体中文**

> 在 NVIDIA Jetson Orin 上本地运行 [HuggingFace TEI](https://github.com/huggingface/text-embeddings-inference)（Text Embeddings Inference），
> 用 GPU 加速 embedding 和 reranker 推理，延迟从 SaaS API 的 80-150ms 降到 5-30ms。

## 目录

- [为什么需要这个项目](#为什么需要这个项目)
- [兼容性](#兼容性)
- [方案演进（10 轮踩坑记录）](#方案演进10-轮踩坑记录)
- [快速开始](#快速开始)
  - [选项 A：下载预编译二进制（推荐）](#选项-a下载预编译二进制推荐1-分钟)
  - [选项 B：自己编译](#选项-b自己编译可定制5-9-小时)
- [验收](#验收)
- [编译耗时实测](#编译耗时实测)
- [使用场景](#使用场景)
- [文件说明](#文件说明)
- [运行时配置要点](#运行时配置要点)
- [已知限制](#已知限制)
- [FAQ](#faq)
- [License](#license)

## 使用场景

### 什么时候需要 TEI

任何需要 **文本向量化（embedding）或相关性重排（reranker）** 的应用：

| 场景 | 用 embedding 做什么 | 用 reranker 做什么 |
| --- | --- | --- |
| **RAG / 知识库问答** | 文档切片向量化，存入向量数据库 | 对召回结果按相关性精排 |
| **语义搜索** | query 向量化，与文档向量比对 | 对 top-N 结果重排提升精度 |
| **文档去重 / 聚类** | 计算文档间相似度 | — |
| **推荐系统** | 用户/物品向量化 | 对候选列表重排 |
| **多语言检索** | 跨语言语义匹配（BGE-m3 支持 100+ 语言） | 跨语言相关性排序 |

### 为什么本地部署而非 SaaS API

| 维度 | SaaS API（硅基流动/OpenAI/Jina） | 本地 TEI |
| --- | --- | --- |
| **延迟** | 80-150ms（公网 RTT 主导） | **5-30ms**（内网/本地） |
| **数据隐私** | 文本发送到第三方服务器 | **数据不出本地**（PII/敏感文档） |
| **成本** | 按调用计费（$0.01-0.1/千次） | **零边际成本**（已摊入硬件） |
| **离线** | 不可用 | **完全离线可用** |
| **并发** | 受 API 限流 | 仅受 GPU 算力限制 |

### 为什么选 Jetson

| 设备类型 | 为什么选 / 不选 |
| --- | --- |
| **Jetson Orin ✅** | ARM64 + GPU + 低功耗（15-60W），适合边缘部署和私有化场景；已有 Jetson 设备可复用 |
| 服务器 GPU（A100/H100） | 性能最强但成本高（$10K+）、功耗高（300W+）、不适合边缘场景 |
| Mac mini | Docker 不支持 Metal GPU 直通；适合个人 LLM 推理但不适合容器化服务部署 |
| DGX Spark | 官方 TEI 预编译镜像，最省心；但 $3,999 需额外购买 |

**典型部署形态**：Jetson Orin NX 16GB 作为 Agent Studio 的推理节点，
embedding/reranker 延迟从 SaaS API 的 80-150ms 降到 5-30ms，
同时所有文本处理不出本地——适合隐私敏感的知识库场景。

---

## 为什么需要这个项目

HuggingFace TEI 官方只发布 **x86 CUDA** 预编译镜像。Jetson 是 **ARM64 + Tegra GPU**，有三重不兼容：

1. 官方镜像没有 arm64 manifest
2. 标准 `nvidia/cuda` 镜像的 cuBLAS 与 Jetson Tegra driver 不兼容
3. candle（TEI 的 ML 后端）有 sm_87 compute capability 检测 bug

本项目记录了完整的解决方案——从编译、调试到部署验证。

---

## 兼容性

### 验证环境

本仓库的二进制和文档基于以下环境编译和测试：

| 组件 | 版本 | 验证命令 |
| --- | --- | --- |
| **设备** | Jetson Orin NX 16GB | `cat /proc/device-tree/model` |
| **JetPack** | 6.2（L4T R36.5.2） | `cat /etc/nv_tegra_release` |
| **Ubuntu** | 22.04.5 LTS（Jammy） | `cat /etc/os-release \| grep VERSION` |
| **内核** | 5.15.199-tegra | `uname -r` |
| **架构** | aarch64 | `uname -m` |
| **glibc** | 2.35 | `ldd --version` |
| **CUDA Toolkit** | 12.6.68 | `nvcc --version` |
| **cuBLAS** | 12.6.1.4 | `dpkg -l \| grep libcublas` |
| **cuDNN** | 9.3.0.75 | `dpkg -l \| grep cudnn` |
| **TensorRT** | 10.3.0.30 | `dpkg -l \| grep libnvinfer` |
| **PyTorch** | 2.5.0a0+nv24.08 | `python3 -c "import torch; print(torch.__version__)"` |
| **Python** | 3.10.12 | `python3 --version` |
| **Docker** | 29.1.3 | `docker --version` |
| **nvidia-container-toolkit** | 1.16.2 | `dpkg -l \| grep libnvidia-container` |
| **Rust（编译容器）** | 1.92.0 | `rustc --version` |

> 如果你的版本差异较大（特别是 CUDA、glibc 或 JetPack），预编译二进制可能不适用。请用选项 B 自行编译。

### 最低要求

| 要求 | 版本 | 原因 |
| --- | --- | --- |
| GPU | Orin 系列（**sm_87**） | nvprune 生成 sm_80+sm_87 SASS 代码 |
| JetPack | **6.x**（CUDA 12.x） | candle 依赖的 cudarc 使用 CUDA 12+ API |
| 内存（编译） | **≥16GB** | candle-flash-attn 编译峰值 ~10GB |
| 内存（运行） | **≥8GB** | 模型常驻 ~3GB + 系统 + GPU 显存 |

### 设备 × JetPack 兼容性

| 设备 | sm | JetPack 6（CUDA 12.6） | JetPack 5（CUDA 11.4） | JetPack 4（CUDA 10.2） |
| --- | --- | --- | --- | --- |
| AGX Orin 64GB | sm_87 | ✅ **最佳** | ❌ CUDA 太低 | ❌ 不支持 |
| AGX Orin 32GB | sm_87 | ✅ | ❌ | ❌ |
| Orin NX 16GB | sm_87 | ✅ **已验证** | ❌ | ❌ |
| Orin NX 8GB | sm_87 | ⚠️ 编译紧张，仅运行 | ❌ | ❌ |
| Orin Nano 8GB | sm_87 | ⚠️ 仅运行预编译 | ❌ | ❌ |
| Orin Nano 4GB | sm_87 | ❌ 内存不够 | ❌ | ❌ |
| Xavier NX / AGX | sm_72 | ❌ 不支持 JP6 | ❌ sm_72 | ❌ |
| TX2 | sm_62 | ❌ | ❌ | ❌ |
| Nano / TX1 | sm_53 | ❌ | ❌ | ❌ |

**JP5（CUDA 11.4）不行**：即使 Orin 在 JP5 上也是 sm_87，candle 的 cudarc 仍需要 CUDA 12+ API。

---

## 方案演进（10 轮踩坑记录）

| # | 方案 | 结果 | 根因 |
| --- | --- | --- | --- |
| ① | Mac buildx 交叉构建 arm64 | ❌ | Mac Docker daemon 连不上 Docker Hub（EOF） |
| ② | NX 原生构建，JOBS=2 | ❌ OOM 死机 | candle-flash-attn 编译每进程 ~4GB，峰值 17GB > 16GB |
| ③ | NX 原生构建，JOBS=1 | ✅ 镜像构建成功 | 耗时 9 小时（candle-flash-attn 占 67%） |
| ④ | 启动容器 `runtime: nvidia` | ❌ | nvidia-container-runtime 版本检查（cuda≥12.9 vs driver 12.6） |
| ⑤ | `NVIDIA_DISABLE_REQUIRE` + 库挂载 | ❌ | entrypoint 的 compat 逻辑把 12.9 driver 加到 LD_LIBRARY_PATH 最前 |
| ⑥ | 覆盖 entrypoint 跳过 compat | ❌ | candle `compute_cap_matching(87,87)` 返回 false（源码 bug） |
| ⑦ | 修复 compute_cap.rs | ❌ | cuBLAS 12.9 静态链接 vs driver 12.6 → `CUBLAS_STATUS_ALLOC_FAILED` |
| ⑧ | **bind mount 宿主 CUDA 12.6 toolkit** | ✅ 编译通过 | cuBLAS 12.6 匹配 driver，但运行仍 `CUBLAS_STATUS_ALLOC_FAILED` |
| ⑨ | **`dynamic-linking` 替代 `static-linking`** | ✅ cuBLAS 解决 | nvprune 静态裁剪的 cuBLAS 损坏；动态链接宿主 `libcublas.so` 成功 |
| ⑩ | PTX 版本不兼容 → 已修复 | ✅ | 清理 12.9 PTX 缓存污染，强制校验 12.6 PTX 8.5 |

### cuBLAS 验证状态

`CUBLAS_STATUS_ALLOC_FAILED` 的根因是 **nvprune**——TEI 的 Dockerfile 在 `static-linking` 模式下
对 `libcublas_static.a` 执行 nvprune，裁剪掉了 Jetson Tegra 特有的 cuBLAS 代码路径。
修复方案是 `-F dynamic-linking`——cuBLAS 动态链接宿主的 `libcublas.so`（通过 bind mount）。

**已验证通过：**

| 测试 | 结果 | 方式 |
| --- | --- | --- |
| 宿主 cuBLAS 初始化 | ✅ `cublasCreate: 0` | 宿主 nvcc + 宿主 cuBLAS 12.6 |
| 宿主 cuBLAS Sgemm | ✅ 结果正确 | 2×2 矩阵乘法 |
| 容器 GPU 内存分配 | ✅ `cudaMalloc: 0` | nvidia runtime 容器 |
| 容器 cuBLAS 初始化 | ✅ `cublasCreate: 0` | 容器 + bind mount 宿主 CUDA 库 |
| 容器 cuBLAS Sgemm | ✅ 结果正确 | 动态链接 `libcublas.so.12.6` |
| **TEI GPU 检测** | ✅ `Cuda(CudaDevice(DeviceId(1)))` | TEI 二进制（dynamic-linking）在容器中 |
| **TEI cuBLAS 初始化** | ✅ 无 `CUBLAS_STATUS_ALLOC_FAILED` | TEI 启动成功，两个模型均可加载 |
| **TEI 端到端推理** | ✅ 20/20 验收通过 | GPU 上执行 `/embed` + `/rerank` |

**PTX 问题解决状态：**

| 检查 | 结果 | 方式 |
| --- | --- | --- |
| CUDA 12.9 遗留 PTX 缓存 | ✅ 最终编译前清理 | `scripts/build-tei-jetson.sh` 隔离 Candle CUDA output、fingerprint 和 rlib |
| Candle `cast.ptx` 版本 | ✅ `.version 8.5` | 不是 CUDA 12.6 PTX 时构建 fail closed |
| TEI kernel 加载 | ✅ 无 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` | 最终运行时验收 |

---

## 快速开始

### 0. 前置准备（Jetson 上，两种路径都需要）

```bash
# 确认环境
cat /etc/nv_tegra_release          # JetPack 6.x (R36.x)
/usr/local/cuda/bin/nvcc --version # CUDA 12.x
free -h                            # ≥16GB（编译）或 ≥8GB（运行预编译）

# 验证宿主 cuBLAS 正常（必须通过）
cat > /tmp/cublas_test.cu << 'EOF'
#include <cstdio>
#include <cublas_v2.h>
int main() {
    cublasHandle_t h;
    printf("cublasCreate: %d\n", (int)cublasCreate(&h));
}
EOF
/usr/local/cuda/bin/nvcc -o /tmp/cublas_test /tmp/cublas_test.cu -lcublas
/tmp/cublas_test  # 必须输出 cublasCreate: 0

# 下载模型（用 hf-mirror，直连 huggingface.co 超时）
pip3 install --user huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
hf download BAAI/bge-m3 --exclude "*.DS_Store" --local-dir ~/models/bge-m3
hf download BAAI/bge-reranker-v2-m3 --exclude "*.DS_Store" --local-dir ~/models/bge-reranker-v2-m3
```

### 选项 A：下载预编译二进制（推荐，~1 分钟）

跳过 5-9 小时编译，直接从 GitHub Releases 下载已编译好的二进制（1.1GB，含 sm_87 修复）。

```bash
# 下载预编译二进制（aarch64, CUDA 12.6, sm_87）
# 替换 VERSION 为实际 release tag（如 v0.1.0）
curl -L "https://github.com/snowolf-zlex/TEI-on-Jetson/releases/download/VERSION/text-embeddings-router-sm87-cuda126" \
  -o text-embeddings-router
chmod +x text-embeddings-router

# 获取 cuda-entrypoint.sh
curl -sL "https://raw.githubusercontent.com/huggingface/text-embeddings-inference/main/cuda-entrypoint.sh" \
  -o cuda-entrypoint.sh && chmod +x cuda-entrypoint.sh

# 构建精简运行时镜像（~30 秒）
docker build -f docker/Dockerfile.tei-runtime -t tei:jetson-runtime .
```

> **二进制兼容性**：`text-embeddings-router-sm87-cuda126` 适用于
> JetPack 6.x + Orin sm_87 + CUDA 12.6。不兼容 JP5/Xavier/其他 CUDA 版本。
> 如需其他配置，用选项 B 自己编译。

### 选项 B：自己编译（可定制，5-9 小时）

适合需要修改源码、禁用 flash attention、或使用不同 CUDA 版本的场景。

```bash
# 获取 TEI 源码（用 codeload tarball，不用 git clone——git-lfs 导致不完整）
mkdir -p tei-src
curl -sL "https://codeload.github.com/huggingface/text-embeddings-inference/tar.gz/refs/heads/main" \
  | tar xz -C tei-src --strip-components=1

# 获取缺失的 cuda-entrypoint.sh
curl -sL "https://raw.githubusercontent.com/huggingface/text-embeddings-inference/main/cuda-entrypoint.sh" \
  -o tei-src/cuda-entrypoint.sh && chmod +x tei-src/cuda-entrypoint.sh

# 应用 compute_cap sm_87 修复
patch -d tei-src -p1 < patches/compute_cap-sm87-fix.patch

# 步骤 A：构建 builder 镜像（cargo chef cook 预编译依赖，4-7 小时）
docker build -f docker/Dockerfile.tei-builder \
  --target builder \
  --platform linux/arm64 \
  --build-arg CUDA_COMPUTE_CAP=87 \
  --build-arg CARGO_BUILD_JOBS=1 \
  --build-arg RAYON_NUM_THREADS=1 \
  -t tei:builder \
  tei-src

# 步骤 B：用硬 CPU/内存限制执行 dynamic-linking 最终编译
TEI_BUILD_CONTAINER=tei-build bash scripts/build-tei-jetson.sh

# 等待编译完成
docker logs -f tei-build   # 看到 BUILD_OK 或 Finished

# 步骤 C：拷贝二进制 + 创建精简运行时镜像
docker cp tei-build:/usr/src/target/release/text-embeddings-router .
docker build -f docker/Dockerfile.tei-runtime -t tei:jetson-runtime .
```

> **想跳过 flash attention 编译（省 6 小时）？**
> 去掉 `--features candle-cuda` 中的 flash-attn 相关参数，
> 设置 `USE_FLASH_ATTENTION=false`。embedding/reranker 性能差异很小（短序列场景）。

### 部署

```bash
# 复制 compose 模板
cp docker/docker-compose.tei.yml docker-compose.override.yml
# 启动 TEI 服务
docker compose up -d tei-embedding tei-reranker
# 等待模型加载（约 30 秒）
docker logs -f agent-studio-tei-embedding-1  # 看到 "Starting ... on Cuda(...)"
```

## 验收

> ✅ **已验证**：2026-08-12，Jetson Orin NX 16GB，JetPack 6.2.3，CUDA 12.6。**20/20 通过。**

```bash
bash scripts/verify-tei.sh all
```

完整验收方法、固定测试输入、实测结果和兼容性说明见
[docs/validation.zh-CN.md](docs/validation.zh-CN.md)。

#### 阶段 1：编译验收

| 检查项 | 预期 | 结果 |
| --- | --- | --- |
| 镜像 `tei:jetson-runtime` 存在 | <500MB | ✅ 161MB |
| 二进制为 ARM64 ELF | `ELF 64-bit ARM aarch64` | ✅ |
| compute_cap sm_87 修复 | 二进制含 `80..=89` 分支 | ✅ |
| 宿主 cuBLAS | `cublasCreate: 0` | ✅ |

#### 阶段 2：安装调试验收

| 检查项 | 预期 | 结果 |
| --- | --- | --- |
| 两容器启动 + healthy | `docker ps` 可见 | ✅ |
| GPU 推理（非 CPU fallback） | 日志含 `Starting.*Cuda` | ✅ |
| `/health` 返回 HTTP 200 | TEI 1.9.3 body 为空 | ✅ |
| GPU 利用率（tegrastats） | embed 调用时 `GR3D_FREQ > 0%` | ✅ |

#### 阶段 3：功能验收

| 检查项 | 预期 | 结果 |
| --- | --- | --- |
| Embedding 维度 | 1024 维 | ✅ 1024 |
| Embedding 延迟 | <50ms（vs SaaS 80-150ms） | ✅ **13.7ms** |
| Rerank 排序正确性 | 按相关性排序 | ✅ |
| Rerank 延迟 | <100ms | ✅ **22.5ms** |

实测输出摘要：
```
✓ 镜像 tei:jetson-runtime 存在（161MB）
✓ 二进制为 ARM64 ELF
✓ compute_cap sm_87 修复已包含
✓ tei-embedding 容器运行中
✓ embedding 使用 GPU（日志含 Cuda）
✓ embedding /health 返回 HTTP 200
✓ embedding 返回 1024 维向量
✓ embedding 平均延迟: 13.7ms
✓ rerank 返回 3 个结果（含 score）
✓ rerank 平均延迟: 22.5ms
═══════════════════════════════════════════════════════════════
  结果: 20 通过 / 0 失败
═══════════════════════════════════════════════════════════════
```

---

## 编译耗时实测

### 首次完整构建（Jetson Orin NX 16GB, JOBS=1）

```
9 小时总时间分解：
├── nvidia/cuda 镜像拉取 ......... 6 分钟
├── apt + sccache ................. 25 分钟（Jetson apt 慢）
├── rustup installer .............. 18.5 分钟（AWS CloudFront 不可控）
├── Rust toolchain ................. 15 分钟（USTC 镜像）
├── ★ candle-flash-attn 33 kernel . 6 小时（占总时间 67%）
├── 非 CUDA Rust crate ............ 40 分钟
├── 最终 cargo build .............. 30 分钟
└── 镜像 export ................... 72 秒
```

### candle-flash-attn 33 个 CUDA kernel 编译耗时

| head_dim | 变体数 | 单个耗时（JOBS=1） |
| --- | --- | --- |
| hdim32 | 4 | 5-8 分钟 |
| hdim64 | 4 | 8-12 分钟 |
| hdim96 | 4 | 10-14 分钟 |
| hdim128 | 4 | 14-20 分钟 |
| hdim160 | 4 | 17-23 分钟 |
| hdim192 | 4 | 20-25 分钟 |
| hdim224 | 4 | 22-28 分钟 |
| hdim256 | 4 | 28-40 分钟 |

历史 JOBS=3 并行构建约 3-4 小时，但这不是安全基线：单个 build script 仍可能启动多个
`nvcc`/`cicc`。请使用 `scripts/build-tei-jetson.sh`，通过 Docker cgroup 给宿主系统保留 CPU 和内存。

### 为什么 JOBS=2 会 OOM 死机

```
内存峰值：
  系统 + Docker + 生产容器 ........ ~4GB
  Tegra GPU 显存预留 ............. ~1GB
  cargo + rustc .................. ~2GB
  cicc × 2 并行 .................. ~8GB（每个 ~4GB）
  rayon 内部并行 ................. ~2GB
  ─────────────────────────────────
  总计 .......................... ~17GB > 16GB → swap 风暴 → 死机
```

**必须使用 `scripts/build-tei-jetson.sh`**。它同时设置 Docker `--cpus=4 --memory=10g`、
`CARGO_BUILD_JOBS=1`、`RAYON_NUM_THREADS=1`、`CMAKE_BUILD_PARALLEL_LEVEL=1` 和
`NVCC_THREADS=1`。

---

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `docker/Dockerfile.tei-builder` | 编译用（Rust 工具链 + nvcc + USTC 镜像 + compute_cap 修复） |
| `docker/Dockerfile.tei-runtime` | 精简运行时镜像（~200MB，不含编译工具链） |
| `docker/docker-compose.tei.yml` | 服务编排（CUDA 库挂载 + nvhost 设备） |
| `patches/compute_cap-sm87-fix.patch` | candle `compute_cap_matching(87,87)` bug 修复 |
| `scripts/verify-tei.sh` | 三阶段验收脚本（编译 → 安装 → 功能） |

---

## 运行时配置要点

标准 NVIDIA Docker 不需要这些，但 Jetson Tegra 必须：

| 配置 | 原因 |
| --- | --- |
| bind mount `/usr/local/cuda/lib64` | 提供宿主 CUDA 12.6 runtime/cuBLAS 动态库 |
| bind mount `/usr/lib/aarch64-linux-gnu/tegra` | Tegra GPU driver 库（`libcuda.so` 的 Jetson 实现） |
| `devices: /dev/nvhost-*` | Jetson GPU 设备节点（CUDA 内存分配需要 nvhost 接口） |
| 直接 router entrypoint | 绕过 CUDA 12.9 compat wrapper，使用 Jetson 宿主库 |

---

## 已知限制

1. **编译成本极高**：首次 5-9 小时，candle-flash-attn 的 33 个 CUDA kernel 占 67%
2. **无增量编译**：bindgen_cuda 每次全量重编译 CUDA kernel，改一行代码 → 4-7 小时
3. **JetPack 版本绑定**：dynamic binary 依赖 CUDA 12 `libcublas.so.12`；CUDA 13 / JetPack 7 需重新编译和验收
4. **仅 Orin sm_87 + JetPack 6.x**：Xavier/TX2/Nano 和 JP5/JP4 不支持
5. **Flash Attention 收益有限**：33 个 kernel 的编译开销巨大，但 embedding/reranker 的短序列推理收益很小

---

## FAQ

**Q: 下载的二进制和镜像为什么不放 Git 仓库？**

Git 单文件限制 100MB，二进制 1.1GB 超限。预编译二进制放在
[GitHub Releases](https://github.com/snowolf-zlex/TEI-on-Jetson/releases)（限制 2GB/文件）。
Dockerfile 在仓库里，下载二进制后 30 秒即可 build 出运行时镜像。

**Q: 能不能跳过 candle-flash-attn 编译？**

可以。修改 Dockerfile 去掉 `USE_FLASH_ATTENTION=true`，用标准 attention 替代。
编译时间从 9 小时降到约 1 小时。embedding/reranker 性能差异很小（短序列场景）。

**Q: 能不能在其他 Jetson 上直接用编译好的镜像？**

同 JetPack 6.2.x 的 Orin 系列（AGX/NX/Nano 8GB+）可以直接 `docker load`。
不同 JetPack 或非 Orin 设备不兼容。详见上方兼容性矩阵。

**Q: 为什么不用 Ollama 跑 embedding？**

Ollama 的 embedding 是"附赠"功能，模型选择少、无 reranker、API 不兼容。
TEI 专门优化 embedding/reranker，支持 OpenAI/Jina 兼容 API。

**Q: DGX Spark 上需要这个方案吗？**

不需要。DGX Spark 有 TEI 官方预编译镜像（Blackwell + CUDA 12.9+），直接 `docker pull` 即用。
本方案仅用于 Jetson（Tegra GPU + 特定 JetPack 版本）。

**Q: Release 里的二进制是什么版本？**

| 文件 | 架构 | CUDA | sm | TEI 版本 | 大小 |
| --- | --- | --- | --- | --- | --- |
| `text-embeddings-router-sm87-cuda126` | aarch64 | 12.6 | sm_87 | 1.9.3 (main) | ~1.1GB |

二进制动态加载宿主 cuBLAS/cuBLASLt，并包含 compute_cap sm_87 修复和 candle-flash-attn kernels。
不包含模型文件（需单独下载）。

---

## License

MIT
