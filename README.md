# TEI on Jetson

**English** | [简体中文](README.zh-CN.md)

> Run [HuggingFace TEI](https://github.com/huggingface/text-embeddings-inference) (Text Embeddings Inference)
> locally on NVIDIA Jetson Orin, using GPU acceleration for embedding and reranker inference.
> Reduces latency from 80-150ms (SaaS API) to 5-30ms (local GPU).

## Table of Contents

- [Why This Project](#why-this-project)
- [Compatibility](#compatibility)
- [Solution Evolution (10 Iterations)](#solution-evolution-10-iterations)
- [Quick Start](#quick-start)
  - [Option A: Download Pre-built Binary (Recommended)](#option-a-download-pre-built-binary-recommended-1-min)
  - [Option B: Build from Source](#option-b-build-from-source-customizable-5-9-hours)
- [Validation](#validation)
- [Build Time Analysis](#build-time-analysis)
- [Files](#files)
- [Runtime Configuration](#runtime-configuration)
- [Use Cases](#use-cases)
- [Known Limitations](#known-limitations)
- [FAQ](#faq)
- [License](#license)

## Use Cases

### When Do You Need TEI

Any application requiring **text embedding or reranking**:

| Scenario | Embedding | Reranker |
| --- | --- | --- |
| **RAG / Knowledge Base QA** | Vectorize document chunks for vector DB | Re-rank retrieved results by relevance |
| **Semantic Search** | Vectorize queries, compare with documents | Re-rank top-N for precision |
| **Dedup / Clustering** | Compute document similarity | — |
| **Recommendation** | User/item embeddings | Re-rank candidate lists |
| **Multilingual Retrieval** | Cross-lingual semantic matching (BGE-m3: 100+ languages) | Cross-lingual relevance scoring |

### Why Local Instead of SaaS API

| Factor | SaaS API (SiliconFlow/OpenAI/Jina) | Local TEI |
| --- | --- | --- |
| **Latency** | 80-150ms (public network RTT) | **5-30ms** (local) |
| **Privacy** | Text sent to third-party servers | **Data never leaves your machine** |
| **Cost** | Per-call pricing ($0.01-0.1/1K calls) | **Zero marginal cost** |
| **Offline** | Not available | **Fully offline** |
| **Concurrency** | API rate limits | Only limited by GPU |

### Why Jetson

| Hardware | Verdict |
| --- | --- |
| **Jetson Orin ✅** | ARM64 + GPU + low power (15-60W); ideal for edge/private deployment; reuse existing Jetson hardware |
| Server GPU (A100/H100) | Best performance but $10K+, 300W+, not suitable for edge |
| Mac mini | Docker doesn't support Metal GPU passthrough; good for personal LLM but not containerized services |
| DGX Spark | Official TEI pre-built image, easiest path; but $3,999 additional cost |

**Typical deployment**: Jetson Orin NX 16GB as an inference node for AI applications,
reducing embedding/reranker latency from 80-150ms (SaaS) to 5-30ms (local),
with all text processing staying on-device — ideal for privacy-sensitive knowledge base scenarios.

---

## Why This Project

HuggingFace TEI only ships **x86 CUDA** pre-built images. Jetson is **ARM64 + Tegra GPU** with three incompatibilities:

1. No arm64 manifest in official images
2. Standard `nvidia/cuda` image's cuBLAS is incompatible with Jetson Tegra driver
3. candle (TEI's ML backend) has an sm_87 compute capability detection bug

This project documents the complete solution — from compilation, debugging, to deployment verification.

---

## Compatibility

### Verified Environment

The binary and instructions in this repo were built and tested on:

| Component | Version | Check Command |
| --- | --- | --- |
| **Device** | Jetson Orin NX 16GB | `cat /proc/device-tree/model` |
| **JetPack** | 6.2 (L4T R36.5.2) | `cat /etc/nv_tegra_release` |
| **Ubuntu** | 22.04.5 LTS (Jammy) | `cat /etc/os-release \| grep VERSION` |
| **Kernel** | 5.15.199-tegra | `uname -r` |
| **Architecture** | aarch64 | `uname -m` |
| **glibc** | 2.35 | `ldd --version` |
| **CUDA Toolkit** | 12.6.68 | `nvcc --version` |
| **cuBLAS** | 12.6.1.4 | `dpkg -l \| grep libcublas` |
| **cuDNN** | 9.3.0.75 | `dpkg -l \| grep cudnn` |
| **TensorRT** | 10.3.0.30 | `dpkg -l \| grep libnvinfer` |
| **PyTorch** | 2.5.0a0+nv24.08 | `python3 -c "import torch; print(torch.__version__)"` |
| **Python** | 3.10.12 | `python3 --version` |
| **Docker** | 29.1.3 | `docker --version` |
| **nvidia-container-toolkit** | 1.16.2 | `dpkg -l \| grep libnvidia-container` |
| **Rust (in builder)** | 1.92.0 | `rustc --version` |

> If your versions differ significantly (especially CUDA, glibc, or JetPack), the pre-built binary may not work. Use Option B to build from source.

### Minimum Requirements

| Requirement | Version | Reason |
| --- | --- | --- |
| GPU | Orin family (**sm_87**) | nvprune generates sm_80+sm_87 SASS code |
| JetPack | **6.x** (CUDA 12.x) | candle's cudarc uses CUDA 12+ APIs |
| RAM (compile) | **≥16GB** | candle-flash-attn compilation peaks at ~10GB |
| RAM (runtime) | **≥8GB** | Model resident ~3GB + system + GPU memory |

### Device × JetPack Matrix

| Device | sm | JetPack 6 (CUDA 12.6) | JetPack 5 (CUDA 11.4) | JetPack 4 (CUDA 10.2) |
| --- | --- | --- | --- | --- |
| AGX Orin 64GB | sm_87 | ✅ **Best** | ❌ CUDA too old | ❌ Not supported |
| AGX Orin 32GB | sm_87 | ✅ | ❌ | ❌ |
| Orin NX 16GB | sm_87 | ✅ **Verified** | ❌ | ❌ |
| Orin NX 8GB | sm_87 | ⚠️ Tight for compile, OK for runtime | ❌ | ❌ |
| Orin Nano 8GB | sm_87 | ⚠️ Runtime only (pre-built) | ❌ | ❌ |
| Orin Nano 4GB | sm_87 | ❌ Not enough RAM | ❌ | ❌ |
| Xavier NX / AGX | sm_72 | ❌ JP6 not supported | ❌ sm_72 | ❌ |
| TX2 | sm_62 | ❌ | ❌ | ❌ |
| Nano / TX1 | sm_53 | ❌ | ❌ | ❌ |

**JP5 (CUDA 11.4) doesn't work**: Even though Orin on JP5 is sm_87, candle's cudarc still requires CUDA 12+ APIs.

---

## Solution Evolution (10 Iterations)

| # | Approach | Result | Root Cause |
| --- | --- | --- | --- |
| ① | Mac buildx cross-compile arm64 | ❌ | Mac Docker daemon can't reach Docker Hub (EOF) |
| ② | NX native build, JOBS=2 | ❌ OOM crash | candle-flash-attn ~4GB/process, peak 17GB > 16GB |
| ③ | NX native build, JOBS=1 | ✅ Image built | 9 hours (flash-attn = 67% of time) |
| ④ | Start container `runtime: nvidia` | ❌ | nvidia-container-runtime version check (cuda≥12.9 vs driver 12.6) |
| ⑤ | `NVIDIA_DISABLE_REQUIRE` + lib mount | ❌ | entrypoint compat logic puts 12.9 driver first in LD_LIBRARY_PATH |
| ⑥ | Override entrypoint to skip compat | ❌ | candle `compute_cap_matching(87,87)` returns false (source bug) |
| ⑦ | Fix compute_cap.rs | ❌ | cuBLAS 12.9 static-linked vs driver 12.6 → `CUBLAS_STATUS_ALLOC_FAILED` |
| ⑧ | **Bind mount host CUDA 12.6 toolkit** | ✅ compiles | cuBLAS 12.6 matches driver, but still `CUBLAS_STATUS_ALLOC_FAILED` |
| ⑨ | **`dynamic-linking` instead of `static-linking`** | ✅ cuBLAS works | nvprune static-cropped cuBLAS broken; dynamic link to host `libcublas.so` works |
| ⑩ | PTX version mismatch → fixed | ✅ | `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` fixed by purging 12.9 cache and validating CUDA 12.6 PTX 8.5 |

### cuBLAS Verification Status

The root cause of `CUBLAS_STATUS_ALLOC_FAILED` was **nvprune** — TEI's Dockerfile runs nvprune on
`libcublas_static.a` (static linking mode), which corrupts the Tegra-specific cuBLAS code paths.
The fix is `-F dynamic-linking` — cuBLAS dynamically links to the host's `libcublas.so` via bind mount.

**What's verified working:**

| Test | Result | Method |
| --- | --- | --- |
| Host cuBLAS init | ✅ `cublasCreate: 0` | Host nvcc + host cuBLAS 12.6 |
| Host cuBLAS Sgemm | ✅ Correct result | 2×2 matrix multiply |
| Container GPU alloc | ✅ `cudaMalloc: 0` | Container with nvidia runtime |
| Container cuBLAS init | ✅ `cublasCreate: 0` | Container + bind mount host CUDA libs |
| Container cuBLAS Sgemm | ✅ Correct result | Dynamic-linked `libcublas.so.12.6` |
| **TEI GPU detection** | ✅ `Cuda(CudaDevice(DeviceId(1)))` | TEI binary (dynamic-linking) in container |
| **TEI cuBLAS init** | ✅ No `CUBLAS_STATUS_ALLOC_FAILED` | TEI binary starts and loads both models |
| **TEI end-to-end inference** | ✅ 20/20 validation passed | `/embed` + `/rerank` on GPU |

**PTX issue resolution:**

| Check | Result | Method |
| --- | --- | --- |
| Stale CUDA 12.9 PTX cache | ✅ Purged before final build | `scripts/build-tei-jetson.sh` quarantines Candle CUDA outputs, fingerprints, and rlibs |
| Candle `cast.ptx` version | ✅ `.version 8.5` | Build fails closed unless CUDA 12.6 PTX is present |
| TEI kernel loading | ✅ No `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` | Final runtime validation |

---

## Quick Start

### Prerequisites (on Jetson, required for both options)

```bash
# Verify environment
cat /etc/nv_tegra_release          # JetPack 6.x (R36.x)
/usr/local/cuda/bin/nvcc --version # CUDA 12.x
free -h                            # ≥16GB (compile) or ≥8GB (runtime only)

# Verify host cuBLAS works (must pass)
cat > /tmp/cublas_test.cu << 'EOF'
#include <cstdio>
#include <cublas_v2.h>
int main() {
    cublasHandle_t h;
    printf("cublasCreate: %d\n", (int)cublasCreate(&h));
}
EOF
/usr/local/cuda/bin/nvcc -o /tmp/cublas_test /tmp/cublas_test.cu -lcublas
/tmp/cublas_test  # Must output: cublasCreate: 0

# Download models (use hf-mirror, direct huggingface.co times out)
pip3 install --user huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
hf download BAAI/bge-m3 --exclude "*.DS_Store" --local-dir ~/models/bge-m3
hf download BAAI/bge-reranker-v2-m3 --exclude "*.DS_Store" --local-dir ~/models/bge-reranker-v2-m3
```

### Option A: Download Pre-built Binary (Recommended, ~1 min)

Skip the 5-9 hour compilation. Download the pre-built binary (1.1GB, includes sm_87 fix) from GitHub Releases.

```bash
# Download pre-built binary (aarch64, CUDA 12.6, sm_87)
# Replace VERSION with actual release tag (e.g., v0.1.0)
curl -L "https://github.com/snowolf-zlex/TEI-on-Jetson/releases/download/VERSION/text-embeddings-router-sm87-cuda126" \
  -o text-embeddings-router
chmod +x text-embeddings-router

# Get cuda-entrypoint.sh
curl -sL "https://raw.githubusercontent.com/huggingface/text-embeddings-inference/main/cuda-entrypoint.sh" \
  -o cuda-entrypoint.sh && chmod +x cuda-entrypoint.sh

# Build slim runtime image (~30 seconds)
docker build -f docker/Dockerfile.tei-runtime -t tei:jetson-runtime .
```

> **Binary compatibility**: `text-embeddings-router-sm87-cuda126` works on
> JetPack 6.x + Orin sm_87 + CUDA 12.6. Not compatible with JP5/Xavier/other CUDA versions.
> For other configurations, use Option B to build from source.

### Option B: Build from Source (Customizable, 5-9 hours)

For scenarios requiring source modifications, disabling flash attention, or different CUDA versions.

```bash
# Get TEI source (use codeload tarball, not git clone — git-lfs causes incomplete checkout)
mkdir -p tei-src
curl -sL "https://codeload.github.com/huggingface/text-embeddings-inference/tar.gz/refs/heads/main" \
  | tar xz -C tei-src --strip-components=1

# Get missing cuda-entrypoint.sh
curl -sL "https://raw.githubusercontent.com/huggingface/text-embeddings-inference/main/cuda-entrypoint.sh" \
  -o tei-src/cuda-entrypoint.sh && chmod +x tei-src/cuda-entrypoint.sh

# Apply compute_cap sm_87 fix
patch -d tei-src -p1 < patches/compute_cap-sm87-fix.patch

# Step A: Build builder image (cargo chef cook pre-compiles dependencies, 4-7 hours)
docker build -f docker/Dockerfile.tei-builder \
  --target builder \
  --platform linux/arm64 \
  --build-arg CUDA_COMPUTE_CAP=87 \
  --build-arg CARGO_BUILD_JOBS=1 \
  --build-arg RAYON_NUM_THREADS=1 \
  -t tei:builder \
  tei-src

# Step B: Final dynamic-linking build with hard CPU/memory limits
TEI_BUILD_CONTAINER=tei-build bash scripts/build-tei-jetson.sh

# Wait for compilation
docker logs -f tei-build   # Look for BUILD_OK or Finished

# Step C: Copy binary + build slim runtime image
docker cp tei-build:/usr/src/target/release/text-embeddings-router .
docker build -f docker/Dockerfile.tei-runtime -t tei:jetson-runtime .
```

> **Want to skip flash attention compilation (saves 6 hours)?**
> Remove flash-attn related args from `--features candle-cuda`,
> set `USE_FLASH_ATTENTION=false`. Minimal performance impact for embedding/reranker (short sequences).

### Deploy

```bash
# Start TEI services
docker compose -f docker/docker-compose.tei.yml up -d tei-embedding tei-reranker
# Wait for model loading (~30 seconds)
docker logs -f agent-studio-tei-embedding-1  # Look for "Starting ... on Cuda(...)"
```

## Validation

> ✅ **Verified**: 2026-08-12 on Jetson Orin NX 16GB, JetPack 6.2.3, CUDA 12.6. **20/20 passed.**

```bash
bash scripts/verify-tei.sh all
```

Full method, fixed test inputs, results, and compatibility notes are in
[docs/validation.md](docs/validation.md).

#### Stage 1: Compilation Verification

| Check | Expected | Result |
| --- | --- | --- |
| Image `tei:jetson-runtime` exists | <500MB | ✅ 161MB |
| Binary is ARM64 ELF | `ELF 64-bit ARM aarch64` | ✅ |
| compute_cap sm_87 fix included | Binary contains `80..=89` branch | ✅ |
| Host cuBLAS works | `cublasCreate: 0` | ✅ |

#### Stage 2: Installation Verification

| Check | Expected | Result |
| --- | --- | --- |
| Both containers up + healthy | `docker ps` shows both | ✅ Up 18min |
| GPU inference (not CPU fallback) | Logs show `Starting Bert model on Cuda(DeviceId(1))` | ✅ |
| `/health` returns HTTP 200 | Empty body is expected in TEI 1.9.3 | ✅ |
| GPU utilization (tegrastats) | `GR3D_FREQ > 0%` during embed call | ✅ |

#### Stage 3: Functional Verification

| Check | Expected | Result |
| --- | --- | --- |
| Embedding dimension | 1024-dim | ✅ 1024 |
| Embedding latency | <50ms (vs SaaS 80-150ms) | ✅ **13.7ms** |
| Rerank correctness | Sorted by relevance | ✅ |
| Rerank latency | <100ms | ✅ **22.5ms** |

Actual output:
```
✓ Image tei:jetson-runtime exists (161MB)
✓ Binary is ARM64 ELF
✓ compute_cap sm_87 fix included
✓ tei-embedding container running
✓ embedding using GPU (Cuda(CudaDevice(DeviceId(1))))
✓ embedding /health returns HTTP 200
✓ embedding returns 1024-dim vector
✓ embedding average latency: 13.7ms
✓ rerank returns 3 results (with score)
✓ rerank average latency: 22.5ms
═══════════════════════════════════════════════════════════════
  Result: 20/20 passed / 0 failed
═══════════════════════════════════════════════════════════════
```

---

## Build Time Analysis

### First Complete Build (Jetson Orin NX 16GB, JOBS=1)

```
9-hour breakdown:
├── nvidia/cuda image pull .......... 6 min
├── apt + sccache ................... 25 min (Jetson apt is slow)
├── rustup installer ............... 18.5 min (AWS CloudFront unreliable)
├── Rust toolchain ................. 15 min (USTC mirror)
├── ★ candle-flash-attn 33 kernels . 6 hours (67% of total time)
├── Non-CUDA Rust crates ........... 40 min
├── Final cargo build .............. 30 min
└── Image export ................... 72 sec
```

### candle-flash-attn 33 CUDA Kernel Compilation Times

| head_dim | Variants | Per-kernel (JOBS=1) |
| --- | --- | --- |
| hdim32 | 4 | 5-8 min |
| hdim64 | 4 | 8-12 min |
| hdim96 | 4 | 10-14 min |
| hdim128 | 4 | 14-20 min |
| hdim160 | 4 | 17-23 min |
| hdim192 | 4 | 20-25 min |
| hdim224 | 4 | 22-28 min |
| hdim256 | 4 | 28-40 min |

Historical JOBS=3 builds finished in ~3-4 hours, but this is not the safe
baseline: a single build script can still spawn many `nvcc`/`cicc` workers.
Use `scripts/build-tei-jetson.sh` so Docker cgroups leave CPU and memory for the host.

### Why JOBS=2 Causes OOM Crash

```
Memory peak:
  System + Docker + prod containers .. ~4GB
  Tegra GPU memory reserve ........... ~1GB
  cargo + rustc ...................... ~2GB
  cicc × 2 parallel .................. ~8GB (each ~4GB)
  rayon internal parallelism ......... ~2GB
  ─────────────────────────────────────
  Total ............................. ~17GB > 16GB → swap storm → crash
```

**Must use `scripts/build-tei-jetson.sh`**. It combines Docker `--cpus=4 --memory=10g`
with `CARGO_BUILD_JOBS=1`, `RAYON_NUM_THREADS=1`,
`CMAKE_BUILD_PARALLEL_LEVEL=1`, and `NVCC_THREADS=1`.

---

## Files

| File | Purpose |
| --- | --- |
| `docker/Dockerfile.tei-builder` | Build image (Rust toolchain + nvcc + USTC mirror + compute_cap fix) |
| `docker/Dockerfile.tei-runtime` | Slim runtime image (~200MB, no build toolchain) |
| `docker/docker-compose.tei.yml` | Service orchestration (CUDA lib mounts + nvhost devices) |
| `patches/compute_cap-sm87-fix.patch` | candle `compute_cap_matching(87,87)` bug fix |
| `scripts/verify-tei.sh` | Three-stage verification script (compile → install → function) |

---

## Runtime Configuration

Standard NVIDIA Docker doesn't need these, but Jetson Tegra requires:

| Config | Reason |
| --- | --- |
| bind mount `/usr/local/cuda/lib64` | Provide host CUDA 12.6 runtime/cuBLAS dynamic libraries |
| bind mount `/usr/lib/aarch64-linux-gnu/tegra` | Tegra GPU driver libs (Jetson's `libcuda.so`) |
| `devices: /dev/nvhost-*` | Jetson GPU device nodes (CUDA memory alloc needs nvhost interface) |
| direct router entrypoint | Avoid CUDA 12.9 compat wrapper and use Jetson host libraries |

---

## Known Limitations

1. **High compilation cost**: First build 5-9 hours; candle-flash-attn's 33 CUDA kernels = 67%
2. **No incremental compilation**: bindgen_cuda recompiles all CUDA kernels every `cargo build`; changing one line = 4-7 hours
3. **JetPack version locked**: the dynamic binary depends on CUDA 12 `libcublas.so.12`; CUDA 13 / JetPack 7 needs rebuild and validation
4. **Only Orin sm_87 + JetPack 6.x**: Xavier/TX2/Nano and JP5/JP4 not supported
5. **Flash Attention limited benefit**: 33 kernels have huge compilation cost but minimal benefit for short-sequence embedding/reranker

---

## FAQ

**Q: Why aren't the binary and image in the Git repo?**

Git limits single files to 100MB; the binary is 1.1GB. Pre-built binaries are on
[GitHub Releases](https://github.com/snowolf-zlex/TEI-on-Jetson/releases) (2GB/file limit).
Dockerfiles are in the repo — download the binary and build the runtime image in 30 seconds.

**Q: Can I skip candle-flash-attn compilation?**

Yes. Remove `USE_FLASH_ATTENTION=true` from the Dockerfile, use standard attention instead.
Compilation drops from 9 hours to ~1 hour. Minimal performance impact for embedding/reranker (short sequences).

**Q: Can I use the pre-built image on other Jetson devices?**

Same JetPack 6.2.x Orin series (AGX/NX/Nano 8GB+) can directly `docker load`.
Different JetPack or non-Orin devices are incompatible. See compatibility matrix above.

**Q: Why not use Ollama for embeddings?**

Ollama's embedding is a "bonus" feature with fewer models, no reranker, and incompatible API.
TEI is purpose-built for embedding/reranker with OpenAI/Jina-compatible API.

**Q: Do I need this on DGX Spark?**

No. DGX Spark has official TEI pre-built images (Blackwell + CUDA 12.9+), just `docker pull`.
This project is only for Jetson (Tegra GPU + specific JetPack).

**Q: What version is the Release binary?**

| File | Arch | CUDA | sm | TEI Version | Size |
| --- | --- | --- | --- | --- | --- |
| `text-embeddings-router-sm87-cuda126` | aarch64 | 12.6 | sm_87 | 1.9.3 (main) | ~1.1GB |

Binary uses dynamic host cuBLAS/cuBLASLt, includes the compute_cap sm_87 fix and candle-flash-attn kernels.
Does not include model files (download separately).

---

## License

MIT
