# Validation

This page is the verification record for the Jetson Orin NX build published by
this repository.

## Verified Target

| Item | Value |
| --- | --- |
| Date | 2026-08-12 |
| Device | Jetson Orin NX 16GB |
| JetPack / L4T | JetPack 6.2.3 / L4T R36.5.2 |
| CUDA toolkit | 12.6.68 |
| TEI | 1.9.3 source snapshot |
| Runtime image | `tei:jetson-runtime` |
| Runtime image size | 161,072,849 bytes |
| Binary SHA-256 | `6841f5fb5fe9ea46d84c1ebab743b018d7f28b6635967245f9d8af9e33ef28b8` |
| CUDA linking mode | dynamic host `libcublas.so.12` / `libcublasLt.so.12` |
| PTX gate | Candle `cast.ptx` must be `.version 8.5` |

The failed static-link image is intentionally not the baseline. The accepted
baseline uses TEI/cudarc `dynamic-linking` and bind-mounts Jetson host CUDA
12.6 libraries.

## How To Run

Start both TEI services first:

```bash
docker compose up -d tei-embedding tei-reranker
```

Then run all three stages:

```bash
bash scripts/verify-tei.sh all
```

Run a single stage when debugging:

```bash
bash scripts/verify-tei.sh 1   # build/runtime image checks
bash scripts/verify-tei.sh 2   # container, health, GPU checks
bash scripts/verify-tei.sh 3   # embedding/rerank API checks
```

The script returns exit code `0` only when all required checks pass.

## Test Data

The validation does not require a large external corpus. It uses deterministic
smoke-test inputs that exercise the real TEI APIs and CUDA backend:

| Endpoint | Input | Expected result |
| --- | --- | --- |
| `/embed` | `Agent Studio 是智能体构建平台` | JSON array with a 1024-dimensional BGE-m3 vector |
| `/embed` latency | `延迟测试` repeated 3 times | Average latency below 50ms |
| `/rerank` | Query `向量检索`; texts: `pgvector 是 PostgreSQL 向量扩展`, `BM25 是全文检索`, `reranker 重排搜索结果` | Three scored results in descending score order |
| `/rerank` relevance | Same fixed rerank input | Result with `index == 0` ranks first |
| `/rerank` latency | Same fixed rerank input repeated 3 times | Average latency below 100ms |

This is an acceptance smoke test, not an IR benchmark. It proves the binary,
runtime image, CUDA/cuBLAS dynamic linking, model loading, GPU execution, and
API contracts are functioning on the target Jetson environment.

## Results

| Stage | Check | Result |
| --- | --- | --- |
| 1 | Image `tei:jetson-runtime` exists | Passed |
| 1 | Image is below 500MiB | Passed, 161,072,849 bytes |
| 1 | Image architecture is ARM64 | Passed |
| 1 | Image labels declare sm_87 and dynamic linking | Passed |
| 1 | `ldd` resolves `libcublas.so.12` and `libcublasLt.so.12` from host CUDA | Passed |
| 1 | Entrypoint is direct `text-embeddings-router` | Passed |
| 1 | Host cuBLAS `cublasCreate` and SGEMM | Passed |
| 2 | Embedding and reranker containers are running | Passed |
| 2 | Logs show CUDA backend, not CPU fallback | Passed |
| 2 | Both `/health` endpoints return HTTP 200 | Passed |
| 2 | `tegrastats` shows `GR3D_FREQ > 0%` during embed calls | Passed |
| 3 | `/embed` returns a 1024-dimensional vector | Passed |
| 3 | `/embed` 3-run average latency | Passed, 13.7ms |
| 3 | `/rerank` returns three scored results in descending order | Passed |
| 3 | `/rerank` ranks the pgvector document first | Passed |
| 3 | `/rerank` 3-run average latency | Passed, 22.5ms |

Final result: **20 passed / 0 failed**.

TEI service logs had no `CUBLAS`, `UNSUPPORTED_PTX`, or `ERROR` failures during
the final validation run.

## Important Compatibility Notes

- TEI 1.9.3 `/health` returns HTTP 200 with an empty body. Do not require a
  JSON `{"status":"ready"}` payload.
- TEI 1.9.3 rerank requests use the field `texts`, not `documents`.
- The runtime image does not need `NVIDIA_DISABLE_REQUIRE` and does not mount
  `/tmp` over `/usr/local/cuda/compat`; the runtime image starts the router
  directly and uses Jetson host CUDA libraries.
- The build must purge stale CUDA 12.9 Candle PTX cache and fail closed unless
  `cast.ptx` is `.version 8.5`.
- CUDA 13 / JetPack 7 and non-Orin devices are not covered by this validation.
