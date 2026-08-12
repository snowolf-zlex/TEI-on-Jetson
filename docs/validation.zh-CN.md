# 验收方法与结果

本文记录本仓库发布的 Jetson Orin NX 构建验收方法和实测结果。

## 已验证目标

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-08-12 |
| 设备 | Jetson Orin NX 16GB |
| JetPack / L4T | JetPack 6.2.3 / L4T R36.5.2 |
| CUDA toolkit | 12.6.68 |
| TEI | 1.9.3 source snapshot |
| 运行时镜像 | `tei:jetson-runtime` |
| 运行时镜像大小 | 161,072,849 bytes |
| 二进制 SHA-256 | `6841f5fb5fe9ea46d84c1ebab743b018d7f28b6635967245f9d8af9e33ef28b8` |
| CUDA 链接方式 | 动态加载宿主 `libcublas.so.12` / `libcublasLt.so.12` |
| PTX 门禁 | Candle `cast.ptx` 必须是 `.version 8.5` |

失败的 static-link 镜像不是验收基线。最终基线使用 TEI/cudarc 的
`dynamic-linking`，并 bind mount Jetson 宿主 CUDA 12.6 动态库。

## 如何运行

先启动两个 TEI 服务：

```bash
docker compose up -d tei-embedding tei-reranker
```

再执行三阶段验收：

```bash
bash verify-tei.sh all
```

排查时可只跑单阶段：

```bash
bash verify-tei.sh 1   # 构建与运行时镜像检查
bash verify-tei.sh 2   # 容器、health、GPU 检查
bash verify-tei.sh 3   # embedding/rerank API 检查
```

只有全部必需检查通过时，脚本才返回退出码 `0`。

## 测试数据

验收不需要大型外部语料库，而是使用固定烟测输入直接调用真实 TEI API 和 CUDA backend：

| Endpoint | 输入 | 预期结果 |
| --- | --- | --- |
| `/embed` | `Agent Studio 是智能体构建平台` | 返回 BGE-m3 的 1024 维向量 |
| `/embed` 延迟 | `延迟测试` 重复 3 次 | 平均延迟低于 50ms |
| `/rerank` | query `向量检索`；texts: `pgvector 是 PostgreSQL 向量扩展`、`BM25 是全文检索`、`reranker 重排搜索结果` | 返回 3 个按 score 降序排列的结果 |
| `/rerank` 相关性 | 同一固定 rerank 输入 | `index == 0` 的 pgvector 文档排第一 |
| `/rerank` 延迟 | 同一固定 rerank 输入重复 3 次 | 平均延迟低于 100ms |

这是一套验收烟测，不是 IR benchmark。它证明二进制、运行时镜像、CUDA/cuBLAS 动态链接、
模型加载、GPU 执行和 API 契约在目标 Jetson 环境上可用。

## 实测结果

| 阶段 | 检查项 | 结果 |
| --- | --- | --- |
| 1 | 镜像 `tei:jetson-runtime` 存在 | 通过 |
| 1 | 镜像小于 500MiB | 通过，161,072,849 bytes |
| 1 | 镜像架构为 ARM64 | 通过 |
| 1 | 镜像 label 声明 sm_87 与 dynamic linking | 通过 |
| 1 | `ldd` 从宿主 CUDA 解析 `libcublas.so.12` 与 `libcublasLt.so.12` | 通过 |
| 1 | entrypoint 直接启动 `text-embeddings-router` | 通过 |
| 1 | 宿主 cuBLAS `cublasCreate` 与 SGEMM | 通过 |
| 2 | embedding 与 reranker 容器运行中 | 通过 |
| 2 | 日志显示 CUDA backend，不是 CPU fallback | 通过 |
| 2 | 两个 `/health` 均返回 HTTP 200 | 通过 |
| 2 | embed 调用期间 `tegrastats` 显示 `GR3D_FREQ > 0%` | 通过 |
| 3 | `/embed` 返回 1024 维向量 | 通过 |
| 3 | `/embed` 3 次平均延迟 | 通过，13.7ms |
| 3 | `/rerank` 返回 3 个按分数降序结果 | 通过 |
| 3 | `/rerank` 将 pgvector 文档排第一 | 通过 |
| 3 | `/rerank` 3 次平均延迟 | 通过，22.5ms |

最终结果：**20 通过 / 0 失败**。

最终验收期间，两个 TEI 服务日志中没有 `CUBLAS`、`UNSUPPORTED_PTX` 或 `ERROR` 失败。

## 重要兼容性说明

- TEI 1.9.3 的 `/health` 返回 HTTP 200，body 为空；不要要求
  `{"status":"ready"}`。
- TEI 1.9.3 的 rerank 请求字段是 `texts`，不是 `documents`。
- 运行时镜像不需要 `NVIDIA_DISABLE_REQUIRE`，也不需要把 `/tmp` 挂到
  `/usr/local/cuda/compat`；镜像直接启动 router，并使用 Jetson 宿主 CUDA 动态库。
- 构建必须清理 CUDA 12.9 遗留的 Candle PTX 缓存，并在 `cast.ptx` 不是 `.version 8.5`
  时 fail closed。
- CUDA 13 / JetPack 7 和非 Orin 设备不在本次验收范围内。
