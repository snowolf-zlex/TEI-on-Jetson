#!/usr/bin/env bash
# =============================================================================
# verify-tei-jetson.sh — TEI on Jetson 三阶段验收脚本
# =============================================================================
# 用法：在 Jetson 上执行（需 TEI 服务已启动）
#   bash scripts/verify-tei-jetson.sh [stage]
#
# 参数：
#   1    = 只跑阶段 1（编译验收）
#   2    = 只跑阶段 2（安装验收）
#   3    = 只跑阶段 3（功能验收）
#   all  = 全部（默认）
#
# 退出码：
#   0 = 全部通过
#   1 = 有失败项（详细输出见 stdout）
# =============================================================================
set -euo pipefail

STAGE="${1:-all}"
PASS=0
FAIL=0
IMAGE="${TEI_IMAGE:-tei:jetson-runtime}"
EMBED_PORT="${TEI_EMBED_PORT:-8083}"
RERANK_PORT="${TEI_RERANK_PORT:-8084}"

green() { printf "\033[32m✓ %s\033[0m\n" "$1"; }
red()   { printf "\033[31m✗ %s\033[0m\n" "$1"; }
info()  { printf "\033[36m▸ %s\033[0m\n" "$1"; }

check() {
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then
    green "$desc"
    PASS=$((PASS+1))
  else
    red "$desc"
    FAIL=$((FAIL+1))
  fi
}

# ─── 阶段 1：编译验收 ─────────────────────────────────────────────────────────
stage1() {
  info "阶段 1：编译验收"
  echo ""

  check "镜像 ${IMAGE} 存在" \
    docker image inspect "${IMAGE}"

  check "镜像小于 500MiB" \
    bash -c "test \"\$(docker image inspect --format '{{.Size}}' '${IMAGE}')\" -lt 524288000"

  check "镜像架构为 ARM64" \
    test "$(docker image inspect --format '{{.Architecture}}' "${IMAGE}")" = "arm64"

  check "镜像声明 compute_cap sm_87" \
    test "$(docker image inspect --format '{{index .Config.Labels "org.agent-studio.tei.cuda-compute-cap"}}' "${IMAGE}")" = "87"

  check "镜像声明 dynamic-linking" \
    test "$(docker image inspect --format '{{index .Config.Labels "org.agent-studio.tei.cuda-linking"}}' "${IMAGE}")" = "dynamic"

  check "CUDA、cuBLAS 与 cuBLASLt 动态库全部解析到 Jetson 宿主" \
    bash -c "output=\$(docker run --rm --entrypoint sh \
      -v /usr/local/cuda/lib64:/usr/local/cuda/lib64:ro \
      -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
      -v /usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/nvidia:ro \
      -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia \
      '${IMAGE}' -c 'ldd /usr/local/bin/text-embeddings-router' 2>&1); \
      printf '%s\n' \"\$output\" | grep -q 'libcublas.so.12 => /usr/local/cuda/lib64/' && \
      printf '%s\n' \"\$output\" | grep -q 'libcublasLt.so.12 => /usr/local/cuda/lib64/' && \
      ! printf '%s\n' \"\$output\" | grep -q 'not found'"

  check "直接启动 router，绕过 CUDA 12.9 compat entrypoint" \
    test "$(docker image inspect --format '{{json .Config.Entrypoint}}' "${IMAGE}")" = '["text-embeddings-router"]'

  # 宿主 cuBLAS 验证（初始化 + 1x1 SGEMM）
  local nvcc_bin="${NVCC:-/usr/local/cuda/bin/nvcc}"
  if [[ -x "${nvcc_bin}" ]]; then
    local tmp_cu; tmp_cu=$(mktemp --suffix=.cu)
    cat > "$tmp_cu" <<'CUDA'
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
int main() {
    cublasHandle_t h;
    int st = cublasCreate(&h);
    printf("cublasCreate: %d\n", st);
    if (st != 0) return 1;
    float a = 2.0f, b = 3.0f, c = 0.0f;
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    if (cudaMalloc(&da, sizeof(float)) != cudaSuccess ||
        cudaMalloc(&db, sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dc, sizeof(float)) != cudaSuccess) return 2;
    cudaMemcpy(da, &a, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice);
    const float alpha = 1.0f, beta = 0.0f;
    st = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                     &alpha, da, 1, db, 1, &beta, dc, 1);
    cudaMemcpy(&c, dc, sizeof(float), cudaMemcpyDeviceToHost);
    printf("cublasSgemm: %d result: %.1f\n", st, c);
    cudaFree(da); cudaFree(db); cudaFree(dc); cublasDestroy(h);
    return st != 0 || std::fabs(c - 6.0f) > 0.001f;
}
CUDA
    local tmp_out; tmp_out=$(mktemp)
    if "${nvcc_bin}" -o "$tmp_out" "$tmp_cu" -lcublas -I/usr/local/cuda/include -L/usr/local/cuda/lib64 2>/dev/null; then
      check "宿主 cuBLAS 初始化与 SGEMM 成功" "$tmp_out"
    else
      red "宿主 cuBLAS 编译失败（nvcc 或库缺失）"
      FAIL=$((FAIL+1))
    fi
    rm -f "$tmp_cu" "$tmp_out"
  else
    info "跳过宿主 cuBLAS 测试（nvcc 不在 PATH）"
  fi
}

# ─── 阶段 2：安装调试验收 ─────────────────────────────────────────────────────
stage2() {
  info "阶段 2：安装调试验收"
  echo ""

  check "tei-embedding 容器运行中" \
    test "$(docker inspect --format '{{.State.Running}}' agent-studio-tei-embedding-1)" = "true"

  check "tei-reranker 容器运行中" \
    test "$(docker inspect --format '{{.State.Running}}' agent-studio-tei-reranker-1)" = "true"

  check "embedding 使用 GPU（日志含 Cuda）" \
    bash -c "docker logs agent-studio-tei-embedding-1 2>&1 | grep -qi 'Starting.*Cuda'"

  check "reranker 使用 GPU（日志含 Cuda）" \
    bash -c "docker logs agent-studio-tei-reranker-1 2>&1 | grep -qi 'Starting.*Cuda'"

  check "embedding /health 返回 HTTP 200" \
    curl -sf -o /dev/null "http://127.0.0.1:${EMBED_PORT}/health"

  check "reranker /health 返回 HTTP 200" \
    curl -sf -o /dev/null "http://127.0.0.1:${RERANK_PORT}/health"

  # GPU 利用率检查（tegrastats 采样 3 秒）
  if command -v tegrastats >/dev/null 2>&1; then
    info "采样 GPU 利用率（5 秒）…"
    local stats_file; stats_file=$(mktemp)
    timeout 5 tegrastats --interval 200 >"${stats_file}" 2>/dev/null &
    local stats_pid=$!
    for _ in $(seq 1 20); do
      curl -sf "http://127.0.0.1:${EMBED_PORT}/embed" \
        -H 'Content-Type: application/json' \
        -d '{"inputs":"GPU utilization test for Agent Studio local inference"}' >/dev/null 2>&1 || true
    done
    wait "${stats_pid}" || true
    local stats; stats=$(<"${stats_file}")
    rm -f "${stats_file}"
    if echo "$stats" | grep -qE "GR3D_FREQ ([1-9]|[1-9][0-9]|100)%"; then
      green "GPU 利用率 >0%（GR3D_FREQ 非 0）"
      PASS=$((PASS+1))
    else
      red "GPU 利用率为 0%（GR3D_FREQ 全 0）"
      FAIL=$((FAIL+1))
    fi
  else
    info "跳过 GPU 利用率测试（tegrastats 不可用）"
  fi
}

# ─── 阶段 3：功能验收 ─────────────────────────────────────────────────────────
stage3() {
  info "阶段 3：功能验收"
  echo ""

  # Embedding 调用 + 维度检查
  info "测试 embedding 调用…"
  local embed_resp; embed_resp=$(curl -sf "http://127.0.0.1:${EMBED_PORT}/embed" \
    -H 'Content-Type: application/json' \
    -d '{"inputs":"Agent Studio 是智能体构建平台"}' 2>&1 || true)

  if echo "$embed_resp" | python3 -c "import json,sys; d=json.load(sys.stdin); assert len(d[0])==1024" 2>/dev/null; then
    green "embedding 返回 1024 维向量"
    PASS=$((PASS+1))
  else
    red "embedding 返回格式或维度错误"
    FAIL=$((FAIL+1))
  fi

  # Embedding 延迟
  info "测试 embedding 延迟（3 次平均）…"
  local total=0
  local latency_ok=true
  for i in 1 2 3; do
    local ms
    if ! ms=$(curl -sf -o /dev/null -w '%{time_total}' \
        "http://127.0.0.1:${EMBED_PORT}/embed" \
        -H 'Content-Type: application/json' \
        -d '{"inputs":"延迟测试"}' 2>/dev/null); then
      latency_ok=false
      break
    fi
    total=$(python3 -c "print($total + $ms * 1000)")
  done
  local avg; avg=$(python3 -c "print(round($total / 3, 1))")
  info "embedding 平均延迟: ${avg}ms"
  if [[ "${latency_ok}" = true ]] && python3 -c "exit(0 if $avg < 50 else 1)" 2>/dev/null; then
    green "embedding 延迟 <50ms"
    PASS=$((PASS+1))
  else
    red "embedding 请求失败或延迟 ≥50ms（${avg}ms）"
    FAIL=$((FAIL+1))
  fi

  # Rerank 调用
  info "测试 rerank 调用…"
  local rerank_resp; rerank_resp=$(curl -sf "http://127.0.0.1:${RERANK_PORT}/rerank" \
    -H 'Content-Type: application/json' \
    -d '{"query":"向量检索","texts":["pgvector 是 PostgreSQL 向量扩展","BM25 是全文检索","reranker 重排搜索结果"]}' 2>&1 || true)

  if echo "$rerank_resp" | python3 -c "import json,sys; d=json.load(sys.stdin); assert len(d)==3; assert all('score' in x and 'index' in x for x in d); assert all(d[i]['score'] >= d[i+1]['score'] for i in range(len(d)-1))" 2>/dev/null; then
    green "rerank 返回 3 个按分数降序的结果"
    PASS=$((PASS+1))
    # 打印排序结果
    echo "$rerank_resp" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for r in d:
    print(f\"  {r['index']}: {r['score']:.4f}\")
" 2>/dev/null || true
  else
    red "rerank 返回格式错误"
    FAIL=$((FAIL+1))
  fi

  if echo "$rerank_resp" | python3 -c "import json,sys; assert json.load(sys.stdin)[0]['index']==0" 2>/dev/null; then
    green "rerank 将 pgvector 文档排在向量检索查询首位"
    PASS=$((PASS+1))
  else
    red "rerank 相关性排序不符合固定用例"
    FAIL=$((FAIL+1))
  fi

  info "测试 rerank 延迟（3 次平均）…"
  total=0
  latency_ok=true
  for i in 1 2 3; do
    local ms
    if ! ms=$(curl -sf -o /dev/null -w '%{time_total}' \
        "http://127.0.0.1:${RERANK_PORT}/rerank" \
        -H 'Content-Type: application/json' \
        -d '{"query":"向量检索","texts":["pgvector 是 PostgreSQL 向量扩展","BM25 是全文检索","reranker 重排搜索结果"]}' 2>/dev/null); then
      latency_ok=false
      break
    fi
    total=$(python3 -c "print($total + $ms * 1000)")
  done
  avg=$(python3 -c "print(round($total / 3, 1))")
  info "rerank 平均延迟: ${avg}ms"
  if [[ "${latency_ok}" = true ]] && python3 -c "exit(0 if $avg < 100 else 1)" 2>/dev/null; then
    green "rerank 延迟 <100ms"
    PASS=$((PASS+1))
  else
    red "rerank 请求失败或延迟 ≥100ms（${avg}ms）"
    FAIL=$((FAIL+1))
  fi
}

# ─── 主流程 ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  TEI on Jetson 验收  (image: ${IMAGE})"
echo "═══════════════════════════════════════════════════════════════"
echo ""

case "$STAGE" in
  1)     stage1 ;;
  2)     stage2 ;;
  3)     stage3 ;;
  all)   stage1; echo ""; stage2; echo ""; stage3 ;;
  *)     echo "用法: $0 [1|2|3|all]"; exit 1 ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
printf "  结果: \033[32m%d 通过\033[0m / \033[31m%d 失败\033[0m\n" "$PASS" "$FAIL"
echo "═══════════════════════════════════════════════════════════════"

exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
