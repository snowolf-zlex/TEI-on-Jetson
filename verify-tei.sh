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
    docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE}$"

  check "二进制为 ARM64 ELF" \
    docker run --rm --entrypoint file "${IMAGE}" /usr/local/bin/text-embeddings-router | grep -q "ELF 64-bit.*ARM aarch64"

  check "compute_cap sm_87 修复已包含" \
    docker run --rm --entrypoint strings "${IMAGE}" /usr/local/bin/text-embeddings-router | grep -q "80..=89.*80..=89"

  check "entrypoint.sh 存在且可执行" \
    docker run --rm --entrypoint test "${IMAGE}" -x /entrypoint.sh

  # 宿主 cuBLAS 验证（需 nvcc）
  if command -v nvcc >/dev/null 2>&1; then
    local tmp_cu; tmp_cu=$(mktemp --suffix=.cu)
    cat > "$tmp_cu" <<'CUDA'
#include <cstdio>
#include <cublas_v2.h>
int main() {
    cublasHandle_t h;
    int st = cublasCreate(&h);
    printf("cublasCreate: %d\n", st);
    if (st == 0) { cublasDestroy(h); }
    return st != 0;
}
CUDA
    local tmp_out; tmp_out=$(mktemp)
    if nvcc -o "$tmp_out" "$tmp_cu" -lcublas -I/usr/local/cuda/include -L/usr/local/cuda/lib64 2>/dev/null; then
      check "宿主 cuBLAS 初始化成功" "$tmp_out"
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
    docker ps --format '{{.Names}}' | grep -q tei-embedding

  check "tei-reranker 容器运行中" \
    docker ps --format '{{.Names}}' | grep -q tei-reranker

  check "embedding 使用 GPU（日志含 Cuda）" \
    docker logs agent-studio-tei-embedding-1 2>&1 | grep -qi "Starting.*Cuda"

  check "reranker 使用 GPU（日志含 Cuda）" \
    docker logs agent-studio-tei-reranker-1 2>&1 | grep -qi "Starting.*Cuda"

  check "embedding /health 返回 ready" \
    curl -sf "http://127.0.0.1:${EMBED_PORT}/health" | grep -qi "ready"

  check "reranker /health 返回 ready" \
    curl -sf "http://127.0.0.1:${RERANK_PORT}/health" | grep -qi "ready"

  # GPU 利用率检查（tegrastats 采样 3 秒）
  if command -v tegrastats >/dev/null 2>&1; then
    info "采样 GPU 利用率（3 秒）…"
    # 触发一次 embedding 调用让 GPU 工作
    curl -sf "http://127.0.0.1:${EMBED_PORT}/embed" \
      -H 'Content-Type: application/json' \
      -d '{"inputs":"GPU utilization test"}' >/dev/null 2>&1 &
    local stats; stats=$(timeout 5 tegrastats --interval 1000 2>/dev/null | head -3 || true)
    if echo "$stats" | grep -qP "GR3D_FREQ [1-9]"; then
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
  for i in 1 2 3; do
    local ms; ms=$(curl -sf -o /dev/null -w '%{time_total}' \
      "http://127.0.0.1:${EMBED_PORT}/embed" \
      -H 'Content-Type: application/json' \
      -d '{"inputs":"延迟测试"}' 2>/dev/null || echo "0")
    total=$(python3 -c "print($total + $ms * 1000)")
  done
  local avg; avg=$(python3 -c "print(round($total / 3, 1))")
  info "embedding 平均延迟: ${avg}ms"
  if python3 -c "exit(0 if $avg < 100 else 1)" 2>/dev/null; then
    green "embedding 延迟 <100ms"
    PASS=$((PASS+1))
  else
    red "embedding 延迟 ≥100ms（${avg}ms）"
    FAIL=$((FAIL+1))
  fi

  # Rerank 调用
  info "测试 rerank 调用…"
  local rerank_resp; rerank_resp=$(curl -sf "http://127.0.0.1:${RERANK_PORT}/rerank" \
    -H 'Content-Type: application/json' \
    -d '{"query":"向量检索","documents":["pgvector 是 PostgreSQL 向量扩展","BM25 是全文检索","reranker 重排搜索结果"]}' 2>&1 || true)

  if echo "$rerank_resp" | python3 -c "import json,sys; d=json.load(sys.stdin); assert len(d)==3; assert 'score' in d[0]" 2>/dev/null; then
    green "rerank 返回 3 个结果（含 score）"
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
