#!/usr/bin/env bash
# Jetson Orin 上执行 TEI 最终编译。Docker cgroup 必须为系统和 SSH 保留资源。
set -euo pipefail

TEI_BUILD_CONTAINER="${TEI_BUILD_CONTAINER:-tei-rebuild}"
TEI_BUILDER_IMAGE="${TEI_BUILDER_IMAGE:-tei:builder}"
TEI_BUILD_CPUS="${TEI_BUILD_CPUS:-4}"
TEI_BUILD_MEMORY="${TEI_BUILD_MEMORY:-10g}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "TEI Jetson build requires an aarch64 host" >&2
  exit 1
fi

if [[ ! "${TEI_BUILD_CPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TEI_BUILD_CPUS must be a positive integer" >&2
  exit 1
fi

host_cpus="$(nproc)"
if (( host_cpus - TEI_BUILD_CPUS < 2 )); then
  echo "Refusing to use ${TEI_BUILD_CPUS}/${host_cpus} CPUs; leave at least 2 CPUs for the host" >&2
  exit 1
fi

if [[ ! -d /usr/local/cuda ]]; then
  echo "/usr/local/cuda is missing; install the JetPack CUDA toolkit first" >&2
  exit 1
fi

cuda_release="$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\),.*/\1/p' | tail -1)"
if [[ "${cuda_release}" != "12.6" ]]; then
  echo "Expected JetPack CUDA 12.6, found ${cuda_release:-unknown}" >&2
  exit 1
fi

if ! docker image inspect "${TEI_BUILDER_IMAGE}" >/dev/null 2>&1; then
  echo "Builder image ${TEI_BUILDER_IMAGE} does not exist" >&2
  exit 1
fi

if docker container inspect "${TEI_BUILD_CONTAINER}" >/dev/null 2>&1; then
  echo "Container ${TEI_BUILD_CONTAINER} already exists; inspect or remove it explicitly" >&2
  exit 1
fi

docker run -d \
  --name "${TEI_BUILD_CONTAINER}" \
  --runtime runc \
  --cpus="${TEI_BUILD_CPUS}" \
  --memory="${TEI_BUILD_MEMORY}" \
  --memory-swap="${TEI_BUILD_MEMORY}" \
  --pids-limit=256 \
  -e CUDA_COMPUTE_CAP=87 \
  -e CARGO_BUILD_JOBS=1 \
  -e RAYON_NUM_THREADS=1 \
  -e CMAKE_BUILD_PARALLEL_LEVEL=1 \
  -e NVCC_THREADS=1 \
  -e CUDA_HOME=/usr/local/cuda \
  -e CUDA_PATH=/usr/local/cuda \
  -e CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -e CUDA_VERSION="${cuda_release}" \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  "${TEI_BUILDER_IMAGE}" \
  bash -lc 'set -o pipefail
    cd /usr/src
    /usr/local/cuda/bin/nvcc --version
    stale_root=/tmp/tei-stale-cuda-cache
    mkdir -p "${stale_root}"
    for pattern in \
      target/release/build/candle-kernels-* \
      target/release/build/candle-flash-attn-* \
      target/release/build/candle-index-select-cu-* \
      target/release/build/candle-layer-norm-* \
      target/release/build/candle-rotary-* \
      target/release/.fingerprint/candle-kernels-* \
      target/release/.fingerprint/candle-flash-attn-* \
      target/release/.fingerprint/candle-index-select-cu-* \
      target/release/.fingerprint/candle-layer-norm-* \
      target/release/.fingerprint/candle-rotary-* \
      target/release/deps/libcandle_kernels-* \
      target/release/deps/libcandle_flash_attn-* \
      target/release/deps/libcandle_index_select_cu-* \
      target/release/deps/libcandle_layer_norm-* \
      target/release/deps/libcandle_rotary-*; do
      for path in ${pattern}; do
        [[ -e "${path}" ]] || continue
        mv "${path}" "${stale_root}/$(printf "%s" "${path}" | tr / _)"
      done
    done
    cargo build --release --bin text-embeddings-router -F candle-cuda -F dynamic-linking -F http --no-default-features 2>&1 | tee /tmp/tei-build.log
    status=${PIPESTATUS[0]}
    if [[ "${status}" -eq 0 ]]; then
      cast_ptx=$(find target/release/build -path "*/candle-kernels-*/out/cast.ptx" -print -quit)
      if [[ -z "${cast_ptx}" ]] || ! grep -qx ".version 8.5" "${cast_ptx}"; then
        echo "Expected CUDA 12.6 PTX .version 8.5, found: ${cast_ptx:-missing}" >&2
        status=1
      fi
    fi
    printf "EXIT_%s\n" "${status}"
    exit "${status}"'

echo "Started ${TEI_BUILD_CONTAINER} with ${TEI_BUILD_CPUS} CPUs and ${TEI_BUILD_MEMORY} memory"
echo "Monitor with: docker logs -f ${TEI_BUILD_CONTAINER}"
