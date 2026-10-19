#!/usr/bin/env bash
# Start the service in the background.
#   usage: scripts/run_server.sh MODEL [-g GPU] [-p PORT] [-n NAME] [-- extra llama-server args]
#
#   MODEL is either an HF id "owner/repo:QUANT" (downloaded into CACHE_DIR on
#   first start) or a .gguf file name inside MODELS_DIR (mounted to /models).
#
# Environment:
#   SLM_BACKEND    cuda|vulkan            (default cuda)
#   SLM_IMAGE      image tag override     (default simple_llm:$SLM_BACKEND)
#   SLM_MODELS_DIR host models dir        (default ./data/models)
#   SLM_CACHE_DIR  host HF download cache (default ./data/cache)
#   SLM_CTX_SIZE   context size           (default 8192)
#   SLM_PARALLEL   concurrent slots       (default 4)
#   SLM_KV_QUANT   KV cache type for K and V, e.g. q8_0 — enables flash
#                  attention with it (container-side, docker/entrypoint.sh)
#   SLM_GPU_DRIVER nvidia|amd             (default nvidia; amd passes /dev/dri
#                                          and selects the Vulkan device via
#                                          GGML_VK_VISIBLE_DEVICES, default 0)
#   SLM_GROUP_ADD  supplementary host group ids, space/comma separated —
#                  needed when /dev/dri/renderD* is group-restricted
#                  (e.g. 0660 root:render); find the gid with
#                  `stat -c %g /dev/dri/renderD128` on the host
#
# The container runs as the invoking user (uid:gid) so HF downloads under
# ./data/cache stay host-user-owned.
#   llama-server native tuning env vars (llama-server flag in parentheses),
#   forwarded into the container when set; SLM_EXTRA_ARGS flags take precedence:
#   LLAMA_ARG_CPU_MOE=1            (--cpu-moe: keep all MoE expert weights on CPU)
#   LLAMA_ARG_N_CPU_MOE=N          (--n-cpu-moe N: first N layers' MoE experts on CPU)
#   LLAMA_ARG_OVERRIDE_TENSOR=...  (-ot: comma-separated tensor=buffer overrides,
#                                   e.g. "exps=CPU" or "blk\.(0|1)\.ffn_.*=CUDA1")
#
# ./secrets.env (git-ignored) is forwarded wholesale via --env-file:
# HF_TOKEN=... downloads gated repos; any other variable in it (e.g.
# LLAMA_ARG_* not listed above) reaches llama-server in the container as-is.
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -ge 1 ] || { echo "usage: $0 MODEL [-g GPU] [-p PORT] [-n NAME] [-- extra args]" >&2; exit 2; }
MODEL=$1; shift

GPU=${SLM_GPU:-1}
PORT=${SLM_PORT:-8011}
NAME=simple_llm
BACKEND=${SLM_BACKEND:-cuda}
IMAGE=${SLM_IMAGE:-simple_llm:${BACKEND}}
MODELS_DIR=${SLM_MODELS_DIR:-$PWD/data/models}
CACHE_DIR=${SLM_CACHE_DIR:-$PWD/data/cache}
CTX=${SLM_CTX_SIZE:-8192}
PARALLEL=${SLM_PARALLEL:-4}
GPU_DRIVER=${SLM_GPU_DRIVER:-nvidia}

case ${GPU_DRIVER} in
  nvidia|amd) ;;
  *) echo "SLM_GPU_DRIVER must be nvidia or amd" >&2; exit 2 ;;
esac

while getopts "g:p:n:" o; do
  case $o in
    g) GPU=$OPTARG ;;
    p) PORT=$OPTARG ;;
    n) NAME=$OPTARG ;;
    *) echo "usage: $0 MODEL [-g GPU] [-p PORT] [-n NAME] [-- extra args]" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA=("$@")
if [ -n "${SLM_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206  # word splitting of flag lists is intended
  EXTRA=(${SLM_EXTRA_ARGS} "${EXTRA[@]}")
fi

docker rm -f "${NAME}" >/dev/null 2>&1 || true

RUN_ARGS=(
  -d --name "${NAME}"
  --ipc=host
  --user "$(id -u):$(id -g)"
)
if [ -n "${SLM_GROUP_ADD:-}" ]; then
  # supplementary host groups (e.g. render/video); word splitting intended
  # shellcheck disable=SC2086
  for g in ${SLM_GROUP_ADD//,/ }; do
    RUN_ARGS+=(--group-add "${g}")
  done
fi
if [ "${GPU_DRIVER}" = amd ]; then
  # AMD: the image's RADV ICD drives the host amdgpu kernel driver through
  # the render nodes; Vulkan device defaults to 0 (SLM_GPU's CUDA-flavored
  # default of 1 rarely matches Vulkan enumeration on single-GPU hosts).
  RUN_ARGS+=(
    --device /dev/dri
    -e "GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES:-${SLM_GPU:-0}}"
  )
else
  RUN_ARGS+=(
    --gpus all
    -e CUDA_VISIBLE_DEVICES="${GPU}"
  )
  # Vulkan on NVIDIA selects devices via GGML_VK_VISIBLE_DEVICES (forward
  # only when the user set it).
  if [ -n "${GGML_VK_VISIBLE_DEVICES:-}" ]; then
    RUN_ARGS+=(-e "GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES}")
  fi
fi
RUN_ARGS+=(
  -e SLM_MODEL="${MODEL}"
  -e SLM_CTX_SIZE="${CTX}"
  -e SLM_PARALLEL="${PARALLEL}"
)
if [ -n "${SLM_KV_QUANT:-}" ]; then
  RUN_ARGS+=(-e "SLM_KV_QUANT=${SLM_KV_QUANT}")
fi
if [ "${#EXTRA[@]}" -gt 0 ]; then
  RUN_ARGS+=(-e "SLM_EXTRA_ARGS=${EXTRA[*]}")
fi
for v in LLAMA_ARG_CPU_MOE LLAMA_ARG_N_CPU_MOE LLAMA_ARG_OVERRIDE_TENSOR; do
  if [ -n "${!v:-}" ]; then
    RUN_ARGS+=(-e "${v}=${!v}")
  fi
done
if [ -f secrets.env ]; then
  RUN_ARGS+=(--env-file secrets.env)
fi

mkdir -p "${MODELS_DIR}" "${CACHE_DIR}"

docker run "${RUN_ARGS[@]}" \
  -v "${MODELS_DIR}":/models \
  -v "${CACHE_DIR}":/data/cache \
  -p "127.0.0.1:${PORT}:8011" \
  "${IMAGE}" >/dev/null

echo "Started ${NAME} (image ${IMAGE}, gpu ${GPU}) -> http://127.0.0.1:${PORT}"
echo "Logs: docker logs -f ${NAME}"
