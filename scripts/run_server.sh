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
#
# Put HF_TOKEN=... into ./secrets.env (git-ignored) to download gated repos.
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
  --gpus all
  --ipc=host
  -e CUDA_VISIBLE_DEVICES="${GPU}"
  -e SLM_MODEL="${MODEL}"
  -e SLM_CTX_SIZE="${CTX}"
  -e SLM_PARALLEL="${PARALLEL}"
)
if [ "${#EXTRA[@]}" -gt 0 ]; then
  RUN_ARGS+=(-e "SLM_EXTRA_ARGS=${EXTRA[*]}")
fi
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
