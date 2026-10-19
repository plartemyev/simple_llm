#!/usr/bin/env bash
# End-to-end integration check: start the service with the given model, run
# the pytest integration suite inside the image, tear the service down.
#
#   usage: scripts/integration_test.sh MODEL [-g GPU] [-p PORT] [-b cuda|vulkan]
#
#   SLM_GPU_DRIVER nvidia|amd  (default nvidia; amd passes /dev/dri and uses
#                              GGML_VK_VISIBLE_DEVICES instead of CUDA_VISIBLE_DEVICES)
#   SLM_GROUP_ADD  supplementary host group ids (space/comma separated) for
#                  group-restricted /dev/dri/renderD* nodes
#   SLM_MMPROJ     forwarded when set: multimodal projector for vision models
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:?usage: integration_test.sh MODEL [-g GPU] [-p PORT] [-b cuda|vulkan]}
shift
GPU=1; PORT=8011; BACKEND=${SLM_BACKEND:-cuda}; NAME=simple_llm_it
GPU_DRIVER=${SLM_GPU_DRIVER:-nvidia}
case ${GPU_DRIVER} in
  nvidia|amd) ;;
  *) echo "SLM_GPU_DRIVER must be nvidia or amd" >&2; exit 2 ;;
esac
while getopts "g:p:b:n:" o; do
  case $o in
    g) GPU=$OPTARG ;;
    p) PORT=$OPTARG ;;
    b) BACKEND=$OPTARG ;;
    n) NAME=$OPTARG ;;
    *) exit 2 ;;
  esac
done

IMAGE=${SLM_IMAGE:-simple_llm:${BACKEND}}
MODELS_DIR=${SLM_MODELS_DIR:-$PWD/data/models}
CACHE_DIR=${SLM_CACHE_DIR:-$PWD/data/cache}
# reasoning models would spend the small test token budgets thinking;
# keep the suite on non-thinking output (matches the production profiles)
EXTRA_ARGS=${SLM_EXTRA_ARGS:-"--reasoning off"}

cleanup() { docker rm -f "${NAME}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

GPU_ARGS=(--gpus all -e CUDA_VISIBLE_DEVICES="${GPU}")
if [ "${GPU_DRIVER}" = amd ]; then
  GPU_ARGS=(--device /dev/dri -e "GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES:-${SLM_GPU:-0}}")
fi
GROUP_ARGS=()
if [ -n "${SLM_GROUP_ADD:-}" ]; then
  # supplementary host groups (e.g. render/video); word splitting intended
  # shellcheck disable=SC2086
  for g in ${SLM_GROUP_ADD//,/ }; do
    GROUP_ARGS+=(--group-add "${g}")
  done
fi

docker run -d --name "${NAME}" --ipc=host \
  --user "$(id -u):$(id -g)" \
  "${GPU_ARGS[@]}" \
  "${GROUP_ARGS[@]}" \
  -e SLM_MODEL="${MODEL}" -e SLM_PORT=8011 -e SLM_CTX_SIZE=8192 -e SLM_PARALLEL=4 \
  -e SLM_EXTRA_ARGS="${EXTRA_ARGS}" \
  $( [ -f secrets.env ] && echo --env-file secrets.env ) \
  $( [ -n "${SLM_MMPROJ:-}" ] && echo "-e SLM_MMPROJ=${SLM_MMPROJ}" ) \
  -v "${MODELS_DIR}":/models -v "${CACHE_DIR}":/data/cache \
  -p "127.0.0.1:${PORT}:8011" \
  "${IMAGE}" >/dev/null
echo "Waiting for ${IMAGE} to serve model ${MODEL} ..."

# wait until /health answers (first start may download the model)
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
  if [ "$(docker inspect -f '{{.State.Status}}' "${NAME}" 2>/dev/null)" != "running" ]; then
    echo "container died during startup:" >&2
    docker logs --tail 50 "${NAME}" >&2
    exit 1
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "service did not become healthy" >&2; exit 1; }
echo "service is up"

MODEL_ID="$(curl -fsS "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
echo "served model id: ${MODEL_ID}"

docker exec -e SLM_BASE_URL="http://127.0.0.1:8011/v1" -e SLM_MODEL_ID="${MODEL_ID}" \
  "${NAME}" python -m pytest tests/ -m integration -v
echo "integration tests passed"
