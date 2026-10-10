#!/usr/bin/env bash
# Build the service image.
#   usage: scripts/build_image.sh [cuda|vulkan] [tag]
#   default backend: cuda (override with SLM_BACKEND)
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND=${1:-${SLM_BACKEND:-cuda}}
TAG=${2:-simple_llm:${BACKEND}}
[ "${BACKEND}" = "cuda" ] || [ "${BACKEND}" = "vulkan" ] || { echo "backend must be cuda or vulkan" >&2; exit 2; }

docker build -f "docker/Dockerfile.${BACKEND}" -t "${TAG}" .
echo "Built ${TAG}"
