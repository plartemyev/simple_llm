#!/usr/bin/env bash
# llama-server launcher.
#
# Configuration via environment variables (all optional except SLM_MODEL):
#   SLM_MODEL       "repo:quant" HF id (downloaded and cached on first start)
#                   or an absolute path to a .gguf file inside the container
#                   (mount it under /models).
#   SLM_HOST        bind address            (default 0.0.0.0)
#   SLM_PORT        port                    (default 8011)
#   SLM_CTX_SIZE    context size, tokens    (default 8192)
#   SLM_PARALLEL    concurrent slots        (default 4)
#   SLM_GPU_LAYERS  layers offloaded to GPU (default 999 = all)
#   SLM_EXTRA_ARGS  extra llama-server flags, word-split
#   HF_TOKEN        optional; required only for gated HF repositories
set -euo pipefail

MODEL=${SLM_MODEL:?set SLM_MODEL to a HF id (repo:quant) or a /models/*.gguf path}
HOST=${SLM_HOST:-0.0.0.0}
PORT=${SLM_PORT:-8011}
CTX=${SLM_CTX_SIZE:-8192}
PARALLEL=${SLM_PARALLEL:-4}
NGL=${SLM_GPU_LAYERS:-999}

args=(
  --host "${HOST}"
  --port "${PORT}"
  --ctx-size "${CTX}"
  --parallel "${PARALLEL}"
  --n-gpu-layers "${NGL}"
  --jinja
)

if [ -f "${MODEL}" ]; then
  args+=(-m "${MODEL}")
elif [ -f "/models/${MODEL}" ]; then
  args+=(-m "/models/${MODEL}")
else
  args+=(-hf "${MODEL}")
fi

# shellcheck disable=SC2086
exec llama-server "${args[@]}" ${SLM_EXTRA_ARGS:-}
