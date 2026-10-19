#!/usr/bin/env bash
# llama-server launcher.
#
# Configuration via environment variables (all optional except SLM_MODEL):
#   SLM_MODEL       "repo:quant" HF id (downloaded and cached on first start)
#                   or an absolute path to a .gguf file inside the container
#                   (mount it under /models).
#   SLM_MMPROJ      vision (image-to-text) control, three modes:
#                     unset (default)  text-only — passes --no-mmproj: a
#                                      projector is never downloaded or loaded
#                     "auto"           llama-server default (--mmproj-auto):
#                                      for HF ids the repo's projector is
#                                      downloaded and loaded; no-op for local
#                                      .gguf files
#                     path or name     explicit projector .gguf, resolved like
#                                      SLM_MODEL (absolute path or a name
#                                      under /models)
#   SLM_HOST        bind address            (default 0.0.0.0)
#   SLM_PORT        port                    (default 8011)
#   SLM_CTX_SIZE    context size, tokens    (default 8192)
#   SLM_PARALLEL    concurrent slots        (default 4)
#   SLM_GPU_LAYERS  layers offloaded to GPU (default 999 = all)
#   SLM_KV_QUANT    KV cache type for K and V, e.g. q8_0 (default f16);
#                   enables flash attention along with it (quantized V
#                   cache requires FA)
#   SLM_EXTRA_ARGS  extra llama-server flags, word-split
#   HF_TOKEN        optional; required only for gated HF repositories
#
# llama-server also honors its native LLAMA_ARG_* env vars directly (e.g.
# LLAMA_ARG_CPU_MOE / LLAMA_ARG_N_CPU_MOE / LLAMA_ARG_OVERRIDE_TENSOR for MoE
# and tensor-placement tuning, see scripts/run_server.sh); flags given here
# via SLM_EXTRA_ARGS take precedence over them.
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

if [ -n "${SLM_KV_QUANT:-}" ]; then
  # quantized KV cache: -fa on -ctk/-ctv <type> (V quantization needs FA)
  args+=(
    --flash-attn on
    --cache-type-k "${SLM_KV_QUANT}"
    --cache-type-v "${SLM_KV_QUANT}"
  )
fi

if [ -f "${MODEL}" ]; then
  args+=(-m "${MODEL}")
elif [ -f "/models/${MODEL}" ]; then
  args+=(-m "/models/${MODEL}")
else
  args+=(-hf "${MODEL}")
fi

if [ -n "${SLM_MMPROJ:-}" ]; then
  if [ "${SLM_MMPROJ}" = auto ]; then
    # llama-server's --mmproj-auto default: for -hf models the projector is
    # discovered in the repo and downloaded; local .gguf files are unaffected
    :
  elif [ -f "${SLM_MMPROJ}" ]; then
    args+=(--mmproj "${SLM_MMPROJ}")
  elif [ -f "/models/${SLM_MMPROJ}" ]; then
    args+=(--mmproj "/models/${SLM_MMPROJ}")
  else
    echo "ERROR: SLM_MMPROJ='${SLM_MMPROJ}': no such file (auto, absolute path, or a name under /models)" >&2
    exit 1
  fi
else
  # text-only default: never download or load a multimodal projector
  args+=(--no-mmproj)
fi

# shellcheck disable=SC2086
exec llama-server "${args[@]}" ${SLM_EXTRA_ARGS:-}
