# simple_llm

Self-hosted, OpenAI-compatible LLM serving for latency-sensitive text
transformation workloads, packaged as an Arch Linux Docker image.

The service is [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
`llama-server`: a single-binary OpenAI-compatible HTTP server
(`/v1/chat/completions`, streaming and non-streaming) that runs small local
models on one GPU. It is a drop-in backend for any client speaking the
OpenAI chat-completions protocol — repoint `base_url` and you have replaced
a hosted LLM API with a local GPU.

- **Backend:** CUDA (primary) and Vulkan compute (secondary), selectable at
  image build time.
- **Base image:** Arch Linux (`archlinux:base-devel`).
- **License:** GPLv2. Upstream model weights keep their own licenses.

## Repository layout

```
docker/            Dockerfile.cuda / Dockerfile.vulkan + entrypoint
scripts/           build / run / data-pull / integration-test helpers
simple_llm/        thin OpenAI-compatible client (phrase + splits profiles)
benchmarks/        benchmark infrastructure (slicer, runner, quality, report)
tests/             unit tests (any host) + integration tests (needs a server)
```

## Quick start

```shell
# 1. build (CUDA backend; see "GPU backend notes" for Vulkan)
scripts/build_image.sh cuda

# 2. (optional) token for gated HF repos — file is git-ignored
echo 'HF_TOKEN=hf_xxx' > secrets.env && chmod 600 secrets.env

# 3. serve a model (HF id:quant or a local .gguf inside ./data/models)
scripts/run_server.sh ornith-ai/Ornith-1.5-9B-GGUF:Q8_0 -g 1 -p 8011

# 4. use it
curl -s http://127.0.0.1:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ornith-ai/Ornith-1.5-9B-GGUF:Q8_0","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}'
```

The first start downloads the model into `./data/cache` (mounted at
`/data/cache`); subsequent starts are instant.

## GPU backend notes

### CUDA

Arch's current `cuda` package is 13.x and needs driver ≥ 580. GPU hosts
running the **570 driver series** (CUDA 12.8 generation) are instead served
with the newest 12.x toolkit from the [Arch Linux Archive]
(https://archive.archlinux.org): **`cuda 12.9.1-2`**, pinned in
`docker/Dockerfile.cuda`:

- CUDA *minor version compatibility* lets binaries built with any 12.x
  toolkit run on any R525+ driver **when they carry SASS for the target
  GPU** — the Dockerfile compiles explicit `sm_XX` cubins
  (`CUDA_ARCH` build-arg, default `86` for Ampere), so no PTX JIT is
  involved.
- nvcc 12.9 accepts host GCC up to 14.x; the build pins `gcc14` from the
  archive (dropped from current repos) and *verifies* the ceiling compiled
  into `/opt/cuda/include/crt/host_config.h` (fails fast on mismatch).
- If a host driver predates R525 or oddities appear, drop
  `CUDA_PKG_URL` to `cuda-12.8.1-2` (also in the archive) — one-ARG change.

### Vulkan

`docker/Dockerfile.vulkan` builds the same server against the system Vulkan
SDK (`vulkan-headers`, `shaderc`). Runtime requirements on NVIDIA hosts:

- driver with Vulkan support,
- container gets the driver's Vulkan producer via nvidia-container-toolkit
  (`NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility`),
- an NVIDIA ICD json visible inside the container (mount
  `/usr/share/vulkan/icd.d/nvidia_icd.json` if the host provides one).

GPU/device selection inside Vulkan runs uses `GGML_VK_VISIBLE_DEVICES`.

## Call profiles

Two workload shapes cover the typical latency-sensitive text transformation
(no task-specific prompt logic lives here — callers bring their own):

| | `phrase` | `splits` |
|---|---|---|
| shape | complete phrase → one call | dictation stream → many small fragment calls |
| streaming | off | on (act on first token) |
| temperature | 0.1 | 0.1 |
| max_tokens | 128 (~2× longest expected output) | 64 |
| timeout | 60 s | 30 s |
| server | `--parallel` ≥ concurrent callers | same, + context fits system prompt + phrase |

`splits` mimics interim speech-recognition output: text arrives in 3-8 word
fragments and each fragment is transformed by its own call. The benchmark
`[splits]` scenario measures exactly this pattern (per-fragment latency +
TTFT at concurrency).

The thin client codifies both profiles:

```python
from simple_llm.client import LLMClient, CallParams

c = LLMClient("http://127.0.0.1:8011/v1", "ornith-ai/Ornith-1.5-9B-GGUF:Q8_0")
text = c.complete("Translate to uppercase tokens: hello world",
                  params=CallParams(temperature=0.1, max_tokens=128))
for delta in c.complete_stream("short fragment", params=CallParams(max_tokens=64)):
    ...
```

### Migrating from hosted OpenAI endpoints

| hosted API | here |
|---|---|
| `client = OpenAI()` | `LLMClient(base_url="http://host:8011/v1", model=<served id>)` |
| `responses.create(model=..., input=...)` | `complete(prompt, params=...)` (chat completions schema) |
| streaming SSE | `complete_stream(...)` |
| `temperature`, `max_tokens` | same fields via `CallParams` |

`llama-server` implements the chat-completions and completions endpoints
(`GET /v1/models`, `GET /health` included); the Responses API is not part of
the surface.

## Models

Candidate list with quantizations: `benchmarks/models.toml`. Selection
criteria: small (≤ 13B), 8-bit quantization (GGUF `Q8_0` — the practical
equivalent of FP8 for llama.cpp), instruction-tuned, multilingual
(English / Russian / Armenian caption text), ungated downloads, permissive
or mirror-available licenses.

| model | params | quant | license |
|---|---|---|---|
| `ornith-ai/Ornith-1.5-9B-GGUF` | 9B | Q8_0 | MIT |
| `unsloth/Qwen3.5-9B-GGUF` | 9B | Q8_0 | apache-2.0 |
| `unsloth/Qwen3.5-4B-GGUF` | 4B | Q8_0 | apache-2.0 |
| `unsloth/gemma-4-12b-it-GGUF` | 12B | Q8_0 | gemma (mirror) |

## Benchmarks

See `benchmarks/README.md`. In short:

- **data** — public talk captions (OPUS TED2020) for each benchmark
  language; `scripts/pull_cc_data.sh en ru hy`;
- **slicing** — `benchmarks/slicer.py` builds whole-phrase samples and
  dictation-style fragment streams from the captions;
- **performance** — `benchmarks/runner.py` measures latency / TTFT /
  throughput at several concurrency levels (VRAM sampled in parallel);
- **quality** — `benchmarks/quality.py` replays a gold JSONL (messages +
  expected tokens + output rules) and scores token overlap (F1) and
  rule compliance;
- **report** — `benchmarks/report.py` merges result JSONs into markdown.

Use-case prompts and gold sets are supplied by the caller (kept out of this
repo on purpose); the schema is documented in `benchmarks/README.md`.

## Tests

```shell
uv run --extra test python -m pytest tests/ -m "not integration"   # unit, any host
scripts/integration_test.sh ornith-ai/Ornith-1.5-9B-GGUF:Q8_0 -g 1  # full stack on a GPU host
```

## License

GPLv2 — see LICENSE. Model weights referenced by `benchmarks/models.toml`
remain under their respective licenses.
