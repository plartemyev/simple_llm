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
compose.yaml       docker compose service definitions (cuda / vulkan-nvidia / vulkan-amd)
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
#    AMD Vulkan host instead (see "GPU backend notes"):
#    SLM_BACKEND=vulkan SLM_GPU_DRIVER=amd scripts/run_server.sh MODEL

# 4. use it
curl -s http://127.0.0.1:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ornith-ai/Ornith-1.5-9B-GGUF:Q8_0","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}'
```

The first start downloads the model into `./data/cache` (mounted at
`/data/cache`); subsequent starts are instant — but an HF id still
re-resolves the repo on `huggingface.co` at every start. For restarts that
never touch the network, serve the cached file directly — see
[Model cache and offline runs](#model-cache-and-offline-runs). The directory
is tracked in
git through its inner `.gitignore` (contents stay local); the container
runs as the invoking user (compose: `SLM_UID`/`SLM_GID`, default 1000), so
downloads remain host-user-owned.

### Docker Compose

`compose.yaml` defines the same service for both backends and is an
alternative to `run_server.sh` (images are built automatically when
missing):

```shell
SLM_MODEL=ornith-ai/Ornith-1.5-9B-GGUF:Q8_0 docker compose up -d cuda   # or: vulkan-nvidia (NVIDIA) / vulkan-amd (AMD)
docker compose logs -f cuda    # then use the endpoint as shown above
docker compose down cuda       # teardown needs the service name: every
                               # service has a profile, so plain `down`
                               # selects nothing
```

The same environment variables configure it (see [Configuration](#configuration));
variables can also live in a `.env` file next to `compose.yaml` (auto-loaded,
git-ignored). `secrets.env` is loaded when present and forwarded wholesale
into the container — see [Configuration](#configuration).

## Home-lab quick start: large MoE on a small AMD GPU

The recipe reproduces the reference deployment of this repo: a 35B-total /
3B-active MoE (`unsloth/Qwen3.6-35B-A3B-GGUF`, 4-bit) served on a Ryzen + RX
Vega 8 GB host — expert weights in system RAM (`--n-cpu-moe 999`), dense
weights plus a 4-bit-quantized KV cache for a 65536-token context on the GPU,
fully offline serving through docker compose. Any AMD GPU works (the image
ships Mesa's RADV; host only needs the amdgpu kernel stack).

1. **Size the quant against RAM, not VRAM.** With all experts on CPU, the GPU
   holds only the dense part (a few GB at 4-bit) plus the KV cache (with
   `q4_0` at 64K ctx, ~2 GB); the experts — most of the file — live in system
   RAM. Check `free -h` and the repo's file sizes (the HF "Files" tab), pick
   the largest 4-bit file that fits available RAM minus a couple GB. A 64 GB
   RAM host comfortably takes `UD-Q4_K_XL` (20.8 GB file).

2. **Download into the shared hub cache** (any host with `uv`; the hf-xet
   bridge hangs behind some proxies, the env var falls back to the plain
   CDN):

   ```shell
   export HF_HOME=$PWD/data/cache/.cache/huggingface HF_HUB_DISABLE_XET=1
   uvx --from huggingface_hub hf download \
     unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
   ```

3. **Hardlink into `data/models`** (zero-copy, and the offline serving layout
   from [Model cache and offline runs](#model-cache-and-offline-runs)). Name
   the link explicitly: the snapshot entry is a symlink into `blobs/<sha>`, so
   `ln` into a directory alone would leave the link named by blob hash.

   ```shell
   ln "$(readlink -f data/cache/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/*/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf)" \
      data/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
   ```

4. **Configure `.env`** next to `compose.yaml` (git-ignored). Values are not
   shell-expanded by compose — generate the API key first, then write the
   literal line:

   ```shell
   cat > .env <<EOF
   SLM_MODEL=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf   # bare name -> offline -m serving
   SLM_CTX_SIZE=65536
   SLM_PARALLEL=4
   SLM_KV_QUANT=q4_0                           # 4-bit KV; enables flash attention
   LLAMA_ARG_N_CPU_MOE=999                     # all expert weights on CPU
   EOF
   echo "SLM_EXTRA_ARGS=--api-key sk-$(openssl rand -hex 24) --offline" >> .env
   chmod 600 .env
   ```

   Together with the entrypoint's default `--n-gpu-layers 999`, the pair
   `999` + `--n-cpu-moe 999` means "everything on the GPU except experts".

5. **Launch** (first run builds `simple_llm:vulkan`; then the container
   restarts on failure and at boot until stopped — `restart: unless-stopped`):

   ```shell
   docker compose up -d vulkan-amd
   docker compose logs -f vulkan-amd
   ```

   Debian-family hosts with `0660` render nodes need the owning group added
   to the container — see [Vulkan](#gpu-backend-notes).

6. **Expose to the LAN (optional)** with a local override. `ports:` lists are
   *appended* across compose files, so the base loopback bind must be
   replaced with the `!override` tag, not appended to:

   ```yaml
   # compose.override.yaml (add to .gitignore; compose loads it automatically)
   services:
     vulkan-amd:
       ports: !override
         - "${SLM_PORT:-8011}:8011"
   ```

   Recreate once more: `docker compose up -d vulkan-amd`.

7. **Verify** (from the server's localhost, or the LAN with step 6):

   ```shell
   curl -s http://127.0.0.1:8011/health        # {"status":"ok"} — probes stay open
   curl -s http://127.0.0.1:8011/v1/models \
     -H "Authorization: Bearer $(grep SLM_EXTRA_ARGS .env | awk '{print $2}')"   # 200
   ```

   Chat requests are the same shape as in the [Quick start](#quick-start)
   curl, plus the bearer header. First request after a start pays a Vulkan
   shader-compile warmup. Reasoning models think by default: budget
   `max_tokens` (~800) or send
   `"chat_template_kwargs": {"enable_thinking": false}`. Swap models by
   pointing `SLM_MODEL` at another file in `data/models` (e.g.
   `GLM-4.7-Flash-UD-Q4_K_XL.gguf`) and re-running step 5.

## Model cache and offline runs

`docker/entrypoint.sh` picks the model source from the shape of `SLM_MODEL`:

| `SLM_MODEL` value | flag | outbound traffic |
|---|---|---|
| HF id `owner/repo:QUANT` | `-hf` | every start resolves the repo on `huggingface.co` (commit + file list) and downloads whatever is missing from the cache |
| path to an existing `.gguf` | `-m` | none |
| bare `.gguf` name (resolved under `/models`) | `-m` | none |

In all modes prompts and completions stay on the host (the service is
published on `127.0.0.1` only); the table is only about HF weight traffic.

### First run (online)

Any HF-id start populates the cache under
`data/cache/.cache/huggingface/hub/` (the container's `HOME=/data/cache` is
bind-mounted from `./data/cache`; llama.cpp's hub-cache layout:
`models--<owner>--<repo>/{refs/main,blobs/,snapshots/<commit>/}`):

```shell
scripts/run_server.sh ornith-ai/Ornith-1.5-9B-GGUF:Q8_0 -g 1
# first start: resolve + download happens before the load_model: log line
```

The resolved commit is recorded in `refs/main`, so the cache knows exactly
which snapshot it holds (gated repos: `HF_TOKEN` as in the Quick start).
Restarting with the same HF id reuses the blobs but still makes the two
resolution requests per start — production should serve from the cache
directly:

### Production runs (no HF traffic)

Any of the following makes restarts fully offline:

- **Copy into `data/models`** (recommended — self-contained, survives cache
  cleanup), then serve the bare file name:

  ```shell
  GGUF=$(ls data/cache/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-9B-GGUF/snapshots/*/Ornith*Q8_0.gguf)
  cp "$GGUF" data/models/ornith-1.5-9b-q8_0.gguf   # or ln (same filesystem)
  scripts/run_server.sh ornith-1.5-9b-q8_0.gguf -g 1
  # compose: SLM_MODEL=ornith-1.5-9b-q8_0.gguf docker compose up -d cuda
  ```

- **Point at the cached snapshot** (no copy, pins the exact commit): pass the
  container path
  `/data/cache/.cache/huggingface/hub/models--<owner>--<repo>/snapshots/<commit>/<file>.gguf`
  as `SLM_MODEL`. The cache volume persists, so the pointer is stable across
  restarts — just not across manual cache pruning.

- **Keep the HF id, add `--offline`**: resolution happens from the local
  cache only (`refs/main` → `snapshots/<commit>`); zero requests, and a
  missing cache fails fast instead of downloading. The `:QUANT` suffix still
  selects among the cached files. Least pinned: serves whatever commit was
  current at the last online run.

  ```shell
  scripts/run_server.sh ornith-ai/Ornith-1.5-9B-GGUF:Q8_0 -g 1 -- --offline
  # compose: SLM_EXTRA_ARGS=--offline (environment or .env)
  ```

To relocate the cache root inside the container (`.cache/huggingface/hub`
under `$HOME` is llama.cpp's default), set `LLAMA_CACHE=/data/cache/hf` —
e.g. via `secrets.env`; the host-side paths above change accordingly.

## Vision (image-to-text)

Vision is **opt-in** so text-only deployments never pay for it: by default
the server runs with `--no-mmproj` — no projector is downloaded and no VRAM
is reserved for one. `SLM_MMPROJ` (scripts and compose) enables it:

| `SLM_MMPROJ` | behavior |
|---|---|
| unset (default) | text-only — passes `--no-mmproj` |
| `auto` | HF-id models: llama-server detects the projector in the repo and downloads it with the model (e.g. `mmproj-BF16.gguf`, ~0.8 GB for Qwen3-VL-4B); no-op for local `.gguf` files |
| path or name in `./data/models` | explicit projector `.gguf`, resolved like `SLM_MODEL` — the way to serve vision from local files |

`GET /props` reports `"modalities": {"vision": true}` once a projector is
loaded — the integration suite skips its vision test without one.

Requests use the OpenAI multimodal content format (images as base64 data
URLs — llama-server does not fetch remote `http(s)://` images):

```shell
IMG=$(base64 -w0 image.png)
curl -s http://127.0.0.1:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"<model id>\",\"messages\":[{\"role\":\"user\",\"content\":[
        {\"type\":\"text\",\"text\":\"Describe this image.\"},
        {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$IMG\"}}]}]}"
```

Projector weights add VRAM on top of the main model (a F16/BF16 projector is
typically 0.5–1.5 GB) — account for it when sizing the quant against the
card.

## Configuration

Environment variables select the backend, size the server, and tune
llama-server. *scripts* = `scripts/run_server.sh` +
`scripts/integration_test.sh`; *compose* = `compose.yaml`; *container* = read
by the image entrypoint / llama-server directly.

| variable | used by | default | purpose |
|---|---|---|---|
| `SLM_MODEL` | scripts, compose | — (required) | HF id `owner/repo:QUANT` or a `.gguf` name inside `./data/models`; `QUANT` is any quantization the repo ships (see [Models](#models)) |
| `SLM_BACKEND` | scripts | `cuda` | image backend: `cuda` or `vulkan` |
| `SLM_IMAGE` | scripts, compose | `simple_llm:$SLM_BACKEND` | image tag override |
| `SLM_GPU` | scripts, compose | `1` | GPU index — CUDA: `CUDA_VISIBLE_DEVICES`; with `SLM_GPU_DRIVER=amd`: overrides the Vulkan device (which defaults to 0) |
| `SLM_GPU_DRIVER` | scripts | `nvidia` | `amd` switches to `/dev/dri` + `GGML_VK_VISIBLE_DEVICES` |
| `SLM_PORT` | scripts, compose | `8011` | host port (the container always listens on 8011) |
| `SLM_CTX_SIZE` | scripts, compose | `8192` | context size, tokens |
| `SLM_PARALLEL` | scripts, compose | `4` | concurrent slots |
| `SLM_MODELS_DIR` / `SLM_CACHE_DIR` | scripts | `./data/models` / `./data/cache` | bind-mount sources |
| `SLM_EXTRA_ARGS` | scripts, compose | — | extra llama-server flags, word-split; overrides `LLAMA_ARG_*` env vars |
| `SLM_KV_QUANT` | scripts, compose | — | K/V cache type (e.g. `q8_0`), enables flash attention — see [KV cache quantization](#kv-cache-quantization) |
| `SLM_MMPROJ` | scripts, compose | — | vision control: `auto` (HF repos download+load their projector), a projector path/name, unset = text-only — see [Vision (image-to-text)](#vision-image-to-text) |
| `SLM_UID` / `SLM_GID` | compose | `1000` | container uid:gid, keeps `./data` host-user-owned |
| `SLM_GROUP_ADD` | scripts | — | supplementary host group ids (space/comma separated) for group-restricted `/dev/dri/renderD*` nodes; compose: `group_add` |
| `LLAMA_ARG_CPU_MOE`, `LLAMA_ARG_N_CPU_MOE`, `LLAMA_ARG_OVERRIDE_TENSOR` | both (passthrough) | — | MoE / tensor-placement tuning — see [MoE offload](#moe-offload--tensor-placement) |
| `HF_TOKEN` | both | — | token for gated HF repos (`secrets.env` or environment) |

`secrets.env` (git-ignored) is loaded when present and forwarded wholesale
(`--env-file` / compose `env_file`): everything in it reaches the container.
Beyond `HF_TOKEN` this makes it the place for tuning variables the helpers
do not declare — e.g. any native `LLAMA_ARG_*` (non-empty values only). The
host environment itself reaches the container only through the variables
listed above (compose: only the keys declared in `compose.yaml`).

Container-only (not forwarded by the helpers): `SLM_HOST`, `SLM_GPU_LAYERS`.
Image build args (`CUDA_ARCH`, `CUDA_PKG_URL`, `LLAMA_CPP_REF`) are
documented in the `docker/Dockerfile.*` headers.

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
SDK (`vulkan-headers`, `shaderc`). The runtime image ships Mesa's RADV ICD
(`vulkan-radeon`), so the GPU vendor is decided by runtime wiring only.

**AMD hosts** work out of the box: the container's RADV drives the host
`amdgpu` kernel driver through `/dev/dri` render nodes (host needs the
standard Mesa/`linux-firmware` stack; Mesa ≥ 24.x exposes
`VK_KHR_cooperative_matrix`, which llama.cpp uses automatically — older
Mesa falls back to slower code paths). Start with
`SLM_GPU_DRIVER=amd scripts/run_server.sh ...` or
`docker compose up -d vulkan-amd`; select the GPU via
`GGML_VK_VISIBLE_DEVICES` (defaults to Vulkan device 0; `SLM_GPU` overrides).

The container runs as a non-root user, so it can only open render nodes it
has permission for. World-accessible nodes (`0666`, the Arch default) need
nothing; group-restricted ones (`0660 root:render` — the common case on
Debian-family hosts) need the owning group added to the container: set
`SLM_GROUP_ADD=<gid>` for the scripts (comma/space separated ids) or add
`group_add: [<gid>]` to the compose service; find the gid with
`stat -c %g /dev/dri/renderD128` on the host.

**NVIDIA hosts** need the driver's Vulkan producer injected, because the
image deliberately carries no NVIDIA userspace (it would version-lock the
image to the host driver):

- driver with Vulkan support,
- container gets the driver's Vulkan producer via nvidia-container-toolkit
  (`NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility`),
- an NVIDIA ICD json visible inside the container (mount
  `/usr/share/vulkan/icd.d/nvidia_icd.json` if the host provides one).

GPU/device selection inside Vulkan runs uses `GGML_VK_VISIBLE_DEVICES`
(`run_server.sh` forwards it when set).

### MoE offload / tensor placement

The llama-server native `LLAMA_ARG_*` tuning options are forwarded from the
host environment into the container by both `scripts/run_server.sh` and
`compose.yaml` when set (flags given via `SLM_EXTRA_ARGS` or `--` take
precedence over them):

| env var | llama-server flag |
|---|---|
| `LLAMA_ARG_CPU_MOE=1` | `--cpu-moe` — keep all MoE expert weights on CPU |
| `LLAMA_ARG_N_CPU_MOE=N` | `--n-cpu-moe N` — same, first N layers only |
| `LLAMA_ARG_OVERRIDE_TENSOR=...` | `-ot` / `--override-tensor` — comma-separated `tensor_pattern=buffer` overrides, e.g. `exps=CPU,iqk=CPU` |

They work with both backends. CPU buffers (`exps=CPU`) are universal;
GPU-side overrides must name the backend's buffers — `CUDA0`, `CUDA1`, ...
on CUDA builds, `VULKAN0`, `VULKAN1`, ... on Vulkan builds. Offloading
expert weights to CPU trades throughput for VRAM; useful when a MoE model's
dense layers fit on the GPU but its expert tensors do not. Set vars only to
non-empty values (an empty `LLAMA_ARG_N_CPU_MOE` fails to parse). Only the
three vars above are declared in the helpers; any other `LLAMA_ARG_*`
reaches the container through `secrets.env` (forwarded wholesale — see
[Configuration](#configuration)).

### KV cache quantization

`SLM_KV_QUANT=q8_0` (scripts and compose) stores the K and V caches in the
given GGUF type and turns flash attention on with it (quantized V cache
requires FA) — it expands to `--flash-attn on --cache-type-k q8_0
--cache-type-v q8_0`. Allowed types (default `f16`): `f32 f16 bf16 q8_0
q4_0 q4_1 iq4_nl q5_0 q5_1`; `q8_0` halves KV VRAM vs `f16`. Flash
attention alone (cache stays `f16`): `SLM_EXTRA_ARGS="--flash-attn on"`.

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

Context sizing: llama-server divides `-c` across `--parallel` slots. Per-call budget is
`prompt + max_tokens` (each fragment call is stateless); the largest production prompt measured
~412 tokens + 128 output → **2048 tokens per slot ≈ 2× worst case, the practical minimum**. Set
`-c = slots × 2048`. Context VRAM cost is small at this scale (measured on a 9B Q8_0: 8192×4 vs
2048×1 differs by only ~0.35 GiB — weights dominate); revisit only if prompts grow (few-shot
examples) or per-slot context exceeds ~8k (then consider `SLM_KV_QUANT=q8_0`).

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
| `unsloth/Qwen3.5-9B-GGUF` | 9B | Q8_0 | apache-2.0 |
| `unsloth/gemma-4-12b-it-GGUF` | 12B | Q8_0 | gemma (mirror) |

`ornith-ai/Ornith-1.5-9B-GGUF` was evaluated and dropped as a clear outsider
(see the parent benchmark report).

Any quantization a GGUF repo ships can be served: pass it as the `SLM_MODEL`
suffix — `:Q4_K_M`, `:UD-Q4_K_XL`, `:Q3_K_XL` (case-insensitive). Without a
suffix llama-server defaults to `Q4_K_M` and falls back to the first `.gguf`
in the repo. The benchmark candidate list pairs 27B+ models with 4/3-bit
quants that fit a single 24 GB card; combine with MoE offload and
`SLM_KV_QUANT` when VRAM is still tight.

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
# AMD host: SLM_GPU_DRIVER=amd scripts/integration_test.sh MODEL -b vulkan
# vision model + vision test: SLM_MMPROJ=auto (see "Vision (image-to-text)")
```

## License

GPLv2 — see LICENSE. Model weights referenced by `benchmarks/models.toml`
remain under their respective licenses.
