# Tooling notes
- Arch installs the toolkit under `/opt/cuda`; add `/opt/cuda/bin` to PATH
  or CMake "Could not find nvcc".
- llama.cpp v0.6.0 `llama-server` needs its full `build/bin/` (companion
  `libllama*.so`) plus `libcudart.so.12` / `libcublas*` from the toolkit —
  it does not bundle them. Host `libcuda.so.1` arrives only with
  `--gpus`.
- ggml-vulkan CMake needs Arch `spirv-headers` (+ `shaderc` for glslc).
- AMD Vulkan needs no host-driver mounts: the image bakes in Mesa's RADV
  (`vulkan-radeon`), which drives the host `amdgpu` kernel driver via
  `/dev/dri` render nodes (stable kernel ABI). Conversely, never bake
  `nvidia-utils` into an image — NVIDIA userspace must be injected by
  nvidia-container-toolkit to match the host driver. AMD runs use
  `--device /dev/dri` + `GGML_VK_VISIBLE_DEVICES` (`SLM_GPU_DRIVER=amd`
  in the scripts; `vulkan-nvidia` / `vulkan-amd` services in compose.yaml).
  Render-node permissions: Arch udev makes `/dev/dri/renderD*` 0666, but
  Debian-family hosts commonly restrict them to `0660 root:render` — with
  non-root containers that needs `--group-add <render-gid>`
  (`SLM_GROUP_ADD` in the scripts, `group_add` in compose); find the gid
  with `stat -c %g /dev/dri/renderD128`.
- llama.cpp v0.6.0 native env options (`LLAMA_ARG_*`, e.g.
  `LLAMA_ARG_CPU_MOE` / `LLAMA_ARG_N_CPU_MOE` / `LLAMA_ARG_OVERRIDE_TENSOR`)
  are parsed before CLI flags (CLI wins) but crash on set-but-empty values
  (`LLAMA_ARG_N_CPU_MOE=` hits `std::stoi("")`). Forward them only when
  non-empty: `run_server.sh` skips empties explicitly; compose.yaml relies on
  null-passthrough (`VAR:`), which sets the var in the container only when
  the host var is set — never write `${VAR:-}` defaults for these.
- llama.cpp v0.6.0 flash-attn / KV cache: `-fa/--flash-attn` takes an
  optional `on|off|auto` (default `auto`; native env
  `LLAMA_ARG_FLASH_ATTN`), `-ctk/-ctv` = `LLAMA_ARG_CACHE_TYPE_K/V`
  (allowed: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1). Quantized V
  cache requires FA on — `SLM_KV_QUANT` wraps the exact trio
  (`--flash-attn on -ctk/-ctv <type>`) in `docker/entrypoint.sh`.
- Compose 5.x: profiled services resolve via `docker compose config <service>`
  (positional); `--profile` flag no longer exists on `config`. Same for
  `down`/`stop`: with every service under a profile, plain `docker compose
  down` is a SILENT NO-OP — pass the service name or `--profile`. A `${VAR:?}`
  interpolation in compose.yaml makes EVERY subcommand (`down`, `logs`, ...)
  require the variable — inline it per command or put it in `.env`
  (auto-loaded from the project dir, already git-ignored here).
- llama.cpp server log timestamps are `MM:SS.mmm` since process start, not
  seconds — "6.20.889" = 6 min 20 s (e.g. the silent `-hf` model download
  happens before the `load_model:` line, so a long pre-load silence with a
  slowly-growing container netIO is a normal download, not a hang).
- The HF download cache under `data/cache` is host-user-owned: the run
  scripts pass `--user $(id -u):$(id -g)` (compose: `user: SLM_UID:SLM_GID`,
  default 1000) and the directory is tracked in git via an inner `.gitignore`
  (`*` / `!.gitignore`) so only the directory itself is committed. To fix
  ownership of files written by older root containers without host sudo:
  `docker run --rm --entrypoint /bin/chown -v "$PWD/data/cache:/c"
  simple_llm:vulkan -R "$(id -u):$(id -g)" /c` (the image ENTRYPOINT
  hijacks plain `docker run <image> chown ...`).
- Granite 4.x models are reasoning models: with the default reasoning
  parser, small `max_tokens` yields empty `content` (text goes to
  `reasoning_content`); budget tokens for the thinking or pass
  `SLM_EXTRA_ARGS="--reasoning off"` like integration_test.sh does.
- The integration suite runs pytest INSIDE the container against the code
  baked in at build time (`COPY . /app`) — after changing `tests/`,
  `simple_llm/client/` or any other in-image code, rebuild the image
  (`scripts/build_image.sh <backend>`; cached llama.cpp layers make it
  fast) or the suite silently runs the stale copy.
- Vision (mmproj): with `-hf`, llama.cpp v0.6.0 auto-selects the projector
  via `find_best_sibling` — closest quant to the tag, first file wins ties;
  for unsloth Qwen3-VL repos that is `mmproj-BF16.gguf` (Q4_K_M tag →
  16-bit tie BF16/F16), and only that file is downloaded.
- For long-running commands (e.g. `docker compose up --build`), detach with `setsid ... </dev/null >path/.tmp/run.log 2>&1 & disown` and poll the log in separate tool calls; do not pipe through `tail` (it buffers until EOF, so a hanging build shows no output and times out the tool).
- When investigating network/services unavailability use `nc -zvw 1 TARGET TARGET_PORT`, `curl -vkL TARGET >/dev/null`, `mtr --json --aslookup --mpls --timeout 30 TARGET`;
- `nc -zuvw2 host port` "succeeding" proves nothing for UDP: a UDP `connect()` never fails on a remote host, so it reports success even when every packet is silently dropped. Verify UDP reachability with a server-side `tcpdump -ni <wan_if> udp port <port>` while generating traffic, or `mtr --udp`;
