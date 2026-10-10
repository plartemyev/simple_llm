# Benchmarks

Infrastructure for measuring local LLM serving quality and performance. The
benchmark is workload-agnostic: it consumes public caption text and a
caller-provided gold set; task-specific prompts stay outside this repo.

## Layout

```
models.toml           candidate models (repo:quant, licenses)
config.example.toml   call profiles + scenario sizes
slicer.py             captions -> phrase samples / dictation fragment streams
runner.py             performance scenarios (latency, TTFT, throughput, VRAM)
quality.py            replays gold message lists, scores F1 + format rules
report.py             aggregates results/*.json into markdown
data/
  samples/            checked-in caption snippets (unit tests, smoke runs)
  cc/                 full corpus (git-ignored; scripts/pull_cc_data.sh)
```

## Steps

```shell
# 1. caption data (public OPUS TED2020 talk captions)
scripts/pull_cc_data.sh en ru hy

# 2. start a model (host-side helper)
scripts/run_server.sh ornith-ai/Ornith-1.5-9B-GGUF:Q8_0 -g 1 -p 8011
MODEL_ID=$(curl -s http://127.0.0.1:8011/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
BASE=http://127.0.0.1:8011/v1

# 3. performance: phrase profile (whole phrase per call)
python -m benchmarks.runner --base-url $BASE --model "$MODEL_ID" \
    --scenario phrase --lang en --concurrency 1 4 --out results/phrase_en.json

# 4. performance: splits profile (small fragments, streaming, TTFT)
python -m benchmarks.runner --base-url $BASE --model "$MODEL_ID" \
    --scenario splits --lang en --concurrency 1 4 --out results/splits_en.json

# 5. quality against a gold set (schema below)
python -m benchmarks.quality --gold gold.jsonl --base-url $BASE \
    --model "$MODEL_ID" --out results/quality.json

# 6. report
python -m benchmarks.report results/
```

Repeat 3-5 per language and per model; keep every JSON in `results/`.

## Gold set schema (quality)

One JSON object per line; `messages` is the exact request to replay, so the
caller's prompt templates are applied on their side:

```json
{"id": "en-001",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "expected": ["TOKEN1", "TOKEN2"],
 "expected_variants": [["TOKEN1", "TOKEN2B"]],
 "rules": {"uppercase": true, "single_word_tokens": true, "banned_chars": "()", "separator": ","},
 "max_tokens": 128}
```

Scores: multiset token precision/recall/F1 after normalization (case fold,
punctuation strip) plus rule compliance rates (uppercase-only tokens,
single-word tokens, banned characters absent).

Normalization is production-parity: token boundaries follow the separator,
then hyphens are treated as separators and multi-word fragments are split on
whitespace before F1 (compliance rules still judge the raw split).
`expected_variants` (optional) lists acceptable alternate gold sequences —
the record is scored against the best-matching variant (use for morphology,
plural, or pronoun-drop alternates).

## Notes

- Everything is deterministic for a fixed `seed`.
- VRAM peaks are sampled via `nvidia-smi` every second (best-effort).
- Sample counts: keep `n_phrase` ≤ 50 per (model, lang, concurrency) for a
  ~1-2 min run on a single RTX 3090-class GPU.
