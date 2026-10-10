"""Performance runner: latency/throughput scenarios against a running server.

Scenarios:
  phrase — n single-shot calls sampled from caption phrases (optionally at a
           given concurrency level);
  splits — dictation-style fragment streams; each stream is processed by one
           worker that issues fragment calls sequentially, several streams
           run in parallel (= concurrency). Streaming mode records
           time-to-first-token per call.

Usage:
    python -m benchmarks.runner --base-url http://127.0.0.1:8011/v1 \
        --model <model-id> --scenario phrase --lang en --n 30 --concurrency 1 4 \
        --out results/phrase_en.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from simple_llm.client import CallParams, LLMClient
from benchmarks.slicer import dictation_streams, load_captions, phrase_samples

# defaults; overridable via --config (TOML with the same keys)
DEFAULTS = {
    "temperature": 0.1,
    "phrase_max_tokens": 128,
    "splits_max_tokens": 64,
    "timeout_s": 60.0,
    "warmup": 3,
    "n_phrase": 30,
    "n_streams": 8,
    "fragments_per_stream": 10,
    "seed": 1234,
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULTS)
    if not path:
        return cfg
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    for section in ("phrase", "splits", "bench"):
        cfg.update({k: v for k, v in data.get(section, {}).items() if k in cfg})
    return cfg


def cc_path(cc_dir: Path, lang: str, use_corpus: bool) -> Path:
    corpus = cc_dir / "cc" / f"{lang}.txt.gz"
    if use_corpus and corpus.exists():
        return corpus
    return cc_dir / "samples" / f"{lang}.txt"


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(q * len(sorted_values)))
    return sorted_values[idx]


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(s, 0.50),
        "p95": percentile(s, 0.95),
        "min": s[0],
        "max": s[-1],
    }


def vram_peaks(interval_s: float, stop: threading.Event) -> tuple[threading.Thread, dict[int, int]]:
    """Start a daemon thread sampling per-GPU used MiB; returns (thread, peak dict)."""
    peak: dict[int, int] = {}

    def sample_loop() -> None:
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=True,
                ).stdout
                for line in out.strip().splitlines():
                    idx, mib = (int(x.strip()) for x in line.split(","))
                    peak[idx] = max(peak.get(idx, 0), mib)
            except Exception:
                pass  # nvidia-smi unavailable — best-effort sampling
            stop.wait(interval_s)

    th = threading.Thread(target=sample_loop, daemon=True)
    th.start()
    return th, peak


def run_phrase(
    client: LLMClient, samples: list[str], concurrency: int, params: CallParams
) -> list[float]:
    latencies: list[float] = []
    lock = threading.Lock()

    def work(i: int) -> None:
        t0 = time.perf_counter()
        client.complete(samples[i], params=params)
        with lock:
            latencies.append(time.perf_counter() - t0)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(work, range(len(samples))))
    return latencies


def run_splits(
    client: LLMClient, streams: list[list[str]], concurrency: int, params: CallParams
) -> tuple[list[float], list[float], list[float]]:
    """Returns (per-call latencies, per-call TTFTs, per-stream wall times)."""
    latencies: list[float] = []
    ttfts: list[float] = []
    stream_walls: list[float] = []
    lock = threading.Lock()

    def work_stream(frags: list[str]) -> None:
        t_stream = time.perf_counter()
        local_lat: list[float] = []
        local_ttft: list[float] = []
        for frag in frags:
            t0 = time.perf_counter()
            first = None
            for _ in client.complete_stream(frag, params=params):
                if first is None:
                    first = time.perf_counter() - t0
            local_lat.append(time.perf_counter() - t0)
            if first is not None:
                local_ttft.append(first)
        with lock:
            latencies.extend(local_lat)
            ttfts.extend(local_ttft)
            stream_walls.append(time.perf_counter() - t_stream)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(work_stream, streams))
    return latencies, ttfts, stream_walls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenario", choices=["phrase", "splits"], required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--cc-dir", default="benchmarks/data")
    ap.add_argument("--n", type=int, default=None, help="override sample/stream count")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cc_dir = Path(args.cc_dir)
    lines = load_captions(cc_path(cc_dir, args.lang, use_corpus=True))

    if args.scenario == "phrase":
        n = args.n or cfg["n_phrase"]
        samples: list[str] | list[list[str]] = phrase_samples(lines, n, seed=cfg["seed"])
        scenario_input: dict = {"n": len(samples)}
    else:
        samples = dictation_streams(
            lines, args.n or cfg["n_streams"], cfg["fragments_per_stream"], seed=cfg["seed"]
        )
        scenario_input = {
            "n_streams": len(samples),
            "fragments_per_stream": cfg["fragments_per_stream"],
        }

    client = LLMClient(args.base_url, args.model, timeout_s=cfg["timeout_s"])
    params = CallParams(
        temperature=cfg["temperature"],
        max_tokens=cfg["phrase_max_tokens" if args.scenario == "phrase" else "splits_max_tokens"],
        seed=cfg["seed"],
    )

    def warm_text(item) -> str:
        return item if isinstance(item, str) else " ".join(item[:2])

    for i in range(cfg["warmup"]):  # model load / allocator warm-up
        client.complete(warm_text(samples[i % len(samples)]), params=CallParams(max_tokens=16))

    results = {
        "model": args.model,
        "scenario": args.scenario,
        "lang": args.lang,
        "input": scenario_input,
        "params": {"temperature": params.temperature, "max_tokens": params.max_tokens},
        "levels": [],
    }

    for conc in args.concurrency:
        stop = threading.Event()
        vram_thread, peak = vram_peaks(1.0, stop)
        t0 = time.perf_counter()
        if args.scenario == "phrase":
            lat = run_phrase(client, samples, conc, params)  # type: ignore[arg-type]
            ttfts: list[float] = []
            stream_walls: list[float] = []
        else:
            lat, ttfts, stream_walls = run_splits(client, samples, conc, params)  # type: ignore[arg-type]
        wall = time.perf_counter() - t0
        stop.set()
        vram_thread.join(timeout=5)

        level = {
            "concurrency": conc,
            "wall_s": wall,
            "latency_s": stats(lat),
            "calls_per_s": len(lat) / wall if wall > 0 else 0.0,
        }
        if args.scenario == "splits":
            level["ttft_s"] = stats(ttfts)
            level["stream_wall_s"] = stats(stream_walls)
        if peak:
            level["vram_peak_mib"] = dict(sorted(peak.items()))
        results["levels"].append(level)
        msg = f"{args.scenario}/{args.lang} conc={conc}: wall={wall:.2f}s p50={level['latency_s'].get('p50', 0):.3f}s p95={level['latency_s'].get('p95', 0):.3f}s"
        if ttfts:
            msg += f" ttft_p50={stats(ttfts)['p50']:.3f}s"
        print(msg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
