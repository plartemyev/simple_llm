"""Aggregate runner/quality JSON outputs into a markdown report.

    python -m benchmarks.report results/ > report.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_all(results_dir: Path) -> list[dict]:
    records = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            records.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            print(f"skipping unparsable {p}", file=sys.stderr)
    return records


def fmt(v: float, unit: str = "") -> str:
    return f"{v:.2f}{unit}" if isinstance(v, (int, float)) else str(v)


def perf_table(records: list[dict]) -> str:
    lines = [
        "| model | scenario | lang | conc | wall_s | lat p50 (s) | lat p95 (s) | ttft p50 (s) | calls/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        if "levels" not in rec:
            continue
        for lvl in rec["levels"]:
            lat = lvl.get("latency_s", {})
            ttft = lvl.get("ttft_s", {})
            lines.append(
                f"| {rec['model']} | {rec['scenario']} | {rec.get('lang', '')} "
                f"| {lvl['concurrency']} | {fmt(lvl.get('wall_s', 0))} "
                f"| {fmt(lat.get('p50', 0), '')} | {fmt(lat.get('p95', 0), '')} "
                f"| {fmt(ttft.get('p50', 0), '') if ttft else '—'} "
                f"| {fmt(lvl.get('calls_per_s', 0))} |"
            )
    return "\n".join(lines)


def vram_table(records: list[dict]) -> str:
    lines = ["| model | scenario | conc | GPU peak MiB |", "|---|---|---|---|"]
    for rec in records:
        for lvl in rec.get("levels", []):
            if "vram_peak_mib" not in lvl:
                continue
            peaks = ", ".join(f"gpu{i}: {v}" for i, v in lvl["vram_peak_mib"].items())
            lines.append(f"| {rec['model']} | {rec['scenario']} | {lvl['concurrency']} | {peaks} |")
    return "\n".join(lines) if len(lines) > 2 else ""


def quality_table(records: list[dict]) -> str:
    lines = [
        "| model | n | F1 mean | F1 min | uppercase | single-word | no-banned | exact |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        if "f1_mean" not in rec:
            continue
        rc = rec.get("rules_compliance", {})
        lines.append(
            f"| {rec.get('model', '')} | {rec.get('n', 0)} | {rec['f1_mean']:.3f} "
            f"| {rec['f1_min']:.3f} "
            f"| {rc.get('uppercase', 0):.2%} | {rc.get('single_word_tokens', 0):.2%} "
            f"| {rc.get('banned_chars_absent', 0):.2%} | {rec.get('exact', 0)} |"
        )
    return "\n".join(lines) if len(lines) > 2 else ""


def main() -> None:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    records = load_all(results_dir)
    print("# Benchmark results\n")
    print("## Performance\n")
    print(perf_table(records))
    vt = vram_table(records)
    if vt:
        print("\n## VRAM peaks\n")
        print(vt)
    qt = quality_table(records)
    if qt:
        print("\n## Quality\n")
        print(qt)


if __name__ == "__main__":
    main()
