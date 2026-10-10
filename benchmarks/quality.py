"""Quality scoring: replay gold message lists against a server, score outputs.

Gold JSONL schema (one record per line):

    {
      "id": "en-001",
      "messages": [{"role": "system", "content": "..."},
                   {"role": "user", "content": "..."}],
      "expected": ["TOKEN1", "TOKEN2"],       # gold token sequence
      "rules": {"uppercase": true, "single_word_tokens": true,
                "banned_chars": "()", "separator": ","}
    }

Scoring is two-fold:
* **token overlap** — multiset precision/recall/F1 between produced and gold
  tokens after normalization (case folding, punctuation stripping);
* **format compliance** — the structural output rules (uppercase-only,
  single-word tokens, banned characters absent).

Usage:
    python -m benchmarks.quality --gold gold.jsonl --base-url http://127.0.0.1:8011/v1 \
        --model <model-id> --out results/quality_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from simple_llm.client import LLMClient

_WORD = re.compile(r"[\w\d]+", re.UNICODE)


@dataclass(frozen=True)
class Rules:
    uppercase: bool = True
    single_word_tokens: bool = True
    banned_chars: str = "()"
    separator: str = ","

    @classmethod
    def from_dict(cls, d: dict | None) -> "Rules":
        d = d or {}
        return cls(
            uppercase=bool(d.get("uppercase", True)),
            single_word_tokens=bool(d.get("single_word_tokens", True)),
            banned_chars=str(d.get("banned_chars", "()")),
            separator=str(d.get("separator", ",")),
        )


def split_tokens(text: str, rules: Rules) -> list[str]:
    """Split raw output into tokens using the separator, whitespace fallback."""
    t = text.strip()
    if rules.separator and rules.separator in t:
        parts = [p.strip() for p in t.split(rules.separator)]
    else:
        parts = t.split()
    return [p for p in parts if p]


def normalize_token(tok: str) -> str:
    """Fold a token for comparison: keep word characters only, uppercase."""
    return "".join(_WORD.findall(tok)).upper()


def rule_flags(tokens: list[str], rules: Rules) -> dict[str, bool]:
    uppercase_ok = all(t == t.upper() for t in tokens) if tokens else False
    single_ok = all(len(t.split()) == 1 for t in tokens) if tokens else False
    banned_ok = not any(c in t for t in tokens for c in rules.banned_chars)
    return {
        "uppercase": uppercase_ok,
        "single_word_tokens": single_ok,
        "banned_chars_absent": banned_ok,
    }


def token_f1(pred: list[str], gold: list[str]) -> dict[str, float]:
    """Multiset precision/recall/F1 over normalized tokens."""
    p = Counter(normalize_token(t) for t in pred if normalize_token(t))
    g = Counter(normalize_token(t) for t in gold if normalize_token(t))
    tp = sum((p & g).values())
    precision = tp / sum(p.values()) if p else 0.0
    recall = tp / sum(g.values()) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(pred_text: str, gold_tokens: list[str], rules: Rules) -> dict:
    tokens = split_tokens(pred_text, rules)
    return {
        "tokens": tokens,
        "overlap": token_f1(tokens, gold_tokens),
        "rules": rule_flags(tokens, rules),
        "raw": pred_text,
    }


def load_gold(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_quality(
    gold: list[dict],
    base_url: str,
    model: str,
    concurrency: int = 4,
) -> dict:
    """Replay every gold record and score the outputs."""
    clients = [
        LLMClient(base_url, model)
        for _ in range(max(1, concurrency))
    ]
    results: list[dict] = [None] * len(gold)  # type: ignore[list-item]

    def work(i: int) -> None:
        rec = gold[i]
        client = clients[i % len(clients)]
        try:
            pred = client.complete_messages(
                rec["messages"],
                temperature=0.1,
                max_tokens=rec.get("max_tokens", 128),
            )
        except Exception as e:  # keep scoring the rest
            pred = f"__ERROR__ {e}"
        rules = Rules.from_dict(rec.get("rules"))
        ev = evaluate(pred, rec["expected"], rules)
        results[i] = {"id": rec["id"], **ev}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        # executor.map keeps submission order and bounds outstanding work
        list(ex.map(work, range(len(gold))))

    f1s = [r["overlap"]["f1"] for r in results if r]
    rule_keys = ["uppercase", "single_word_tokens", "banned_chars_absent"]
    compliance = {
        k: (
            sum(1 for r in results if r and r["rules"][k]) / len(results)
            if results
            else 0.0
        )
        for k in rule_keys
    }
    exact = sum(
        1
        for rec, r in zip(gold, results)
        if r and r["tokens"] and r["tokens"] == rec["expected"]
    )

    return {
        "n": len(results),
        "f1_mean": statistics.fmean(f1s) if f1s else 0.0,
        "f1_min": min(f1s) if f1s else 0.0,
        "rules_compliance": compliance,
        "exact": exact,
        "records": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True, help="gold JSONL file")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    summary = run_quality(gold, args.base_url, args.model, args.concurrency)
    summary["model"] = args.model
    summary["base_url"] = args.base_url

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"{args.model}: n={summary['n']} f1_mean={summary['f1_mean']:.3f} "
        f"compliance={ {k: round(v, 3) for k, v in summary['rules_compliance'].items()} }"
    )


if __name__ == "__main__":
    main()
