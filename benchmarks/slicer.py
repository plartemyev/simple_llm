"""Turn public caption text (one caption line per line) into benchmark inputs.

Two views of the same corpus:

* **phrase samples** — 1-2 consecutive caption lines joined into a complete
  spoken phrase, for single-shot ("phrase") latency/quality scenarios;
* **dictation streams** — sentences cut into small word fragments, mimicking
  interim speech-recognition output that arrives in pieces and must be
  processed incrementally ("splits" scenario).

All functions are deterministic for a given seed.
"""

from __future__ import annotations

import gzip
import random
import re
from pathlib import Path

# Captions noise: bracketed stage directions, music markers, speaker credit
# lines, dash-led dialogue, ALL-CAPS credits.
_NOISE = re.compile(r"^\s*(\[|\(|\{|♪|>>>?|—|-{1,2}\s|\*[A-Z])")
_MIN_WORDS_PHRASE = 4
_MAX_WORDS_PHRASE = 30


def is_noise(line: str) -> bool:
    return bool(_NOISE.match(line)) or line.isupper()


def load_captions(path: str | Path) -> list[str]:
    """Load caption lines, dropping noise lines and empties."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        lines = (ln.strip() for ln in f)
        return [ln for ln in lines if ln and not is_noise(ln)]


def phrase_samples(
    lines: list[str],
    n: int,
    *,
    min_words: int = _MIN_WORDS_PHRASE,
    max_words: int = _MAX_WORDS_PHRASE,
    group_max: int = 2,
    seed: int = 1234,
) -> list[str]:
    """Sample n phrases of 1..group_max joined caption lines."""
    if n < 0:
        raise ValueError("n must be >= 0")
    rnd = random.Random(seed)
    out: list[str] = []
    i = 0
    scans = 0
    while len(out) < n and i < len(lines) and scans <= len(lines):
        scans += 1
        k = rnd.randint(1, group_max)
        chunk = lines[i : i + k]
        i += k
        text = " ".join(chunk)
        w = len(text.split())
        if min_words <= w <= max_words:
            out.append(text)
    rnd.shuffle(out)
    return out[:n]


def fragment(text: str, *, min_words: int = 3, max_words: int = 8, seed: int = 0) -> list[str]:
    """Cut text into consecutive word fragments (interim-dictation style)."""
    if min_words <= 0 or max_words < min_words:
        raise ValueError("require 0 < min_words <= max_words")
    words = text.split()
    rnd = random.Random(seed)
    out: list[str] = []
    i = 0
    while i < len(words):
        k = rnd.randint(min_words, max_words)
        out.append(" ".join(words[i : i + k]))
        i += k
    return out


def dictation_streams(
    lines: list[str],
    n_streams: int,
    fragments_per_stream: int,
    *,
    seed: int = 1234,
) -> list[list[str]]:
    """Sample n_streams sentences, each cut into fragment lists."""
    if n_streams < 0 or fragments_per_stream < 1:
        raise ValueError("bad stream parameters")
    rnd = random.Random(seed)
    streams: list[list[str]] = []
    i = 0
    scans = 0
    while len(streams) < n_streams and i < len(lines) and scans <= len(lines):
        scans += 1
        sentence = lines[i]
        i += 1
        w = len(sentence.split())
        if not 6 <= w <= 40:
            continue
        frags = fragment(sentence, seed=rnd.randint(0, 2**31 - 1))
        if len(frags) >= 2:
            streams.append(frags[:fragments_per_stream])
    return streams
