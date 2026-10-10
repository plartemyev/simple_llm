import gzip

import pytest

from benchmarks.slicer import dictation_streams, fragment, is_noise, load_captions, phrase_samples

SAMPLE = "benchmarks/data/samples/en.txt"


def test_is_noise_filters_stage_directions_and_credits():
    assert is_noise("[Applause]")
    assert is_noise("(Laughter)")
    assert is_noise("♪ music ♪")
    assert is_noise(">> SPEAKER NAME")
    assert is_noise("- dialogue line")
    assert is_noise("TED STAGE 2019")
    assert not is_noise("This is a normal caption line.")


def test_load_captions_reads_gz_and_plain(tmp_path):
    lines = ["Hello world.", "[Applause]", "Second line here.", ""]
    plain = tmp_path / "a.txt"
    plain.write_text("\n".join(lines), encoding="utf-8")
    assert load_captions(plain) == ["Hello world.", "Second line here."]

    gz = tmp_path / "b.txt.gz"
    gz.write_bytes(gzip.compress("\n".join(lines).encode("utf-8")))
    assert load_captions(gz) == ["Hello world.", "Second line here."]


def test_phrase_samples_respects_bounds_and_count():
    lines = load_captions(SAMPLE)
    n = 5
    phrases = phrase_samples(lines, n, min_words=3, max_words=40, seed=7)
    assert len(phrases) <= n
    for p in phrases:
        assert 3 <= len(p.split()) <= 40


def test_phrase_samples_deterministic_for_same_seed():
    lines = load_captions(SAMPLE)
    assert phrase_samples(lines, 4, seed=42) == phrase_samples(lines, 4, seed=42)


def test_fragment_covers_all_words_within_bounds():
    text = "one two three four five six seven eight nine ten eleven twelve"
    frags = fragment(text, min_words=3, max_words=5, seed=1)
    assert " ".join(frags) == text
    for f in frags:
        assert 1 <= len(f.split()) <= 5


def test_fragment_rejects_bad_bounds():
    with pytest.raises(ValueError):
        fragment("x y z", min_words=0)
    with pytest.raises(ValueError):
        fragment("x y z", min_words=5, max_words=3)


def test_dictation_streams_shapes():
    lines = load_captions(SAMPLE)
    streams = dictation_streams(lines, 3, 4, seed=5)
    assert len(streams) <= 3
    for s in streams:
        assert 2 <= len(s) <= 4
        for frag in s:
            assert frag  # non-empty fragments


def test_dictation_streams_finite_when_no_candidate():
    assert dictation_streams(["tiny", "lines"], 5, 10) == []
