import pytest
from benchmarks.quality import Rules, evaluate, rule_flags, split_tokens, token_f1

GOLD = ["YESTERDAY", "ME", "GO", "STORE"]


def test_split_tokens_separator_and_whitespace_fallback():
    rules = Rules()
    assert split_tokens("A, B, C", rules) == ["A", "B", "C"]
    assert split_tokens("A B C", rules) == ["A", "B", "C"]
    assert split_tokens("", rules) == []


def test_token_f1_perfect_and_partial():
    assert token_f1(GOLD, GOLD)["f1"] == 1.0
    scores = token_f1(["YESTERDAY", "ME", "SHOP"], GOLD)
    assert 0.0 < scores["f1"] < 1.0
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(0.5)
    assert token_f1([], GOLD)["f1"] == 0.0


def test_token_f1_handles_case_and_punctuation():
    scores = token_f1(["yesterday,", "me", "go", "store."], GOLD)
    assert scores["f1"] == 1.0


def test_rule_flags():
    rules = Rules()
    ok = rule_flags(["A", "B"], rules)
    assert ok == {"uppercase": True, "single_word_tokens": True, "banned_chars_absent": True}

    assert rule_flags(["A", "bb c"], rules)["single_word_tokens"] is False
    assert rule_flags(["A", "b"], rules)["uppercase"] is False
    assert rule_flags(["A", "(B)"], rules)["banned_chars_absent"] is False
    assert rule_flags([], rules)["uppercase"] is False


def test_evaluate_combines_overlap_and_rules():
    res = evaluate("YESTERDAY, ME, GO, STORE", GOLD, Rules())
    assert res["overlap"]["f1"] == 1.0
    assert all(res["rules"].values())
    assert res["tokens"] == GOLD
