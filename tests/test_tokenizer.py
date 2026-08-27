"""Digit pre-tokenization regression tests.

``train_tokenizer`` isolates digit runs into <=3-digit chunks before BPE
sees them (the tiktoken-lineage scheme Llama 3 / GPT-4 popularized, still
what current frontier tokenizers ship). This used to be a no-op: passing a
plain ``str`` pattern to ``pre_tokenizers.Split`` makes it match the
pattern as a literal substring, not a regex, so ``pattern=r"\\d"`` never
matched anything and digits fell through to BPE-frequency-dependent
merging (e.g. "24" and "4567" each became their own single token). These
tests pin the fixed, regex-wrapped behavior so it can't silently regress.
"""

from __future__ import annotations

import re

import pytest

from pretrain.data.tokenizer import assert_fresh_digit_pretokenizer, train_tokenizer


def _train(tmp_path, texts, name="tokenizer.json"):
    out = tmp_path / name
    train_tokenizer(iter(texts), out, vocab_size=600, min_frequency=1)
    from tokenizers import Tokenizer as HFTokenizer

    return HFTokenizer.from_file(str(out)), out


def test_pretokenizer_splits_digit_runs_into_chunks_of_at_most_three(tmp_path):
    tok, _ = _train(tmp_path, ["hello world " * 50, "0123456789 " * 50])
    pieces = tok.pre_tokenizer.pre_tokenize_str("4567")
    chunks = [text for text, _ in pieces]
    assert chunks == ["456", "7"]
    assert all(len(c) <= 3 for c in chunks)


def test_pretokenizer_does_not_swallow_whole_number_as_one_chunk(tmp_path):
    """Regression guard for the literal-string-pattern bug: a >3-digit
    number must never come back as a single pre-token chunk."""
    tok, _ = _train(tmp_path, ["hello world " * 50, "0123456789 " * 50])
    pieces = tok.pre_tokenizer.pre_tokenize_str("1234567")
    chunks = [text for text, _ in pieces]
    assert "1234567" not in chunks
    assert all(len(c) <= 3 for c in chunks)


def test_trained_vocab_has_no_stale_multi_digit_tokens(tmp_path):
    """The artifact-level observable that actually damaged us: no vocab
    entry should ever be a run of 4+ digits (with or without the ByteLevel
    leading-space marker). This survives future refactors of the
    Split+ByteLevel pipeline in a way pre_tokenize_str() unit checks don't
    — if someone collapses them into one combined regex step, this still
    catches a reintroduced digit-merging bug from the trained artifact
    itself, not just from calling the pre-tokenizer in isolation."""
    tok, _ = _train(tmp_path, ["The 4567 apples cost 100000 dollars. " * 200])
    stale = [t for t in tok.get_vocab() if re.fullmatch(r"Ġ?\d{4,}", t)]
    assert stale == []


def test_space_before_number_is_its_own_token(tmp_path):
    """Digit chunks are never Ġ-prefixed — a preceding space becomes a
    standalone "Ġ" token instead (matches cl100k/Llama-3's combined-regex
    convention). Locking this in so a future "fix" for the lone-Ġ
    appearance has to consciously touch this test rather than silently
    changing the convention."""
    tok, _ = _train(tmp_path, ["saw 365 days " * 50, "hello world " * 50])
    pieces = tok.pre_tokenizer.pre_tokenize_str(" 365")
    chunks = [text for text, _ in pieces]
    assert chunks == ["Ġ", "365"]


def test_encode_decode_roundtrip_preserves_numbers(tmp_path):
    """NOTE: with a ByteLevel decoder this roundtrip holds for *any* input,
    trained-on or not — it guards general decoder wiring, not anything
    digit-specific. It would pass even against the pre-fix broken
    tokenizer; don't count it as regression coverage for that bug."""
    tok, _ = _train(tmp_path, ["The 4567 apples cost 100000 dollars. " * 50])
    text = "The 4567 apples cost 100000 dollars."
    ids = tok.encode(text).ids
    assert tok.decode(ids) == text


def test_assert_fresh_digit_pretokenizer_accepts_current_artifact(tmp_path):
    _, path = _train(tmp_path, ["hello world " * 50, "0123456789 " * 50])
    assert_fresh_digit_pretokenizer(path)  # must not raise


def test_assert_fresh_digit_pretokenizer_rejects_pre_fix_artifact(tmp_path):
    """Simulates a tokenizer.json trained by the pre-fix code: the digit
    Split step's pattern serialized as a literal string, not a regex."""
    _, path = _train(tmp_path, ["hello world " * 50, "0123456789 " * 50])
    import json

    data = json.loads(path.read_text())
    for pt in data["pre_tokenizer"]["pretokenizers"]:
        if pt.get("type") == "Split":
            pt["pattern"] = {"String": "\\d"}
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="stale-tokenizer trap"):
        assert_fresh_digit_pretokenizer(path)
