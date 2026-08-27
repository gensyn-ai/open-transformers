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

from pretrain.data.tokenizer import train_tokenizer


def _train(tmp_path, texts):
    out = tmp_path / "tokenizer.json"
    train_tokenizer(iter(texts), out, vocab_size=600, min_frequency=1)
    from tokenizers import Tokenizer as HFTokenizer

    return HFTokenizer.from_file(str(out))


def test_pretokenizer_splits_digit_runs_into_chunks_of_at_most_three(tmp_path):
    tok = _train(tmp_path, ["hello world " * 50, "0123456789 " * 50])
    pieces = tok.pre_tokenizer.pre_tokenize_str("4567")
    chunks = [text for text, _ in pieces]
    assert chunks == ["456", "7"]
    assert all(len(c) <= 3 for c in chunks)


def test_pretokenizer_does_not_swallow_whole_number_as_one_chunk(tmp_path):
    """Regression guard for the literal-string-pattern bug: a >3-digit
    number must never come back as a single pre-token chunk."""
    tok = _train(tmp_path, ["hello world " * 50, "0123456789 " * 50])
    pieces = tok.pre_tokenizer.pre_tokenize_str("1234567")
    chunks = [text for text, _ in pieces]
    assert "1234567" not in chunks
    assert all(len(c) <= 3 for c in chunks)


def test_encode_decode_roundtrip_preserves_numbers(tmp_path):
    tok = _train(tmp_path, ["The 4567 apples cost 100000 dollars. " * 50])
    text = "The 4567 apples cost 100000 dollars."
    ids = tok.encode(text).ids
    assert tok.decode(ids) == text
