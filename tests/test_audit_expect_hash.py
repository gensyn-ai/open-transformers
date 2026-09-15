"""``--expect-hash`` must be validated eagerly and strictly.

The old check accepted any hex string of >= 16 chars as a digest. Run stdout
truncates digests to 16 chars, so a pasted prefix passed validation and failed
the comparison only at the end of a multi-hour replay. Now only the canonical
64-char (blake2b-32) form is a digest; other pure-hex strings are rejected as
truncated, and a file must actually contain a digest.

Pure unit tests on ``_resolve_expect_hash`` — no repop, no checkpoint needed.
"""

from __future__ import annotations

import pytest

from pretrain.cli.audit_replay import _DIGEST_HEX_LEN, _resolve_expect_hash

FULL = "AB" * 32  # 64 hex chars, uppercase to exercise normalization


def test_full_length_digest_is_accepted_and_lowercased():
    assert len(FULL) == _DIGEST_HEX_LEN
    assert _resolve_expect_hash(FULL) == FULL.lower()


def test_surrounding_whitespace_is_tolerated():
    assert _resolve_expect_hash(f"  {FULL}\n") == FULL.lower()


@pytest.mark.parametrize("n", [16, 32, 63, 65, 128])
def test_wrong_length_hex_is_rejected_as_truncated(n):
    s = ("a1" * 64)[:n]
    with pytest.raises(ValueError, match="truncated"):
        _resolve_expect_hash(s)


def test_missing_path_is_rejected_eagerly(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_expect_hash(str(tmp_path / "state_hash.txt"))


def test_digest_file_is_read_and_lowercased(tmp_path):
    f = tmp_path / "state_hash.txt"
    f.write_text(FULL + "\n")
    assert _resolve_expect_hash(str(f)) == FULL.lower()


def test_non_digest_file_is_rejected(tmp_path):
    f = tmp_path / "results.json"
    f.write_text('{"step": 10}')
    with pytest.raises(ValueError, match="does not contain"):
        _resolve_expect_hash(str(f))


def test_truncated_digest_file_is_rejected(tmp_path):
    f = tmp_path / "state_hash.txt"
    f.write_text(FULL[:16])
    with pytest.raises(ValueError, match="does not contain"):
        _resolve_expect_hash(str(f))


def test_hexlike_existing_path_is_still_a_path(tmp_path, monkeypatch):
    """A short hex string that IS an existing file resolves as a path (the
    file's content wins), not as a truncated digest."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "deadbeef"
    f.write_text(FULL)
    assert _resolve_expect_hash("deadbeef") == FULL.lower()


def test_digest_len_is_derived_from_state_hash_size():
    """_DIGEST_HEX_LEN must track state_hash._DIGEST_SIZE, not a hardcoded 64,
    so a future digest-size change can't make this validator reject every
    genuine digest as truncated."""
    from pretrain.train.state_hash import _DIGEST_SIZE

    assert _DIGEST_HEX_LEN == 2 * _DIGEST_SIZE


def test_read_digest_file_validates_content(tmp_path):
    """The shared helper the from-init sibling reads now use must reject a
    wrong/truncated file (the same class of bug --expect-hash guards)."""
    from pretrain.cli.audit_replay import _read_digest_file

    good = tmp_path / "state_hash_init.txt"
    good.write_text(FULL + "\n")
    assert _read_digest_file(good, what="state_hash_init.txt") == FULL.lower()

    bad = tmp_path / "bad.txt"
    bad.write_text(FULL[:16])  # truncated
    with pytest.raises(ValueError, match="does not contain"):
        _read_digest_file(bad, what="state_hash_init.txt")

    junk = tmp_path / "results.json"
    junk.write_text('{"state_hash": "..."}')
    with pytest.raises(ValueError, match="does not contain"):
        _read_digest_file(junk, what="state_hash.txt")
