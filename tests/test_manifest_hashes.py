"""Audit-hash fields on SourceManifest + ShardInfo + the merge step.

The hash algorithm is blake2b-32, matching pretrain.train.state_hash so
every "bytes we committed to" artifact in the project speaks one hash.
"""

from __future__ import annotations

import argparse
import hashlib

import pytest

from pretrain.cli.prepare_data import _cmd_merge_manifests
from pretrain.data.manifest import (
    ShardInfo,
    SourceManifest,
    blake2b_file,
)
from pretrain.data.prepare import write_shards


def test_blake2b_file_matches_known_value(tmp_path):
    payload = b"the quick brown fox"
    p = tmp_path / "x.bin"
    p.write_bytes(payload)
    expected = hashlib.blake2b(payload, digest_size=32).hexdigest()
    assert blake2b_file(p) == expected
    # 32-byte digest → 64 hex chars (same length as a sha256 hex, but a
    # different algorithm — important to keep names accurate).
    assert len(blake2b_file(p)) == 64


def test_manifest_roundtrip_with_hashes(tmp_path):
    m = SourceManifest(
        name="src",
        tokenizer_hash="tok123",
        dtype="uint32",
        shards=[
            ShardInfo(
                prefix="src_00000",
                num_documents=3,
                token_count=42,
                bin_blake2b="aa" * 32,
                idx_blake2b="bb" * 32,
            ),
        ],
        raw_jsonl_blake2b={"src.jsonl": "cc" * 32},
    )
    p = tmp_path / "manifest.yaml"
    m.save(p)
    loaded = SourceManifest.load(p)
    assert loaded == m


def test_legacy_manifest_loads_without_hash_fields(tmp_path):
    """Manifests written before the audit fields existed must still load."""
    p = tmp_path / "manifest.yaml"
    p.write_text(
        "name: src\n"
        "tokenizer_hash: tok123\n"
        "dtype: uint32\n"
        "shards:\n"
        "  - prefix: src_00000\n"
        "    num_documents: 3\n"
        "    token_count: 42\n",
        encoding="utf-8",
    )
    loaded = SourceManifest.load(p)
    assert loaded.shards[0].bin_blake2b == ""
    assert loaded.shards[0].idx_blake2b == ""
    assert loaded.raw_jsonl_blake2b == {}


class _FixedTok:
    """Minimal tokenizer stub for write_shards — bypasses HF tokenizers."""

    hash = "fixedtok"

    def encode_iter(self, texts):
        for t in texts:
            yield list(t.encode("utf-8"))  # bytes-as-ids; fine for round-trip


def test_write_shards_records_bin_idx_hashes(tmp_path):
    out = tmp_path / "shards"
    texts = ["hello world", "another doc", "third document text"]

    manifest = write_shards(
        text_iter=iter(texts),
        tokenizer=_FixedTok(),
        output_dir=out,
        source_name="src",
        target_shard_bytes=1 << 30,
        raw_jsonl_blake2b={"src.jsonl": "deadbeef"},
    )

    assert manifest.raw_jsonl_blake2b == {"src.jsonl": "deadbeef"}
    assert len(manifest.shards) == 1
    sh = manifest.shards[0]
    assert sh.bin_blake2b and sh.idx_blake2b
    # Recorded hashes must match the actual on-disk bytes.
    bin_path = (out / sh.prefix).with_suffix(".bin")
    idx_path = (out / sh.prefix).with_suffix(".idx")
    assert sh.bin_blake2b == blake2b_file(bin_path)
    assert sh.idx_blake2b == blake2b_file(idx_path)


def test_write_shards_is_deterministic(tmp_path):
    """Same input bytes + same tokenizer ⟹ byte-identical shards.

    Hashes the .bin / .idx / raw JSONL produced by two independent
    write_shards runs and asserts equality. This is the contract the
    manifest's audit fields exist to enforce — if it ever fails, our
    "re-running prep reproduces the corpus" claim is broken.
    """
    texts = ["hello world", "another doc", "third document text"] * 50
    raw_blob = b"".join(t.encode("utf-8") + b"\n" for t in texts)
    raw_jsonl = tmp_path / "src.jsonl"
    raw_jsonl.write_bytes(raw_blob)
    raw_hash = blake2b_file(raw_jsonl)

    def shard_once(out_dir):
        return write_shards(
            text_iter=iter(texts),
            tokenizer=_FixedTok(),
            output_dir=out_dir,
            source_name="src",
            target_shard_bytes=1 << 30,
            raw_jsonl_blake2b={raw_jsonl.name: raw_hash},
        )

    m1 = shard_once(tmp_path / "shards_a")
    m2 = shard_once(tmp_path / "shards_b")

    assert m1.raw_jsonl_blake2b == m2.raw_jsonl_blake2b == {raw_jsonl.name: raw_hash}
    assert len(m1.shards) == len(m2.shards) > 0
    for s1, s2 in zip(m1.shards, m2.shards):
        assert s1.bin_blake2b == s2.bin_blake2b
        assert s1.idx_blake2b == s2.idx_blake2b
        assert s1.num_documents == s2.num_documents
        assert s1.token_count == s2.token_count


def _merge(tmp_path, *manifests: SourceManifest) -> SourceManifest:
    out_dir = tmp_path / "shards"
    out_dir.mkdir()
    for i, m in enumerate(manifests):
        m.save(out_dir / f"manifest.w{i:02d}.yaml")
    args = argparse.Namespace(
        output_dir=str(out_dir),
        expected_workers=len(manifests),
        remove_worker_manifests=False,
    )
    _cmd_merge_manifests(args)
    return SourceManifest.load(out_dir / "manifest.yaml")


def _mk(name: str, raw: dict[str, str]) -> SourceManifest:
    return SourceManifest(
        name=name,
        tokenizer_hash="t",
        dtype="uint32",
        shards=[ShardInfo(prefix="x_00000", num_documents=1, token_count=1)],
        raw_jsonl_blake2b=raw,
    )


def test_merge_unions_raw_jsonl_hashes(tmp_path):
    """Distinct per-worker JSONLs each contribute one entry to the merged dict."""
    merged = _merge(
        tmp_path,
        _mk("src", {"src_w00.jsonl": "aa" * 32}),
        _mk("src", {"src_w01.jsonl": "bb" * 32}),
    )
    assert merged.raw_jsonl_blake2b == {
        "src_w00.jsonl": "aa" * 32,
        "src_w01.jsonl": "bb" * 32,
    }


def test_merge_dedupes_identical_entries(tmp_path):
    """Multiple workers reading slices of one JSONL each record the same
    {filename: hash}. Merge collapses them silently."""
    h = "aa" * 32
    merged = _merge(
        tmp_path,
        _mk("src", {"shared.jsonl": h}),
        _mk("src", {"shared.jsonl": h}),
    )
    assert merged.raw_jsonl_blake2b == {"shared.jsonl": h}


def test_merge_fails_loudly_on_hash_conflict(tmp_path):
    with pytest.raises(SystemExit, match="raw_jsonl_blake2b conflict"):
        _merge(
            tmp_path,
            _mk("src", {"shared.jsonl": "aa" * 32}),
            _mk("src", {"shared.jsonl": "bb" * 32}),
        )
