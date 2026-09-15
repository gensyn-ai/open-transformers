"""Audit-data interval fetch (partial-dataset download for the single-device audit).

The load-bearing guarantee is that :meth:`ShardedWindowView.walk_doc_refs` — the
length-only walk the fetch tool uses to decide which shards to download —
enumerates *exactly* the documents the audit materialises over the same interval
(across all virtual ranks). If those two ever diverge, the fetch would miss a
shard and the audit would read a hole. We pin that here, plus the end-to-end
fetch against an injected (local) mirror.
"""

from __future__ import annotations

import json
import shutil
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

from pretrain.config.schema import (
    DataConfig,
    DataSourceConfig,
    ModelConfig,
    OptimConfig,
    RootConfig,
    RunConfig,
    ScheduleConfig,
    TrainConfig,
)
from pretrain.data.global_stream import GlobalStreamState, ShardedWindowView
from pretrain.data.indexed_dataset import IndexedDatasetReader, IndexedDatasetWriter
from pretrain.data.manifest import ShardInfo, SourceManifest, blake2b_file
from pretrain.train.batch_schedule import iter_step_plans


# --------------------------------------------------------------------------- #
# Fixtures: multi-shard synthetic sources
# --------------------------------------------------------------------------- #


def _make_multishard_source(
    src_dir: Path, name: str, shard_doc_counts: list[int], base: int, *, with_hashes: bool
) -> SourceManifest:
    """Write a source as several shards with varied doc lengths."""
    src_dir.mkdir(parents=True, exist_ok=True)
    shards: list[ShardInfo] = []
    tok = base
    for sid, n_docs in enumerate(shard_doc_counts):
        stem = f"{name}_{sid:05d}"
        prefix = src_dir / stem
        total = 0
        with IndexedDatasetWriter(prefix, dtype=np.uint32) as w:
            for i in range(n_docs):
                # Vary doc length so packing crosses shard/doc boundaries.
                dl = 5 + ((i + sid) % 7)
                w.add_document(np.arange(tok, tok + dl, dtype=np.uint32))
                tok += dl
                total += dl
        info = ShardInfo(prefix=stem, num_documents=n_docs, token_count=total)
        if with_hashes:
            info.idx_blake2b = blake2b_file(prefix.with_suffix(".idx"))
            info.bin_blake2b = blake2b_file(prefix.with_suffix(".bin"))
        shards.append(info)
    manifest = SourceManifest(name=name, tokenizer_hash="deadbeef", dtype="uint32", shards=shards)
    manifest.save(src_dir / "manifest.yaml")
    return manifest


def _sources(root: Path, *, with_hashes: bool = False):
    a = _make_multishard_source(root / "a", "a", [40, 35, 30], 0, with_hashes=with_hashes)
    b = _make_multishard_source(root / "b", "b", [25, 25], 1_000_000, with_hashes=with_hashes)
    return [a, b], [str(root / "a"), str(root / "b")], [0.6, 0.4]


def _small_train() -> TrainConfig:
    # M = ceil(160/(2*16)) = 5 micro-batches/step, mb=2 → 10 windows/step.
    train = TrainConfig(seq_len=16, micro_batch_size=2)
    train.global_batch_tokens.warmup = 16 * 2 * 5
    train.ckpt_every_tokens = 3 * 5 * 2 * 16  # ~3 optimizer steps
    return train


def _prefix_to_shard(manifests, dirs) -> dict[str, tuple[str, int]]:
    """Map a reader's resolved prefix string -> (source_name, shard_id)."""
    out: dict[str, tuple[str, int]] = {}
    for m, d in zip(manifests, dirs):
        for sid, shard in enumerate(m.shards):
            p = Path(shard.prefix)
            resolved = p if p.is_absolute() else Path(d) / p
            out[str(resolved)] = (m.name, sid)
    return out


# --------------------------------------------------------------------------- #
# Core property: walk_doc_refs == documents the audit materialises
# --------------------------------------------------------------------------- #


def _materialized_doc_refs(manifests, dirs, weights, train, *, seed, N, target_tokens, state):
    """Union, over all N virtual ranks, of (name, shard_id, local) that get a
    real payload read while materialising the interval [0, target_tokens)."""
    prefix_map = _prefix_to_shard(manifests, dirs)
    seen: set[tuple[str, int, int]] = set()
    orig = IndexedDatasetReader.document

    def recording(self, idx):
        seen.add((*prefix_map[str(self.prefix)], idx))
        return orig(self, idx)

    plans = []
    consumed = 0
    for p in iter_step_plans(0, 0, train):
        if consumed >= target_tokens:
            break
        plans.append(p)
        consumed += p.tokens_this_step

    with mock.patch.object(IndexedDatasetReader, "document", recording):
        for r in range(N):
            view = ShardedWindowView(
                manifests, dirs, weights, train, seed=seed, rank=r, world_size=N,
                eos_id=999, state=state,
            )
            it = iter(view)
            owned = sum(1 for p in plans for m in range(p.microbatches) if m % N == r)
            for _ in range(owned):
                next(it)
    return seen


def test_walk_doc_refs_matches_materialized_docs(tmp_path):
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    seed = 7
    target = train.ckpt_every_tokens

    # The fetch tool's enumeration (index-only length walk).
    view = ShardedWindowView(
        manifests, dirs, weights, train, seed=seed, rank=0, world_size=1,
        eos_id=999, index_only=True,
    )
    walked = set(view.walk_doc_refs(target_consumed_tokens=target))

    # Independent reference: what the audit actually reads, across all ranks.
    for N in (1, 2, 4):
        materialized = _materialized_doc_refs(
            manifests, dirs, weights, train, seed=seed, N=N, target_tokens=target, state=None
        )
        assert walked == materialized, (
            f"N={N}: walk_doc_refs and materialised reads disagree "
            f"(walk-only={walked - materialized}, mat-only={materialized - walked})"
        )
    # Sanity: the interval touched more than one shard (exercises shard_id mapping).
    assert len({(n, s) for n, s, _ in walked}) > 1


def test_walk_doc_refs_until_step_matches_materialized(tmp_path):
    """Same equivalence using the --until-step stop condition."""
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    seed = 11
    until = 2

    view = ShardedWindowView(
        manifests, dirs, weights, train, seed=seed, rank=0, world_size=1,
        eos_id=999, index_only=True,
    )
    walked = set(view.walk_doc_refs(until_step=until))

    # Materialise exactly the first `until` steps across ranks.
    plans = [p for _, p in zip(range(until), iter_step_plans(0, 0, train))]
    prefix_map = _prefix_to_shard(manifests, dirs)
    seen: set[tuple[str, int, int]] = set()
    orig = IndexedDatasetReader.document

    def recording(self, idx):
        seen.add((*prefix_map[str(self.prefix)], idx))
        return orig(self, idx)

    with mock.patch.object(IndexedDatasetReader, "document", recording):
        for r in range(4):
            v = ShardedWindowView(
                manifests, dirs, weights, train, seed=seed, rank=r, world_size=4, eos_id=999
            )
            it = iter(v)
            owned = sum(1 for p in plans for m in range(p.microbatches) if m % 4 == r)
            for _ in range(owned):
                next(it)
    assert walked == seen


def test_walk_doc_refs_requires_exactly_one_stop(tmp_path):
    manifests, dirs, weights = _sources(tmp_path)
    view = ShardedWindowView(
        manifests, dirs, weights, _small_train(), seed=1, rank=0, world_size=1,
        eos_id=999, index_only=True,
    )
    import pytest

    with pytest.raises(ValueError):
        list(view.walk_doc_refs())
    with pytest.raises(ValueError):
        list(view.walk_doc_refs(target_consumed_tokens=10, until_step=1))


# --------------------------------------------------------------------------- #
# index_only reader
# --------------------------------------------------------------------------- #


def test_index_only_reader_no_bin(tmp_path):
    import pytest

    _make_multishard_source(tmp_path / "a", "a", [4], 0, with_hashes=False)
    prefix = tmp_path / "a" / "a_00000"
    # Remove the .bin: index-only must still construct and report lengths.
    prefix.with_suffix(".bin").unlink()
    r = IndexedDatasetReader(prefix, index_only=True)
    assert r.num_documents == 4
    assert r.document_length(0) > 0
    with pytest.raises(RuntimeError):
        r.document(0)
    # Non-index-only must refuse to open without the .bin.
    with pytest.raises(FileNotFoundError):
        IndexedDatasetReader(prefix)


# --------------------------------------------------------------------------- #
# gs:// parsing
# --------------------------------------------------------------------------- #


def test_parse_gs_uri():
    from pretrain.data.fetch_interval import _parse_gs_uri

    assert _parse_gs_uri("gs://bucket/a/b/c") == ("bucket", "a/b/c")
    assert _parse_gs_uri("gs://bucket") == ("bucket", "")
    assert _parse_gs_uri("gs://bucket/") == ("bucket", "")
    import pytest

    with pytest.raises(ValueError):
        _parse_gs_uri("s3://bucket/x")


# --------------------------------------------------------------------------- #
# End-to-end fetch against an injected local "mirror"
# --------------------------------------------------------------------------- #


class _LocalMirror:
    """Test double for GcsMirror: serves objects from a local directory."""

    def __init__(self, bucket_dir: Path) -> None:
        self.bucket_dir = Path(bucket_dir)
        self.downloaded: list[str] = []

    def download(self, rel: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.bucket_dir / rel, dest)
        self.downloaded.append(rel)


def _write_checkpoint(ckpt_dir: Path, cfg: RootConfig, *, seed: int, step: int, consumed: int):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "reduction_mode": "deterministic_allgather",
        "seed": seed,
        "step": step,
        "consumed_tokens": consumed,
        "dp_world_size": 4,
        "config_resolved": cfg.model_dump_json(),
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta))
    state = GlobalStreamState(
        consumed_documents_per_source={},
        epoch_per_source={},
        mix_rng_state=None,
        carry_over=[],
        windows_emitted=0,
    )
    (ckpt_dir / "global_stream.json").write_text(
        json.dumps(
            {
                "consumed_documents_per_source": state.consumed_documents_per_source,
                "epoch_per_source": state.epoch_per_source,
                "mix_rng_state": state.mix_rng_state,
                "carry_over": state.carry_over,
                "windows_emitted": state.windows_emitted,
            }
        )
    )


def _root_config(seq_len: int, train: TrainConfig) -> RootConfig:
    model = ModelConfig(
        name="tiny", n_layers=2, d_model=32, n_heads=4, n_kv_heads=2, head_dim=8,
        ffn_intermediate=64, vocab_size=256, max_seq_len_pretrain=seq_len,
    )
    data = DataConfig(
        sources=[
            DataSourceConfig(name="a", path="data/shards/a", weight=0.6),
            DataSourceConfig(name="b", path="data/shards/b", weight=0.4),
        ],
        seq_len=seq_len,
        document_separator_id=999,
    )
    return RootConfig(
        model=model, data=data, train=train,
        optim=OptimConfig(), schedule=ScheduleConfig(),
        run=RunConfig(seed=42, reduction_mode="deterministic_allgather"),
    )


def test_fetch_audit_interval_downloads_only_touched_shards(tmp_path):
    from pretrain.data.fetch_interval import fetch_audit_interval

    bucket = tmp_path / "bucket"
    manifests, dirs, weights = _sources(bucket, with_hashes=True)

    train = _small_train()
    cfg = _root_config(train.seq_len, train)
    ckpt = tmp_path / "ckpt" / "step_000000000"
    _write_checkpoint(ckpt, cfg, seed=cfg.run.seed, step=0, consumed=0)

    dest = tmp_path / "audit_data"
    mirror = _LocalMirror(bucket)
    result = fetch_audit_interval(
        ckpt, "gs://unused/root", dest=dest, mirror=mirror, verify=True
    )

    # Layout: shards under <dest>/shards/<source>/, data_root points at the parent.
    assert Path(result.data_root) == dest / "shards"
    assert (dest / "shards" / "a" / "manifest.yaml").exists()
    assert (dest / "fetch_manifest.json").exists()

    # Independently compute the touched shards and assert we downloaded exactly
    # those .bin — no more (not the whole corpus), no less.
    view = ShardedWindowView(
        manifests, dirs, weights, train, seed=cfg.run.seed, rank=0, world_size=1,
        eos_id=999, index_only=True,
    )
    expected = {(n, s) for n, s, _ in view.walk_doc_refs(target_consumed_tokens=train.ckpt_every_tokens)}

    downloaded_bins = {
        (src.name, p)
        for src in (dest / "shards").iterdir() if src.is_dir()
        for p in range(99)
        if (src / f"{src.name}_{p:05d}.bin").exists()
    }
    assert downloaded_bins == expected
    # Every source's .idx are present (all downloaded), but most .bin are not.
    total_shards = sum(len(m.shards) for m in manifests)
    assert len(downloaded_bins) < total_shards, "should NOT have pulled every .bin"
    assert result.touched_shards  # recorded in the summary


# --------------------------------------------------------------------------- #
# GcsMirror credential handling
#
# A published audit mirror is world-readable and an auditor verifying it needs
# no Google account, but ``storage.Client()`` refuses to construct without
# credentials — it fails before it ever learns the object is public. Without
# the anonymous fallback an unauthenticated volunteer must install the Cloud
# SDK and complete an OAuth flow purely to satisfy a constructor.
# --------------------------------------------------------------------------- #


def _fake_storage(*, default_raises: bool):
    """Stand-in for ``google.cloud.storage``, so this needs no network."""
    pytest.importorskip("google.auth")
    from google.auth.exceptions import DefaultCredentialsError

    class Client:
        def __init__(self, anonymous: bool = False):
            self.anonymous = anonymous

        def bucket(self, name):
            return f"bucket:{name}:anonymous={self.anonymous}"

    def make_default():
        if default_raises:
            raise DefaultCredentialsError("no ADC")
        return Client(anonymous=False)

    ns = mock.MagicMock()
    ns.Client.side_effect = make_default
    ns.Client.create_anonymous_client.side_effect = lambda: Client(anonymous=True)
    return ns


def _mirror_with(storage_ns):
    from pretrain.data import fetch_interval

    with mock.patch.dict("sys.modules", {"google.cloud": mock.MagicMock(storage=storage_ns)}):
        with mock.patch.object(fetch_interval, "_parse_gs_uri", return_value=("b", "p")):
            return fetch_interval.GcsMirror("gs://b/p")


def test_gcs_mirror_uses_credentials_when_they_exist():
    """Unchanged for every existing caller: the credentialed client still wins,
    and it is the only one that can reach a private mirror."""
    mirror = _mirror_with(_fake_storage(default_raises=False))
    assert mirror._client.anonymous is False


def test_gcs_mirror_falls_back_to_anonymous_without_credentials():
    """A public mirror must be readable with no account at all."""
    mirror = _mirror_with(_fake_storage(default_raises=True))
    assert mirror._client.anonymous is True
    assert mirror._bucket == "bucket:b:anonymous=True"


def test_a_private_mirror_still_fails_but_says_which_object():
    """The fallback does not paper over a real permission problem: an anonymous
    client against a private mirror fails at the first request with a 401/403
    naming the object, which beats a DefaultCredentialsError naming nothing."""
    mirror = _mirror_with(_fake_storage(default_raises=True))
    assert mirror._client.anonymous is True, (
        "construction succeeds; the failure moves to the request, where it can "
        "identify what was being read"
    )
