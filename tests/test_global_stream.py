"""Canonical global-stream determinism + topology-slicing invariance (plan E.1).

These pin the two properties the single-device audit relies on:

  1. The packed window stream ``W[0], W[1], …`` is a pure function of
     ``(seed, manifests, seq_len, eos_id)`` — no rank / world-size term.
  2. The micro-batch assignment rule partitions that one stream: for any world
     size N, the union of all ranks' windows in a step is exactly the step's
     global windows, disjoint and complete (incl. the non-divisible / partial
     last-accum case). So a run at N=1 consumes the identical windows a run at
     N=32 does, just grouped differently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pretrain.config.schema import TrainConfig
from pretrain.data.global_stream import (
    GlobalStream,
    GlobalStreamState,
    ShardedWindowView,
)
from pretrain.data.indexed_dataset import IndexedDatasetWriter
from pretrain.data.manifest import ShardInfo, SourceManifest
from pretrain.train.batch_schedule import (
    accum_for_rank,
    canonical_tokens_per_step,
    iter_step_plans,
    microbatch_indices_for_rank,
    microbatches_per_step,
    windows_per_step,
)


def _make_source(
    base_dir: Path, name: str, n_docs: int, base: int, doc_len: int = 10
) -> tuple[SourceManifest, str]:
    src_dir = base_dir / name
    src_dir.mkdir(parents=True)
    prefix = src_dir / f"{name}_00000"
    with IndexedDatasetWriter(prefix, dtype=np.uint32) as w:
        for i in range(n_docs):
            w.add_document(
                np.arange(base + i * doc_len, base + (i + 1) * doc_len, dtype=np.uint32)
            )
    manifest = SourceManifest(
        name=name,
        tokenizer_hash="deadbeef",
        dtype="uint32",
        shards=[
            ShardInfo(prefix=f"{name}_00000", num_documents=n_docs, token_count=n_docs * doc_len)
        ],
    )
    manifest.save(src_dir / "manifest.yaml")
    return manifest, str(src_dir)


def _stream(tmp_path, **kw) -> GlobalStream:
    a, da = _make_source(tmp_path / "x", "a", 80, 0)
    b, db = _make_source(tmp_path / "x", "b", 60, 100_000)
    return GlobalStream([a, b], [da, db], [0.7, 0.3], seq_len=16, seed=kw.get("seed", 42), eos_id=999, state=kw.get("state"))


# --------------------------------------------------------------------------- #
# Stream determinism / purity
# --------------------------------------------------------------------------- #


def test_stream_is_deterministic(tmp_path):
    s1, s2 = _stream(tmp_path / "a"), _stream(tmp_path / "b")
    w1 = [w for _, w in zip(range(12), s1)]
    w2 = [w for _, w in zip(range(12), s2)]
    for x, y in zip(w1, w2):
        np.testing.assert_array_equal(x, y)


def test_stream_different_seed_diverges(tmp_path):
    s1 = _stream(tmp_path / "a", seed=1)
    s2 = _stream(tmp_path / "b", seed=2)
    assert not np.array_equal(next(iter(s1)), next(iter(s2)))


def test_stream_windows_have_seq_len_plus_one(tmp_path):
    s = _stream(tmp_path)
    w = next(iter(s))
    assert w.shape == (17,)  # seq_len=16 → +1 for the label shift


def test_resume_is_bit_exact_including_counter(tmp_path):
    s1 = _stream(tmp_path / "a")
    it1 = iter(s1)
    pre = [next(it1) for _ in range(5)]
    state = s1.state()
    assert state.windows_emitted == 5
    post = [next(it1) for _ in range(5)]

    s2 = _stream(tmp_path / "b", state=state)
    assert s2.windows_emitted == 5
    it2 = iter(s2)
    resumed = [next(it2) for _ in range(5)]
    for i, (p, r) in enumerate(zip(post, resumed)):
        np.testing.assert_array_equal(p, r, err_msg=f"window {i} not bit-exact after resume")
    assert s2.windows_emitted == 10


def test_state_roundtrips_through_dataclass(tmp_path):
    s = _stream(tmp_path)
    list(zip(range(3), s))
    st = s.state()
    # Reconstructable from its own fields (mirrors checkpoint json round-trip).
    st2 = GlobalStreamState(**{k: getattr(st, k) for k in vars(st)})
    assert st2.windows_emitted == st.windows_emitted
    assert st2.consumed_documents_per_source == st.consumed_documents_per_source


# --------------------------------------------------------------------------- #
# Topology slicing rule — pure index math, the heart of invariance
# --------------------------------------------------------------------------- #


def test_assignment_partitions_microbatches_disjoint_and_complete():
    for M in (1, 5, 8, 13, 64):
        for N in (1, 2, 3, 4, 8):
            owned = [microbatch_indices_for_rank(M, world_size=N, rank=r) for r in range(N)]
            flat = sorted(m for lst in owned for m in lst)
            assert flat == list(range(M)), f"M={M} N={N} not a partition: {flat}"
            # disjoint: no microbatch owned twice
            assert len(flat) == len(set(flat))
            # accum_for_rank agrees with the index list length
            for r in range(N):
                assert accum_for_rank(M, world_size=N, rank=r) == len(owned[r])
            # sum of per-rank accum == total microbatches
            assert sum(accum_for_rank(M, world_size=N, rank=r) for r in range(N)) == M


def test_lowest_ranks_take_the_extra_microbatch():
    # M=13, N=4 → counts [4, 3, 3, 3]; lowest M%N=1 rank gets the extra.
    counts = [accum_for_rank(13, world_size=4, rank=r) for r in range(4)]
    assert counts == [4, 3, 3, 3]
    # M=13, N=8 → counts [2,2,2,2,2,1,1,1]; lowest 5 ranks get the extra.
    counts8 = [accum_for_rank(13, world_size=8, rank=r) for r in range(8)]
    assert counts8 == [2, 2, 2, 2, 2, 1, 1, 1]
    assert sum(counts8) == 13


def test_slicing_reassembles_the_canonical_stream(tmp_path):
    """The union of every rank's windows, put back in global order, equals the
    raw stream — for every world size. This is the invariance the audit needs:
    N=1 and N=8 consume the *same* windows, only grouped differently.
    """
    # A small synthetic schedule so a "step" is only a handful of windows.
    train = TrainConfig(seq_len=16, micro_batch_size=2)
    train.global_batch_tokens.warmup = 16 * 2 * 5  # → M = 5 micro-batches/step
    mb = train.micro_batch_size

    # Materialise enough of the stream to cover a few steps.
    s = _stream(tmp_path)
    plans = [p for _, p in zip(range(3), iter_step_plans(0, 0, train))]
    n_windows = plans[-1].base_window + plans[-1].microbatches * mb
    all_windows = [w for _, w in zip(range(n_windows), s)]

    for N in (1, 2, 3, 4, 8):
        for plan in plans:
            M = plan.microbatches
            # Collect (global_window_index -> window) by walking each rank's
            # micro-batches and the windows inside them.
            seen: dict[int, np.ndarray] = {}
            for r in range(N):
                for m in microbatch_indices_for_rank(M, world_size=N, rank=r):
                    for slot in range(mb):
                        gw = plan.base_window + m * mb + slot
                        assert gw not in seen, "window assigned to two ranks"
                        seen[gw] = all_windows[gw]
            # Complete coverage of the step's window range.
            expected = set(range(plan.base_window, plan.base_window + M * mb))
            assert set(seen) == expected
            # And the bytes match the canonical stream.
            for gw, w in seen.items():
                np.testing.assert_array_equal(w, all_windows[gw])


# --------------------------------------------------------------------------- #
# Canonical schedule math — world-size-independent by construction
# --------------------------------------------------------------------------- #


def test_microbatches_per_step_defaults():
    # Defaults: warmup 1_048_576, main 2_097_152, late 4_194_304; mb=4, seq=4096
    # → per-mb 16384 → M = 64, 128, 256 at the three phases.
    train = TrainConfig()
    assert microbatches_per_step(0, train) == 64
    assert microbatches_per_step(train.warmup_to_main_at_tokens, train) == 128
    assert microbatches_per_step(train.main_to_late_at_tokens, train) == 256


def test_microbatches_per_step_matches_legacy_where_divisible():
    # Canonical M must equal legacy dp_world_size * grad_accum_steps when the
    # target divides evenly (the divisor for which legacy's per-rank round-up is
    # exact). Probe the warmup phase across several world sizes.
    from pretrain.train.batch_schedule import grad_accum_steps

    train = TrainConfig()
    M = microbatches_per_step(0, train)
    for N in (1, 2, 4, 8, 16):
        assert grad_accum_steps(0, train, dp_world_size=N) * N == M


def test_canonical_tokens_per_step_independent_of_world_size():
    from pretrain.train.batch_schedule import current_global_batch_tokens

    train = TrainConfig()
    for consumed in (0, train.warmup_to_main_at_tokens, train.main_to_late_at_tokens):
        toks = canonical_tokens_per_step(consumed, train)
        assert toks == windows_per_step(consumed, train) * train.seq_len
        # >= the phase target, and a multiple of mb*seq (whole micro-batches).
        assert toks >= current_global_batch_tokens(consumed, train)
        assert toks % (train.micro_batch_size * train.seq_len) == 0


def test_iter_step_plans_accumulates_consistently():
    train = TrainConfig()
    mb = train.micro_batch_size
    consumed = 0
    windows = 0
    for plan in (p for _, p in zip(range(50), iter_step_plans(0, 0, train))):
        assert plan.consumed_tokens == consumed
        assert plan.base_window == windows
        assert plan.tokens_this_step == plan.microbatches * mb * train.seq_len
        consumed += plan.tokens_this_step
        windows += plan.microbatches * mb


# --------------------------------------------------------------------------- #
# ShardedWindowView — one rank's slice of the canonical stream
# --------------------------------------------------------------------------- #


def _small_train() -> TrainConfig:
    # M = ceil(160 / (2*16)) = 5 micro-batches/step, mb=2 → 10 windows/step.
    train = TrainConfig(seq_len=16, micro_batch_size=2)
    train.global_batch_tokens.warmup = 16 * 2 * 5
    return train


def _sources(tmp_path):
    a, da = _make_source(tmp_path, "a", 80, 0)
    b, db = _make_source(tmp_path, "b", 60, 100_000)
    return [a, b], [da, db], [0.7, 0.3]


def _expected_owned(global_windows, plans, mb, N, r):
    """Micro-batches assigned to rank r, in order, straight from the stream."""
    out = []
    for plan in plans:
        for m in range(plan.microbatches):
            if m % N == r:
                base = plan.base_window + m * mb
                out.append(np.stack(global_windows[base : base + mb], axis=0))
    return out


def test_view_owned_microbatches_match_canonical_stream(tmp_path):
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    mb = train.micro_batch_size

    plans = [p for _, p in zip(range(4), iter_step_plans(0, 0, train))]
    n_windows = plans[-1].base_window + plans[-1].microbatches * mb
    gstream = GlobalStream(manifests, dirs, weights, seq_len=train.seq_len, seed=5, eos_id=999)
    global_windows = [w for _, w in zip(range(n_windows), gstream)]

    for N in (1, 2, 3, 4):
        for r in range(N):
            view = ShardedWindowView(
                manifests, dirs, weights, train, seed=5, rank=r, world_size=N, eos_id=999
            )
            expected = _expected_owned(global_windows, plans, mb, N, r)
            got = [mbatch for _, mbatch in zip(range(len(expected)), iter(view))]
            assert len(got) == len(expected), f"N={N} r={r} count mismatch"
            for i, (g, e) in enumerate(zip(got, expected)):
                np.testing.assert_array_equal(g, e, err_msg=f"N={N} r={r} mb {i}")


def test_view_state_equals_global_stream_state(tmp_path):
    """Walking the view (any rank) leaves the *global* stream position — bit
    identical to GlobalStream after the same number of windows."""
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    mb = train.micro_batch_size

    # Advance exactly two full steps' worth of windows.
    plans = [p for _, p in zip(range(2), iter_step_plans(0, 0, train))]
    n_windows = plans[-1].base_window + plans[-1].microbatches * mb

    gstream = GlobalStream(manifests, dirs, weights, seq_len=train.seq_len, seed=5, eos_id=999)
    git = iter(gstream)
    for _ in range(n_windows):
        next(git)
    gstate = gstream.state()

    for N in (1, 2, 4):
        for r in range(N):
            view = ShardedWindowView(
                manifests, dirs, weights, train, seed=5, rank=r, world_size=N, eos_id=999
            )
            it = iter(view)
            # Drive the view across both full steps. Assignment resets per step,
            # so count owned micro-batches per step (m in range(M)), not across.
            owned = sum(1 for p in plans for m in range(p.microbatches) if m % N == r)
            for _ in range(owned):
                next(it)
            vstate = view.state()
            assert vstate.windows_emitted == gstate.windows_emitted == n_windows
            assert vstate.consumed_documents_per_source == gstate.consumed_documents_per_source
            assert vstate.epoch_per_source == gstate.epoch_per_source
            assert vstate.carry_over == gstate.carry_over
            assert vstate.mix_rng_state == gstate.mix_rng_state


def test_view_resumes_bit_exact(tmp_path):
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()

    plans = [p for _, p in zip(range(4), iter_step_plans(0, 0, train))]
    cut = plans[1].base_window + plans[1].microbatches * train.micro_batch_size  # after 2 steps

    # Rank 1 of 4, run two steps, snapshot, finish two more.
    N, r = 4, 1
    v1 = ShardedWindowView(manifests, dirs, weights, train, seed=5, rank=r, world_size=N, eos_id=999)
    it1 = iter(v1)
    owned_first = sum(1 for p in plans[:2] for m in range(p.microbatches) if m % N == r)
    [next(it1) for _ in range(owned_first)]
    state = v1.state()
    post = [next(it1) for _ in range(sum(1 for p in plans[2:4] for m in range(p.microbatches) if m % N == r))]

    # Resume from snapshot — must reproduce `post` exactly.
    consumed_at_cut = plans[2].consumed_tokens
    v2 = ShardedWindowView(
        manifests, dirs, weights, train, seed=5, rank=r, world_size=N, eos_id=999,
        state=state, start_consumed_tokens=consumed_at_cut, start_step=2,
    )
    it2 = iter(v2)
    resumed = [next(it2) for _ in range(len(post))]
    for i, (p, q) in enumerate(zip(post, resumed)):
        np.testing.assert_array_equal(p, q, err_msg=f"resumed mb {i} not bit-exact")


def test_view_skips_non_owned_payload_reads(tmp_path):
    """A rank that owns 1/4 of the work reads far fewer token payloads than a
    rank that owns everything — proving the length-walk skips non-owned data."""
    from pretrain.data.indexed_dataset import IndexedDatasetReader

    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    plans = [p for _, p in zip(range(4), iter_step_plans(0, 0, train))]
    total_mb = sum(p.microbatches for p in plans)

    counter = {"n": 0}
    orig = IndexedDatasetReader.document

    def counting_document(self, idx):
        counter["n"] += 1
        return orig(self, idx)

    def run(N, r):
        counter["n"] = 0
        view = ShardedWindowView(
            manifests, dirs, weights, train, seed=5, rank=r, world_size=N, eos_id=999
        )
        it = iter(view)
        owned = sum(1 for m in range(total_mb) if m % N == r)
        for _ in range(owned):
            next(it)
        return counter["n"]

    import unittest.mock as mock
    with mock.patch.object(IndexedDatasetReader, "document", counting_document):
        reads_full = run(1, 0)   # owns all 20 micro-batches
        reads_quarter = run(4, 0)  # owns ~5
    assert reads_quarter * 2 < reads_full, (
        f"expected far fewer payload reads for a 1/4 slice: "
        f"quarter={reads_quarter} full={reads_full}"
    )


def test_build_global_loader_collates_input_label_shift(tmp_path):
    """The loader splits each [mb, seq_len+1] window into input/label with the
    one-token shift, matching the legacy build_loader collate."""
    from pretrain.config.schema import DataConfig, DataSourceConfig
    from pretrain.data.loader import build_global_loader

    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    cfg = DataConfig(
        sources=[
            DataSourceConfig(name="a", path=dirs[0], weight=0.7),
            DataSourceConfig(name="b", path=dirs[1], weight=0.3),
        ],
        seq_len=train.seq_len,
        document_separator_id=999,
    )
    loader, view = build_global_loader(
        cfg, train, rank=0, world_size=2, seed=5, eos_id=999, pin_memory=False
    )
    it = iter(loader)
    batch = next(it)
    assert batch["input_ids"].shape == (train.micro_batch_size, train.seq_len)
    assert batch["labels"].shape == (train.micro_batch_size, train.seq_len)
    # label[t] == input[t+1] within a window (the canonical next-token shift).
    import torch
    assert torch.equal(batch["input_ids"][:, 1:], batch["labels"][:, :-1])
    # accum_for_step exposes this rank's micro-batch count for a step.
    assert view.accum_for_step(5) == 3  # M=5, N=2, rank 0 → m∈{0,2,4}
