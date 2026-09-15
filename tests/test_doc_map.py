"""Step -> document map (training data explorer).

Pins that the index-only span walk describes exactly the windows the cluster
trains on: per window, the recorded document fragments reassemble the
materialised window (minus EOS separators), the fragment set equals
``walk_doc_refs``, and the walk's final position matches a real stream state.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pretrain.data.doc_map import compare_position, step_rows, write_doc_map
from pretrain.data.global_stream import ShardedWindowView
from pretrain.data.indexed_dataset import IndexedDatasetReader
from pretrain.train.batch_schedule import iter_step_plans
from tests.test_fetch_interval import _prefix_to_shard, _small_train, _sources

EOS = 999
SEED = 5
UNTIL = 4


def _index_only_view(manifests, dirs, weights, train):
    return ShardedWindowView(
        manifests, dirs, weights, train, seed=SEED, rank=0, world_size=1, eos_id=EOS, index_only=True
    )


def _materialized_windows(manifests, dirs, weights, train, until_step):
    """Global windows for steps < until_step from a full (rank 0 of 1) view."""
    view = ShardedWindowView(manifests, dirs, weights, train, seed=SEED, rank=0, world_size=1, eos_id=EOS)
    plans = [p for _, p in zip(range(until_step), iter_step_plans(0, 0, train))]
    it = iter(view)
    windows = []
    for p in plans:
        for _ in range(p.microbatches):
            windows.extend(list(next(it)))
    return windows, view


def _readers(manifests, dirs):
    out = {}
    for m, d in zip(manifests, dirs):
        for sid, shard in enumerate(m.shards):
            pfx = Path(shard.prefix)
            out[(m.name, sid)] = IndexedDatasetReader(pfx if pfx.is_absolute() else Path(d) / pfx)
    return out


def test_spans_reassemble_materialized_windows(tmp_path):
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    windows, _ = _materialized_windows(manifests, dirs, weights, train, UNTIL)
    readers = _readers(manifests, dirs)

    view = _index_only_view(manifests, dirs, weights, train)
    by_window: dict[int, list] = defaultdict(list)
    n_steps = 0
    for plan, spans in view.walk_step_spans(until_step=UNTIL):
        n_steps += 1
        for gw, src, sid, local, s, e in spans:
            assert plan.base_window <= gw < plan.base_window + plan.microbatches * train.micro_batch_size
            by_window[gw].append((src, sid, local, s, e))
    assert n_steps == UNTIL
    assert sorted(by_window) == list(range(len(windows)))

    for gw, win in enumerate(windows):
        frags = [readers[(src, sid)].document(local)[s:e] for src, sid, local, s, e in by_window[gw]]
        got = np.concatenate(frags).astype(np.int64)
        np.testing.assert_array_equal(got, win[win != EOS])
        # fragments + EOS tokens fill the window exactly
        assert sum(e - s for _, _, _, s, e in by_window[gw]) + int((win == EOS).sum()) == train.seq_len + 1
        # fragments never overlap or run out of order inside a document
        for src, sid, local, s, e in by_window[gw]:
            assert 0 <= s < e <= readers[(src, sid)].document_length(local)


def test_spans_cover_exactly_walk_doc_refs(tmp_path):
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    refs = set(_index_only_view(manifests, dirs, weights, train).walk_doc_refs(until_step=UNTIL))
    spans = set()
    for _, ss in _index_only_view(manifests, dirs, weights, train).walk_step_spans(until_step=UNTIL):
        spans.update((src, sid, local) for _, src, sid, local, _, _ in ss)
    assert spans == refs


def test_span_recording_does_not_perturb_stream(tmp_path):
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    a = _index_only_view(manifests, dirs, weights, train)
    list(a.walk_step_spans(until_step=UNTIL))
    b = _index_only_view(manifests, dirs, weights, train)
    list(b.walk_doc_refs(until_step=UNTIL))
    assert a.position() == b.position()
    assert a._span_sink is None and a._ref_sink is None


def test_write_doc_map_and_position_match_checkpoint_state(tmp_path):
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    windows, full_view = _materialized_windows(manifests, dirs, weights, train, UNTIL)
    ckpt_state = full_view.state()  # what a checkpoint would write as global_stream.json

    view = _index_only_view(manifests, dirs, weights, train)
    out = tmp_path / "step_docs"
    summary = write_doc_map(
        view, out, until_step=UNTIL, micro_batch_size=train.micro_batch_size, dp_world_size=3, chunk_steps=2
    )
    files = sorted(out.glob("step_docs-*.parquet"))
    assert [Path(f).name for f in summary["files"]] == [f.name for f in files]
    assert len(files) == 2  # 4 steps / 2 per chunk
    table = pq.read_table(files[0]).to_pandas()
    for f in files[1:]:
        table = __import__("pandas").concat([table, pq.read_table(f).to_pandas()])
    assert len(table) == summary["rows"] == sum(summary["rows_per_source"].values())
    assert sorted(table["step"].unique()) == list(range(UNTIL))
    assert summary["windows"] == len(windows)
    # ownership columns follow the cluster rule
    mb = train.micro_batch_size
    plans = {p.step: p for _, p in zip(range(UNTIL), iter_step_plans(0, 0, train))}
    for row in table.itertuples():
        rel = row.global_window - plans[row.step].base_window
        assert row.microbatch == rel // mb and row.slot == rel % mb
        assert row.dp_rank == row.microbatch % 3
    # frag_idx restarts per window and counts fragments in order
    for gw, grp in table.groupby("global_window"):
        assert list(grp.sort_values("frag_idx")["frag_idx"]) == list(range(len(grp)))
    # final position agrees with the materialised stream's checkpoint state
    gs = json.loads(json.dumps(ckpt_state.__dict__))
    assert compare_position(summary["position"], gs) == []
    gs["windows_emitted"] += 1
    assert compare_position(summary["position"], gs)


def test_step_rows_shape():
    from pretrain.train.batch_schedule import StepPlan

    plan = StepPlan(step=7, consumed_tokens=0, base_window=100, microbatches=3, tokens_this_step=0)
    spans = [(100, "a", 0, 5, 0, 10), (100, "a", 0, 6, 0, 7), (101, "b", 1, 2, 3, 17), (105, "a", 2, 9, 0, 17)]
    cols = step_rows(plan, spans, micro_batch_size=2, dp_world_size=2)
    assert cols["microbatch"] == [0, 0, 0, 2]
    assert cols["slot"] == [0, 0, 1, 1]
    assert cols["frag_idx"] == [0, 1, 0, 0]
    assert cols["dp_rank"] == [0, 0, 0, 0]
    assert cols["tok_end"] == [10, 7, 17, 17]


def test_dp_rank_follows_the_training_ownership_rule(tmp_path):
    """The doc map's dp_rank must be derived from the same function the training
    loop uses, not a re-statement of the modulo that happens to agree today."""
    from pretrain.train.batch_schedule import microbatch_indices_for_rank

    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    world = 3
    view = _index_only_view(manifests, dirs, weights, train)
    for plan, spans in view.walk_step_spans(until_step=UNTIL):
        cols = step_rows(plan, spans, micro_batch_size=train.micro_batch_size, dp_world_size=world)
        for rank in range(world):
            owned = sorted({m for m, r in zip(cols["microbatch"], cols["dp_rank"]) if r == rank})
            assert owned == microbatch_indices_for_rank(plan.microbatches, world_size=world, rank=rank)


def test_walk_doc_refs_guard_and_cleanup(tmp_path):
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    view = _index_only_view(manifests, dirs, weights, train)
    # A span walk in progress refuses a ref walk on the same cursor.
    spans = view.walk_step_spans(until_step=UNTIL)
    next(spans)
    try:
        next(view.walk_doc_refs(until_step=UNTIL))
        assert False, "walk_doc_refs advanced a view that walk_step_spans was driving"
    except RuntimeError:
        pass
    spans.close()
    assert view._span_sink is None
    # An abandoned ref walk leaves the view walkable.
    view = _index_only_view(manifests, dirs, weights, train)
    refs = view.walk_doc_refs(until_step=UNTIL)
    next(refs)
    refs.close()
    assert view._ref_sink is None
    assert list(view.walk_step_spans(until_step=UNTIL))


def test_span_walk_refuses_resumed_carry_over(tmp_path):
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    _, full_view = _materialized_windows(manifests, dirs, weights, train, UNTIL)
    state = full_view.state()
    if not state.carry_over:
        import pytest

        pytest.skip("this corpus left no carry-over at the step boundary")
    resumed = ShardedWindowView(
        manifests, dirs, weights, train, seed=SEED, rank=0, world_size=1, eos_id=EOS,
        state=state, start_step=UNTIL, index_only=True,
    )
    assert resumed.has_carry_over
    try:
        next(resumed.walk_step_spans(until_step=UNTIL + 1))
        assert False, "carry-over tokens would have gone unattributed"
    except ValueError as e:
        assert "carry-over" in str(e)
    fresh = ShardedWindowView(
        manifests, dirs, weights, train, seed=SEED, rank=0, world_size=1, eos_id=EOS, index_only=True
    )
    assert not fresh.has_carry_over


def test_write_doc_map_counts_steps_walked(tmp_path):
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    view = _index_only_view(manifests, dirs, weights, train)
    summary = write_doc_map(
        view, tmp_path / "out", until_step=UNTIL, micro_batch_size=train.micro_batch_size, dp_world_size=2
    )
    assert summary["steps_walked"] == UNTIL


def test_compare_position_flags_a_source_missing_from_the_checkpoint():
    position = {
        "consumed_documents_per_source": {"a": 3, "b": 4},
        "epoch_per_source": {"a": 0, "b": 0},
        "windows_emitted": 7,
    }
    older = {"consumed_documents_per_source": {"a": 3}, "epoch_per_source": {"a": 0}, "windows_emitted": 7}
    problems = compare_position(position, older)
    assert any(p.startswith("consumed_documents_per_source[b]") for p in problems)
    assert any(p.startswith("epoch_per_source[b]") for p in problems)


def test_resolve_until_step_requires_a_forward_walk_from_checkpoint_state():
    import pytest

    from pretrain.cli.dump_doc_map import resolve_until_step

    assert resolve_until_step(None, 80957, from_checkpoint_state=False) == 80957
    assert resolve_until_step(20, 80957, from_checkpoint_state=False) == 20
    assert resolve_until_step(80960, 80957, from_checkpoint_state=True) == 80960
    with pytest.raises(SystemExit):
        resolve_until_step(None, 80957, from_checkpoint_state=True)
    with pytest.raises(SystemExit):
        resolve_until_step(80957, 80957, from_checkpoint_state=True)


def test_rebase_sources_keeps_every_other_field(tmp_path):
    from pretrain.data.fetch_interval import rebase_sources

    manifests, dirs, weights = _sources(tmp_path)
    from pretrain.config.schema import DataConfig, DataSourceConfig

    cfg = DataConfig(
        sources=[DataSourceConfig(name=m.name, path=f"/elsewhere/{Path(d).name}", weight=w) for m, d, w in zip(manifests, dirs, weights)],
        seq_len=_small_train().seq_len,
        document_separator_id=EOS,
    )
    out = rebase_sources(cfg, tmp_path)
    assert [s.path for s in out.sources] == [str(tmp_path / Path(d).name) for d in dirs]
    assert out.model_dump(exclude={"sources"}) == cfg.model_dump(exclude={"sources"})
    assert [s.path for s in cfg.sources] == [f"/elsewhere/{Path(d).name}" for d in dirs]  # input untouched
