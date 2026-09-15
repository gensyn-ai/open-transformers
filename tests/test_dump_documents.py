"""Document dump + doc-map spot check (training data explorer).

Pins the review findings on PR #62: dumped text keeps a special-token id that
sits inside a document, output files cannot collide across same-basename
shards, and the spot check rebuilds windows from fragments plus the packer's
separators instead of stripping every EOS-valued token.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pretrain.cli import dump_documents
from pretrain.data.doc_map import write_doc_map
from pretrain.data.global_stream import ShardedWindowView
from pretrain.data.indexed_dataset import IndexedDatasetReader, IndexedDatasetWriter
from pretrain.data.manifest import ShardInfo, SourceManifest
from pretrain.data.tokenizer import Tokenizer, train_tokenizer
from pretrain.train.batch_schedule import iter_step_plans
from tests.test_fetch_interval import _root_config, _small_train, _sources, _write_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SPOT_CHECK = ROOT / "scripts" / "explorer" / "spot_check_doc_map.py"
SEED = 5
UNTIL = 4


# --------------------------------------------------------------------------- #
# [3] special tokens inside a document survive the dump
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory) -> Path:
    corpus = ["hello world " * 5, "the quick brown fox", "lorem ipsum dolor sit amet"]
    return train_tokenizer(corpus, tmp_path_factory.mktemp("tok") / "tokenizer.json", vocab_size=300)


def test_decode_keeps_mid_document_special_token(tiny_tokenizer):
    tok = Tokenizer(tiny_tokenizer)
    ids = tok.encode("hello <|endoftext|> world")
    assert tok.eos_id in ids[1:-1], "fixture must place the special id inside the document"
    assert tok.decode(ids, skip_special_tokens=False) == "hello <|endoftext|> world"
    assert tok.decode_batch([ids], skip_special_tokens=False) == ["hello <|endoftext|> world"]
    # The default is unchanged so existing callers keep their behaviour.
    assert "<|endoftext|>" not in tok.decode(ids)


def test_dump_shard_text_keeps_special_token_and_counts_it(tiny_tokenizer, tmp_path):
    tok = Tokenizer(tiny_tokenizer)
    docs = [tok.encode("hello <|endoftext|> world"), tok.encode("the quick brown fox")]
    prefix = tmp_path / "src_00000"
    with IndexedDatasetWriter(prefix, dtype=np.uint32) as w:
        for d in docs:
            w.add_document(np.asarray(d, dtype=np.uint32))
    out = tmp_path / "out" / "00000_src_00000.parquet"
    dump_documents._init_worker(str(tiny_tokenizer))
    source, sid, n, _ = dump_documents.dump_shard(("src", 0, str(prefix), str(out), 1))
    assert (source, sid, n) == ("src", 0, 2)
    table = pq.read_table(out).to_pylist()
    assert table[0]["text"] == "hello <|endoftext|> world"
    assert table[0]["n_tokens"] == len(docs[0])
    assert table[1]["text"] == "the quick brown fox"


# --------------------------------------------------------------------------- #
# [6] output path keyed on shard_id; duplicates rejected
# --------------------------------------------------------------------------- #


def _write_source(root: Path, name: str, prefixes: list[str]) -> None:
    src = root / name
    shards = []
    for p in prefixes:
        full = src / p
        full.parent.mkdir(parents=True, exist_ok=True)
        with IndexedDatasetWriter(full, dtype=np.uint32) as w:
            w.add_document(np.arange(3, dtype=np.uint32))
        shards.append(ShardInfo(prefix=p, num_documents=1, token_count=3))
    SourceManifest(name=name, tokenizer_hash="x", dtype="uint32", shards=shards).save(src / "manifest.yaml")


def test_build_tasks_keys_output_on_shard_id(tmp_path):
    # Same basename under two parent dirs: the old <prefix.name>.parquet key
    # mapped both shards onto one file and the resume logic skipped one.
    _write_source(tmp_path / "data", "a", ["w0/part_00000", "w1/part_00000"])
    tasks = dump_documents.build_tasks(tmp_path / "data", tmp_path / "out", None, 8, None)
    outs = [Path(t[3]) for t in tasks]
    assert [t[1] for t in tasks] == [0, 1]
    assert [o.name for o in outs] == ["00000_part_00000.parquet", "00001_part_00000.parquet"]
    assert len(set(outs)) == len(outs)
    assert all(o.parent == tmp_path / "out" / "a" for o in outs)


def test_build_tasks_rejects_duplicate_outputs(tmp_path):
    # Two manifests naming the same source collide even with shard_id keying.
    _write_source(tmp_path / "data", "x", ["x_00000"])
    _write_source(tmp_path / "data", "y", ["x_00000"])
    m = SourceManifest.load(tmp_path / "data" / "y" / "manifest.yaml")
    m.name = "x"
    m.save(tmp_path / "data" / "y" / "manifest.yaml")
    with pytest.raises(ValueError, match="duplicate output"):
        dump_documents.build_tasks(tmp_path / "data", tmp_path / "out", None, 8, None)


# --------------------------------------------------------------------------- #
# [4] + [7] spot check: fragment-by-fragment window rebuild, lazy readers
# --------------------------------------------------------------------------- #


def _spot_check_module():
    spec = importlib.util.spec_from_file_location("spot_check_doc_map", SPOT_CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _doc_map(tmp_path, manifests, dirs, weights, train, *, eos):
    view = ShardedWindowView(
        manifests, dirs, weights, train, seed=SEED, rank=0, world_size=1, eos_id=eos, index_only=True
    )
    out = tmp_path / f"step_docs_{eos}"
    write_doc_map(view, out, until_step=UNTIL, micro_batch_size=train.micro_batch_size, dp_world_size=2)
    return out


def _windows(manifests, dirs, weights, train, *, eos):
    view = ShardedWindowView(manifests, dirs, weights, train, seed=SEED, rank=0, world_size=1, eos_id=eos)
    it = iter(view)
    wins = []
    for p in [p for _, p in zip(range(UNTIL), iter_step_plans(0, 0, train))]:
        for _ in range(p.microbatches):
            wins.extend(list(next(it)))
    return wins


def test_spot_check_passes_on_correct_dump(tmp_path):
    sc = _spot_check_module()
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    out = _doc_map(tmp_path, manifests, dirs, weights, train, eos=999)
    steps = {0, 2}
    rows = sc.load_rows(out, steps)
    checked, bad = sc.check_steps(manifests, dirs, weights, train, seed=SEED, eos=999, rows=rows, steps=steps)
    assert (checked, bad) == (2 * 5 * train.micro_batch_size, 0)


def test_spot_check_keeps_in_document_separator_ids(tmp_path):
    """A document token equal to the separator id is data, not a separator."""
    sc = _spot_check_module()
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    eos = 7  # source "a" holds np.arange tokens, so 7 sits inside a document
    wins = _windows(manifests, dirs, weights, train, eos=eos)
    out = _doc_map(tmp_path, manifests, dirs, weights, train, eos=eos)
    steps = set(range(UNTIL))
    rows = sc.load_rows(out, steps)
    # The old ``win[win != eos]`` reference would have dropped this token.
    n_eos_valued = sum(int((w == eos).sum()) for w in wins)
    n_separators = sum(train.seq_len + 1 - sum(r["tok_end"] - r["tok_start"] for r in frs) for frs in rows.values())
    assert n_eos_valued > n_separators
    checked, bad = sc.check_steps(manifests, dirs, weights, train, seed=SEED, eos=eos, rows=rows, steps=steps)
    assert (checked, bad) == (len(wins), 0)


def test_spot_check_flags_corrupt_fragment(tmp_path, capsys):
    sc = _spot_check_module()
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    out = _doc_map(tmp_path, manifests, dirs, weights, train, eos=999)
    rows = sc.load_rows(out, {1})
    gw = min(rows)
    rows[gw][0]["tok_end"] -= 1
    checked, bad = sc.check_steps(manifests, dirs, weights, train, seed=SEED, eos=999, rows=rows, steps={1})
    assert bad == 1 and checked == 5 * train.micro_batch_size
    msg = capsys.readouterr().out
    assert f"MISMATCH step=1 window={gw}" in msg and "frag 0" in msg


def test_spot_check_materializes_only_requested_steps(tmp_path):
    sc = _spot_check_module()
    manifests, dirs, weights = _sources(tmp_path)
    train = _small_train()
    out = _doc_map(tmp_path, manifests, dirs, weights, train, eos=999)
    rows = sc.load_rows(out, {2})
    calls = []
    orig = ShardedWindowView._take_window

    def spy(self, *, materialize):
        calls.append(materialize)
        return orig(self, materialize=materialize)

    opened = []

    def spy_reader(prefix, **kw):
        opened.append(Path(prefix))
        return IndexedDatasetReader(prefix, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ShardedWindowView, "_take_window", spy)
        # Only the spot check's own opens; the view's walkers import the class separately.
        mp.setattr(sc, "IndexedDatasetReader", spy_reader)
        checked, bad = sc.check_steps(manifests, dirs, weights, train, seed=SEED, eos=999, rows=rows, steps={2})
    per_step = 5 * train.micro_batch_size
    assert bad == 0 and checked == per_step
    # steps 0..2 walked, only step 2 materialised, step 3 never touched
    assert calls == [False] * (2 * per_step) + [True] * per_step
    # One reader per shard the checked rows reference, none for the rest.
    touched = {(r["source"], r["shard_id"]) for frs in rows.values() for r in frs}
    assert 0 < len(touched) < sum(len(m.shards) for m in manifests)
    assert sorted(opened) == sorted(Path(dirs[[m.name for m in manifests].index(s)]) / f"{s}_{sid:05d}" for s, sid in touched)


def test_spot_check_cli_end_to_end(tmp_path):
    manifests, dirs, weights = _sources(tmp_path / "shards")
    train = _small_train()
    cfg = _root_config(train.seq_len, train)
    ckpt = tmp_path / "ckpt" / "step_000000000"
    _write_checkpoint(ckpt, cfg, seed=SEED, step=0, consumed=0)
    view = ShardedWindowView(
        manifests, dirs, weights, train, seed=SEED, rank=0, world_size=1, eos_id=999, index_only=True
    )
    out = tmp_path / "step_docs"
    write_doc_map(view, out, until_step=UNTIL, micro_batch_size=train.micro_batch_size, dp_world_size=4)

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    cmd = [
        sys.executable, str(SPOT_CHECK), "--checkpoint", str(ckpt), "--step-docs", str(out),
        "--steps", "0", "3", "--data-root", str(tmp_path / "shards"),
    ]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "checked 20 windows across steps [0, 3]: 0 mismatches" in r.stdout

    # Corrupt one fragment on disk and the CLI must exit 1 naming the window.
    f = sorted(out.glob("step_docs-*.parquet"))[0]
    t = pq.read_table(f).to_pandas()
    i = t.index[t["step"] == 3][0]
    t.loc[i, "tok_start"] += 1
    pq.write_table(pa.Table.from_pandas(t, schema=pq.read_schema(f), preserve_index=False), f)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert f"MISMATCH step=3 window={int(t.loc[i, 'global_window'])}" in r.stdout
