"""Document dump (training data explorer).

Pins two review findings: dumped text keeps a special-token id that sits
inside a document, and output files cannot collide across same-basename
shards.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from pretrain.cli import dump_documents
from pretrain.data.indexed_dataset import IndexedDatasetReader, IndexedDatasetWriter
from pretrain.data.manifest import ShardInfo, SourceManifest
from pretrain.data.tokenizer import Tokenizer, train_tokenizer



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
