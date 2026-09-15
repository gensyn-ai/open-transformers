"""Indexed binary round-trip tests."""

from __future__ import annotations

import numpy as np
import pytest

from pretrain.data.indexed_dataset import (
    IndexedDatasetReader,
    IndexedDatasetWriter,
)


def test_round_trip_uint32(tmp_path):
    docs = [
        np.array([1, 2, 3, 4, 5], dtype=np.uint32),
        np.array([10, 20, 30], dtype=np.uint32),
        np.array([100], dtype=np.uint32),
        np.arange(1000, dtype=np.uint32),
    ]
    prefix = tmp_path / "shard"

    with IndexedDatasetWriter(prefix, dtype=np.uint32) as w:
        for d in docs:
            w.add_document(d)

    r = IndexedDatasetReader(prefix)
    assert r.num_documents == len(docs)
    assert r.token_count == sum(len(d) for d in docs)
    for i, d in enumerate(docs):
        np.testing.assert_array_equal(np.asarray(r.document(i)), d)


def test_negative_index(tmp_path):
    prefix = tmp_path / "shard"
    with IndexedDatasetWriter(prefix, dtype=np.uint16) as w:
        w.add_document([1, 2, 3])
        w.add_document([4, 5])
    r = IndexedDatasetReader(prefix)
    np.testing.assert_array_equal(np.asarray(r.document(-1)), np.array([4, 5]))


def test_iter_matches_index(tmp_path):
    prefix = tmp_path / "shard"
    docs = [np.arange(i, i + 5, dtype=np.uint32) for i in range(0, 30, 5)]
    with IndexedDatasetWriter(prefix, dtype=np.uint32) as w:
        for d in docs:
            w.add_document(d)
    r = IndexedDatasetReader(prefix)
    for i, doc in enumerate(r):
        np.testing.assert_array_equal(np.asarray(doc), docs[i])


def test_bad_magic(tmp_path):
    prefix = tmp_path / "shard"
    with IndexedDatasetWriter(prefix, dtype=np.uint32) as w:
        w.add_document([1, 2])
    # Corrupt the .idx magic.
    idx_path = prefix.with_suffix(".idx")
    data = bytearray(idx_path.read_bytes())
    data[0:8] = b"BADMAGIC"
    idx_path.write_bytes(bytes(data))

    with pytest.raises(ValueError):
        IndexedDatasetReader(prefix)
