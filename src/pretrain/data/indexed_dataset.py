"""Megatron-style indexed binary dataset (`.bin` + `.idx`).

Vendored implementation; ~ 200 LOC. We do **not** depend on the Megatron
package itself (~ 1 GB transitive deps for what is fundamentally a fixed
file format). See ADR-002.

File format (matches Megatron-Core ``IndexedDataset`` v1):

    .idx layout:
        bytes  0..8   : 8-byte magic 'MMIDIDX\0'
        bytes  8..16  : version (uint64 little-endian, currently 1)
        bytes 16..17  : dtype code (uint8) — see DTYPE_CODES
        bytes 17..25  : num_documents (int64)
        bytes 25..33  : token_count (int64)
        bytes 33..    : doc_offsets[num_documents+1] (int64) — token-offset
                        of each document, plus final sentinel
        followed by   : doc_byte_offsets[num_documents+1] (int64) — byte
                        offset of each document in the .bin

    .bin layout:
        Concatenation of token IDs in the chosen dtype, no framing.
        Document boundaries are recovered from .idx.

This is the format that `prepare.py` writes and `IndexedDatasetReader`
reads. It is mmap-backed for zero-copy reads in the hot path.
"""

from __future__ import annotations

import dataclasses
import os
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

MAGIC = b"MMIDIDX\0"
VERSION = 1

# Dtype code table (matches Megatron-Core).
DTYPE_CODES: dict[int, np.dtype] = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.int8),
    3: np.dtype(np.int16),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    6: np.dtype(np.float32),
    7: np.dtype(np.float64),
    8: np.dtype(np.uint16),
    9: np.dtype(np.uint32),
}

CODE_BY_DTYPE = {v: k for k, v in DTYPE_CODES.items()}


def _validate_dtype(dt: np.dtype) -> int:
    if dt not in CODE_BY_DTYPE:
        raise ValueError(
            f"unsupported dtype {dt}; choose one of {list(CODE_BY_DTYPE)}"
        )
    return CODE_BY_DTYPE[dt]


@dataclasses.dataclass
class _IndexHeader:
    dtype: np.dtype
    num_documents: int
    token_count: int


def _read_header(path: Path) -> tuple[_IndexHeader, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        version = struct.unpack("<Q", f.read(8))[0]
        if version != VERSION:
            raise ValueError(f"{path}: unknown version {version}")
        dtype_code = struct.unpack("<B", f.read(1))[0]
        num_documents = struct.unpack("<q", f.read(8))[0]
        token_count = struct.unpack("<q", f.read(8))[0]
        # doc_offsets[N+1] then doc_byte_offsets[N+1]
        offsets = np.frombuffer(
            f.read(8 * (num_documents + 1)), dtype=np.int64
        ).copy()
        byte_offsets = np.frombuffer(
            f.read(8 * (num_documents + 1)), dtype=np.int64
        ).copy()
    header = _IndexHeader(
        dtype=DTYPE_CODES[dtype_code],
        num_documents=num_documents,
        token_count=token_count,
    )
    return header, offsets, byte_offsets


class IndexedDatasetReader:
    """Read-only mmap view of one shard.

    A "shard" is one ``.bin`` + ``.idx`` pair. Multi-shard datasets are
    handled by the dataloader (which holds many readers and shuffles
    their concatenation; see :class:`MixSampler`).
    """

    def __init__(
        self, prefix: str | os.PathLike[str], *, index_only: bool = False
    ) -> None:
        prefix = Path(prefix)
        self.prefix = prefix
        self.index_only = index_only
        idx_path = prefix.with_suffix(".idx")
        bin_path = prefix.with_suffix(".bin")
        # ``index_only`` (audit-data fetch): we only need the .idx — document
        # *lengths* and *byte ranges* — to walk the canonical stream and decide
        # which shards' .bin to download. The .bin may legitimately be absent at
        # this point, so don't require or mmap it; ``document()`` then raises.
        if not idx_path.exists():
            raise FileNotFoundError(f"missing shard index: {idx_path}")
        if not index_only and not bin_path.exists():
            raise FileNotFoundError(f"missing shard payload: {bin_path}")
        self._header, self._token_offsets, self._byte_offsets = _read_header(idx_path)
        # mmap the .bin lazily so file descriptors are cheap.
        self._mmap = (
            None if index_only else np.memmap(bin_path, dtype=self._header.dtype, mode="r")
        )

    @property
    def dtype(self) -> np.dtype:
        return self._header.dtype

    @property
    def num_documents(self) -> int:
        return self._header.num_documents

    @property
    def token_count(self) -> int:
        return self._header.token_count

    def document(self, idx: int) -> np.ndarray:
        if self._mmap is None:
            raise RuntimeError(
                f"{self.prefix}: reader opened index_only; .bin payload is not "
                "available (cannot read document tokens, only lengths/offsets)"
            )
        if idx < 0:
            idx += self.num_documents
        if not 0 <= idx < self.num_documents:
            raise IndexError(idx)
        start = int(self._token_offsets[idx])
        end = int(self._token_offsets[idx + 1])
        # Returned array is a *view* over the mmap. Caller must copy if
        # they hold it past the next mmap touch.
        return np.asarray(self._mmap[start:end])

    def document_length(self, idx: int) -> int:
        """Token count of document ``idx`` without touching the ``.bin``.

        Read straight from the ``.idx`` token-offset table, so this never
        faults a page of the mmap'd payload. The canonical global stream's
        cluster-side slicing walks every document's length to compute window
        boundaries while reading token payloads only for the windows the
        local rank actually owns (see ``data/global_stream.py``).
        """
        if idx < 0:
            idx += self.num_documents
        if not 0 <= idx < self.num_documents:
            raise IndexError(idx)
        return int(self._token_offsets[idx + 1]) - int(self._token_offsets[idx])

    def __len__(self) -> int:
        return self.num_documents

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.num_documents):
            yield self.document(i)


class IndexedDatasetWriter:
    """Streaming writer. Use as a context manager:

        with IndexedDatasetWriter("path/to/shard", dtype=np.uint32) as w:
            for doc_tokens in stream:
                w.add_document(doc_tokens)

    On ``__exit__`` the ``.idx`` is flushed.
    """

    def __init__(self, prefix: str | os.PathLike[str], dtype: np.dtype) -> None:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self._prefix = prefix
        self._dtype = np.dtype(dtype)
        self._dtype_code = _validate_dtype(self._dtype)
        self._bin_file = open(prefix.with_suffix(".bin"), "wb")
        self._token_offsets: list[int] = [0]
        self._byte_offsets: list[int] = [0]
        self._closed = False

    def add_document(self, tokens: np.ndarray | list[int]) -> None:
        if self._closed:
            raise RuntimeError("writer already closed")
        arr = np.asarray(tokens, dtype=self._dtype)
        if arr.ndim != 1:
            raise ValueError(f"expected 1-D token array, got shape {arr.shape}")
        self._bin_file.write(arr.tobytes())
        self._token_offsets.append(self._token_offsets[-1] + arr.size)
        self._byte_offsets.append(
            self._byte_offsets[-1] + arr.size * self._dtype.itemsize
        )

    def close(self) -> None:
        if self._closed:
            return
        self._bin_file.close()
        idx_path = self._prefix.with_suffix(".idx")
        with open(idx_path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<Q", VERSION))
            f.write(struct.pack("<B", self._dtype_code))
            f.write(struct.pack("<q", len(self._token_offsets) - 1))
            f.write(struct.pack("<q", self._token_offsets[-1]))
            f.write(np.asarray(self._token_offsets, dtype=np.int64).tobytes())
            f.write(np.asarray(self._byte_offsets, dtype=np.int64).tobytes())
        self._closed = True

    def __enter__(self) -> "IndexedDatasetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
