"""Deterministic, rank-aware, weighted multi-source sampler.

For each global step the sampler:
  1. Treats sources as a categorical distribution with normalised weights.
  2. Per *document slot*, draws a source.
  3. Within that source, walks a deterministic shard-and-document
     iterator (shard-level shuffle keyed by ``(epoch_seed, source_id)``,
     document-level shuffle keyed by ``(epoch_seed, source_id, shard_id)``).

The sampler is rank-aware: rank ``r`` of ``R`` reads documents whose
``document_id mod R == r``. This is the simplest correct way to keep
the loader DDP/FSDP-clean today and unchanged at multi-node tomorrow.

Resume: per-source ``consumed_documents`` counters are held in state and
restored on checkpoint load. The sampler fast-forwards using the seed +
counters; the same seed always produces the same documents at the same
counter value.
"""

from __future__ import annotations

import dataclasses
import hashlib
import weakref
from typing import Iterator

import numpy as np

from pretrain.data.indexed_dataset import IndexedDatasetReader
from pretrain.data.manifest import SourceManifest

# Process-wide cache of per-(seed, source, epoch) document permutations. The
# permutation is a pure function of those keys (rank-independent) and is only
# ever read, so identical walkers can share one array. This matters for the
# single-device audit, which builds N ShardedWindowViews (one per virtual rank)
# in ONE process: without sharing, each view materialises its own full
# permutation — at 32 ranks over a large corpus that's ~32x ~3 GB ≈ 100 GB and
# OOMs the box. A WeakValueDictionary frees each array once no walker references
# it, so the cluster path (one walker per process) keeps a single live entry and
# leaks nothing across epochs.
_GLOBAL_PERM_CACHE: "weakref.WeakValueDictionary[tuple, np.ndarray]" = (
    weakref.WeakValueDictionary()
)

# Process-wide cache of shard readers, keyed by resolved prefix. Each reader
# holds the shard's document offset index in RAM (np.frombuffer of the .idx, not
# mmap), so the single-device audit's N ShardedWindowViews would otherwise each
# build their own readers — N× the index (~3 GB/source at 32 ranks → OOM).
# Readers are read-only (mmap'd payload + immutable index), so sharing one per
# shard across views is safe. A plain dict is fine: the set of shards is bounded
# and the readers are live for the whole run (cluster path: one process touches
# each shard once anyway).
_READER_CACHE: dict = {}


def _cached_reader(prefix, *, index_only: bool = False) -> "IndexedDatasetReader":
    # Key on ``index_only`` too: an index-only reader has no .bin mmap, so it
    # must not be handed back to a caller that needs payloads (and vice versa).
    key = (str(prefix), index_only)
    reader = _READER_CACHE.get(key)
    if reader is None:
        reader = IndexedDatasetReader(prefix, index_only=index_only)
        _READER_CACHE[key] = reader
    return reader


def _stable_name_hash(name: str) -> int:
    """Process-independent 32-bit hash of a source name.

    Python's built-in ``hash()`` is randomised per interpreter, so different
    torchrun ranks (and different re-runs of the same job) would seed the
    sampler RNG differently — breaking both rank-stride disjointness and
    cross-run reproducibility. SHA-256 of the UTF-8 bytes is stable across
    processes, machines, and Python versions.
    """
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")


@dataclasses.dataclass
class MixSamplerState:
    """Everything needed to resume the sampler bit-exactly."""

    consumed_documents_per_source: dict[str, int]
    epoch_per_source: dict[str, int]
    # The mix RNG state, encoded as the bit-generator state dict that
    # ``np.random.default_rng().bit_generator.state`` returns. Restored on
    # resume so source picks remain deterministic across save points.
    mix_rng_state: dict | None = None
    # Tokens left in the carry-over buffer at save time. Concatenated to
    # the front of the buffer on resume so packing stays consistent.
    carry_over: list[int] = dataclasses.field(default_factory=list)


class _SourceWalker:
    """One source's deterministic document walk. Holds shard readers and
    a permutation per epoch. Rank-aware: only emits documents whose global
    document index ``% world_size == rank``.
    """

    def __init__(
        self,
        manifest: SourceManifest,
        manifest_dir: str,
        seed: int,
        rank: int,
        world_size: int,
        start_consumed: int = 0,
        start_epoch: int = 0,
        index_only: bool = False,
    ) -> None:
        from pathlib import Path

        self.name = manifest.name
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # Resolve shard prefixes relative to manifest_dir if they are not
        # already absolute. Manifest stores stems without suffixes.
        # ``index_only`` opens readers from the .idx alone (no .bin) — the
        # canonical-stream length-walk used by the audit-data fetch tool to
        # decide which shards to download (see data/global_stream.py).
        base = Path(manifest_dir)
        self._readers = [
            _cached_reader(
                shard.prefix if Path(shard.prefix).is_absolute() else base / shard.prefix,
                index_only=index_only,
            )
            for shard in manifest.shards
        ]
        self._shard_doc_counts = np.asarray(
            [r.num_documents for r in self._readers], dtype=np.int64
        )
        self._cum_docs = np.concatenate(
            [[0], np.cumsum(self._shard_doc_counts)]
        )
        self.total_documents = int(self._cum_docs[-1])

        self.consumed = start_consumed
        self.epoch = start_epoch
        self._global_perm = self._build_global_perm(self.epoch)

    def _build_global_perm(self, epoch: int) -> np.ndarray:
        """Permutation over all documents in the source for this epoch.

        Implementation: shard-level shuffle with key ``(seed, source, epoch)``,
        then document-level shuffle within each shard with key
        ``(seed, source, epoch, shard_id)``. Concatenated, this yields a
        deterministic permutation of [0, total_documents).
        """
        # Share an identical permutation across walkers (see _GLOBAL_PERM_CACHE).
        # Keyed by everything the result depends on: seed, source name, epoch,
        # and total_documents (a cheap guard against a name hash colliding with a
        # differently-sized source). Read-only, so sharing is bitwise-safe.
        key = (self.seed, _stable_name_hash(self.name), epoch, self.total_documents)
        cached = _GLOBAL_PERM_CACHE.get(key)
        if cached is not None:
            return cached

        rng_shards = np.random.default_rng(
            np.array(
                [self.seed, _stable_name_hash(self.name), epoch], dtype=np.uint64
            )
        )
        shard_order = rng_shards.permutation(len(self._readers))

        out: list[np.ndarray] = []
        for shard_id in shard_order:
            n = int(self._shard_doc_counts[shard_id])
            offset = int(self._cum_docs[shard_id])
            rng_docs = np.random.default_rng(
                np.array(
                    [
                        self.seed,
                        _stable_name_hash(self.name),
                        epoch,
                        int(shard_id),
                    ],
                    dtype=np.uint64,
                )
            )
            local = rng_docs.permutation(n) + offset
            out.append(local)
        perm = np.concatenate(out)
        _GLOBAL_PERM_CACHE[key] = perm
        return perm

    def _resolve_doc(self, global_doc_index: int) -> np.ndarray:
        # Translate to shard + within-shard index via cumulative bounds.
        shard_id = int(np.searchsorted(self._cum_docs, global_doc_index, side="right") - 1)
        local = global_doc_index - int(self._cum_docs[shard_id])
        return self._readers[shard_id].document(local)

    def next_document_ref(self) -> tuple[int, int]:
        """Advance the walker by one document and return ``(shard_id, local)``.

        We advance ``self.consumed`` by 1 each call (rank-local count); the
        position in the global permutation is computed from
        ``consumed * world_size + rank`` so the rank-stride is implicit.
        Returning the location (rather than the payload) lets the canonical
        global stream's cluster-side slicing walk every document's *length*
        without faulting the ``.bin`` for documents it won't keep.
        """
        # Rank-aware: we pick every R-th entry of the global perm.
        idx_in_perm = self.consumed * self.world_size + self.rank
        if idx_in_perm >= self._global_perm.size:
            self.epoch += 1
            self.consumed = 0
            self._global_perm = self._build_global_perm(self.epoch)
            idx_in_perm = self.rank
        global_doc = int(self._global_perm[idx_in_perm])
        self.consumed += 1
        shard_id = int(np.searchsorted(self._cum_docs, global_doc, side="right") - 1)
        local = global_doc - int(self._cum_docs[shard_id])
        return shard_id, local

    def next_document(self) -> np.ndarray:
        """Advance the walker by one document this rank should see."""
        shard_id, local = self.next_document_ref()
        return self._readers[shard_id].document(local)

    def reader(self, shard_id: int) -> IndexedDatasetReader:
        return self._readers[shard_id]


class MixSampler:
    """Iterable that yields packed token sequences of length ``seq_len``.

    Tokens are concatenated across documents with an EOS separator; we
    do not bound packing by document. This matches Llama 3 / DCLM /
    FineWeb training recipes.

    Use as ``for batch in MixSampler(...)``. Each yield is a 1-D ``np.ndarray``
    of length ``seq_len + 1`` so the caller can split into ``input/labels``.
    """

    def __init__(
        self,
        manifests: list[SourceManifest],
        manifest_dirs: list[str],
        weights: list[float],
        seq_len: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        eos_id: int = 0,
        state: MixSamplerState | None = None,
    ) -> None:
        if len(manifests) != len(weights) or len(manifests) != len(manifest_dirs):
            raise ValueError("manifests / dirs / weights length mismatch")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self._weights = np.asarray([w / total for w in weights], dtype=np.float64)
        self._eos_id = eos_id
        self._seq_len = seq_len
        self._rng = np.random.default_rng(
            np.array([seed, rank, world_size], dtype=np.uint64)
        )

        consumed = (state.consumed_documents_per_source if state else None) or {}
        epochs = (state.epoch_per_source if state else None) or {}
        self._walkers: list[_SourceWalker] = []
        for m, d in zip(manifests, manifest_dirs):
            self._walkers.append(
                _SourceWalker(
                    manifest=m,
                    manifest_dir=d,
                    seed=seed,
                    rank=rank,
                    world_size=world_size,
                    start_consumed=consumed.get(m.name, 0),
                    start_epoch=epochs.get(m.name, 0),
                )
            )
        # Carry-over buffer: tokens from a partially-consumed document.
        if state and state.carry_over:
            self._buf: list[int] = list(state.carry_over)
        else:
            self._buf = []
        if state and state.mix_rng_state is not None:
            self._rng.bit_generator.state = state.mix_rng_state

    def state(self) -> MixSamplerState:
        return MixSamplerState(
            consumed_documents_per_source={w.name: w.consumed for w in self._walkers},
            epoch_per_source={w.name: w.epoch for w in self._walkers},
            mix_rng_state=dict(self._rng.bit_generator.state),
            carry_over=list(self._buf),
        )

    def _refill(self) -> None:
        """Pull one more document into the buffer; choose the source by weight."""
        src = int(self._rng.choice(len(self._walkers), p=self._weights))
        doc = self._walkers[src].next_document()
        self._buf.extend(int(t) for t in doc.tolist())
        self._buf.append(self._eos_id)

    def __iter__(self) -> Iterator[np.ndarray]:
        target_len = self._seq_len + 1   # +1 so caller can shift for labels
        while True:
            while len(self._buf) < target_len:
                self._refill()
            chunk = self._buf[:target_len]
            self._buf = self._buf[target_len:]
            yield np.asarray(chunk, dtype=np.int64)
