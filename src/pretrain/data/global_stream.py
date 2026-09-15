"""Canonical, topology-independent packed-window stream.

The training data is a single logical sequence of packed windows
``W[0], W[1], W[2], …`` — each a 1-D ``int64`` array of length ``seq_len + 1``
(so the caller can shift into input/labels) — defined purely as a function of
``(seed, manifests, seq_len, eos_id)``. It does **not** depend on world size or
rank. Any topology slices this one stream deterministically (see
``train/batch_schedule.py`` for the global-step → window-index → (rank, accum,
slot) assignment); the single-device audit walks the whole thing in order.

This replaces the per-rank coupling in :mod:`pretrain.data.mix_sampler`, where
the document walk strided by ``consumed*world_size + rank``, the source-mix RNG
was seeded ``[seed, rank, world_size]``, and packing happened in a per-rank
buffer — so two world sizes produced entirely different packed sequences and
there was no canonical "global sequence #i" to audit against.

Implementation reuse: the per-source document walk is exactly the existing
``_SourceWalker`` driven at ``rank=0, world_size=1`` (its stride collapses to
``consumed`` — i.e. the full per-epoch permutation in order). Only two things
change relative to ``MixSampler``: the source-mix RNG is seeded on ``[seed]``
alone, and packing/bookkeeping is global (a ``windows_emitted`` counter anchors
resume).
"""

from __future__ import annotations

import dataclasses
from typing import Iterator

import numpy as np

from pretrain.config.schema import TrainConfig
from pretrain.data.manifest import SourceManifest
from pretrain.data.mix_sampler import _SourceWalker
from pretrain.train.batch_schedule import StepPlan, iter_step_plans, owner_rank


@dataclasses.dataclass
class GlobalStreamState:
    """Everything needed to resume the canonical stream bit-exactly.

    Identical on every rank (it describes the global stream position, not a
    rank's slice), so a checkpoint stores one copy rather than one per rank.
    """

    # Total documents drawn from each source so far (global, not per-rank).
    consumed_documents_per_source: dict[str, int]
    epoch_per_source: dict[str, int]
    # The single global mix-RNG bit-generator state.
    mix_rng_state: dict | None = None
    # Tokens of a partially-consumed document left in the packing buffer at
    # save time; concatenated to the front of the buffer on resume.
    carry_over: list[int] = dataclasses.field(default_factory=list)
    # How many windows ``W[i]`` have been emitted. Resume anchor / cross-check:
    # equals ``windows_consumed_before_step(step)`` derived from the token
    # schedule, so the audit can verify it reconstructed the same position.
    windows_emitted: int = 0


class GlobalStream:
    """Canonical iterable of packed ``seq_len + 1`` windows.

    Pure function of ``(seed, manifests, seq_len, eos_id)``. Use as
    ``for window in GlobalStream(...)`` — each yield is a 1-D ``np.ndarray`` of
    length ``seq_len + 1``. Call :meth:`state` to snapshot for checkpointing.
    """

    def __init__(
        self,
        manifests: list[SourceManifest],
        manifest_dirs: list[str],
        weights: list[float],
        seq_len: int,
        seed: int,
        eos_id: int = 0,
        state: GlobalStreamState | None = None,
    ) -> None:
        if len(manifests) != len(weights) or len(manifests) != len(manifest_dirs):
            raise ValueError("manifests / dirs / weights length mismatch")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self._weights = np.asarray([w / total for w in weights], dtype=np.float64)
        self._eos_id = eos_id
        self._seq_len = seq_len
        # Single global mix RNG — seeded on [seed] only, NOT [seed, rank,
        # world_size]. This is the whole point: one canonical source-draw
        # interleaving for the entire run regardless of topology.
        self._rng = np.random.default_rng(np.array([seed], dtype=np.uint64))

        consumed = (state.consumed_documents_per_source if state else None) or {}
        epochs = (state.epoch_per_source if state else None) or {}
        # rank=0, world_size=1 → the walker's stride collapses to ``consumed``,
        # yielding the full per-epoch permutation in global order.
        self._walkers: list[_SourceWalker] = [
            _SourceWalker(
                manifest=m,
                manifest_dir=d,
                seed=seed,
                rank=0,
                world_size=1,
                start_consumed=consumed.get(m.name, 0),
                start_epoch=epochs.get(m.name, 0),
            )
            for m, d in zip(manifests, manifest_dirs)
        ]
        self._buf: list[int] = list(state.carry_over) if state and state.carry_over else []
        self._windows_emitted = state.windows_emitted if state else 0
        if state and state.mix_rng_state is not None:
            self._rng.bit_generator.state = state.mix_rng_state

    @property
    def windows_emitted(self) -> int:
        return self._windows_emitted

    def state(self) -> GlobalStreamState:
        return GlobalStreamState(
            consumed_documents_per_source={w.name: w.consumed for w in self._walkers},
            epoch_per_source={w.name: w.epoch for w in self._walkers},
            mix_rng_state=dict(self._rng.bit_generator.state),
            carry_over=list(self._buf),
            windows_emitted=self._windows_emitted,
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
            self._windows_emitted += 1
            yield np.asarray(chunk, dtype=np.int64)


# Segment kinds for the length-walk buffer (see ShardedWindowView).
_DOC = 0   # (kind, reader, local, length)
_EOS = 1   # (kind, None, None, 1)
_LIT = 2   # (kind, ndarray, None, length) — resumed carry-over tokens


class ShardedWindowView:
    """One rank's slice of :class:`GlobalStream`, materialised lazily.

    Yields the micro-batches (``[micro_batch_size, seq_len + 1]`` arrays) that
    micro-batch assignment gives to ``rank`` of ``world_size``, in consumption
    order across optimizer steps. The single global window stream is identical
    to :class:`GlobalStream`; this view simply *walks every window* (advancing
    the same source-draw RNG and document walk in lockstep) but reads token
    payloads only for the windows this rank owns — non-owned windows cost only a
    cheap ``.idx`` length lookup (``next_document_ref`` + ``document_length``).

    Because the length-walk is bit-identical on every rank, :meth:`state` returns
    the **global** stream position (same on all ranks) — so a checkpoint stores
    one ``GlobalStreamState`` rather than one per rank, and the single-device
    audit reconstructs it from ``seed + manifests + step`` alone.
    """

    def __init__(
        self,
        manifests: list[SourceManifest],
        manifest_dirs: list[str],
        weights: list[float],
        train: TrainConfig,
        seed: int,
        *,
        rank: int,
        world_size: int,
        micro_batch_size: int | None = None,
        eos_id: int = 0,
        state: GlobalStreamState | None = None,
        start_consumed_tokens: int = 0,
        start_step: int = 0,
        index_only: bool = False,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self._weights = np.asarray([w / total for w in weights], dtype=np.float64)
        self._train = train
        self._seq_len = train.seq_len
        self._mb = micro_batch_size if micro_batch_size is not None else train.micro_batch_size
        self._eos_id = eos_id
        self._rank = rank
        self._world_size = world_size
        self._rng = np.random.default_rng(np.array([seed], dtype=np.uint64))

        consumed = (state.consumed_documents_per_source if state else None) or {}
        epochs = (state.epoch_per_source if state else None) or {}
        self._walkers: list[_SourceWalker] = [
            _SourceWalker(
                manifest=m,
                manifest_dir=d,
                seed=seed,
                rank=0,
                world_size=1,
                start_consumed=consumed.get(m.name, 0),
                start_epoch=epochs.get(m.name, 0),
                index_only=index_only,
            )
            for m, d in zip(manifests, manifest_dirs)
        ]
        if state and state.mix_rng_state is not None:
            self._rng.bit_generator.state = state.mix_rng_state
        # Optional sink for :meth:`walk_doc_refs`: when set, every refilled
        # document's ``(source_name, shard_id, local)`` ref is appended here.
        # ``None`` in the training/audit hot path (no overhead).
        self._ref_sink: list[tuple[str, int, int]] | None = None
        # Optional sink for :meth:`walk_step_spans`: every document fragment a
        # window consumes is appended as
        # ``(global_window, source_name, shard_id, local, tok_start, tok_end)``.
        self._span_sink: list[tuple[int, str, int, int, int, int]] | None = None

        # Length-walk buffer: a list of segments + an offset into the first one.
        self._segs: list[tuple] = []
        self._head_off = 0
        self._buf_len = 0
        if state and state.carry_over:
            arr = np.asarray(state.carry_over, dtype=np.int64)
            self._segs.append((_LIT, arr, None, int(arr.size), None))
            self._buf_len = int(arr.size)

        self._windows_emitted = state.windows_emitted if state else 0
        self._start_consumed_tokens = start_consumed_tokens
        self._start_step = start_step

    @property
    def windows_emitted(self) -> int:
        return self._windows_emitted

    def accum_for_step(self, microbatches: int) -> int:
        """How many micro-batches this rank yields for a step of ``microbatches``."""
        from pretrain.train.batch_schedule import accum_for_rank

        return accum_for_rank(microbatches, world_size=self._world_size, rank=self._rank)

    # -- length-walk buffer ------------------------------------------------- #

    def _refill(self) -> None:
        src = int(self._rng.choice(len(self._walkers), p=self._weights))
        walker = self._walkers[src]
        shard_id, local = walker.next_document_ref()
        if self._ref_sink is not None:
            self._ref_sink.append((walker.name, shard_id, local))
        reader = walker.reader(shard_id)
        doc_len = reader.document_length(local)
        self._segs.append((_DOC, reader, local, doc_len, (walker.name, shard_id)))
        self._segs.append((_EOS, None, None, 1, None))
        self._buf_len += doc_len + 1

    def _take(self, n: int, *, materialize: bool) -> np.ndarray | None:
        """Consume ``n`` tokens from the buffer front. Read payload only if asked."""
        pieces: list[np.ndarray] = [] if materialize else None
        remaining = n
        while remaining > 0:
            kind, a, b, length, ref = self._segs[0]
            avail = length - self._head_off
            take = min(avail, remaining)
            if self._span_sink is not None and kind == _DOC:
                self._span_sink.append(
                    (self._windows_emitted, ref[0], ref[1], b, self._head_off, self._head_off + take)
                )
            if materialize:
                if kind == _DOC:
                    doc = a.document(b)  # mmap view, length == `length`
                    pieces.append(
                        np.asarray(doc[self._head_off : self._head_off + take], dtype=np.int64)
                    )
                elif kind == _EOS:
                    pieces.append(np.full(take, self._eos_id, dtype=np.int64))
                else:  # _LIT
                    pieces.append(a[self._head_off : self._head_off + take].astype(np.int64))
            self._head_off += take
            remaining -= take
            if self._head_off == length:
                self._segs.pop(0)
                self._head_off = 0
        self._buf_len -= n
        if materialize:
            return np.concatenate(pieces)
        return None

    def _take_window(self, *, materialize: bool) -> np.ndarray | None:
        target_len = self._seq_len + 1
        while self._buf_len < target_len:
            self._refill()
        win = self._take(target_len, materialize=materialize)
        self._windows_emitted += 1
        return win

    # -- state -------------------------------------------------------------- #

    def _materialize_buffer(self) -> list[int]:
        """Read the (small, < seq_len+1) leftover buffer into token values.

        Identical on every rank, so it round-trips into the single global
        ``carry_over``. Reads at most one window's worth of payload.
        """
        out: list[int] = []
        off = self._head_off
        for kind, a, b, length, _ref in self._segs:
            if kind == _DOC:
                out.extend(int(t) for t in a.document(b)[off:length].tolist())
            elif kind == _EOS:
                if off < length:
                    out.append(self._eos_id)
            else:  # _LIT
                out.extend(int(t) for t in a[off:length].tolist())
            off = 0
        return out

    def state(self) -> GlobalStreamState:
        return GlobalStreamState(
            consumed_documents_per_source={w.name: w.consumed for w in self._walkers},
            epoch_per_source={w.name: w.epoch for w in self._walkers},
            mix_rng_state=dict(self._rng.bit_generator.state),
            carry_over=self._materialize_buffer(),
            windows_emitted=self._windows_emitted,
        )

    # -- iteration ---------------------------------------------------------- #

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield this rank's micro-batches across successive optimizer steps.

        Each step is walked in full — all ``M`` micro-batches, materialising the
        owned ones and length-skipping the rest — *before* its owned micro-batches
        are yielded. That keeps the internal window cursor on a step boundary once
        the caller has consumed a step's worth, so :meth:`state` taken between
        steps captures the exact global stream position (the trailing non-owned
        windows of the step are already accounted for).
        """
        plans = iter_step_plans(
            self._start_consumed_tokens,
            self._windows_emitted,
            self._train,
            start_step=self._start_step,
        )
        for plan in plans:
            owned_microbatches: list[np.ndarray] = []
            for m in range(plan.microbatches):
                owned = owner_rank(m, world_size=self._world_size) == self._rank
                windows = [self._take_window(materialize=owned) for _ in range(self._mb)]
                if owned:
                    owned_microbatches.append(np.stack(windows, axis=0))  # [mb, seq_len+1]
            yield from owned_microbatches

    # -- audit-data fetch: enumerate the documents an interval consumes ------ #

    def walk_doc_refs(
        self,
        *,
        target_consumed_tokens: int | None = None,
        until_step: int | None = None,
    ) -> Iterator[tuple[str, int, int]]:
        """Yield ``(source_name, shard_id, local)`` for every document the global
        stream refills while advancing from this view's start position up to a
        stop point — without reading any ``.bin`` payload.

        This is the canonical-stream length-walk: it drives the *same*
        :meth:`_refill` / window cursor as :meth:`__iter__` (so the documents it
        enumerates are exactly those the cluster — and the single-device audit —
        consume over the interval), but never materialises tokens. Build the view
        with ``index_only=True`` so it runs against ``.idx`` alone.

        Stop condition mirrors ``audit_replay``'s ``_done``: process whole
        optimizer steps while ``step < until_step`` (if given) else while
        ``consumed_tokens < target_consumed_tokens``. Exactly one of the two must
        be supplied. Documents only *partially* consumed by the final step are
        still yielded (their refill happened), matching what the audit reads.

        The view is single-use: this advances the walkers/RNG just like
        iteration would.
        """
        if (target_consumed_tokens is None) == (until_step is None):
            raise ValueError(
                "walk_doc_refs needs exactly one of target_consumed_tokens / until_step"
            )
        # Both sinks share the one cursor: a second walk would advance it under
        # the first and attribute refs or spans to the wrong steps.
        if self._span_sink is not None or self._ref_sink is not None:
            raise RuntimeError("view is already being walked")
        self._ref_sink = []
        try:
            plans = iter_step_plans(
                self._start_consumed_tokens,
                self._windows_emitted,
                self._train,
                start_step=self._start_step,
            )
            for plan in plans:
                done = (
                    plan.step >= until_step
                    if until_step is not None
                    else plan.consumed_tokens >= target_consumed_tokens
                )
                if done:
                    break
                # Walk every window of the step (all micro-batches), length-only.
                for _ in range(plan.microbatches * self._mb):
                    self._take_window(materialize=False)
                while self._ref_sink:
                    yield self._ref_sink.pop(0)
        finally:
            # An abandoned generator must not leave the view looking walked.
            self._ref_sink = None

    @property
    def has_carry_over(self) -> bool:
        """True when the buffer holds resumed literal tokens with no document behind them."""
        return any(seg[0] == _LIT for seg in self._segs)

    def walk_step_spans(
        self, *, until_step: int
    ) -> Iterator[tuple[StepPlan, list[tuple[int, str, int, int, int, int]]]]:
        """Yield ``(plan, spans)`` for every optimizer step ``< until_step``.

        ``spans`` lists, in stream order, every document fragment packed into
        the step's windows as ``(global_window, source_name, shard_id, local,
        tok_start, tok_end)`` — the half-open token range of that document
        that landed in that window. EOS separators are not listed; a window's
        fragment lengths plus its EOS tokens sum to ``seq_len + 1``.

        Drives the same ``_refill`` / ``_take`` cursor as :meth:`__iter__` and
        :meth:`walk_doc_refs`, so the fragments are exactly what the cluster
        trained on. Works with ``index_only=True``: only ``.idx`` is read.
        Single-use, like :meth:`walk_doc_refs`.

        A view resumed from a checkpoint ``state`` is refused: its carry-over
        tokens are literal ids with no document behind them, so the first
        window would be short of fragments with nothing to say so. Attribute a
        resumed run by walking a fresh view from step 0.
        """
        if self._span_sink is not None or self._ref_sink is not None:
            raise RuntimeError("view is already being walked")
        if self.has_carry_over:
            raise ValueError(
                "walk_step_spans cannot attribute resumed carry-over tokens to documents; "
                "build the view without state and walk from step 0"
            )
        self._span_sink = []
        try:
            for plan in iter_step_plans(
                self._start_consumed_tokens,
                self._windows_emitted,
                self._train,
                start_step=self._start_step,
            ):
                if plan.step >= until_step:
                    break
                for _ in range(plan.microbatches * self._mb):
                    self._take_window(materialize=False)
                spans = self._span_sink
                self._span_sink = []
                yield plan, spans
        finally:
            self._span_sink = None

    def position(self) -> dict:
        """Stream position without the carry-over buffer, so it is safe on an
        ``index_only`` view. Same fields as ``global_stream.json`` minus
        ``mix_rng_state`` / ``carry_over``."""
        return {
            "consumed_documents_per_source": {w.name: w.consumed for w in self._walkers},
            "epoch_per_source": {w.name: w.epoch for w in self._walkers},
            "windows_emitted": self._windows_emitted,
        }
