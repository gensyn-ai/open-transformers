"""Topology-independent gradient reduce-scatter for auditable runs.

FSDP2's default reduce-scatter routes through NCCL, whose cross-rank summation
order depends on the world size and the chosen algorithm/channels. Float
addition isn't associative, so the reduced gradient's *bits* differ across
device counts (verified at N>=4: a single-device sum reproduces NCCL's result to
~1e-11 but not bitwise — see ``scripts/repro/``). That defeats auditing a cluster
run on a single device.

:class:`DeterministicReduceScatter` replaces the comm with one whose summation
order is fixed (ascending rank), independent of topology: an all-to-all routes
each rank's bucket chunks to their destination ranks, then each rank sums the
per-element contributions it received in rank order (and divides by world size
for AVG). The single-device audit reconstructs the same value by summing the N
virtual ranks' partials in the same ascending order — bitwise.

It is wired in only when ``run.reduction_mode == "deterministic_allgather"`` (the
default ``"nccl"`` path is untouched; the config name is historical — the
mechanism is now all-to-all, not all-gather), via FSDP2's public
``FSDPModule.set_custom_reduce_scatter``. Cost vs. native reduce-scatter: the
cross-rank summation runs locally after the transfer instead of being pipelined
into it; the data volume (``(ws-1)·k`` received per rank) is the same — acceptable
for auditable runs.

Validated bitwise at N=2 and N=4 on H100 in ``scripts/repro/validate_deterministic_rs.py``.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

LOG = logging.getLogger(__name__)

# Diagnostic timing for the cross-pod (replicate) reduction. Gated by the SAME
# env var as the loop's fwd/bwd CUDA-phase timings (PRETRAIN_CUDA_EVENT_TIMING),
# so one flag turns on all per-step timing — and the per-step host sync those
# phase metrics already pay covers this too. When on, the hook records CUDA
# events per call; the training loop drains them once per step via
# ``pop_stats()`` and logs the cross-replica reduction cost to W&B alongside
# ``phase_*_ms``. Off by default (it perturbs comm/compute overlap slightly).
_CUDA_EVENT_TIMING = os.environ.get("PRETRAIN_CUDA_EVENT_TIMING", "0") == "1"

# The pluggable ReduceScatter comm interface + ``set_custom_reduce_scatter`` are
# a newer FSDP2 capability (torch >= ~2.8; present in the cluster's 2.11 NGC
# build, absent in the <2.8 CPU wheel the ``dev`` extra pins). Import it if
# available; otherwise fall back to a plain base so this module still imports on
# older torch — using the feature then raises a clear error.
try:  # public alias, if a future torch exposes it
    from torch.distributed.fsdp import ReduceScatter  # type: ignore

    _HAVE_CUSTOM_REDUCE_SCATTER = True
except ImportError:
    try:  # torch 2.8–2.11: internal home
        from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatter

        _HAVE_CUSTOM_REDUCE_SCATTER = True
    except ImportError:  # torch < 2.8: no pluggable reduce-scatter
        ReduceScatter = object  # type: ignore
        _HAVE_CUSTOM_REDUCE_SCATTER = False


class DeterministicReduceScatter(ReduceScatter):
    """Reduce-scatter that sums in ascending rank order, then /world_size for AVG.

    Bitwise-identical regardless of world size, so a checkpoint produced under
    any topology can be advanced to the next checkpoint on a single device (which
    sums the same per-rank partials in the same order) and match exactly.
    """

    def allocate(self, size, dtype, device):  # required by the Comm interface
        return torch.empty(size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op,
        async_op: bool = False,
    ):
        ws = group.size()
        k = output_tensor.numel()
        # Decompose reduce-scatter as all-to-all (transpose) + local sum. The input
        # is the unsharded bucket, laid out as ws chunks of k where chunk d is
        # destined for rank d. all_to_all_single (equal split) sends chunk d to
        # rank d and hands this rank chunk r from every source — so we receive
        # exactly the ws·k elements we need and (ws-1)·k over the wire, vs. the old
        # all-gather's ws²·k buffer and (ws-1)·ws·k received (of which (ws-1)/ws was
        # discarded). all_to_all is pure data movement (no arithmetic) → bitwise
        # deterministic and topology-independent, exactly like the all-gather it
        # replaces, so the summation below is unchanged and the audit still matches.
        recvd = torch.empty(ws * k, dtype=input_tensor.dtype, device=input_tensor.device)
        dist.all_to_all_single(recvd, input_tensor.contiguous(), group=group)
        # recvd[src] = source ``src``'s contribution destined for this rank.
        contribs = recvd.view(ws, k)  # [src, k]
        acc = contribs[0].clone()
        for s in range(1, ws):
            acc.add_(contribs[s])  # fixed ascending-rank order, in place (no temporaries)
        if op == dist.ReduceOp.AVG:
            # AVG by × (1/ws), NOT acc.div_(ws). An fp32 tensor ÷ python-scalar is
            # lowered to a reciprocal-multiply on CUDA but to a correctly-rounded
            # divide on CPU, so the two disagree by 1 ULP for non-power-of-2 ws
            # (e.g. /6 on ~1/3 of elements; /2,/4,/8 are exact). The explicit
            # × (1.0/ws) is a reciprocal-multiply on every backend, so the gradient
            # is bit-identical across CUDA/CPU/MPS and the single-device audit
            # reproduces it. (Mirrors repop.qat.lsq's "× const, not ÷".)
            acc.mul_(1.0 / ws)
        output_tensor.copy_(acc)
        if async_op:
            fut: torch.futures.Future = torch.futures.Future()
            fut.set_result(output_tensor)
            return _CompletedWork(fut)
        return None


class _CompletedWork:
    """Minimal Work-like object for the synchronous-but-async_op=True path."""

    def __init__(self, fut: "torch.futures.Future") -> None:
        self._fut = fut

    def wait(self) -> None:
        return None

    def get_future(self) -> "torch.futures.Future":
        return self._fut


# Algorithm tag persisted in CheckpointMeta.replicate_reduce_algo so the audit
# replays the SAME cross-replica combination order a given checkpoint was
# trained with. Bump this string if the order ever changes again; the audit
# branches on it and treats a missing field as the legacy ascending all-gather
# (older checkpoints predate this field). See cli/audit_replay.py.
REPLICATE_REDUCE_ALGO = "recursive_doubling"


def _binary_blocks(n: int) -> list[tuple[int, int]]:
    """Decompose ``n`` ranks into consecutive power-of-2 blocks, largest first.

    One block per set bit of ``n``, in descending size order, so the block starts
    are ascending and contiguous. ``6 = 0b110 -> [(0, 4), (4, 2)]``; ``7 = 0b111
    -> [(0, 4), (4, 2), (6, 1)]``; a power of two is a single block ``[(0, n)]``.
    This is the canonical partition shared by :func:`tree_reduce_sum` (the audit
    reference) and :func:`_recursive_doubling_allreduce_avg` (the cluster path),
    so both realise the identical combination tree for any ``n``.
    """
    blocks: list[tuple[int, int]] = []
    start = 0
    bit = 1 << (n.bit_length() - 1)
    while bit:
        if n & bit:
            blocks.append((start, bit))
            start += bit
        bit >>= 1
    return blocks


def tree_reduce_sum(parts: list[torch.Tensor]) -> torch.Tensor:
    """Binary-blocks fold — the exact combination order produced by
    :class:`DeterministicReplicateAllReduce`'s recursive-doubling all-reduce.

    Partitions ``parts`` into the :func:`_binary_blocks` of ``len(parts)``; each
    block is reduced as a balanced adjacent-pair tree ``((g0+g1)+(g2+g3)) + ...``
    (lower index always the left operand), then the block sums are combined by an
    ascending left-fold (largest/lowest-rank block first). For a power-of-2 count
    this is exactly the balanced tree ``((g0+g1)+(g2+g3))+...`` — unchanged from
    before; ``6`` folds as ``((g0+g1)+(g2+g3)) + (g4+g5)``. Float add is IEEE-
    commutative (``a+b`` is bit-equal to ``b+a``) but **not** associative, so this
    grouping is what the single-device audit must replay to match the cluster
    bitwise.
    """
    n = len(parts)
    if n == 0:
        raise ValueError("tree_reduce_sum: empty input")
    acc = None
    for start, size in _binary_blocks(n):
        level = list(parts[start : start + size])  # one power-of-2 block
        while len(level) > 1:
            level = [level[i] + level[i + 1] for i in range(0, len(level), 2)]
        # Ascending left-fold across blocks: lower-rank block stays on the left.
        acc = level[0] if acc is None else acc + level[0]
    return acc


def _p2p_full(op, peer_global: int, flat: torch.Tensor, grp) -> None:
    """One blocking p2p of the whole buffer (``op`` is ``dist.isend``/``irecv``)."""
    for work in dist.batch_isend_irecv([dist.P2POp(op, flat, peer_global, group=grp)]):
        work.wait()


def _binomial_broadcast(
    flat: torch.Tensor, grp: dist.ProcessGroup, r: int, ws: int
) -> tuple[torch.Tensor, int]:
    """Broadcast ``flat`` from in-group rank 0 to all ranks (binomial tree).

    Pure data movement — no arithmetic — so every rank ends bit-identical to
    rank 0's buffer. ``partner < ws`` guards make the standard root-0 binomial
    tree valid for non-power-of-2 ``ws``. Returns ``(buf, p2p_steps)``.
    """
    steps = 0
    mask = 1
    while mask < ws:  # receive phase: take the buffer once, from r - lowest_set_bit
        if r & mask:
            recv = torch.empty_like(flat)
            _p2p_full(dist.irecv, dist.get_global_rank(grp, r - mask), recv, grp)
            flat = recv
            steps += 1
            break
        mask <<= 1
    mask >>= 1
    while mask > 0:  # send phase: forward to r + mask for each lower bit
        if r + mask < ws:
            _p2p_full(dist.isend, dist.get_global_rank(grp, r + mask), flat, grp)
            steps += 1
        mask >>= 1
    return flat, steps


def _recursive_doubling_allreduce_avg(
    flat: torch.Tensor, grp: dist.ProcessGroup, ws: int
) -> tuple[torch.Tensor, int]:
    """Recursive-doubling all-reduce-AVG on a 1D buffer. Returns ``(reduced, steps)``.

    The combination tree is :func:`tree_reduce_sum`'s binary-blocks fold — the
    ranks split into the :func:`_binary_blocks` of ``ws`` (largest first), each a
    power-of-2 block. **Phase A** runs the classic recursive-doubling butterfly
    *within* each block: at step ``k`` every rank exchanges its whole buffer with
    block-local partner ``local ^ 2^k`` (mapped to that partner's GLOBAL rank so
    the p2p ops ride the replicate subgroup's own communicator and stay isolated
    from default-PG collectives) and computes ``own + recv``. IEEE add is
    commutative so partners hold bit-identical values at every step, and every
    rank finishes with its block's balanced-tree sum. For a power-of-2 ``ws`` this
    is a single block and is byte-identical to the original pure-butterfly path —
    so existing power-of-2 checkpoints reduce exactly as before.

    **Phase B** (only when ``ws`` is not a power of two, i.e. >1 block) combines
    the block sums into the total on rank 0 in **ascending block order** — the
    same lower-rank-left fold :func:`tree_reduce_sum` and the single-device audit
    replay — then binomially broadcasts the total back so all ranks finish
    bit-identical. E.g. ``ws=6`` reduces as ``((g0+g1)+(g2+g3)) + (g4+g5)``.

    Crucially the reduction stays **element-wise**: element ``i`` only ever
    combines with element ``i`` on other ranks, through a tree fixed by rank
    indices alone. So the per-element result is independent of the buffer's length
    and of how separate gradients are packed into it — which is what makes
    coalescing many modules' shards into one buffer (the bucketed flush) bitwise-
    identical to reducing each module's shard separately.
    """
    r = grp.rank()
    blocks = _binary_blocks(ws)
    my_start, my_size = next(
        (s, sz) for s, sz in blocks if s <= r < s + sz
    )
    recv = torch.empty_like(flat)
    steps = 0
    # Phase A: balanced recursive-doubling within my power-of-2 block.
    local = r - my_start
    mask = 1
    while mask < my_size:
        partner_global = dist.get_global_rank(grp, my_start + (local ^ mask))
        ops = [
            dist.P2POp(dist.isend, flat, partner_global, group=grp),
            dist.P2POp(dist.irecv, recv, partner_global, group=grp),
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        flat = flat + recv  # own + partner (commutative ⇒ identical on both)
        steps += 1
        mask <<= 1
    # Phase B: cross-block combine on rank 0 (ascending) + broadcast back.
    if len(blocks) > 1:
        if r == 0:
            for start, _size in blocks[1:]:
                _p2p_full(dist.irecv, dist.get_global_rank(grp, start), recv, grp)
                flat = flat + recv  # acc + block sum (lower-rank acc on the left)
                steps += 1
        elif r == my_start:  # block representative: hand its sum to rank 0
            _p2p_full(dist.isend, dist.get_global_rank(grp, 0), flat, grp)
        flat, bsteps = _binomial_broadcast(flat, grp, r, ws)
        steps += bsteps
    # AVG by × (1/ws), NOT flat / ws — fp32 ÷ python-scalar is a reciprocal-multiply
    # on CUDA but a correctly-rounded divide on CPU, differing by 1 ULP for
    # non-power-of-2 ws. The explicit reciprocal-multiply is portable across
    # backends, so the cross-replica-averaged gradient reproduces bitwise on a
    # single-device audit. See DeterministicReduceScatter for the full rationale.
    return flat * (1.0 / ws), steps  # AVG across replicas


class DeterministicReplicateAllReduce:
    """Fixed-order cross-replica all-reduce for HSDP (the ``set_all_reduce_hook``).

    Native HSDP does the cross-replica reduction with an NCCL all-reduce whose
    order isn't reproducible on a single device for >2 replicas (and FSDP2's hook
    is post-only, so it can't be replaced there). Instead we wrap FSDP with a 1D
    *shard* mesh — so its reduce-scatter (made deterministic separately) covers
    only sharding — and drive the replicate reduction through this hook, which
    FSDP2 invokes post-reduce-scatter.

    The reduction is a **recursive-doubling all-reduce** over the replicate group
    (see :func:`_recursive_doubling_allreduce_avg`): for a power-of-2 group of size
    ``P`` it runs ``log2(P)`` butterfly steps. This cut the inter-node volume from
    the old all-gather's ``P·shard`` (received per rank) to ``log2(P)·shard`` and
    the serial adds from ``P-1`` to ``log2(P)``. Non-power-of-2 ``P`` is supported
    via a binary-blocks fold (butterfly within each power-of-2 block, then a small
    cross-block combine + broadcast); cost stays ``~log2(P)`` plus a couple of
    full-buffer transfers, and ``P=2^k`` is unchanged.

    **Bucketing.** FSDP2 fires the all-reduce hook once per ``fully_shard`` module
    (~one per transformer block + embeddings — dozens for a real model). Running a
    separate butterfly per module means dozens of independent, latency-bound
    inter-node round-trips per step — the dominant HSDP cost on a socket-NCCL
    cluster with no GPUDirect. In the default **bucketed** mode the per-module hook
    only *stashes* each module's reduce-scattered shard during backward; the
    training loop then calls :meth:`flush` once after backward to coalesce every
    stashed shard into a single contiguous buffer and run ONE butterfly, cutting
    the per-step round-trips from ``n_modules·log2(P)`` to ``log2(P)``. Because the
    butterfly is element-wise (above), the coalesced reduction is bitwise-identical
    to the per-module one, so the single-device audit is unaffected
    (``replicate_reduce_algo`` stays ``recursive_doubling``; no hash change).
    ``bucketed=False`` restores the immediate per-module path (for A/B timing).

    Determinism relies on a power-of-2 replicate degree — validated in
    :func:`apply_deterministic_replicate_all_reduce`.
    """

    def __init__(
        self, replicate_group: dist.ProcessGroup, *, bucketed: bool = True
    ) -> None:
        self._group = replicate_group
        self._bucketed = bucketed
        # Bucketed mode: each module's post-reduce-scatter shard is stashed here
        # during backward (keyed by install index so the coalesced buffer's layout
        # is identical on every rank regardless of the order FSDP fired the hooks)
        # and reduced in one collective at flush().
        self._pending: dict[int, torch.Tensor] = {}
        # Timing state (only touched when ``_CUDA_EVENT_TIMING``). One hook
        # instance is shared across every FSDP module, so these counters
        # aggregate over all modules within a step; the training loop drains
        # them once per step via ``pop_stats()``.
        self._timing = _CUDA_EVENT_TIMING and torch.cuda.is_available()
        self._bytes = 0  # inter-node bytes moved per rank this window (log2(P)·shard)
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        # How many FSDP modules installed this hook — set by the installer for
        # the startup log line. Not used in the hot path.
        self.n_installed = 0

    def on_module(self, idx: int, reduce_output: torch.Tensor) -> None:
        """Per-module hook body (installed once per ``fully_shard`` module).

        Bucketed: stash this module's reduce-scattered shard for the coalesced
        :meth:`flush`. Immediate: reduce it now (legacy per-module path)."""
        if self._group.size() <= 1:
            return
        if self._bucketed:
            self._pending[idx] = reduce_output
        else:
            self._reduce_buffers([reduce_output])

    def flush(self) -> None:
        """Run the single coalesced cross-replica all-reduce over every stashed
        shard, in place. The training loop calls this once per optimizer step
        after backward and before the grad-norm fold / optimizer read ``.grad``
        (which aliases these reduce-scatter buffers via ``torch.as_strided`` —
        an in-place write here is therefore seen by ``.grad``). No-op when not
        bucketed, ``dp_replicate==1``, or nothing was stashed (e.g. a micro-batch
        that didn't trigger a grad sync)."""
        if not self._bucketed or self._group.size() <= 1 or not self._pending:
            self._pending.clear()
            return
        # Ascending install index ⇒ identical buffer layout on every rank, so the
        # element-wise butterfly's cross-rank correspondence holds regardless of
        # the (reverse-graph) order FSDP fired the per-module hooks during backward.
        bufs = [self._pending[i] for i in sorted(self._pending)]
        self._reduce_buffers(bufs)
        self._pending.clear()

    def __call__(self, reduce_output: torch.Tensor) -> None:
        """Immediately reduce a single buffer in place. The direct-call entry
        point used by the immediate (non-bucketed) install and the repro tests."""
        if self._group.size() <= 1:
            return
        self._reduce_buffers([reduce_output])

    def _reduce_buffers(self, bufs: list[torch.Tensor]) -> None:
        """Coalesce ``bufs`` into one contiguous staging buffer, run a single
        recursive-doubling all-reduce-AVG, and scatter the result back into each
        buffer in place. One buffer = the immediate per-module path; many = the
        bucketed path — bitwise-identical either way, since the butterfly is
        element-wise (see :func:`_recursive_doubling_allreduce_avg`): concatenation
        only changes message packing, never a single element's float adds."""
        ws = self._group.size()
        if self._timing:
            e_start = torch.cuda.Event(enable_timing=True)
            e_end = torch.cuda.Event(enable_timing=True)
            e_start.record()
        flats = [b.contiguous().reshape(-1) for b in bufs]
        sizes = [f.numel() for f in flats]
        # cat copies (multi); clone keeps the single-buffer input intact until the
        # copy-back. Either way the butterfly runs on a private staging buffer.
        staging = torch.cat(flats) if len(flats) > 1 else flats[0].clone()
        reduced, steps = _recursive_doubling_allreduce_avg(staging, self._group, ws)
        off = 0
        for b, sz in zip(bufs, sizes):
            b.copy_(reduced[off : off + sz].view_as(b))  # in place ⇒ .grad sees it
            off += sz
        if self._timing:
            e_end.record()
            self._events.append((e_start, e_end))
            # Each butterfly step sends+receives the whole (coalesced) buffer once
            # → per-call inter-node volume is log2(P)·numel.
            self._bytes += steps * reduced.numel() * reduced.element_size()

    def pop_stats(self) -> dict[str, float] | None:
        """Resolve this step's pending CUDA events and return aggregated timing,
        then reset. Called once per optimizer step by the training loop when
        ``PRETRAIN_CUDA_EVENT_TIMING`` is on, so the cross-replica reduction cost
        lands in W&B next to the fwd/bwd phase timings.

        ``replicate_reduce_ms`` is the summed wall time of every replicate
        all-reduce in the step (one per FSDP module); ``..._MB`` is the total
        inter-node bytes moved per rank. Returns ``None`` when timing is off or
        no reduction ran this step (e.g. ``dp_replicate == 1``).

        Must be called on EVERY rank when timing is on (not just rank 0) — it's
        what drains the accumulated CUDA events, so skipping it on a rank would
        leak events. Syncing the last event suffices: all pairs were recorded
        in order on the same stream, so the final ``end`` firing implies all
        prior events have too.
        """
        if not self._timing or not self._events:
            return None
        self._events[-1][1].synchronize()
        total_ms = sum(s.elapsed_time(e) for s, e in self._events)
        n = len(self._events)
        moved_mb = self._bytes / 1e6
        self._events.clear()
        self._bytes = 0
        return {
            "replicate_reduce_ms": total_ms,
            "replicate_reduce_calls": float(n),
            "replicate_reduce_MB": moved_mb,
        }


def _make_module_hook(hook: DeterministicReplicateAllReduce, idx: int):
    """A per-module ``set_all_reduce_hook`` callable that tags its module with a
    stable index, so :meth:`DeterministicReplicateAllReduce.flush` can order the
    coalesced buffer identically on every rank. ``idx`` is bound per closure."""

    def _hook(reduce_output: torch.Tensor) -> None:
        hook.on_module(idx, reduce_output)

    return _hook


def apply_deterministic_replicate_all_reduce(
    modules, replicate_group, *, bucketed: bool | None = None
) -> DeterministicReplicateAllReduce:
    """Install :class:`DeterministicReplicateAllReduce` on each FSDP2 module.

    Used for HSDP runs (``dp_replicate > 1``) under deterministic reduction: the
    modules are wrapped with a 1D shard mesh and this hook does the cross-replica
    reduction deterministically. Returns the (single, shared) hook instance so
    the caller can stash it for the per-step :meth:`DeterministicReplicateAllReduce.flush`
    and timing drains; ``hook.n_installed`` is the number of modules it was
    attached to.

    ``bucketed`` (default: env ``PRETRAIN_REPLICATE_BUCKET`` != "0", i.e. on)
    coalesces every module's cross-replica all-reduce into a single per-step
    collective; ``False`` keeps the legacy per-module reduction. Bitwise-identical
    either way (see the class docstring) — the toggle exists only for A/B timing.

    Any replicate degree ``>= 1`` is supported: power-of-2 runs a single
    butterfly, non-power-of-2 a binary-blocks fold (see
    :func:`_recursive_doubling_allreduce_avg`). Both realise the
    :func:`tree_reduce_sum` combination order the single-device audit replays.
    """
    if not _HAVE_CUSTOM_REDUCE_SCATTER:
        raise RuntimeError(
            "deterministic HSDP needs FSDP2's set_all_reduce_hook (torch >= ~2.8)."
        )
    if bucketed is None:
        bucketed = os.environ.get("PRETRAIN_REPLICATE_BUCKET", "1") != "0"
    hook = DeterministicReplicateAllReduce(replicate_group, bucketed=bucketed)
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    n = 0
    for m in modules:
        if hasattr(m, "set_all_reduce_hook"):
            # n is the install index (contiguous 0..n-1, identical on every rank).
            m.set_all_reduce_hook(_make_module_hook(hook, n), stream=stream)
            n += 1
    hook.n_installed = n
    LOG.info(
        "deterministic replicate all-reduce: bucketed=%s (%d modules → %s)",
        bucketed, n, "1 coalesced collective/step" if bucketed else "per-module",
    )
    return hook


def apply_deterministic_reduce_scatter(modules) -> int:
    """Install :class:`DeterministicReduceScatter` on each FSDP2-wrapped module.

    ``modules`` is the iterable of ``fully_shard``-wrapped modules (each owns a
    reduce-scatter comm). Returns the count installed. No-op for modules without
    the setter (e.g. world_size==1, where there is no reduce-scatter).
    """
    if not _HAVE_CUSTOM_REDUCE_SCATTER:
        raise RuntimeError(
            "run.reduction_mode='deterministic_allgather' needs FSDP2's pluggable "
            "reduce-scatter (set_custom_reduce_scatter), available in torch >= ~2.8. "
            f"This torch ({torch.__version__}) lacks it — use the cluster NGC image."
        )
    comm = DeterministicReduceScatter()
    n = 0
    for m in modules:
        if hasattr(m, "set_custom_reduce_scatter"):
            m.set_custom_reduce_scatter(comm)
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Topology-invariant gradient-norm fold (used by gradient clipping AND the
# spike protocol's grad-norm trigger).
#
# The global grad norm is the one reduction the deterministic reduce-scatter /
# replicate all-reduce above do NOT cover. The norm goes through
# ``DTensor.full_tensor()``, an NCCL all-reduce of the per-shard partial norms
# whose summation order is topology-dependent — so the clip coefficient (and any
# near-threshold spike decision) differs bitwise across device counts and can't
# be reproduced on the single-device audit. That is why clipping has had to run
# at an effectively-disabled value and the spike threshold kept very high.
#
# We replace it with a fixed ascending-shard fold of per-shard sums-of-squares.
# Key property for HSDP: the cross-replica axis is NOT a mesh dim on the
# auditable path (it is driven by ``DeterministicReplicateAllReduce``'s hook),
# so it never enters this fold — after that all-reduce every replica holds
# identical shards and independently reaches the same norm. Only the intra-pod
# shard axis (NVLink-local) is folded, so the internode reduction stays free.
#
# Canonical contract (the cluster and the single-device audit MUST agree
# bitwise):
#     ‖g‖ = sqrt( Σ_{r ascending} s_r ),
#     s_r = Σ_{p in named_parameters order} ‖shard_r(p.grad)‖²
# with:
#   * ‖·‖² via ``repop.ops.sum_dim`` — a fixed-order fp32 reduction, byte-equal
#     cpu/cuda/mps (``torch.dot``/``torch.sum`` pick a device-specific tree and
#     broke the single-device audit);
#   * ``shard_r`` matching DTensor ``Shard(0)`` exactly (ceil split, zero-padded
#     tail). The pad adds 0.0 to the dot AND keeps the reduction-kernel input
#     shape identical on both sides, so the kernel's internal tree matches;
#   * cross-shard and cross-param sums done explicitly left-to-right (no
#     kernel-chosen reduction tree), so the audit reproduces the grouping.
#
# Bump GRAD_NORM_ALGO if any of the above changes; it is persisted in
# CheckpointMeta and the audit branches on it (older checkpoints, which used the
# non-deterministic full_tensor norm, record the absence as "full_tensor").
GRAD_NORM_ALGO = "ascending_shard_sos_v2_bfr_sumdim"


def grad_sum_of_squares(t: torch.Tensor) -> torch.Tensor:
    """``‖t‖²`` via repop's fixed-order reduction.

    NOT ``torch.dot``/``torch.sum``: those pick a device-specific reduction tree
    (cuda/cpu/mps each fold in a different order), so the grad norm — and the
    gradient clipping it drives — diverged across devices and broke the
    single-device audit. ``repop.ops.sum_dim`` folds in a fixed order, byte
    identical cpu/cuda/mps.
    """
    import repop.ops as repop_ops

    flat = t.reshape(-1)
    if flat.dtype != torch.float32:
        flat = flat.float()
    if flat.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=t.device)
    return repop_ops.sum_dim((flat * flat).view(1, -1), dim=1).reshape(())


def shard0_full_chunk_size(dim0: int, world_size: int) -> int:
    """DTensor ``Shard(0)`` per-rank shard length: ``ceil(dim0 / world_size)``."""
    n = max(world_size, 1)
    return (dim0 + n - 1) // n


def shard0_chunk(t: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """The ``rank``-th DTensor ``Shard(0)`` local shard of ``t``.

    Zero-padded to the full chunk size so its shape — and therefore the dot
    reduction tree — matches FSDP2's padded local shard on the cluster exactly.
    Returns a view when no padding is needed (the common, allocation-free case).
    """
    fcs = shard0_full_chunk_size(t.shape[0], world_size)
    start = rank * fcs
    chunk = t[start : start + fcs]
    if chunk.shape[0] < fcs:  # tail shard: pad with zeros like DTensor
        pad = torch.zeros(
            (fcs - chunk.shape[0], *t.shape[1:]), dtype=t.dtype, device=t.device
        )
        chunk = torch.cat([chunk, pad], dim=0) if chunk.shape[0] else pad
    return chunk


def shard0_logical_chunk(t: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """The ``rank``-th DTensor ``Shard(0)`` local shard of ``t`` WITHOUT the
    tail-rank zero padding — i.e. only the rows that logically belong to this
    rank (``t[rank*fcs : min((rank+1)*fcs, dim0)]``), which may be fewer than
    ``fcs`` on the last rank and empty when ``dim0 < rank*fcs``.

    Use this (not :func:`shard0_chunk`) anywhere the *bytes* are consumed —
    e.g. the sharded state hash — so the result never depends on FSDP2's pad
    fill value (which is not guaranteed zero across torch versions). Always a
    view; never allocates.
    """
    fcs = shard0_full_chunk_size(t.shape[0], world_size)
    start = rank * fcs
    return t[start : min(start + fcs, t.shape[0])]


def _ascending_sum(parts: list[torch.Tensor]) -> torch.Tensor:
    """Explicit left-to-right sum (index 0 first) — the order the audit replays."""
    acc = parts[0].clone()
    for p in parts[1:]:
        acc = acc + p
    return acc


def deterministic_scalar_sum(
    partial: torch.Tensor, group: dist.ProcessGroup | None = None
) -> list[float]:
    """Cross-rank sum of a small 1-D vector of per-rank scalar partials, folded
    in ascending rank order — deterministic telemetry reductions (e.g. the
    global-batch loss), where NCCL's topology-dependent summation order would
    make the logged value non-reproducible.

    ``partial`` is this rank's partial sums as an fp64 tensor (host Python-float
    accumulations, moved to the collective's device by the caller). The
    ``all_gather`` is pure data movement — deterministic and topology-free; the
    fold runs on the HOST in Python floats (IEEE fp64 adds, ascending rank
    order), so no fp64 device kernel is involved and the arithmetic is
    byte-portable across CUDA/CPU/MPS (MPS has no fp64).

    Telemetry-grade by design: the result must never feed weights, gradients,
    optimizer state, control flow, or the state hash — those go through the
    audited reductions above.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [float(x) for x in partial]
    world = dist.get_world_size(group)
    if world == 1:
        return [float(x) for x in partial]
    gathered = [torch.empty_like(partial) for _ in range(world)]
    dist.all_gather(gathered, partial, group=group)
    rows = [g.cpu().tolist() for g in gathered]
    sums = list(rows[0])
    for row in rows[1:]:
        for i, v in enumerate(row):
            sums[i] += v
    return sums


def deterministic_total_norm(
    model: torch.nn.Module, device: torch.device
) -> torch.Tensor:
    """Cluster-side global L2 grad norm via the canonical ascending-shard fold.

    Each rank sums the sum-of-squares of its own parameter shards (in
    ``named_parameters`` order); the shard group then all-gathers those scalars
    and sums them in ascending rank order. ``all_gather`` is pure data movement
    (deterministic, topology-free); the only arithmetic is the local SoS and the
    ascending fold, both reproduced bitwise by :func:`audit_total_norm`.
    """
    from torch.distributed.tensor import DTensor
    import repop.ops as repop_ops

    local_sq = torch.zeros((), dtype=torch.float32, device=device)
    shard_group = None
    for _, p in model.named_parameters():
        g = p.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            # On the auditable path the grad mesh is the 1D fsdp shard mesh
            # (Shard(0)); the replicate axis is the hook's, not a mesh dim, so
            # the only reduction axis is this shard group. (tp>1 would add a
            # Replicate embedding that must be counted once — not supported by
            # the audit today; the loop asserts tp_size==1 for auditable runs.)
            local = g.to_local()
            if shard_group is None and g.device_mesh.size() > 1:
                shard_group = g.device_mesh.get_group()
        else:
            local = g
        local_sq = local_sq + grad_sum_of_squares(local)

    if shard_group is None:  # nothing sharded (world_size 1) → already global
        return repop_ops.sqrt(local_sq.reshape(1)).reshape(())
    world = shard_group.size()
    gathered = [torch.empty_like(local_sq) for _ in range(world)]
    dist.all_gather(gathered, local_sq, group=shard_group)
    return repop_ops.sqrt(_ascending_sum(gathered).reshape(1)).reshape(())


# Identifier recorded in checkpoint meta for the deterministic ascending-shard
# per-tensor sum-of-squares fold — the norm source for the stateless global
# clip (pretrain.train.global_clip), the spike protocol, and the per-param
# telemetry. Bump the version suffix if the per-tensor SoS grouping, the sqrt,
# or the fold ordering ever changes — the identifier is persisted in
# CheckpointMeta.grad_norm_algo as provenance for the recorded norms. Nothing
# consumes this field programmatically (informational provenance only).
GRAD_NORM_ALGO = "deterministic_per_tensor_sos_v1"


def deterministic_per_tensor_and_global_norm(
    model: torch.nn.Module, device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Per-tensor L2 grad norms AND the global norm, from ONE ``all_gather``.

    Per-tensor norms (not just the global norm) are needed by the per-param
    grad-norm telemetry; the
    global clip and spike protocol read the derived global norm. Under FSDP2 each parameter is
    ``Shard(0)`` across the shard group, so a tensor's full norm requires its
    sum-of-squares reduced across that group. We batch all ``P`` per-tensor
    partial sums into a single ``all_gather`` of a length-``P`` vector (one
    collective, same call count as :func:`deterministic_total_norm`, payload
    ``P`` fp32 scalars instead of one), fold across ranks in ascending order
    per tensor, and ``sqrt`` element-wise. The global norm is derived from the
    SAME per-tensor global sums (ascending fold over tensors → ``sqrt``), so the
    spike-protocol trigger and the clip share the one collective.

    Returns ``(norms, global_norm)`` where ``norms`` maps ``named_parameters``
    name → 0-dim fp32 norm and ``global_norm`` is a 0-dim fp32 scalar. Both are
    reproduced bit-for-bit on a single device by
    :func:`audit_per_tensor_and_global_norm`.

    NOTE: this fold groups the global sum as (ranks-inner per tensor, then
    tensors-outer), a different — but equally deterministic — order than
    :func:`deterministic_total_norm` ((all-params-inner per rank, then
    ranks-outer)). The two global norms differ only in the last ULPs; runs
    record ``GRAD_NORM_ALGO`` in checkpoint meta so the audit uses the
    matching fold.
    """
    from torch.distributed.tensor import DTensor
    import repop.ops as repop_ops

    names: list[str] = []
    local_parts: list[torch.Tensor] = []
    shard_group = None
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            local = g.to_local()
            if shard_group is None and g.device_mesh.size() > 1:
                shard_group = g.device_mesh.get_group()
        else:
            local = g
        names.append(name)
        local_parts.append(grad_sum_of_squares(local))

    if not names:
        return {}, torch.zeros((), dtype=torch.float32, device=device)

    local_vec = torch.stack(local_parts)  # [P] fp32, this rank's shard SoS
    if shard_group is None:  # world_size 1 → already global
        global_sq = local_vec
    else:
        world = shard_group.size()
        gathered = [torch.empty_like(local_vec) for _ in range(world)]
        dist.all_gather(gathered, local_vec, group=shard_group)
        global_sq = _ascending_sum(gathered)  # element-wise ascending fold → [P]

    return _finish_per_tensor(names, global_sq, repop_ops)


def audit_per_tensor_and_global_norm(
    model: torch.nn.Module, shard_world_size: int, device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Single-device counterpart of :func:`deterministic_per_tensor_and_global_norm`.

    Re-slices each reconstructed full gradient into the ``shard_world_size``
    ``Shard(0)`` shards the cluster folded over and reproduces the per-tensor
    ascending SoS fold bitwise — no collective. This is the speed-for-correctness
    audit path: ``P × world`` sequential local folds instead of one collective.
    """
    import repop.ops as repop_ops

    world = max(shard_world_size, 1)
    names: list[str] = []
    global_sq_parts: list[torch.Tensor] = []
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            continue
        parts = [grad_sum_of_squares(shard0_chunk(g, r, world)) for r in range(world)]
        names.append(name)
        global_sq_parts.append(_ascending_sum(parts))  # ascending over ranks

    if not names:
        return {}, torch.zeros((), dtype=torch.float32, device=device)

    global_sq = torch.stack(global_sq_parts)  # [P]
    return _finish_per_tensor(names, global_sq, repop_ops)


def _finish_per_tensor(
    names: list[str], global_sq: torch.Tensor, repop_ops
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Shared tail of the per-tensor folds: element-wise ``sqrt`` for the
    per-tensor norms, and the ascending-over-tensors fold + ``sqrt`` for the
    global norm. Kept in one place so the cluster and audit paths are byte-equal.
    """
    per_tensor = repop_ops.sqrt(global_sq)  # [P], element-wise, BFR cpu/cuda/mps
    global_norm = repop_ops.sqrt(
        _ascending_sum([global_sq[i].reshape(1) for i in range(global_sq.shape[0])])
    ).reshape(())
    norms = {name: per_tensor[i] for i, name in enumerate(names)}
    return norms, global_norm


def audit_total_norm(
    model: torch.nn.Module, shard_world_size: int, device: torch.device
) -> torch.Tensor:
    """Audit-side counterpart of :func:`deterministic_total_norm`.

    The audit holds each parameter's full (already cross-rank-reduced) gradient,
    so it re-slices into the ``shard_world_size`` DTensor ``Shard(0)`` shards the
    cluster folded over and reproduces the per-shard SoS + ascending sum bitwise.
    Only the intra-pod shard degree is folded — the cross-replica axis is already
    collapsed into the reconstructed full gradient. Memory: one shard view at a
    time, dot is fused, so no full-tensor squared temporaries are allocated.
    """
    import repop.ops as repop_ops

    world = max(shard_world_size, 1)
    s = [torch.zeros((), dtype=torch.float32, device=device) for _ in range(world)]
    for _, p in model.named_parameters():
        g = p.grad
        if g is None:
            continue
        for r in range(world):
            s[r] = s[r] + grad_sum_of_squares(shard0_chunk(g, r, world))
    return repop_ops.sqrt(_ascending_sum(s).reshape(1)).reshape(())
