"""Bitwise state hashing for cross-run equivalence checks.

Produces a topology-invariant fingerprint of model weights, optimizer
moments, gradients, and (optionally) a precomputed cross-rank batch
digest + a previous hash for chaining. Two runs that are bit-for-bit
identical produce the same digest at the same point regardless of how
params/grads are sharded across the mesh.

Topology invariance comes from ``DTensor.full_tensor()`` — under FSDP /
TP / HSDP each parameter and moment is reassembled to its unsharded form
before bytes are extracted, so a checkpoint loaded under (fsdp=8) hashes
to the same value as the same checkpoint loaded under (tp=2, fsdp=4).

``full_tensor()`` is a collective, so this function must be invoked on
every rank under a distributed run. All ranks compute the same digest,
so no broadcast is required — *including the batch term*, provided the
caller used :class:`RunningBatchHasher` to produce a precomputed
``batch_digest`` via DP all-gather (see class docstring). The legacy
``batch=Mapping`` path is kept for CPU-only tests; under distributed
that path produces a rank-local digest and must be avoided.

Sharded-local path (v3) — the one the hot loop and audit now use
-----------------------------------------------------------------
:func:`compute_state_hash` above all-gathers every parameter, moment and
gradient to its full unsharded form on *every* rank before hashing — a
collective + a redundant full-model blake2b per rank, which dominates a
frequent-hash run. The sharded path instead hashes only each rank's
*local* shard and combines 32-byte digests:

    local  = local_shard_state_digest(model, optimizer, include_grads,
                                       extract=to_local)   # no collective
    shard  = combine_shard_state(local, dp_group)          # tiny all_gather
    digest = finalize_state_hash(prev_hash=…, shard_state_digest=shard,
                                 optimizer=…, batch_digest=…)

The single-device audit reconstructs the same value from the full master
via :func:`audit_shard_state_digest`, which slices each full tensor with
``shard0_chunk(t, s, num_shards)`` — byte-identical to the cluster's
``to_local()`` including the zero pad (verified) — and combines over the
DP ranks in the same rep-major order the cluster's ``dp_group`` all_gather
and the gradient fold already use. NO arithmetic enters the combine (pure
byte concatenation + blake2b), so it is bitwise-reproducible across
CPU/CUDA/MPS by construction — the only reproducibility burden is the
``to_local()`` ⇔ ``shard0_chunk`` byte layout, shared with the grad-norm
audit. Trade-off vs the full path: the digest now depends on the shard
layout (``num_shards``), so it is NOT cross-topology-invariant — same as
the batch term, and acceptable because the audit replays the recorded
``dp_shard``.

Byte stream layout (blake2b, 32-byte digest):

    schema_tag  (v2 — bumped from v1 when batch_digest was added)
    prev_hash (32 bytes; zeros if None)
    for name, param in sorted(model.named_parameters()):
        name
        weight: dtype, shape, bytes
        if optimizer is given and param has state:
            step (int64 LE), exp_avg, exp_avg_sq, max_exp_avg_sq?
        if include_grads:
            grad: dtype, shape, bytes  (or "grad_none" marker)
    if optimizer is given:
        param_groups minus 'params' as canonical JSON
    if batch_digest is given:
        32 raw bytes
    if batch is given (legacy CPU-test path):
        for k in sorted(batch): k, dtype, shape, bytes
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Mapping

import torch

LOG = logging.getLogger(__name__)

_SCHEMA = b"pretrain.state_hash.v2\n"
_DIGEST_SIZE = 32


def _full(t: torch.Tensor) -> torch.Tensor:
    """Materialize a DTensor to its unsharded full form; pass-through for
    plain tensors. ``full_tensor`` is a collective under FSDP/TP — every
    rank must call it.
    """
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def tensor_bytes(t: torch.Tensor) -> memoryview:
    """Reinterpret the tensor's storage as raw bytes.

    ``view(torch.uint8)`` works for any element dtype — including bf16,
    which has no numpy equivalent and would break a ``.numpy().tobytes()``
    path. Flattening first avoids the multi-dim ``view(dtype)`` stride
    restriction. The returned view retains the backing storage without a
    bytes copy. Callers consume it synchronously and must not mutate the
    tensor while hashing.
    """
    t = t.detach().contiguous().cpu().reshape(-1)
    return memoryview(t.view(torch.uint8).numpy())


def feed_tensor(h: Any, tag: bytes, t: torch.Tensor) -> None:
    full = _full(t)
    h.update(tag)
    h.update(str(full.dtype).encode("utf-8"))
    h.update(b"\0")
    h.update(str(tuple(full.shape)).encode("utf-8"))
    h.update(b"\0")
    h.update(tensor_bytes(full))


def _coerce_prev(prev_hash: str | bytes | None) -> bytes:
    if prev_hash is None:
        return b"\0" * _DIGEST_SIZE
    if isinstance(prev_hash, str):
        raw = bytes.fromhex(prev_hash)
    else:
        raw = bytes(prev_hash)
    if len(raw) < _DIGEST_SIZE:
        return raw + b"\0" * (_DIGEST_SIZE - len(raw))
    return raw[:_DIGEST_SIZE]


def _jsonable(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    return str(v)


class RunningBatchHasher:
    """Per-rank rolling blake2b digest over every batch consumed.

    Update with pinned-CPU loader output **before** ``.to(device,
    non_blocking=True)`` so the hash never forces a host sync. The
    per-rank digest is 32 bytes; the global digest is produced by an
    ``all_gather`` over a DP-only ProcessGroup and a deterministic
    concatenation in DP-rank order so every rank arrives at the same
    bytes.

    Internally the running hash is a *chained* 32-byte digest rather
    than a single long-lived ``hashlib.blake2b`` object: each batch
    updates the chain as ``chain_n = blake2b(chain_{n-1} || batch_n)``.
    The full state of the hasher is exactly its current 32-byte chain
    value, which makes the hasher saveable/restorable across a
    checkpoint resume — ``hashlib.blake2b`` itself doesn't expose its
    internal Merkle-Damgård state, so a single-context blake2b can't be
    snapshotted byte-for-byte mid-stream. Two runs that consume the
    same sequence of batches produce identical chain values; a resumed
    run primed with the saved chain at step ``k`` produces the same
    chain at step ``k+m`` as a continuous run at step ``k+m``.

    Under HSDP+TP: TP-paired ranks consume identical batches (the data
    loader strides by ``dp_rank``, not global rank). The caller must
    therefore pass a DP-only ``dp_group`` to :meth:`global_digest`;
    including TP duplicates would make the digest depend on ``tp_size``
    even when the data is identical.

    Cross-topology caveat: two runs with different ``dp_world_size`` get
    different per-dp-rank document slicing, so their per-rank running
    digests differ and the combined digest differs too. This is
    inherent to data parallelism, not a bug in the hash. The
    weight/grad/moment portion of :func:`compute_state_hash` remains
    cross-topology-invariant via ``full_tensor()`` — turn off
    ``include_batch`` for cross-topology comparison.
    """

    def __init__(self, prev_digest: bytes | None = None) -> None:
        if prev_digest is None:
            self._chain = b"\0" * _DIGEST_SIZE
        else:
            if len(prev_digest) != _DIGEST_SIZE:
                raise ValueError(
                    f"prev_digest must be {_DIGEST_SIZE} bytes, got {len(prev_digest)}"
                )
            self._chain = bytes(prev_digest)

    def update(self, batch: Mapping[str, torch.Tensor]) -> None:
        h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
        h.update(self._chain)
        for k in sorted(batch.keys()):
            t = batch[k]
            h.update(k.encode("utf-8"))
            h.update(b"\0")
            h.update(str(t.dtype).encode("utf-8"))
            h.update(b"\0")
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(b"\0")
            h.update(tensor_bytes(t))
        self._chain = h.digest()

    def local_digest(self) -> bytes:
        return self._chain

    def global_digest(self, dp_group: Any = None) -> bytes:
        """Combine per-rank running digests via all_gather over ``dp_group``.

        ``dp_group=None`` (or no distributed init) → returns the local
        digest unchanged. Pass a DP-only group when running distributed
        — typically derived from ``ParallelDims`` as
        ``parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"])
        ._flatten().get_group()``.
        """
        return combine_batch_digests(_all_gather_bytes(self.local_digest(), dp_group))


def combine_batch_digests(digests: list[bytes]) -> bytes:
    """Combine per-DP-rank running batch digests into the global batch digest.

    The single-process form of :meth:`RunningBatchHasher.global_digest`: one
    rank contributes its own chain unchanged, several are blake2b'd in DP-rank
    order. Both the audit replay and the hand-off verifier reconstruct the
    cluster's batch term this way, from the ``batch_hasher.rank_<r>.bin`` files
    a checkpoint carries, so the rule lives next to the hasher that defines it.
    """
    if not digests:
        raise ValueError("no per-rank batch digests to combine")
    if any(len(d) != _DIGEST_SIZE for d in digests):
        raise ValueError(f"each per-rank batch digest must be {_DIGEST_SIZE} bytes")
    if len(digests) == 1:
        return digests[0]
    combined = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for d in digests:
        combined.update(d)
    return combined.digest()


def compute_state_hash(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    include_grads: bool = False,
    batch: Mapping[str, torch.Tensor] | None = None,
    batch_digest: bytes | None = None,
    prev_hash: str | bytes | None = None,
) -> str:
    """Return a hex digest fingerprinting the requested state.

    Under distributed: must be invoked on every rank; returns the same
    digest on every rank when batch info is supplied via
    ``batch_digest`` (precomputed cross-rank). The legacy ``batch=``
    parameter hashes rank-local bytes and is intended for CPU tests.
    """
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    h.update(_SCHEMA)
    h.update(b"prev\0")
    h.update(_coerce_prev(prev_hash))

    # Look up optimizer state by Parameter object. ``optimizer.state`` is
    # keyed by the live Parameter the optimizer was built with — same
    # object that ``model.named_parameters()`` yields (FSDP2 wrap mutates
    # parameters in-place; build_optimizer runs after wrap).
    opt_state: dict[torch.nn.Parameter, dict] = (
        dict(optimizer.state) if optimizer is not None else {}
    )

    h.update(b"params\0")
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        feed_tensor(h, b"weight\0", p.data)

        if optimizer is not None and p in opt_state:
            state = opt_state[p]
            h.update(b"optim_step\0")
            step = state.get("step", 0)
            if isinstance(step, torch.Tensor):
                step = int(step.item())
            h.update(int(step).to_bytes(8, "little", signed=True))
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state and isinstance(state[key], torch.Tensor):
                    # Keep the wire delimiter as a byte literal: Cython 3.1.2
                    # miscompiles a NUL escape inside this formatted string.
                    feed_tensor(h, b"optim_" + key.encode("utf-8") + b"\0", state[key])

        if include_grads:
            if p.grad is None:
                h.update(b"grad_none\0")
            else:
                feed_tensor(h, b"grad\0", p.grad)

    if optimizer is not None:
        h.update(b"param_groups\0")
        groups = [
            {k: _jsonable(v) for k, v in g.items() if k != "params"}
            for g in optimizer.param_groups
        ]
        h.update(json.dumps(groups, sort_keys=True, default=str).encode("utf-8"))

    if batch_digest is not None:
        if len(batch_digest) != _DIGEST_SIZE:
            raise ValueError(
                f"batch_digest must be {_DIGEST_SIZE} bytes, got {len(batch_digest)}"
            )
        h.update(b"batch_digest\0")
        h.update(batch_digest)

    if batch is not None:
        h.update(b"batch\0")
        for k in sorted(batch.keys()):
            h.update(k.encode("utf-8"))
            h.update(b"\0")
            feed_tensor(h, b"batch_tensor\0", batch[k])

    return h.hexdigest()


# -----------------------------------------------------------------------------
# Sharded-local hashing (v3). See the module docstring's "Sharded-local path".
# -----------------------------------------------------------------------------

_SCHEMA_V3 = b"pretrain.state_hash.v3\n"


def _local_logical(t: torch.Tensor) -> torch.Tensor:
    """This rank's local Shard(0) shard with the FSDP2 tail padding STRIPPED —
    the logical rows only — so the hash never depends on the pad fill value
    (not guaranteed zero across torch versions). NO collective.

    Plain (non-DTensor) tensors and tensors not sharded on dim 0 (e.g. a
    Replicate placement) pass their local view through unchanged. For a dim-0
    Shard, the local view is ``[fcs]`` rows (zero-padded on the tail rank); we
    return its first ``real`` rows, where ``real`` is this rank's logical share
    derived from the global ``dim0`` and the rank's mesh coordinate — the exact
    rows :func:`shard0_logical_chunk` reproduces on the audit's full master.
    """
    if not hasattr(t, "to_local"):
        return t
    from torch.distributed.tensor import Shard

    from pretrain.parallel.deterministic_reduce import shard0_full_chunk_size

    loc = t.to_local()
    mesh = t.device_mesh
    shard_mesh_dim = None
    for i, pl in enumerate(t.placements):
        if isinstance(pl, Shard) and pl.dim == 0:
            shard_mesh_dim = i
            break
    if shard_mesh_dim is None:
        return loc  # not sharded on dim 0 → the whole local view is logical
    world = mesh.size(shard_mesh_dim)
    coord = mesh.get_coordinate()[shard_mesh_dim]
    dim0 = t.shape[0]  # global (logical) size for a DTensor
    fcs = shard0_full_chunk_size(dim0, world)
    real = max(0, min(fcs, dim0 - coord * fcs))
    return loc[:real]


def _feed_local(h: Any, tag: bytes, t: torch.Tensor) -> None:
    """Like :func:`feed_tensor` but ``t`` is already this shard's local slice — no
    ``full_tensor()`` collective. Hashes dtype, shape and raw bytes."""
    h.update(tag)
    h.update(str(t.dtype).encode("utf-8"))
    h.update(b"\0")
    h.update(str(tuple(t.shape)).encode("utf-8"))
    h.update(b"\0")
    h.update(tensor_bytes(t))


def local_shard_state_digest(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    include_grads: bool = False,
    extract: Any = None,
) -> bytes:
    """32-byte blake2b over one shard's local slice of every parameter,
    optimizer moment and (optionally) gradient — NO collective.

    ``extract(t)`` maps a full/DTensor tensor to the local-shard tensor whose
    bytes are hashed:

      * cluster:  :func:`_local_logical` (the default) — the rank's own shard,
                  tail padding stripped.
      * audit:    ``lambda t: shard0_logical_chunk(t, s, num_shards)`` — slice
                  the reconstructed full master into the cluster's logical
                  Shard(0) rows.

    Both hash only the logical (unpadded) rows, so the digest never depends on
    FSDP2's pad fill value (verified byte-identical across the cluster's
    stripped ``to_local()`` and the audit's logical chunk).
    ``param_groups`` / ``prev_hash`` / ``batch_digest`` are NOT folded in here;
    they are rank-identical and are hashed once by :func:`finalize_state_hash`
    after the cross-shard combine. Byte layout mirrors :func:`compute_state_hash`
    per-parameter (name, weight, optim_step+moments, grad) so a single-bit
    change anywhere in a shard flips its digest.
    """
    if extract is None:
        extract = _local_logical
    opt_state: dict[torch.nn.Parameter, dict] = (
        dict(optimizer.state) if optimizer is not None else {}
    )
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        _feed_local(h, b"weight\0", extract(p.data))

        if optimizer is not None and p in opt_state:
            state = opt_state[p]
            h.update(b"optim_step\0")
            step = state.get("step", 0)
            if isinstance(step, torch.Tensor):
                step = int(step.item())
            h.update(int(step).to_bytes(8, "little", signed=True))
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state and isinstance(state[key], torch.Tensor):
                    _feed_local(h, b"optim_" + key.encode("utf-8") + b"\0", extract(state[key]))

        if include_grads:
            if p.grad is None:
                h.update(b"grad_none\0")
            else:
                _feed_local(h, b"grad\0", extract(p.grad))
    return h.digest()


def _all_gather_bytes(local: bytes, dp_group: Any) -> list[bytes]:
    """all_gather a fixed-size digest over ``dp_group`` → digests in rank order.

    Returns ``[local]`` when not distributed / ``dp_group is None`` / ws==1.
    Same gloo CPU all_gather mechanism as :meth:`RunningBatchHasher.global_digest`.
    """
    if dp_group is None or not torch.distributed.is_initialized():
        return [local]
    ws = torch.distributed.get_world_size(group=dp_group)
    if ws == 1:
        return [local]
    n = len(local)
    gathered = [torch.empty(n, dtype=torch.uint8) for _ in range(ws)]
    src = torch.frombuffer(bytearray(local), dtype=torch.uint8)
    torch.distributed.all_gather(gathered, src, group=dp_group)
    return [bytes(g.numpy()) for g in gathered]


def combine_shard_state(local: bytes, dp_group: Any = None) -> bytes:
    """Cross-shard combine of per-rank local shard-state digests.

    all_gather the per-rank :func:`local_shard_state_digest` over the DP group,
    then blake2b over the gathered digests in rank order. ALWAYS wraps in an
    outer blake2b (even ws==1) so a single-process audit that combines ``N>=1``
    per-shard digests under the same outer hash (see :func:`audit_shard_state_digest`)
    matches the cluster bitwise.
    """
    combined = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for d in _all_gather_bytes(local, dp_group):
        combined.update(d)
    return combined.digest()


def audit_shard_state_digest(
    model: torch.nn.Module,
    num_shards: int,
    num_ranks: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    include_grads: bool = False,
) -> bytes:
    """Single-device reconstruction of the cluster's cross-shard combine.

    Computes ``num_shards`` distinct per-shard digests by slicing each full
    tensor of the reconstructed master with ``shard0_chunk(t, s, num_shards)``,
    then combines over ``num_ranks`` virtual DP ranks in rank order
    (rank ``r`` → shard ``r % num_shards``) — the rep-major mapping the
    cluster's ``dp_group`` all_gather and the gradient fold already use, so the
    outer blake2b sees the same digest sequence :func:`combine_shard_state`
    produced on the cluster.
    """
    from pretrain.parallel.deterministic_reduce import shard0_logical_chunk

    per_shard = [
        local_shard_state_digest(
            model,
            optimizer=optimizer,
            include_grads=include_grads,
            extract=(lambda t, s=s: shard0_logical_chunk(t, s, num_shards)),
        )
        for s in range(num_shards)
    ]
    combined = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for r in range(num_ranks):
        combined.update(per_shard[r % num_shards])
    return combined.digest()


def finalize_state_hash(
    *,
    prev_hash: str | bytes | None = None,
    shard_state_digest: bytes,
    optimizer: torch.optim.Optimizer | None = None,
    batch_digest: bytes | None = None,
) -> str:
    """Fold the cross-shard state digest, optimizer ``param_groups``, the
    running batch digest and the previous chained hash into the final hex
    digest. Rank-identical — every input is rank-identical after the combine,
    so all ranks return the same value without a broadcast.
    """
    if len(shard_state_digest) != _DIGEST_SIZE:
        raise ValueError(
            f"shard_state_digest must be {_DIGEST_SIZE} bytes, got {len(shard_state_digest)}"
        )
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    h.update(_SCHEMA_V3)
    h.update(b"prev\0")
    h.update(_coerce_prev(prev_hash))
    h.update(b"shard_state\0")
    h.update(shard_state_digest)
    if optimizer is not None:
        h.update(b"param_groups\0")
        groups = [
            {k: _jsonable(v) for k, v in g.items() if k != "params"}
            for g in optimizer.param_groups
        ]
        h.update(json.dumps(groups, sort_keys=True, default=str).encode("utf-8"))
    if batch_digest is not None:
        if len(batch_digest) != _DIGEST_SIZE:
            raise ValueError(
                f"batch_digest must be {_DIGEST_SIZE} bytes, got {len(batch_digest)}"
            )
        h.update(b"batch_digest\0")
        h.update(batch_digest)
    return h.hexdigest()
