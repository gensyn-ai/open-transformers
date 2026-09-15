"""Single-device audit: reproduce a cluster checkpoint interval bitwise.

Given a checkpoint from an *auditable* cluster run (``run.reduction_mode ==
'deterministic_allgather'`` — canonical data stream + fixed-order reduce-scatter),
this replays forward on ONE device and, on reaching the next checkpoint, should
produce a bit-identical model+optimizer state (verified via ``state_hash``).

How it stays bitwise-identical despite running on one device:
  * Data — the canonical ``GlobalStream`` is a pure function of (seed, manifests,
    seq_len); we rebuild each of the cluster's N virtual ranks' slices from the
    saved global-stream state and feed the same windows in the same order.
  * Reduction — two levels, emulated exactly. The shard-inner reduce-scatter
    sums partials in ascending rank order then ``/dp_shard``. The replicate-outer
    all-reduce replays the cluster's ``recursive_doubling`` order — the balanced
    adjacent-pair tree of ``parallel.deterministic_reduce.tree_reduce_sum`` (the
    ``_DiskTreeFold`` below) — then ``/dp_replicate``.
  * Kernels — repop's cross-device-reproducible mode makes the per-op math
    identical to the cluster (run on a matching GPU arch).
  * Grad clip + spike — clipping is the stateless deterministic global-norm
    clip (see ``pretrain.train.global_clip``). Its global norm comes from the
    same deterministic ascending-shard sum-of-squares fold the cluster used
    (re-slicing the reconstructed full gradient into ``dp_shard`` Shard(0)
    shards; see ``parallel.deterministic_reduce``), so the clip coefficient —
    and the pre-clip global norm the spike protocol reads to replay the
    cluster's skip/cooldown decisions — are bit-identical.

Scope: single-device replay of an FSDP/HSDP run (TP was removed). HSDP
(``dp_replicate > 1``) IS supported via the two-level fold below, for any
``dp_replicate`` — the recursive-doubling replicate path replays as the
binary-blocks tree (a single balanced tree when ``dp_replicate`` is a power of
two, e.g. ``((g0+g1)+(g2+g3)) + (g4+g5)`` at ``dp_replicate=6``).

Usage:
    python -m pretrain.cli.audit_replay --checkpoint runs/<id>/checkpoints/step_000000010 \\
        [--config-name 1b_repop_v2] [--device cuda] [--until-step N]

    # Config-only init audit (no checkpoint needed): regenerate init from the
    # config's seed and report/compare its hash. --checkpoint is omitted.
    python -m pretrain.cli.audit_replay --from-init --until-step 0 \\
        --config-name 1b_repop_v2 [--expect-hash <digest|path/to/state_hash_init.txt>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm

# Module-level so tests can monkeypatch ``audit_replay.load_config`` and so the
# config-only init audit can be redirected; pretrain.config is pure
# hydra/pydantic (no repop), so importing it here is safe w.r.t. the
# "set repop_env before importing repop-backed modules" ordering below.
from pretrain.config import load_config

# The spill digest below hashes tensors through the same walk the state hash
# uses, so the two cannot drift. state_hash is pure-Python (no repop) — the
# same reason given at the _DIGEST_SIZE import further down — so importing it
# here is safe w.r.t. the repop import ordering.
from pretrain.train.state_hash import feed_tensor

LOG = logging.getLogger("pretrain.audit")

# --------------------------------------------------------------------------
# Verified disk spill
#
# The offload paths below stream the fold partials, the AdamW moments and the
# fp32 master through ``torch.save``/``torch.load``. At the 1.6 B audit shape
# (dp_replicate=6, 192 micro-batches) that is ~140 GB of serialisation per
# step, and on a Mac whose peak working set exceeds physical RAM it happens
# while the machine is swapping continuously for hours.
#
# ``torch.load`` does not checksum. A single flipped bit in a payload comes
# back as a valid float and the replay continues: measured on torch 2.12,
# flipping one bit under a ``weights_only=True`` load returned 4.0001220703125
# where 4.0 was written, with no error raised. The run then produces a wrong
# state hash and reports NO MATCH against a step that is in fact correct --
# the worst outcome an audit can produce, because it is indistinguishable from
# a real reproducibility failure.
#
# So each spill records a digest of the tensors as they were handed to
# ``torch.save``, and each load re-derives it from the tensors that came back.
# That brackets the whole exposure -- serialisation buffer, page cache, disk,
# read buffer, deserialised tensor -- rather than only what is at rest on
# disk, which matters because re-reading a spill file after the fact cannot
# distinguish a clean file that was read wrongly from a file that was never
# corrupted at all.
#
# Cost is a hash pass over data already in hand: blake2b-32 measures 1.33 GB/s
# on an M-series CPU, so ~105 s against a 6-18 h step. Verification is read-only
# and cannot move a single result bit; it only decides whether the replay
# raises instead of returning a wrong answer. ``PRETRAIN_AUDIT_SPILL_VERIFY=0``
# disables it.
# --------------------------------------------------------------------------


class SpillCorruption(RuntimeError):
    """A spilled tensor did not survive its disk round-trip.

    Raised rather than returned: every caller is mid-replay, and continuing
    past this point is what produces the wrong hash we are trying to prevent.
    """


def _spill_verify_enabled() -> bool:
    return os.environ.get("PRETRAIN_AUDIT_SPILL_VERIFY", "1").lower() not in (
        "0",
        "false",
        "no",
        "",
    )


def _digest_payload(obj) -> str:
    """Order-stable blake2b-32 over a tensor, or a str->tensor mapping.

    The tensor walk is ``state_hash.feed_tensor``, not a second copy of it.
    Two byte-serialisation conventions for tensors in one repository drift
    apart, and a drifted one surfaces as a phantom audit mismatch, which is
    the failure class this verification exists to remove. Reuse also inherits
    that walk's properties: dtype and shape are hashed alongside the bytes, so
    a payload that survives with the right bits under the wrong metadata is
    still caught, and its ``memoryview`` avoids a second full copy of the
    payload on every save and load, which matters on the machines that are
    already swapping.

    blake2b-32 for the same reason. ``pretrain.data.manifest.blake2b_file``
    states the convention that every "bytes I committed to" artifact in the
    project speaks one hash algorithm. sha256 is ~2.4x faster here (3.20 GB/s
    against 1.33 GB/s, hardware SHA on M-series), but that buys about 61 s on
    a step that runs for 6 to 18 hours, which does not justify a second
    algorithm.

    Mapping keys are walked in sorted order so the digest does not depend on
    dict ordering. ``None`` entries (an absent ``max_exp_avg_sq``) are
    recorded as such, so absent stays distinguishable from present.
    """
    h = hashlib.blake2b(digest_size=32)

    def feed(name: str, t) -> None:
        if t is None:
            h.update(name.encode() + b"\0none\0")
            return
        feed_tensor(h, name.encode() + b"\0", t)

    if isinstance(obj, torch.Tensor):
        feed("", obj)
    else:
        for k in sorted(obj):
            feed(k, obj[k])
    return h.hexdigest()


def _digest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".blake2b")


def _save_verified(obj, path: Path) -> None:
    """``torch.save`` plus a sidecar digest of what was handed over.

    The digest is taken from the in-RAM object before serialisation, so it
    describes the values the replay intended to persist rather than whatever
    reached the file.
    """
    if _spill_verify_enabled():
        digest = _digest_payload(obj)
        torch.save(obj, path)
        _digest_path(path).write_text(digest)
    else:
        torch.save(obj, path)


def _load_verified(path: Path, map_location):
    """``torch.load`` that refuses to return a payload which changed in transit.

    Always deserialises to CPU first so the check reads real bytes; the move to
    ``map_location`` happens after it passes. ``torch.load`` stages through host
    memory anyway, so this does not add a copy that was not already there.

    A spill written before verification existed, or with it disabled, has no
    sidecar; that loads unchecked rather than failing, so an in-flight run is
    never broken by the upgrade.
    """
    obj = torch.load(path, map_location="cpu", weights_only=True)
    sidecar = _digest_path(path)
    if _spill_verify_enabled() and sidecar.exists():
        want = sidecar.read_text().strip()
        got = _digest_payload(obj)
        if got != want:
            raise SpillCorruption(
                f"{path.name} changed between write and read: wrote {want[:16]}..., "
                f"read back {got[:16]}.... The bytes did not survive the disk "
                f"round-trip, so this replay cannot produce a trustworthy state "
                f"hash. This is a fault on this machine, not a reproducibility "
                f"failure of the step being audited -- the most common cause is "
                f"a working set larger than physical RAM, which makes the host "
                f"swap for the length of the run."
            )
    if map_location is not None and str(map_location) != "cpu":
        if isinstance(obj, torch.Tensor):
            return obj.to(map_location)
        return {
            k: (v.to(map_location) if isinstance(v, torch.Tensor) else v)
            for k, v in obj.items()
        }
    return obj


def _unlink_verified(path: Path) -> None:
    """Remove a spill file and its sidecar together."""
    path.unlink()
    _digest_path(path).unlink(missing_ok=True)


def mismatch_message(digest: str, expected: str) -> str:
    """The one line a state-hash mismatch is reported on, from either path.

    A mismatch can surface from two places: ``main`` when the replay simply
    did not reproduce, and the ``--save-checkpoint-dir`` guard inside
    ``audit_replay`` that refuses to hand off an unreproduced state. The guard
    runs first and used to truncate both digests to 16 hex, so on the chained
    flow -- the only one the audit CLI uses -- the reproduced digest appeared
    at full length nowhere at all: the result JSON carrying it is printed only
    on success. A reader was left with a 16-hex prefix for the one number the
    failure is about.

    Both digests in full, one shape, one function, so the two paths cannot
    drift apart again. Downstream parses ``AUDIT FAILED: state_hash X != Y``.
    """
    return f"AUDIT FAILED: state_hash {digest} != {expected}"


def _log_rss(tag: str) -> None:
    """Log host RSS + accelerator allocation at a phase boundary, gated by
    PRETRAIN_AUDIT_MEMLOG=1. Flushed per line so the last line before an
    OOM-kill identifies the phase that blew up memory.

    Reports CUDA allocation on NVIDIA, and MPS current + driver allocation on
    Apple Silicon. On unified memory the MPS *driver* figure (what Metal has
    reserved from the shared pool, cache included) is the number that actually
    competes with host RAM, so it's the one to watch for an MPS OOM.
    """
    if os.environ.get("PRETRAIN_AUDIT_MEMLOG", "0") != "1":
        return
    rss_gb = -1.0
    try:
        with open("/proc/self/status") as f:  # Linux: current RSS
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_gb = int(line.split()[1]) / 1048576.0
                    break
    except Exception:
        try:
            # macOS has no /proc; fall back to peak RSS (bytes on Darwin, KiB on
            # Linux) so the line still carries a host-memory figure.
            import resource
            import sys

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_gb = ru / (1024**3 if sys.platform == "darwin" else 1024**2)
        except Exception:
            pass
    acc = ""
    try:
        if torch.cuda.is_available():
            # Peak matters more than current for an OOM: the allocation that
            # fails is almost never live at a phase boundary. ``peak`` is since
            # the previous MEMLOG line, so it attributes to THIS phase; reserved
            # is what the caching allocator holds from the driver, which is what
            # the card actually has committed.
            free_b, total_b = torch.cuda.mem_get_info()
            acc = " cuda_alloc=%.1fGB peak=%.1fGB reserved=%.1fGB free=%.1fGB/%.1fGB" % (
                torch.cuda.memory_allocated() / 1e9,
                torch.cuda.max_memory_allocated() / 1e9,
                torch.cuda.memory_reserved() / 1e9,
                free_b / 1e9,
                total_b / 1e9,
            )
            torch.cuda.reset_peak_memory_stats()
        elif torch.backends.mps.is_available():
            cur = torch.mps.current_allocated_memory() / 1e9
            drv = torch.mps.driver_allocated_memory() / 1e9
            acc = " mps_alloc=%.1fGB mps_driver=%.1fGB" % (cur, drv)
    except Exception:
        pass
    LOG.info("MEMLOG[%s] host_RSS=%.1fGB%s", tag, rss_gb, acc)


def _tree_push(
    stack: list[tuple[int, dict[str, torch.Tensor]]],
    part: dict[str, torch.Tensor],
) -> None:
    """Binary-carry fold realising ``tree_reduce_sum``'s combination order.

    Push one replicate's (inner-averaged) gradient — a ``name -> tensor`` dict —
    and combine any equal-"level" entry below it, so after pushing all
    ``dp_replicate`` partials in ascending replicate order the stack holds the
    binary-blocks tree the cluster's recursive-doubling all-reduce produces
    (one residual entry per power-of-2 block, i.e. per set bit of
    ``dp_replicate``). Keeps only ``O(log2(dp_replicate))`` full-gradient
    accumulators instead of all ``dp_replicate`` (a flat list would OOM at
    scale). The earlier-pushed (lower-replicate) operand stays on the left —
    matching the lower-rank-left grouping of ``tree_reduce_sum`` — and is mutated
    in place. For a power-of-2 count the stack collapses to exactly one entry;
    otherwise the residual blocks are combined in ascending order by
    :meth:`_DiskTreeFold.result`.
    """
    level = 0
    while stack and stack[-1][0] == level:
        _, lower = stack.pop()
        for name in lower:
            lower[name].add_(part[name])  # lower-rep (left) + higher-rep (right)
        part = lower
        level += 1
    stack.append((level, part))


class _DiskTreeFold:
    """Streaming balanced-tree fold (same order as :func:`_tree_push`) that
    spills dormant accumulators to disk, so RAM holds at most ~2 full gradients
    regardless of ``dp_replicate``.

    The recursive-doubling tree otherwise needs ``log2(dp_replicate)+1``
    accumulators resident (4 at ``dp_replicate=8`` → ~26 GB for the 1.6B model,
    which OOMs a 64 GiB pod). Here only the two operands of the *current* add are
    in RAM; every waiting partial sits on disk. Bitwise-identical to the in-RAM
    fold: ``torch.save``/``load`` round-trips fp32 byte-exactly and the adds keep
    the lower-replicate operand on the left. ``spill_dir=None`` ⇒ all in RAM (no
    I/O), for the shallow stacks at ``dp_replicate<=2``.
    """

    def __init__(self, spill_dir: Path | None) -> None:
        self._spill_dir = spill_dir
        self._stack: list[tuple[int, object]] = []  # entry: dict (RAM) | Path (disk)
        self._seq = 0

    def _store(self, grad: dict[str, torch.Tensor]):
        if self._spill_dir is None:
            return grad
        path = self._spill_dir / f"fold_{self._seq}.pt"
        self._seq += 1
        _save_verified(grad, path)
        return path

    def _fetch(self, entry) -> dict[str, torch.Tensor]:
        if isinstance(entry, Path):
            grad = _load_verified(entry, "cpu")
            _unlink_verified(entry)  # reclaim disk as soon as the partial is consumed
            return grad
        return entry

    def push(self, part: dict[str, torch.Tensor]) -> None:
        level = 0
        while self._stack and self._stack[-1][0] == level:
            _, lower = self._stack.pop()
            lower = self._fetch(lower)          # dormant accumulator ← disk
            for name in lower:
                lower[name].add_(part[name])    # lower-rep (left) + higher-rep
            part = lower
            level += 1
        self._stack.append((level, self._store(part)))  # result → disk

    def result(self) -> dict[str, torch.Tensor]:
        if not self._stack:
            raise RuntimeError("tree fold is empty (no replicate partials pushed)")
        # Combine the residual per-block accumulators in ascending block order
        # (largest/lowest-rank block first) — the cluster's binary-blocks
        # cross-block fold (see deterministic_reduce._recursive_doubling_allreduce_avg).
        # For a power-of-2 dp_replicate the stack is a single entry and this
        # returns it unchanged; for e.g. dp_replicate=6 it folds
        # ``block(0..3) + block(4..5)`` with the lower-rank block on the left.
        acc = self._fetch(self._stack[0][1])
        for _level, entry in self._stack[1:]:
            part = self._fetch(entry)
            for name in acc:
                acc[name].add_(part[name])  # lower-rank block (left) + higher block
        return acc


def _prepopulate_cpu_optim_state(optimizer) -> None:
    """Seed the repop-AdamW state template (step / exp_avg / exp_avg_sq) on CPU
    for every parameter, so DCP loads the checkpoint's optimizer shards into CPU
    tensors — never the GPU — and they stay there for per-parameter streaming.

    Mirrors ``FSDPAwareRepopAdamW.step``'s ``len(state)==0`` init but pins the
    moments to CPU (`device="cpu"`). With this in place, ``Checkpointer.load``
    is called with ``optim_state_offload=True`` so it skips the GPU-materialising
    primer and the device-moving ``load_state_dict``.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = {
                "step": torch.zeros((), dtype=torch.int64),
                "exp_avg": torch.zeros(p.shape, dtype=torch.float32, device="cpu"),
                "exp_avg_sq": torch.zeros(p.shape, dtype=torch.float32, device="cpu"),
            }
            if group.get("amsgrad"):
                st["max_exp_avg_sq"] = torch.zeros(
                    p.shape, dtype=torch.float32, device="cpu"
                )
            optimizer.state[p] = st


def _spill_moments_to_disk(optimizer, spill_dir: Path) -> dict:
    """Move each param's loaded ``exp_avg``/``exp_avg_sq`` from CPU to disk and
    drop the CPU tensors (keep the tiny ``step``). Returns a param→path map.

    For 8B the moments are ~64 GB — leaving them on CPU alongside the fold's
    ~2-gradient working set (~64 GB) overruns a 128 GB pod. On disk during the
    fold, host RAM holds only the fold; moments are streamed back one param at a
    time for the step.
    """
    paths: dict = {}
    idx = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p)
            if not st or "exp_avg" not in st or st["exp_avg"] is None:
                continue
            path = spill_dir / f"opt_{idx}.pt"
            idx += 1
            blob = {"exp_avg": st["exp_avg"], "exp_avg_sq": st["exp_avg_sq"]}
            if "max_exp_avg_sq" in st:
                blob["max_exp_avg_sq"] = st["max_exp_avg_sq"]
            _save_verified(blob, path)
            st["exp_avg"] = None  # free CPU; step stays for bias correction
            st["exp_avg_sq"] = None
            if "max_exp_avg_sq" in st:
                st["max_exp_avg_sq"] = None
            paths[p] = path
    return paths


def _offload_optimizer_step(optimizer, dev: torch.device, moment_paths: dict) -> None:
    """AdamW step with optimizer moments streamed disk→GPU→disk per parameter.

    The optimizer is idle through the whole micro-batch/fold phase; its moments
    live on disk so neither GPU nor host RAM carries them then. Each param's pair
    is read to the GPU only for its (unchanged) repop kernel call, then written
    back — GPU and RAM hold at most one param's moments at a time. The kernel,
    args, and per-element math are identical to ``FSDPAwareRepopAdamW.step``, so
    the result is bitwise-identical; only where the moments rest changes.

    NOTE: mirrors the per-parameter body of
    ``src/pretrain/optim/adamw_repop.py::FSDPAwareRepopAdamW.step`` — keep in
    sync if that kernel call changes. The single-device audit has plain (non-
    DTensor) params, so no ``to_local`` unwrap is needed.
    """
    from repop import ops

    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            st = optimizer.state[p]
            st["step"] += 1
            blob = _load_verified(moment_paths[p], dev)
            exp_avg = blob["exp_avg"]
            exp_avg_sq = blob["exp_avg_sq"]
            max_eas = blob.get("max_exp_avg_sq")
            ops.adamw_kernel_step(
                param=p.data,
                grad=p.grad,
                exp_avg=exp_avg,
                exp_avg_sq=exp_avg_sq,
                max_exp_avg_sq=max_eas,
                step=int(st["step"].item()),
                lr=group["lr"],
                beta1=beta1,
                beta2=beta2,
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                amsgrad=group["amsgrad"],
            )
            out = {"exp_avg": exp_avg.to("cpu"), "exp_avg_sq": exp_avg_sq.to("cpu")}
            if max_eas is not None:
                out["max_exp_avg_sq"] = max_eas.to("cpu")
            _save_verified(out, moment_paths[p])
            del exp_avg, exp_avg_sq, max_eas, blob


def _materialize_moments(optimizer, moment_paths: dict) -> None:
    """Reload the disk-resident moments back into ``optimizer.state`` (on CPU) so
    ``compute_state_hash`` can read them. Called once at the end, after the fold
    is freed — the ~64 GB (8B) fits when nothing else large is resident."""
    for p, path in moment_paths.items():
        blob = _load_verified(path, "cpu")
        st = optimizer.state[p]
        st["exp_avg"] = blob["exp_avg"]
        st["exp_avg_sq"] = blob["exp_avg_sq"]
        if blob.get("max_exp_avg_sq") is not None:
            st["max_exp_avg_sq"] = blob["max_exp_avg_sq"]


# --------------------------------------------------------------------------
# Automatic memory plan (CUDA).
#
# A volunteer running the published kit on a 24 GB consumer card should not have
# to know which offload flags to pass. These two helpers turn the existing
# offloads on when the card cannot hold the default plan, and leave a card with
# room behaving exactly as before.
#
# Both decide from MEASURED free VRAM rather than a full analytic model: by the
# time each runs, the expensive-to-predict things (the model, and later the bf16
# grad model) are already resident, so the only predicted term is the one with an
# exact closed form. An explicit flag is never turned OFF — auto only escalates.
# --------------------------------------------------------------------------

#: Headroom left for per-layer activations and the loss's fp32 chunk temporaries,
#: which are not worth modelling exactly. Empirical; override with
#: PRETRAIN_AUDIT_MEM_RESERVE_GB when tuning on a new card.
_MEM_RESERVE_GB = float(os.environ.get("PRETRAIN_AUDIT_MEM_RESERVE_GB", "3.0"))


def _cuda_free_bytes(dev: torch.device) -> int | None:
    """Free VRAM as the driver sees it, or None when that is not meaningful."""
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        # Hand back anything the caching allocator is sitting on first, so "free"
        # reflects the card rather than our own cache.
        torch.cuda.empty_cache()
        return int(torch.cuda.mem_get_info()[0])
    except Exception:
        return None


def _loss_transient_bytes(cfg) -> int:
    """Bytes the fused CE/z-loss holds at its peak for ONE micro-batch.

    Two ``[micro_batch * seq_len, vocab]`` matrices: the logits saved for the
    backward, and the grad-logits written in the backward. Both are the model's
    output dtype (bf16 under mixed precision), because ``_FusedCEZLoss`` writes
    each chunk straight into the output dtype. These dominate every other
    activation at a 128 k vocab, and unlike the transformer blocks their size is
    exact rather than estimated.
    """
    rows = int(cfg.train.micro_batch_size) * int(cfg.train.seq_len)
    elem = 2 if bool(cfg.run.mixed_precision) else 4
    return 2 * rows * int(cfg.model.vocab_size) * elem


def _plan_optimizer_offload(
    dev, model, cfg, *, requested: bool, from_init: bool, world_size: int
) -> bool:
    """Decide ``--offload-optimizer`` with the model resident but no optimizer yet.

    AdamW's moments are ``exp_avg`` + ``exp_avg_sq`` in fp32, i.e. 8 bytes per
    parameter, which at 1.6 B params is 12.9 GB — the single largest block on the
    card and the one most likely to be what a 24 GB card cannot find room for.
    """
    if requested:
        return True
    free = _cuda_free_bytes(dev)
    if free is None or from_init or cfg.optim.name != "adamw_repop":
        # --offload-optimizer is unsupported on those paths; say nothing and let
        # the existing errors speak if the user asked for it explicitly.
        return requested
    n = sum(p.numel() for p in model.parameters())
    moments = 8 * n
    # Same gate as the loop: a separate grad model only exists when the cluster
    # was sharded (parallelize_llama3_repop returns early at ws <= 1). Its cost
    # differs by path. MPS emulates mixed precision with an in-place bf16 param
    # cast, so 2 bytes/param. CUDA builds a REAL ``fully_shard`` model, and
    # FSDP2's MixedPrecisionPolicy keeps the sharded parameter in its built
    # dtype (fp32) and casts only the all-gathered copy — so 4 bytes/param
    # resident plus a 2-byte/param unsharded bf16 buffer at ws=1.
    if not (bool(cfg.run.mixed_precision) and world_size > 1):
        grad_model = 0
    elif dev.type == "mps":
        grad_model = 2 * n
    else:
        grad_model = 6 * n
    dev_grads = 4 * n
    reserve = int(_MEM_RESERVE_GB * 1024**3)
    need = moments + grad_model + dev_grads + _loss_transient_bytes(cfg) + reserve
    GB = 1024**3
    if free >= need:
        LOG.info(
            "memory plan: optimizer stays on device (free %.1f GiB >= need %.1f GiB)",
            free / GB, need / GB,
        )
        return False
    LOG.info(
        "memory plan: optimizer-state offload AUTO-ENABLED — free %.1f GiB < "
        "need %.1f GiB (moments %.1f + grad model %.1f + device grads %.1f "
        "+ loss transient %.1f + reserve %.1f). Pass --offload-optimizer to make "
        "this explicit, or raise PRETRAIN_AUDIT_MEM_RESERVE_GB to tune it.",
        free / GB, need / GB, moments / GB, grad_model / GB, dev_grads / GB,
        _loss_transient_bytes(cfg) / GB, reserve / GB,
    )
    return True


def _plan_master_offload(
    dev, cfg, *, requested: bool, has_grad_model: bool, n_params: int
) -> bool:
    """Decide ``--offload-master`` just before the micro-batch loop.

    By this point the master, the grad model and any optimizer state are all
    placed, so free VRAM is measured. What is still to come is the backward's
    gradients and the loss's two ``[rows, vocab]`` matrices, and those are what
    ``need`` has to cover: predicting only the loss transient understates it by
    the whole gradient set and lets a card that cannot fit sail past the check.
    Spilling the master is only possible when a separate grad model owns the
    forward/backward; otherwise the master IS the compute model.
    """
    if requested:
        return True
    if not has_grad_model:
        return False
    free = _cuda_free_bytes(dev)
    if free is None:
        return False
    # FSDP2 reduces to fp32 (reduce_dtype), so the gradients are 4 bytes per
    # parameter and none of them exist yet.
    dev_grads = 4 * n_params
    loss = _loss_transient_bytes(cfg)
    reserve = int(_MEM_RESERVE_GB * 1024**3)
    need = dev_grads + loss + reserve
    GB = 1024**3
    if free >= need:
        LOG.info(
            "memory plan: fp32 master stays on device (free %.1f GiB >= need "
            "%.1f GiB)", free / GB, need / GB,
        )
        return False
    LOG.info(
        "memory plan: fp32-master offload AUTO-ENABLED — free %.1f GiB < need "
        "%.1f GiB (device grads %.1f + loss transient %.1f + reserve %.1f). The "
        "master is idle between _sync_grad_model and the step, so this is "
        "bitwise-neutral.",
        free / GB, need / GB, dev_grads / GB, loss / GB, reserve / GB,
    )
    return True


def _plan_grad_offload(
    dev, cfg, *, requested: bool, has_grad_model: bool, n_params: int
) -> bool:
    """Decide host-side gradient accumulation, just before the micro-batch loop.

    Gradients persist across an accumulation window by design, and one
    micro-batch's transient is large: measured on a 4090 at the audit shape, a
    bare FSDP2 grad model sits at 6.43 GB and one micro-batch peaks at 23.12 GB,
    so the transient is about 16.7 GB (this micro-batch's gradients 6.45, the
    logits 4.2, the grad-logits 4.2, and about 1.8 of all-gather and
    activations). The first micro-batch therefore fits on a 24 GB card and the
    second does not, because it starts with the previous one's 6.45 GB of
    gradients already resident.

    Draining each micro-batch's gradients to the host as fp32 and clearing the
    device leaf makes every micro-batch look like the first. It costs one
    device-to-host copy per micro-batch.
    """
    if requested:
        return True
    if not has_grad_model:
        return False
    free = _cuda_free_bytes(dev)
    if free is None:
        return False
    # The worst micro-batch is the second one onward: it carries the previous
    # micro-batch's gradients and builds its own. The all-gather and the
    # activations are left to the reserve; FSDP2 reshards per module, so only
    # one module's bf16 copy is live at a time and the pair measured about
    # 1.8 GB, comfortably inside it.
    carried = 4 * n_params
    this_mb = 4 * n_params
    loss = _loss_transient_bytes(cfg)
    reserve = int(_MEM_RESERVE_GB * 1024**3)
    need = carried + this_mb + loss + reserve
    GB = 1024**3
    if free >= need:
        LOG.info(
            "memory plan: gradients stay on device (free %.1f GiB >= need "
            "%.1f GiB)", free / GB, need / GB,
        )
        return False
    LOG.info(
        "memory plan: host gradient accumulation AUTO-ENABLED — free %.1f GiB < "
        "need %.1f GiB (gradients carried from the previous micro-batch %.1f + "
        "this micro-batch's %.1f + loss transient %.1f + reserve %.1f). "
        "Reduce-scatter is the identity at world_size 1, so the host fp32 sum "
        "is the same additions in the same order.",
        free / GB, need / GB, carried / GB, this_mb / GB, loss / GB, reserve / GB,
    )
    return True


def _spill_master_to_disk(model, spill_dir: Path) -> dict:
    """Spill the fp32 master params to disk and free them from the device.

    Wherever a separate ``grad_model`` owns the forward/backward, the fp32 master
    is idle from ``_sync_grad_model`` until the optimizer step. That is the MPS
    bf16-emulation path AND the CUDA path, which wraps a real ``fully_shard``
    grad model; only a single-rank fp32 run has no grad model, and there the
    master is the compute model. Spilling it during the micro-batch/fold phase
    keeps its ~6 GB (1.6 B) / ~32 GB (8 B) out of the pool exactly when the pool
    peaks. Nothing here is device-specific: the caller frees the CUDA allocator's
    now-unused blocks after the spill, as it drains the MPS cache. Reload with ``_reload_master_from_disk`` before the step. Mirrors the
    optimizer-moment offload; the disk round-trip changes no value, so the audit
    stays bitwise-identical. Returns a param→path map (keyed by Parameter, whose
    identity the optimizer state also keys on). Per-param so RAM holds at most one
    extra param's worth during the copy."""
    paths: dict = {}
    for idx, p in enumerate(model.parameters()):
        path = spill_dir / f"master_{idx}.pt"
        _save_verified(p.data.to("cpu"), path)
        # Drop the device storage (nothing else references it: grad_model holds
        # its own bf16 copy, the optimizer keys on the Parameter not its data).
        p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        paths[p] = path
    return paths


def _reload_master_from_disk(model, master_paths: dict, dev: torch.device) -> None:
    """Restore the spilled fp32 master params to ``dev`` for grad assignment,
    grad-norm/clip, the optimizer step, LSQ refresh, and the state hash."""
    for p in model.parameters():
        path = master_paths.get(p)
        if path is not None:
            p.data = _load_verified(path, dev)


def _empty_cache(dev: torch.device) -> None:
    """Return the allocator's freed-but-unreturned buffers to the device pool.

    With ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`` the MPS allocator never
    auto-evicts its cache, so the large fp32 logit/softmax transients from one
    micro-batch stay reserved and pile up across the accum×N inner loop until the
    pool is exhausted. Reclaiming keeps the working set to ~one micro-batch's
    peak. CUDA's caching allocator reuses freed blocks on its own, so this is
    only worth doing there when block SIZES change between phases and reuse
    stops being possible: releasing lets the next phase's differently-shaped
    allocation come from contiguous memory instead of failing between the old
    blocks. Purely an allocator hint — it frees nothing live, so it cannot change
    any result bit."""
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


# ---- Gradient sidecar (audit handoff) ----------------------------------------
# The chained-audit handoff (--save-checkpoint-dir) additionally exports the
# target step's final gradients as ``gradients.safetensors`` so a recipient can
# recompute the logged state hash without replaying the preceding interval.
# The gradients are the full logical master's
# reduced/post-clipped gradients — exactly the tensors
# ``audit_shard_state_digest(include_grads=True)`` hashed — stored losslessly
# (dtype/shape/bits preserved). Tensor keys are ``model.named_parameters()``
# names; parameters whose ``grad is None`` are listed in the metadata's
# ``none_grad_names`` rather than written as a tensor, so a verifier can tell a
# legitimate None gradient from a missing/corrupt entry.
_GRAD_SIDECAR_FILENAME = "gradients.safetensors"
_GRAD_SIDECAR_FORMAT = "pretrain-audit-gradients"
_GRAD_SIDECAR_VERSION = 1


def _write_gradient_sidecar(grads: dict[str, torch.Tensor | None], path: Path) -> Path:
    """Write the held target-step gradients to ``path`` as a safetensors sidecar.

    ``grads`` maps ``model.named_parameters()`` names to the detached gradient
    (or ``None`` for a parameter whose ``grad`` was None). Each gradient is
    stored losslessly (``.detach().contiguous().cpu()``; no cast — bf16
    round-trips). ``None`` entries are recorded in the metadata's
    ``none_grad_names`` list rather than as a tensor, so a verifier can tell a
    legitimate None gradient from a missing/corrupt entry.
    """
    import safetensors.torch as sft

    tensors: dict[str, torch.Tensor] = {}
    none_names: list[str] = []
    for name, g in grads.items():
        if g is None:
            none_names.append(name)
        else:
            tensors[name] = g.detach().contiguous().cpu()
    sft.save_file(
        tensors,
        str(path),
        metadata={
            "format": _GRAD_SIDECAR_FORMAT,
            "format_version": str(_GRAD_SIDECAR_VERSION),
            "none_grad_names": json.dumps(none_names),
        },
    )
    return path


def _save_chained_audit_checkpoint(
    *,
    save_dir: str,
    step: int,
    consumed: int,
    digest: str,
    chained_hash_meta: str | None,
    meta_obj: dict,
    stream_state,
    model,
    optimizer,
    spike_state: dict,
    batch_hashers,
    batch_digest,
    N: int,
    gradients: dict[str, torch.Tensor | None] | None = None,
) -> Path:
    """Write a fresh, fully-loadable checkpoint capturing this audit's post-step
    state, so the NEXT interval can be audited starting from it.

    Reuses ``Checkpointer.save`` for the DCP model/optimizer shards and
    the rank-0 metadata (meta.json, spike_protocol.json, global_stream.json,
    rng.rank_0.pt, batch_hasher.rank_0.bin, state_hash.txt, _COMPLETE). The
    ``meta`` copies the loaded checkpoint's descriptor verbatim (seed, config,
    repop_env, topology, reduction/clip algos, ...) and
    advances only step / consumed_tokens / chained_hash / windows_emitted, so the
    next audit rebuilds an identical run and resumes the hash chain from ``digest``.

    ``state_hash`` is passed verbatim (the chained ``digest``) — NOT recomputed —
    because the loop's canonical hash is taken pre-``zero_grad`` (grads live) and
    the grads are already cleared here. ``sampler_state`` is None because an
    auditable run's stream position lives in the single global_stream.json.

    ``Checkpointer.save`` writes only rank 0's per-rank files in a single-process
    run, but the next audit primes one batch hasher per virtual rank r in
    range(N) from ``batch_hasher.rank_{r}.bin``. Ranks 1..N-1 are written here
    from this audit's per-rank hashers so the handoff carries the full set. (The
    audit only reads rng.rank_0.pt, so per-rank RNG needs no extra files.)

    Takes ownership of ``gradients``: writes them directly into the unpublished
    checkpoint and clears the dictionary before DCP save to release host memory.
    All files are created on the destination filesystem; one directory rename
    publishes the complete handoff. No separate gradient temp file/copy is needed.
    """
    from pretrain.train.checkpoint import CheckpointMeta, Checkpointer

    save_meta_dict = dict(meta_obj)
    save_meta_dict["step"] = step
    save_meta_dict["consumed_tokens"] = consumed
    # Mirrors loop._save_checkpoint exactly: meta carries the RUNNING chain
    # (what the next interval's hashes chain from), while state_hash.txt gets
    # ``digest`` — the target-step checkpoint hash. They coincide on a
    # hash-due target; on an off-cadence target the chain is older than the
    # side-link and the next audit must seed from the chain, not the link.
    save_meta_dict["chained_hash"] = chained_hash_meta
    save_meta_dict["windows_emitted"] = getattr(stream_state, "windows_emitted", 0)
    # from_dict: ``meta_obj`` came from the loaded checkpoint's meta.json,
    # which may carry retired keys written by older code.
    save_meta = CheckpointMeta.from_dict(save_meta_dict)

    output_root = Path(save_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"step_{step:09d}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite handoff: {destination}")

    # The hash gate has already passed. Stage only publication, not replay:
    # readers must never select a checkpoint whose sidecar write failed.
    with tempfile.TemporaryDirectory(prefix=".handoff-", dir=output_root) as staging:
        staged_dir = Path(staging) / destination.name
        staged_dir.mkdir()
        if gradients is not None:
            _write_gradient_sidecar(gradients, staged_dir / _GRAD_SIDECAR_FILENAME)
            gradients.clear()
        saved_dir = Checkpointer(staging).save(
            step,
            model,
            optimizer,
            None,  # auditable/canonical stream → global_stream.json
            save_meta,
            batch_digest=batch_digest,
            batch_hasher_digest=(
                batch_hashers[0].local_digest() if batch_hashers is not None else None
            ),
            spike_state=spike_state,
            global_stream_state=stream_state,
            state_hash=digest,
        )
        if batch_hashers is not None:
            for r in range(1, N):
                (saved_dir / f"batch_hasher.rank_{r}.bin").write_bytes(
                    batch_hashers[r].local_digest()
                )
        saved_dir.rename(destination)
        return destination


# The env the audit may re-apply from a checkpoint's meta.json. Mirrors what
# ``loop._capture_repop_env`` can WRITE — every ``REPOP*``-prefixed var plus
# the two contract keys — so a legitimate meta always passes. Anything else
# (LD_*, *_PROXY, GOOGLE_APPLICATION_CREDENTIALS, allocator knobs, ...) is an
# injection attempt or corruption: meta.json is untrusted input (the audit is
# designed to load checkpoints produced by someone else) and these vars are
# applied to ``os.environ`` before kernels dispatch and before any network
# fetch runs, so an unrestricted apply would hand the checkpoint author
# control of the auditor's process environment.
#
# These hand-copy ``loop._CONTRACT_ENV_KEYS`` + the ``startswith("REPOP")``
# rule because loop is repop-backed and importing it here would violate this
# file's "apply repop_env BEFORE importing repop-backed modules" ordering.
# ``tests/test_audit_repop_env.py`` asserts they stay equal to loop's, so the
# copy cannot drift silently.
_REPOP_ENV_CONTRACT_KEYS = ("CUBLAS_WORKSPACE_CONFIG", "TORCH_CUDA_ARCH_LIST")

# Even within the ``REPOP*`` namespace, a filesystem/loader REDIRECT key hands
# repop an attacker-controlled path — e.g. ``REPOP_METAL_SHADER_DIR`` points
# the Metal backend at an arbitrary shader/kernel source dir, so a doctored
# meta could get the auditor to compile and run attacker kernels while the run
# still "reproduces". A genuine H100 cluster writer never sets these (they are
# dev/Mac-only overrides), so the prefix rule gives no protection for the
# doctored case; refuse any REPOP key that denotes a path/dir/plugin. Matched
# by shape (suffix/substring) rather than an exact name so a future path knob
# is refused by default until explicitly reviewed.
_REPOP_ENV_REDIRECT_SUFFIXES = ("_DIR", "_PATH", "_FILE", "_ROOT")
_REPOP_ENV_REDIRECT_SUBSTRINGS = ("SHADER", "LIB", "PLUGIN", "PRELOAD", "LD_")


def _is_redirect_key(k: str) -> bool:
    return k.endswith(_REPOP_ENV_REDIRECT_SUFFIXES) or any(
        s in k for s in _REPOP_ENV_REDIRECT_SUBSTRINGS
    )


def _apply_repop_env(repop_env: object) -> None:
    """Validate then apply a checkpoint meta's ``repop_env`` to ``os.environ``.

    Allowlist: ``REPOP*``-prefixed keys + ``_REPOP_ENV_CONTRACT_KEYS`` (the set
    ``loop._capture_repop_env`` records), EXCLUDING filesystem/loader redirect
    keys (see ``_is_redirect_key``); values must be scalars. Validates the
    WHOLE dict before applying anything, so a rejected meta leaves the
    environment untouched. Raises ``ValueError`` on any violation.
    """
    if repop_env is None:
        return
    if not isinstance(repop_env, dict):
        raise ValueError(
            f"meta.json repop_env must be a mapping of env vars, got "
            f"{type(repop_env).__name__} — corrupt or doctored checkpoint meta."
        )
    for k, v in repop_env.items():
        if not isinstance(k, str) or not (
            k.startswith("REPOP") or k in _REPOP_ENV_CONTRACT_KEYS
        ):
            raise ValueError(
                f"meta.json repop_env contains disallowed variable {k!r}: only "
                f"REPOP*-prefixed vars and {list(_REPOP_ENV_CONTRACT_KEYS)} are "
                "ever recorded by the training loop (loop._capture_repop_env). "
                "Refusing to apply it — a checkpoint's meta.json is untrusted "
                "input and must not set arbitrary environment variables in the "
                "auditor's process."
            )
        if k == "REPOP_FORCE_FSDP_WS1":
            raise ValueError(
                "meta.json must not set REPOP_FORCE_FSDP_WS1: single-rank FSDP "
                "wrapping is controlled by the auditor, not the checkpoint."
            )
        if _is_redirect_key(k):
            raise ValueError(
                f"meta.json repop_env contains a filesystem/loader redirect "
                f"variable {k!r}: it would point repop at an attacker-controlled "
                "path (e.g. a shader/kernel source dir). A genuine cluster run "
                "never records these; refusing to apply it."
            )
        if not isinstance(v, (str, int, float, bool)):
            raise ValueError(
                f"meta.json repop_env[{k!r}] must be a scalar, got "
                f"{type(v).__name__}."
            )
    for k, v in repop_env.items():
        os.environ[k] = str(v)
        LOG.info("repop_env: %s=%s", k, v)


# Canonical digest length: every state-hash digest this repo writes
# (state_hash.txt, state_hash_init.txt, state_hashes.jsonl) is
# blake2b hex — derived from the writer's digest size so a future change to
# state_hash._DIGEST_SIZE can't make this validator reject every genuine
# digest as "truncated". pretrain.train.state_hash is pure-Python (no repop),
# so importing it here is safe w.r.t. the repop import-ordering below.
from pretrain.train.state_hash import _DIGEST_SIZE as _STATE_HASH_DIGEST_SIZE

_DIGEST_HEX_LEN = 2 * _STATE_HASH_DIGEST_SIZE


def _looks_like_digest(s: str) -> bool:
    """True iff ``s`` is exactly one canonical hex digest."""
    return len(s) == _DIGEST_HEX_LEN and re.fullmatch(r"[0-9a-fA-F]+", s) is not None


def _read_digest_file(path: Path, *, what: str) -> str:
    """Read a ``state_hash.txt``-style file and return its (lower-cased) digest,
    validating that it actually contains one canonical digest. Used for the
    from-init auto-target reads (state_hash_init.txt / a step-0 state_hash.txt)
    as well as ``--expect-hash`` file resolution, so a wrong/truncated sibling
    file fails fast here instead of after a full replay. Caller checks
    existence; a malformed content raises ``ValueError``."""
    content = path.read_text().strip()
    if not _looks_like_digest(content):
        raise ValueError(
            f"{what} file {path} does not contain a {_DIGEST_HEX_LEN}-char hex "
            f"digest (starts with {content[:40]!r}) — expected a "
            "state_hash.txt-style file."
        )
    return content.lower()


def _resolve_expect_hash(expect_hash: str) -> str:
    """Resolve ``--expect-hash`` to a literal digest (lower-cased). Accepts
    either the digest itself (a full ``_DIGEST_HEX_LEN``-char hex string) or a
    path to a ``state_hash.txt``-style file containing one.

    Validated *eagerly* (caller runs this before the expensive replay) so every
    malformed input fails fast instead of after a multi-hour audit:
      * a pure-hex string of any OTHER length that names no existing file is
        rejected as a truncated digest — run stdout truncates digests to 16
        chars, and a pasted prefix used to pass the old ``{16,}`` check and
        fail the comparison only at the very end of the replay;
      * a path must exist AND contain a full-length hex digest (a wrong file —
        a log, a JSON — used to be read verbatim and could never match).
    """
    s = expect_hash.strip()
    is_hex = re.fullmatch(r"[0-9a-fA-F]+", s) is not None
    if _looks_like_digest(s):
        return s.lower()  # literal digest
    p = Path(expect_hash)
    if not p.is_file():
        if is_hex:
            raise ValueError(
                f"--expect-hash {s!r} looks like a truncated hex digest "
                f"({len(s)} chars; the canonical state-hash digest is "
                f"{_DIGEST_HEX_LEN}). Run stdout truncates digests — take the "
                "full value from state_hash.txt / state_hashes.jsonl."
            )
        raise FileNotFoundError(
            f"--expect-hash {expect_hash!r} is neither a {_DIGEST_HEX_LEN}-char "
            f"hex digest nor an existing file (treated as a path, but it does "
            f"not exist)."
        )
    return _read_digest_file(p, what="--expect-hash")


# Upper bound on the dp_world_size the audit will emulate. meta.json is
# untrusted input; the audit's memory and wall-clock scale linearly with N
# (one virtual-rank loader + one rank slice per step each), so anything beyond
# the real cluster is corruption or a resource-exhaustion attempt. Set to the
# current cluster's dp degree (48); bump this one constant when the cluster
# grows (e.g. to 64).
_MAX_AUDIT_WORLD = 48

# meta.json keys that describe the SEGMENT a checkpoint terminates (the code,
# kernels, topology and clipper that RAN those steps) — as opposed to the keys
# that describe the checkpoint's own POSITION in the run (step, consumed_tokens,
# windows_emitted, chained_hash) or the whole run (seed). A resume that changes
# any of these forks a new segment: to audit the boundary interval, the state
# loads from the OLD checkpoint while the descriptor must come from the NEW
# segment (--descriptor-checkpoint). Absence matters as much as presence — a
# key missing from the descriptor meta is REMOVED so its legacy default applies
# (e.g. rewarm_anchor_tokens absent ⇒ -1 ⇒ no re-warm ramp).
_SEGMENT_META_KEYS = (
    "config_resolved",
    "repop_env",
    "reduction_mode",
    "dp_world_size",
    "dp_replicate",
    "dp_shard",
    "replicate_reduce_algo",
    "grad_norm_algo",
    "clip_algo",
    # Segment-scoped despite being a token count: the LR re-warm anchor is the
    # segment's FORK point, recorded identically in every checkpoint the
    # segment writes. A boundary audit (state from the pre-fork checkpoint,
    # whose meta has no anchor) must take it from the descriptor or it replays
    # the first post-fork step at full schedule LR while the live run took it
    # at lr = 0 exactly (found the hard way: Option-A boundary 50200→50201,
    # got b3c05da9… vs expected 350ed7b0…).
    "rewarm_anchor_tokens",
)


def _overlay_descriptor_meta(meta_obj: dict, desc_obj: dict) -> list[str]:
    """Overlay the segment-scoped keys of ``desc_obj`` onto ``meta_obj`` in
    place (see ``_SEGMENT_META_KEYS``). Position keys are never touched. The
    two metas must agree on ``seed`` — it drives the canonical data stream for
    the whole run, so a mismatch means the descriptor belongs to a different
    run, not a different segment. Returns the keys whose values changed, for
    loud logging at the call site."""
    if "seed" in meta_obj and "seed" in desc_obj and meta_obj["seed"] != desc_obj["seed"]:
        raise ValueError(
            f"--descriptor-checkpoint belongs to a different run: seed "
            f"{desc_obj['seed']} != --checkpoint's seed {meta_obj['seed']}."
        )
    changed: list[str] = []
    for k in _SEGMENT_META_KEYS:
        if k in desc_obj:
            if meta_obj.get(k) != desc_obj[k]:
                changed.append(k)
            meta_obj[k] = desc_obj[k]
        elif k in meta_obj:
            del meta_obj[k]
            changed.append(k)
    return changed


def _require_repop_backend(dev: torch.device, build_info: dict) -> None:
    """Fail fast when the repop build cannot serve the requested device.

    A replay result that cannot name the kernel build that produced it is not
    audit evidence: the state hash is a claim about repop's kernels as much as
    about this repo's model code, which is why ``build_info`` is recorded into
    every result. On MPS the Metal backend must actually be compiled in — a
    CPU-only repop on an mps device would fall back to CPU kernels op by op
    and "pass" without exercising the hardware the verification claims to
    cover. Same shape as the device-availability gate: loud and up front,
    never a silent downgrade.
    """
    if dev.type == "mps" and "metal" not in build_info.get("backends", []):
        raise RuntimeError(
            f"--device {dev} requested but this repop build has no Metal "
            f"backend (build_info: commit={build_info.get('commit')!r}, "
            f"backends={build_info.get('backends')!r}). Install the macOS "
            "audit wheel (repop-*-macosx_*_arm64.whl), or pass --device cpu "
            "to verify on CPU kernels explicitly."
        )
    if dev.type == "cuda" and "cuda" not in build_info.get("backends", []):
        raise RuntimeError(
            f"--device {dev} requested but this repop build has no CUDA "
            f"backend (build_info: commit={build_info.get('commit')!r}, "
            f"backends={build_info.get('backends')!r}). Install the Linux "
            "audit wheel (repop-*-linux_x86_64.whl), or pass --device cpu "
            "to verify on CPU kernels explicitly."
        )


def _parse_args():
    p = argparse.ArgumentParser(prog="pretrain.cli.audit_replay")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="cluster checkpoint dir to start from. Optional ONLY for a "
        "config-only init audit (--from-init --until-step 0 with --config-name), "
        "which loads no weights; required for every replay and to read a "
        "recorded init/checkpoint hash from disk.",
    )
    p.add_argument("--config-name", default=None, help="train config (else read from meta.json)")
    p.add_argument(
        "--data-root",
        default=None,
        help="redirect every data source to <data-root>/<source-basename> instead "
        "of the path baked into the config. Point this at an already-staged tree "
        "(e.g. ./data/audit_data/shards) to audit against a partial dataset. "
        "Mutually exclusive with --gcs-root, which fetches that tree for you.",
    )
    p.add_argument(
        "--gcs-root",
        default=None,
        help="GCS mirror of the dataset (gs://bucket/.../data/shards). When set, "
        "the audit validates the checkpoint, then downloads ONLY the shards this "
        "interval consumes into --fetch-dest and replays against them — so you "
        "only need the checkpoint locally, not the full corpus.",
    )
    p.add_argument(
        "--fetch-dest",
        default="./data/audit_data",
        help="local destination for --gcs-root fetches; shards land under "
        "<fetch-dest>/shards/<source>/ (default: ./data/audit_data)",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    from pretrain.cli.cpu_threads import parse_cpu_threads

    p.add_argument(
        "--cpu-threads",
        default="auto",
        type=parse_cpu_threads,
        help="CPU worker count: auto respects OMP/MKL settings, otherwise prefers "
        "Apple performance cores; other hosts retain the Torch default. "
        "A positive integer overrides the count. Applies only to --device cpu.",
    )
    p.add_argument(
        "--from-init",
        action="store_true",
        help="reconstruct the START state from the seed instead of loading "
        "--checkpoint, then replay forward to --checkpoint's step. Rebuilds "
        "build_model + init_weights(seed) (init is device-independent), verifies "
        "the regenerated init reproduces the run's recorded init hash "
        "(state_hash_init.txt, next to the checkpoints), then runs the standard "
        "replay to --checkpoint and compares to its state_hash.txt. So "
        "`--from-init --checkpoint .../step_000000010` audits init -> step 10. "
        "Add `--until-step 0` for a pure init-hash verification (no replay); "
        "--expect-hash then overrides the init target. --checkpoint supplies the "
        "run descriptor (seed/config/repop_env/topology) from its meta.json. Only "
        "meaningful for runs trained with train.state_hash.at_init=true.",
    )
    p.add_argument(
        "--descriptor-checkpoint",
        default=None,
        help="checkpoint whose meta.json supplies the SEGMENT descriptor "
        "(repop_env, resolved config, topology, reduction/clip algorithms) "
        "instead of --checkpoint's. For "
        "auditing a resume-boundary interval: the starting STATE (weights / "
        "moments / gamma / RNG / stream position / chained hash) still loads "
        "from --checkpoint, but the interval was TRAINED by the resumed run, "
        "whose descriptor lives in ITS first checkpoint. Example — old run "
        "forked at step 50,200 into a fixed run whose first checkpoint is "
        "step 50,300: --checkpoint <old>/step_000050200 --descriptor-checkpoint "
        "<new>/step_000050300 --until-step 50300 --expect-hash "
        "<new>/step_000050300/state_hash.txt. Seeds must agree; position "
        "fields are never taken from the descriptor.",
    )
    p.add_argument(
        "--until-step",
        type=int,
        default=None,
        help="replay until this optimizer step (default: one ckpt_every_tokens "
        "interval; with --from-init, default is --checkpoint's step; 0 = verify "
        "init only, no replay)",
    )
    p.add_argument(
        "--expect-hash",
        default=None,
        help="expected next-checkpoint digest to compare against: either the "
        "literal 64-char hex digest or a path to its state_hash.txt. Validated "
        "up front, before the replay — a truncated digest (run stdout cuts "
        "them to 16 chars; use state_hashes.jsonl for the full value) or a "
        "missing/malformed file fails fast (default: don't compare, just "
        "print the digest)",
    )
    p.add_argument(
        "--fold-spill-dir",
        default=None,
        help="directory for the cross-replica fold's disk spill (bounds RAM to "
        "~2 gradients). Point at fast local scratch. Default: auto temp dir when "
        "dp_replicate>2, else in-RAM. Use --no-fold-spill to force in-RAM.",
    )
    p.add_argument(
        "--no-fold-spill",
        action="store_true",
        help="keep the cross-replica fold entirely in RAM (no disk spill).",
    )
    p.add_argument(
        "--offload-optimizer",
        action="store_true",
        help="keep AdamW moments on disk and stream them per-parameter to the GPU "
        "for the step. Frees ~2x model size from BOTH VRAM and host RAM during the "
        "fold — required to audit models whose params+moments don't fit one GPU "
        "(e.g. 8B). repop-AdamW only.",
    )
    p.add_argument(
        "--optimizer-offload-dir",
        default=None,
        help="directory for the offloaded AdamW moments (default: auto temp dir). "
        "Point at fast local scratch with room for ~2x model size.",
    )
    p.add_argument(
        "--offload-master",
        action="store_true",
        help="spill the fp32 master params to disk during the micro-batch/fold "
        "phase, reloading them only for the step/hash. Frees ~1x model size from "
        "the unified pool at peak. MPS bf16-emulation path only (the fwd/bwd runs "
        "on the bf16 grad_model, so the master is idle then); a no-op without a "
        "separate grad_model. Bitwise-identical.",
    )
    p.add_argument(
        "--offload-grads",
        action="store_true",
        help="accumulate each micro-batch's gradients on the host as fp32 and "
        "clear the device leaf, so a micro-batch does not start with the "
        "previous one's gradients resident. Frees ~1x model size at peak, which "
        "on a 24 GB card is the difference between the second micro-batch "
        "fitting and not. Costs one device-to-host copy per micro-batch. "
        "Bitwise-identical: at world_size 1 the reduce-scatter is the identity, "
        "so this is the same fp32 additions in the same order.",
    )
    p.add_argument(
        "--master-offload-dir",
        default=None,
        help="directory for the offloaded fp32 master (default: auto temp dir). "
        "Needs room for ~1x model size.",
    )
    p.add_argument(
        "--save-checkpoint-dir",
        default=None,
        help="after the replay reaches the target step and its state_hash is "
        "computed, save a NEW checkpoint under <dir>/step_<target> containing the "
        "updated weights/optimizer, the replayed RNG + per-rank batch-hasher "
        "chains + global-stream position + spike state, and a meta.json that copies "
        "the loaded checkpoint's metadata with step/consumed_tokens/chained_hash/"
        "windows_emitted advanced to the target. This lets another user audit the "
        "NEXT interval by pointing --checkpoint at that dir, without re-auditing "
        "this one. Any target step works (the target-step hash is chained from "
        "the running chain exactly as the loop stamps every checkpoint it "
        "saves); requires periodic state hashing (state_hash.every_n_steps > 0).",
    )
    p.add_argument(
        "--loss-log",
        default=None,
        help="write virtual rank 0's per-step mean CE / z-loss to this JSON "
        "file ({\"records\": [{step, consumed_tokens, loss_ce, loss_zloss}, "
        "...]}), atomically rewritten after every replayed step. An honest "
        "replay reproduces the cluster rank 0's logged loss_ce / loss_zloss "
        "(logs/metrics.jsonl) bit-for-bit. Also returned in the result "
        "JSON as rank0_losses.",
    )
    return p.parse_args()


def _write_loss_log(path: Path, records: list[dict]) -> None:
    """Atomically (re)write the rank-0 loss log as a single JSON document.

    Rewritten after every replayed step (the records are a few floats each, so
    a full rewrite is trivially cheap) via write-to-temp + ``os.replace`` so a
    client polling the file never observes a torn/partial JSON document.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def audit_replay(
    checkpoint: str,
    *,
    config_name: str | None = None,
    device: str = "cuda",
    from_init: bool = False,
    until_step: int | None = None,
    expect_hash: str | None = None,
    data_root: str | None = None,
    gcs_root: str | None = None,
    fetch_dest: str = "./data/audit_data",
    fold_spill_dir: str | None = None,
    no_fold_spill: bool = False,
    offload_optimizer: bool = False,
    optimizer_offload_dir: str | None = None,
    offload_master: bool = False,
    offload_grads: bool = False,
    master_offload_dir: str | None = None,
    save_checkpoint_dir: str | None = None,
    descriptor_checkpoint: str | None = None,
    loss_log: str | None = None,
) -> dict:
    """Replay one interval on a single device. Returns a result dict with the
    final step, consumed_tokens, and state_hash digest.

    ``loss_log`` (rank-0 loss verification): when set, write virtual rank 0's
    per-step mean CE / z-loss to this JSON file ({"records": [{step,
    consumed_tokens, loss_ce, loss_zloss}, ...]}), atomically rewritten after
    every replayed step. The values use the same fp64 host fold over rank 0's
    microbatches as the training loop's rank-local ``loss_ce`` /
    ``loss_zloss`` metrics keys, so an honest replay reproduces the cluster's
    logs/metrics.jsonl values bit-for-bit — a cheap streaming cross-check a
    verifier can run against the run's logged losses before accepting the
    audit's hand-off checkpoint. The records are also returned in the result dict as
    ``rank0_losses`` (always, even without ``loss_log``).

    ``save_checkpoint_dir`` (chained auditing): when set, persist a fresh,
    fully-loadable checkpoint at ``<dir>/step_<target>`` once the replay finishes,
    so a second user can audit the NEXT interval starting from it (no need to
    re-audit this one). The saved state carries the updated weights and optimizer
    moments, plus the replayed RNG, per-rank batch-hasher chains,
    global-stream position, and spike state; its meta.json copies the loaded
    checkpoint's metadata with step / consumed_tokens / chained_hash /
    windows_emitted advanced to the target. The returned dict then also carries
    ``saved_checkpoint``. Any target step works — the saved state_hash.txt is
    the target-step hash chained from the running chain, exactly as the loop
    stamps every checkpoint it saves — but periodic state hashing must be on
    (``state_hash.every_n_steps > 0``).

    ``descriptor_checkpoint`` (resume-boundary auditing): a mid-run fork that
    changes any segment-scoped descriptor field (repop_env, config,
    clipper) makes the boundary interval unauditable
    from --checkpoint's own meta: the state to load is the OLD checkpoint's,
    but the steps were trained under the NEW segment's descriptor. Pass the
    new segment's first checkpoint here to overlay its segment-scoped meta
    keys (``_SEGMENT_META_KEYS``) before the replay; position fields and the
    chained hash still come from --checkpoint. One meta.json still equals one
    clipper/env — this flag stitches exactly ONE boundary interval; chains on
    either side audit as usual from their own segments' checkpoints."""
    # Resolve the expected digest first — if it's a path, fail fast on a bad
    # path before any of the expensive model build / load / replay work.
    expected_digest = _resolve_expect_hash(expect_hash) if expect_hash is not None else None

    # Config-only init audit: no checkpoint at all. The from-init path
    # regenerates the START state from the seed and loads nothing from disk, so
    # for a pure init verification (no replay) meta.json is unnecessary —
    # everything needed comes from --config-name (seed via cfg.run.seed, model
    # arch) and, optionally, --expect-hash (the target). Reject the combinations
    # that genuinely DO need a checkpoint (any replay, or no config to build from).
    config_only = checkpoint is None
    if descriptor_checkpoint is not None and (from_init or config_only):
        raise ValueError(
            "--descriptor-checkpoint audits a resume-boundary REPLAY interval; "
            "it cannot be combined with --from-init or a config-only init audit "
            "(init belongs to the original segment, whose descriptor is "
            "--checkpoint's own meta)."
        )
    if config_only:
        if not from_init:
            raise ValueError(
                "--checkpoint is required unless auditing init with --from-init."
            )
        if config_name is None:
            raise ValueError(
                "a config-only audit (no --checkpoint) needs --config-name to "
                "supply the seed and model architecture."
            )
        if until_step not in (None, 0):
            raise ValueError(
                "without a --checkpoint only init can be verified (there is no "
                "saved state to replay toward); pass --until-step 0 (or omit it)."
            )
        if save_checkpoint_dir is not None:
            raise ValueError(
                "--save-checkpoint-dir needs a real replay to a target step; it is "
                "meaningless for a config-only init audit (no --checkpoint)."
            )

    # Determinism mirrors the cluster auditable run.
    os.environ.setdefault("REPOP_EXECUTION_MODE", "cross_device_reproducible")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # The audit unshards the whole model onto one device; reduce allocator
    # fragmentation so the large transient tensors don't trip a spurious OOM.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Read meta BEFORE importing repop-backed modules so the run's full kernel/
    # determinism env (REPOP_* backward-variant selection, CUBLAS_WORKSPACE_CONFIG,
    # TORCH_CUDA_ARCH_LIST — see loop._capture_repop_env) is in place when repop
    # dispatches. These OVERRIDE
    # the setdefaults above; otherwise QAT/int8/cuBLAS paths silently fall back
    # to a different kernel and the replay diverges. Config-only mode has no meta
    # to read (and no kernels run for a pure init hash).
    meta_obj: dict = {}
    clip_algo = "global"  # config-only init audits never clip; replays overwrite below
    if not config_only:
        meta_obj = json.loads((Path(checkpoint) / "meta.json").read_text())
        if descriptor_checkpoint is not None:
            # Resume-boundary audit: the interval's steps were trained by the
            # segment that WROTE descriptor_checkpoint, so its meta supplies
            # everything that shaped the compute (repop_env / config /
            # topology / clipper), while the starting
            # state and position keep coming from --checkpoint. Must happen
            # HERE, before the repop_env application below and every
            # meta_obj.get() after it.
            desc_obj = json.loads(
                (Path(descriptor_checkpoint) / "meta.json").read_text()
            )
            changed = _overlay_descriptor_meta(meta_obj, desc_obj)
            LOG.info(
                "descriptor overlay from %s: %s",
                descriptor_checkpoint,
                ", ".join(changed) if changed else "no differences",
            )
        _apply_repop_env(meta_obj.get("repop_env"))

    from pretrain.data.loader import build_global_loader
    from pretrain.model import build_model
    from pretrain.optim.registry import build_optimizer
    from pretrain.parallel.env import set_seed
    from pretrain.train.batch_schedule import microbatches_per_step

    from pretrain.train.checkpoint import Checkpointer
    from pretrain.model.fused_loss import fused_ce_z_loss
    from pretrain.train.loop import (
        _autocast_ctx,
        _eos_id_from_data_cfg,
    )
    from pretrain.train.spike_protocol import Halt, SpikeProtocol
    from pretrain.train.state_hash import (
        RunningBatchHasher,
        audit_shard_state_digest,
        combine_batch_digests,
        compute_state_hash,
        finalize_state_hash,
    )

    # The requested device must actually be available — no silent fallback.
    # The old behaviour quietly downgraded a missing cuda/mps to CPU, which
    # "works" but is intractable at real model sizes (a 1.6B replay that takes
    # ~12 h on MPS never finishes on CPU): the audit just looked hung, hours
    # into a replay that could not end. An immediate error names the fix.
    #
    # Match on ``torch.device(device).type`` — NOT the raw string — so an
    # indexed device (``"cuda:0"`` / ``"mps:0"``, which ``audit_replay()``
    # accepts programmatically from the chained-audit tooling) is still gated,
    # and an unknown backend name raises here rather than deep in torch.
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {device} requested but CUDA is not available on this "
            "host. Pass --device cpu explicitly if a CPU replay is really "
            "intended (bitwise-equivalent, but only tractable for tiny models)."
        )
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            f"--device {device} requested but MPS is not available on this "
            "host. Pass --device cpu explicitly if a CPU replay is really "
            "intended (bitwise-equivalent, but only tractable for tiny models)."
        )
    # ---- Kernel-build provenance (the audit-kit contract) ----------------
    import repop

    repop_provenance = repop.build_info()
    _require_repop_backend(dev, repop_provenance)
    # NOTE: no GPU-arch gate. This audit is intentionally arch-agnostic — the
    # whole point is that repop's cross-device-reproducible kernels (and the
    # device-independent trunc_normal init) yield the SAME bits on CUDA, Metal,
    # or CPU. Refusing a "mismatched" arch would defeat that, so we just run.
    ckptr = None if config_only else Checkpointer(os.path.dirname(checkpoint) or ".")
    if config_name is None:
        # The resolved config is stored in meta; load it back through pydantic
        # (parse_config_resolved drops retired keys older code serialized).
        from pretrain.config import parse_config_resolved

        cfg = parse_config_resolved(meta_obj["config_resolved"])
    else:
        cfg = load_config(config_name)

    # ---- Init-audit mode --------------------------------------------------
    # ``--from-init`` reconstructs the START state from the seed instead of
    # loading --checkpoint: repop's trunc_normal init is device-independent, so
    # build_model + init_weights(seed) yields byte-identical weights to the
    # cluster's init. It then (a) verifies that regenerated init reproduces the
    # run's recorded init hash (``state_hash_init.txt``) and (b) replays forward
    # to --checkpoint's step on the SAME codepath as an interval audit, comparing
    # to that checkpoint's state_hash.txt. With ``--until-step 0`` it stops after
    # (a) — a pure init-hash verification, no replay. The actual reconstruction +
    # gate live just below where the model is built (search "from-init").

    if gcs_root is not None and data_root is not None:
        raise ValueError("pass only one of --data-root / --gcs-root")

    # The deterministic-reduction requirement only applies when we REPLAY. A pure
    # init verification (--from-init --until-step 0) has no cross-rank reduction,
    # so it is valid even on a non-auditable (nccl) run.
    if from_init:
        _gate_target = until_step if until_step is not None else int(meta_obj.get("step", 0))
        will_replay = _gate_target > 0
    else:
        will_replay = True
    if will_replay and meta_obj.get("reduction_mode") != "deterministic_allgather":
        raise ValueError(
            "checkpoint was not produced in auditable mode "
            f"(reduction_mode={meta_obj.get('reduction_mode')!r}); cannot reproduce bitwise."
        )

    # A replay must advance BEYOND the start checkpoint's step. With
    # --until-step <= the start step the step loop runs zero iterations, the
    # "reproduced" digest degenerates to the start checkpoint's own
    # chained_hash, and comparing it against that checkpoint's state_hash.txt
    # trivially "matches" — a vacuous PASS that verified nothing. Refuse up
    # front (before the expensive model build / DCP load). From-init replays
    # start at step 0 and their target<=0 case is the init-only mode, which is
    # a real (init-hash) verification, not a replay.
    if not from_init and until_step is not None:
        _start_step = int(meta_obj.get("step", 0))
        if until_step <= _start_step:
            raise ValueError(
                f"--until-step {until_step} is not beyond the checkpoint's own "
                f"step ({_start_step}): the replay would run zero steps and "
                "re-report the checkpoint's stored chained hash, so any "
                "--expect-hash comparison against it would be a vacuous PASS. "
                "Pass a target beyond the start step, or --from-init to audit "
                "up to this checkpoint from init."
            )

    # Topology + reduction descriptors drive only the REPLAY (cross-rank fold,
    # grad-norm, spike). A config-only init audit never replays, so they're
    # parsed only when a checkpoint is present; there the seed comes from meta,
    # otherwise from the config.
    seed = int(meta_obj["seed"]) if not config_only else int(cfg.run.seed)
    # Default the mesh to 1x1 before the branch. A config-only audit never
    # replays, so it never reads these for a fold, but the memory plan is sized
    # from the world size and is called unconditionally.
    N = 1
    dp_replicate = 1
    dp_shard = 1
    if config_only:
        LOG.info(
            "config-only init audit: seed=%d config=%s device=%s (no checkpoint; "
            "init verification only)", seed, config_name, dev.type,
        )
    else:
        # Tolerate a minimal init descriptor (a step-0 checkpoint may carry only
        # seed/config/repop_env): these replay descriptors default to a 1×1 mesh
        # and are unused on a verify-only (--until-step 0) from-init audit.
        # ``or``-defaults: CheckpointMeta leaves these 0 when unset (a minimal
        # init descriptor), which must fall back to a 1×1 mesh, not literal 0.
        N = int(meta_obj.get("dp_world_size") or 1)
        dp_replicate = int(meta_obj.get("dp_replicate") or 1)
        dp_shard = int(meta_obj.get("dp_shard") or N)
        # Bounds gate BEFORE anything scales with these values: meta.json is
        # untrusted input, and the audit builds one virtual-rank loader per dp
        # rank and walks N rank slices per step — a corrupt/doctored
        # dp_world_size would exhaust memory long before any other check
        # fails. _MAX_AUDIT_WORLD is far above any run we could ever audit on
        # one device (a replay's wall-clock scales linearly with N).
        if not (1 <= N <= _MAX_AUDIT_WORLD):
            raise ValueError(
                f"implausible dp_world_size {N} in meta.json (expected "
                f"1..{_MAX_AUDIT_WORLD}) — corrupt or doctored checkpoint meta."
            )
        if not (1 <= dp_replicate <= N) or not (1 <= dp_shard <= N):
            raise ValueError(
                f"implausible mesh in meta.json: dp_replicate={dp_replicate} "
                f"dp_shard={dp_shard} (each expected 1..dp_world_size={N})."
            )
        if dp_replicate * dp_shard != N:
            raise ValueError(f"mesh mismatch: dp_replicate({dp_replicate})*dp_shard({dp_shard}) != N({N})")
        if int(meta_obj.get("step", 0)) < 0 or int(meta_obj.get("consumed_tokens", 0)) < 0:
            raise ValueError(
                f"negative position in meta.json (step="
                f"{meta_obj.get('step')}, consumed_tokens="
                f"{meta_obj.get('consumed_tokens')}) — corrupt checkpoint meta."
            )
        # Cross-replica reduction order. The cluster path is recursive_doubling,
        # which the audit replays via the balanced tree fold (_DiskTreeFold). The
        # legacy ascending_allgather fold was dropped with the pre-reset run — a
        # cold start always records recursive_doubling — so refuse anything else
        # loudly rather than silently mis-folding.
        replicate_algo = meta_obj.get("replicate_reduce_algo", "recursive_doubling")
        if replicate_algo != "recursive_doubling":
            raise ValueError(
                f"this audit only supports replicate_reduce_algo='recursive_doubling'; "
                f"got {replicate_algo!r} (a pre-reset checkpoint; audit it from an "
                f"older checkout)."
            )
        # Clipping is the stateless deterministic global-norm clip. Its global
        # norm uses the deterministic ascending-shard fold, so the clip
        # coefficient and the spike-trigger norm are single-device
        # reproducible. Refuse anything else loudly.
        clip_algo = meta_obj.get("clip_algo", "global")
        if clip_algo != "global":
            raise ValueError(
                f"this audit only supports clip_algo='global' (stateless "
                f"deterministic global-norm clip); got {clip_algo!r} (audit it "
                f"from an older checkout)."
            )
        LOG.info("auditing %s: N=%d (dp_replicate=%d dp_shard=%d) seed=%d start_step=%d consumed_tokens=%d",
                 checkpoint, N, dp_replicate, dp_shard, seed,
                 int(meta_obj.get("step", 0)), int(meta_obj.get("consumed_tokens", 0)))

    # Data: the checkpoint is now validated as auditable, so pull this interval's
    # shards on demand if asked. --gcs-root downloads only the documents the
    # canonical stream consumes from here to the next checkpoint (a fraction of a
    # percent of the corpus); the user need only have staged the checkpoint.
    # --data-root points at an already-staged tree instead. Either way we then
    # swap each source's parent dir (keeping its basename) to read from there —
    # the canonical stream is path-independent, so this changes where bytes are
    # read, never which bytes.
    if gcs_root is not None:
        from pretrain.data.fetch_interval import fetch_audit_interval

        fr = fetch_audit_interval(
            checkpoint, gcs_root, dest=fetch_dest,
            config_name=config_name, until_step=until_step, from_init=from_init,
        )
        data_root = fr.data_root
        LOG.info(
            "fetched %d shard(s) (%.2f GB) for the interval → %s",
            sum(len(v) for v in fr.touched_shards.values()),
            fr.bytes_fetched / 1e9,
            data_root,
        )
    if data_root is not None:
        for s in cfg.data.sources:
            s.path = str(Path(data_root) / Path(s.path).name)

    # Cross-replica fold disk spill (bounds RAM to ~2 gradients). Only the
    # recursive-doubling tree needs it (>2 replicates → stack deeper than 1);
    # the ascending fold is already a single accumulator. Honour explicit
    # --fold-spill-dir / --no-fold-spill; otherwise auto-spill to a temp dir
    # when it helps. ``_spill_owned`` marks an auto-created dir we must clean up.
    spill_dir: Path | None = None
    _spill_owned = False
    if (
        not config_only
        and not no_fold_spill
        and dp_replicate > 2
    ):
        if fold_spill_dir is not None:
            spill_dir = Path(fold_spill_dir)
            spill_dir.mkdir(parents=True, exist_ok=True)
        else:
            spill_dir = Path(tempfile.mkdtemp(prefix="audit_fold_"))
            _spill_owned = True
        LOG.info(
            "cross-replica fold spilling to %s (RAM bounded to ~2 gradients; "
            "dp_replicate=%d). Point --fold-spill-dir at fast local scratch if "
            "this temp location is slow or space-limited.", spill_dir, dp_replicate,
        )

    set_seed(seed)

    # Build model and apply the SAME wrapping the cluster used. Activation
    # checkpointing (checkpoint_wrapper) renames params with a
    # ``_checkpoint_wrapped_module`` prefix, so the audit must apply it too or
    # the model's FQNs won't match the checkpoint's and DCP silently loads
    # nothing (the model then runs from init). At world_size=1 the FSDP step is
    # skipped, leaving the full unsharded model on one device. Load the
    # checkpoint; DCP reshards the N-way state onto this single rank.
    from pretrain.parallel.parallel_dims import ParallelDims
    from pretrain.parallel.parallelize_llama3_repop import parallelize_llama3_repop

    # Loss reproduction. The model forward no longer computes the loss — CE and
    # z-loss are FUSED over the logits (one shared softmax, one combined
    # grad-logits; see pretrain.model.fused_loss), matching the training loop.
    LOG.info("loss: fused CE+z-loss (matches the training loop)")

    model = build_model(cfg.model, device=dev)
    model = parallelize_llama3_repop(
        model, cfg, ParallelDims(dp_replicate=1, dp_shard=1, world_size=1)
    )
    _log_rss("model_built")
    # Decide the optimizer-state offload BEFORE the optimizer exists, because it
    # changes the checkpoint LOAD path (moments must never touch the device).
    # The model is already resident, so free VRAM here is measured, not guessed;
    # only the moment size is predicted, and that is exact from the param count.
    offload_optimizer = _plan_optimizer_offload(
        dev, model, cfg, requested=offload_optimizer, from_init=from_init,
        world_size=N,
    )
    optimizer = build_optimizer(model, cfg.optim)
    # Mirrors loop.py: eager zero-state priming so from-init replays carry the
    # same optim-state key set (and hash) as the cluster from step 1.
    from pretrain.optim.adamw_repop import prime_optimizer_state

    prime_optimizer_state(optimizer)
    optim_dir: Path | None = None
    _optim_dir_owned = False
    moment_paths: dict = {}
    # fp32-master offload (set up here; spilled/reloaded per step in the loop, and
    # only on the bf16-emul path where a separate grad_model exists).
    master_dir: Path | None = None
    _master_dir_owned = False
    _master_plan_done = False
    _grad_plan_done = False
    _pre_forward_logged = False
    # Counted once: both memory planners size their predictions off it, and at
    # 340 tensors the sum is not worth repeating inside the step loop.
    _n_master_params = sum(p.numel() for p in model.parameters())
    master_paths: dict = {}
    if offload_master:
        if master_offload_dir is not None:
            master_dir = Path(master_offload_dir)
            master_dir.mkdir(parents=True, exist_ok=True)
        else:
            master_dir = Path(tempfile.mkdtemp(prefix="audit_master_"))
            _master_dir_owned = True

    # ---- Resolve the audit START state ------------------------------------
    # --from-init regenerates init from the seed (no checkpoint load); otherwise
    # DCP-load --checkpoint's model + optimizer. Each branch sets stream_state,
    # consumed, step, and target_step; target_tokens is derived afterwards.
    if from_init:
        if offload_optimizer:
            raise NotImplementedError(
                "--offload-optimizer is not supported with --from-init (init has no "
                "optimizer moments to offload; they accrue during the replay)."
            )
        from pretrain.model.init import init_weights

        # Regenerate init in place on the already-parallelized model so param
        # names (activation-checkpoint wrapper prefixes) match what the loop
        # hashed; the optimizer stays freshly-built (lazy/empty), identical to
        # the cluster's optimizer at init (moments lazily created as zeros on the
        # first .step()).
        init_weights(model, seed=seed)
        # Load-fidelity gate: the regenerated init must reproduce the loop's
        # recorded init hash before we trust it as a replay start. Mirrors the
        # loop's at_init hash exactly (weights + optimizer-config, grads
        # excluded, standalone prev_hash=None).
        init_digest = compute_state_hash(
            model, optimizer=optimizer, include_grads=False, prev_hash=None
        )
        target_step = until_step if until_step is not None else int(meta_obj.get("step", 0))
        init_only = target_step <= 0
        # Target init hash to compare against. With a checkpoint: prefer the
        # run's state_hash_init.txt (the loop's init artifact, next to the
        # checkpoints), else the checkpoint's own state_hash.txt (a step-0/init
        # checkpoint stores the init hash there). Config-only mode has no such
        # file, so the target is whatever --expect-hash supplies (else None ⇒
        # just report the digest). --expect-hash overrides on a verify-only run.
        init_target = None
        if not config_only:
            init_file = Path(os.path.dirname(checkpoint) or ".") / "state_hash_init.txt"
            if init_file.is_file():
                init_target = _read_digest_file(init_file, what="state_hash_init.txt")
            elif int(meta_obj.get("step", 0)) == 0:
                # Only a genuine step-0/init checkpoint stores the init hash in its
                # own state_hash.txt. For any later checkpoint that file is the
                # post-step chained hash, NOT init — using it would compare the
                # regenerated init against an unrelated step hash and always fail.
                cp0 = Path(checkpoint) / "state_hash.txt"
                if cp0.is_file():
                    init_target = _read_digest_file(cp0, what="state_hash.txt")
        if init_only and expected_digest is not None:
            init_target = expected_digest
        init_match = None if init_target is None else (init_digest == init_target)
        LOG.info(
            "from-init: regenerated init state_hash=%s expected=%s MATCH=%s",
            init_digest[:16], (init_target[:16] if init_target else "none"), init_match,
        )
        if init_only or init_match is False:
            # Verify-only, or the start itself is wrong — replaying would be
            # meaningless, so stop here and report the init comparison.
            res = {
                "step": 0,
                "consumed_tokens": 0,
                "state_hash": init_digest,
                "mode": "init",
                "repop": repop_provenance,
                "device": dev.type,
            }
            if init_target is not None:
                res["expected"] = init_target
                res["match"] = init_match
            return res
        # Replay from init. Default the final target to --checkpoint's own
        # recorded hash (we replay up TO it) when --expect-hash wasn't given.
        if expected_digest is None and target_step == int(meta_obj.get("step", -1)):
            cp_sh = Path(checkpoint) / "state_hash.txt"
            if cp_sh.is_file():
                expected_digest = _read_digest_file(cp_sh, what="state_hash.txt")
        stream_state = None  # fresh canonical stream from position 0
        consumed = 0
        step = 0
        LOG.info("from-init: init verified; replaying steps 0 -> %d", target_step)
    elif offload_optimizer:
        if cfg.optim.name != "adamw_repop":
            raise NotImplementedError(
                f"--offload-optimizer is implemented for the repop AdamW only "
                f"(its step is per-parameter); got optim.name={cfg.optim.name!r}."
            )
        # Seed the moment template on CPU and load into it in place, so the
        # optimizer state never touches the GPU at load — then spill it to disk
        # so it's off the host RAM through the fold too. See
        # _offload_optimizer_step / _spill_moments_to_disk.
        _prepopulate_cpu_optim_state(optimizer)
        stream_state, meta, _ = ckptr.load(
            checkpoint, model, optimizer, optim_state_offload=True,
            model_weights_only=cfg.train.resume_reset_optimizer,
        )
        if optimizer_offload_dir is not None:
            optim_dir = Path(optimizer_offload_dir)
            optim_dir.mkdir(parents=True, exist_ok=True)
        else:
            optim_dir = Path(tempfile.mkdtemp(prefix="audit_optim_"))
            _optim_dir_owned = True
        moment_paths = _spill_moments_to_disk(optimizer, optim_dir)
        LOG.info("optimizer-state offload ON: %d params' AdamW moments spilled to "
                 "%s, streamed per-parameter to the GPU for the step.",
                 len(moment_paths), optim_dir)
        consumed = meta.consumed_tokens
        step = meta.step
        target_step = until_step
    else:
        stream_state, meta, _ = ckptr.load(
            checkpoint, model, optimizer,
            model_weights_only=cfg.train.resume_reset_optimizer,
        )
        consumed = meta.consumed_tokens
        step = meta.step
        target_step = until_step
    if cfg.train.resume_reset_optimizer:
        # Mirror the phase-boundary run: model weights loaded from the (pre-reset)
        # checkpoint, optimizer started cold. The chained-hash anchor + data
        # stream still come from this checkpoint's meta, so the replay reproduces
        # the phase-2 run that resumed here. Audit the phase-2 boundary by pointing
        # --checkpoint at the pre-reset (phase-1) step and forcing the phase-2
        # config via --config-name; phase-2 intervals after the boundary need no
        # flag (their checkpoints already carry the phase-2 optimizer state).
        LOG.info("resume_reset_optimizer: loaded model weights only from %s; "
                 "optimizer starts cold (phase-boundary audit).", checkpoint)
    _log_rss("ckpt_loaded")

    from pretrain.optim.schedules import build_schedule, schedule_lr

    lr_schedule = build_schedule(cfg.schedule, cfg.train, cfg.optim)

    # Replay until `target_step` (absolute optimizer step), or — if not given —
    # one ckpt_every_tokens interval from the start.
    target_tokens = None
    if target_step is None:
        if cfg.train.ckpt_every_steps > 0:
            # Step-based ckpt cadence: one default interval = ckpt_every_steps
            # (matches how the loop now spaces checkpoints).
            target_step = step + cfg.train.ckpt_every_steps
        else:
            target_tokens = consumed + cfg.train.ckpt_every_tokens

    eos = _eos_id_from_data_cfg(cfg)
    mb = cfg.train.micro_batch_size

    # One loader per virtual rank, all seeded from the same global-stream state.
    # Keep each rank's ShardedWindowView (the build_global_loader ``[1]``): every
    # view length-walks the FULL global window sequence (materialising only its
    # owned windows), so all N views advance the same global walkers/RNG/carry-over
    # in lockstep and ``view.state()`` returns the identical GLOBAL stream position
    # on every rank. --save-checkpoint-dir reads views[0].state() at the target
    # step to persist the stream position for the NEXT interval's audit.
    _loader_pairs = [
        build_global_loader(
            cfg.data, cfg.train, rank=r, world_size=N, seed=seed, eos_id=eos,
            state=stream_state, start_consumed_tokens=consumed, start_step=step,
            pin_memory=False,
        )
        for r in range(N)
    ]
    iters = [iter(p[0]) for p in _loader_pairs]
    views = [p[1] for p in _loader_pairs]
    _log_rss(f"loaders_built_N{N}")

    # Batch-digest emulation. If the run included the batch in its state hash,
    # reproduce it: one RunningBatchHasher per virtual rank, primed from that
    # rank's saved chain in the checkpoint, updated as the audit feeds that
    # rank's microbatches — exactly the per-rank chained hash the cluster did.
    # The per-rank assignment (m % N) is topology-specific, so this only matches
    # the SAME N the run used (which is what the audit emulates).
    sh = cfg.train.state_hash
    batch_hashers: list[RunningBatchHasher] | None = None
    if sh.every_n_steps > 0 and sh.include_batch:
        if from_init:
            # Init consumed no batches, so each rank's chain starts fresh —
            # exactly the cluster's batch-hasher state at init.
            batch_hashers = [RunningBatchHasher() for _ in range(N)]
        else:
            digs = []
            for r in range(N):
                p = Path(checkpoint) / f"batch_hasher.rank_{r}.bin"
                if not p.exists():
                    raise FileNotFoundError(
                        f"include_batch is on but {p} is missing — the checkpoint "
                        "lacks the per-rank batch-hasher chains needed to reproduce "
                        "the batch digest."
                    )
                digs.append(p.read_bytes())
            batch_hashers = [RunningBatchHasher(prev_digest=d) for d in digs]

    # Spike protocol — replayed so the audit makes the SAME skip / cooldown
    # decisions the cluster did, off the now-deterministic global grad norm.
    # Without this the audit would step every iteration and diverge from any
    # cluster interval in which a spike fired. Restore mid-cooldown / halt-window
    # state from the checkpoint so a cooldown straddling the boundary continues.
    # The pre-clip global norm is deterministic (single-device reproducible),
    # so the spike decision driver replays exactly.
    spike = SpikeProtocol(
        threshold=cfg.train.spike.grad_norm_threshold,
        skips_in_window_to_halt=cfg.train.spike.skips_in_window_to_halt,
        halt_window_steps=cfg.train.spike.halt_window_steps,
        skip_steps_on_spike=cfg.train.spike.skip_steps_on_spike,
        start_step=cfg.train.spike.start_step,
    )
    # From init the protocol starts fresh (matching the cluster at init); for an
    # interval, restore the start checkpoint's mid-cooldown / halt-window state.
    sp_path = Path(checkpoint) / "spike_protocol.json"
    if not from_init and sp_path.exists():
        spike.load_state_dict(json.loads(sp_path.read_text()))

    def _reached(step_val: int, consumed_val: int) -> bool:
        """Single encoding of the interval-end predicate, shared by the loop
        guard (_done) and the target-step hash gate below — so the two can't
        drift if the default-target rule ever changes."""
        if target_step is not None:
            return step_val >= target_step
        return consumed_val >= target_tokens

    def _done() -> bool:
        return _reached(step, consumed)

    # Pre-count the interval's optimizer steps (cheap integer walk of the same
    # schedule the loop follows) so the progress bar has a real total — the
    # replay can run ~80 min and otherwise leaves the terminal blank.
    def _count_total_steps() -> int:
        from pretrain.train.batch_schedule import iter_step_plans

        n = 0
        for plan in iter_step_plans(consumed, 0, cfg.train, start_step=step):
            if (target_step is not None and plan.step >= target_step) or (
                target_step is None and plan.consumed_tokens >= target_tokens
            ):
                break
            n += 1
        return n

    total_steps = _count_total_steps()
    # Start the bar AT the checkpoint's step and count up to the target step, so
    # it reads e.g. `10/20` → `20/20` (the absolute optimizer step), matching the
    # checkpoint dir names, rather than a 0-based interval counter.
    pbar = tqdm(
        initial=step,
        total=step + total_steps,
        desc="audit replay",
        unit="step",
        dynamic_ncols=True,
        position=0,
    )
    # Second bar: micro-batches completed within the CURRENT step. M =
    # microbatches_per_step = accum * dp_world_size varies by phase (warmup/main/
    # late), so it's reset to the step's M at the top of each step and advanced
    # once per micro-batch. Purely a display aid (each micro-batch is minutes on
    # MPS) — device-agnostic, touches no hashed value or reduction, so it cannot
    # affect the CUDA path or bitwise results. leave=False so a finished per-step
    # bar isn't left behind each step.
    mb_pbar = tqdm(
        total=None,
        desc="  microbatches",
        unit="mb",
        dynamic_ncols=True,
        position=1,
        leave=False,
    )
    LOG.info("replaying steps %d → %d", step, step + total_steps)

    # ---- bf16: per-rank gradients via real FSDP2 ------------------------------
    # On the cluster, mixed-precision FSDP2 all-gathers every param as bf16 for
    # forward/backward while keeping the sharded master fp32 and reducing the
    # gradient in reduce_dtype=fp32. A hand-rolled "cast the leaf to bf16"
    # emulation is NOT byte-faithful: its gradients round to bf16, whereas FSDP
    # keeps them fp32 (measured divergence at the LM head / wv weights). So for
    # bf16 runs we compute each virtual rank's gradient through a *real* FSDP2
    # copy of the model — forced 1-rank ``fully_shard`` with the run's mp_policy
    # — and fold those fp32 grads onto the plain fp32 ``model`` exactly as the
    # validated fp32 path does. The plain ``model``/``optimizer`` remain the
    # canonical master we clip, step, and hash. For fp32 runs grad_model is None
    # and the path is unchanged.
    # The cluster only wraps with fully_shard (bf16 param_dtype) at
    # world_size > 1 — parallelize_llama3_repop returns early at ws <= 1, so a
    # single-process run trains the PLAIN fp32 model under autocast (fp32
    # residual stream). Reproduce the regime the run actually trained in:
    # forcing the FSDP2 bf16 grad model onto a ws=1 run replays a cluster that
    # never existed (bf16 block outputs) and diverges from the first forward.
    _bf16_params = bool(cfg.run.mixed_precision) and N > 1
    grad_model = None
    _emul_bf16 = False  # MPS-only: bf16 grad model emulating FSDP2 mixed precision
    if _bf16_params and dev.type == "mps":
        # FSDP2 ``fully_shard`` cannot run on Apple MPS: it relies on CUDA-only
        # device-stream/event APIs and ``UntypedStorage.resize_`` (free of the
        # unsharded shard), none of which the MPS backend implements in torch
        # 2.11. At world_size=1 FSDP2 mixed precision is numerically just "run
        # forward/backward with bf16-cast params, keep the fp32 master, reduce
        # the gradient in reduce_dtype=fp32"; the reduce-scatter is identity. We
        # emulate it with a plain bf16 copy of the model for the forward/backward.
        #
        # Gradient accumulation across the accum>1 microbatches/rank must be in
        # **fp32**: FSDP2 with reduce_dtype=fp32 upcasts each microbatch's bf16
        # grad to fp32 and accumulates in fp32 (NOT on the bf16 leaf). VERIFIED on
        # CUDA against real ws=1 fully_shard at accum=8: fp32-per-microbatch accum
        # is byte-exact (0/99 params differ) whereas naive bf16-leaf accumulation
        # drifts up to 1.5e-4 (87/99). ``_emul_bf16`` makes the rank loop do that
        # fp32 accumulation. Param FQNs keep the ``_checkpoint_wrapped_module``
        # prefix (AC applies in dev mode), so the folded grads map onto the fp32
        # master by name as fully_shard does; fold/clip/step/hash run on the master.
        grad_model = build_model(cfg.model, device=dev)
        grad_model = parallelize_llama3_repop(
            grad_model, cfg,
            ParallelDims(dp_replicate=1, dp_shard=1, world_size=1),
        )
        # Cast PARAMS ONLY to bf16 — NOT buffers. FSDP2's MixedPrecisionPolicy
        # (param_dtype=bf16) casts parameters during all-gather but leaves buffers
        # at their built dtype, so on the cluster (and the CUDA/CPU audit's real
        # fully_shard) the RoPE cos/sin tables stay fp32. A blanket
        # ``.to(torch.bfloat16)`` here would also cast those buffers to bf16, and
        # ``apply_rope`` (rx = x1*cos - x2*sin, no internal upcast) would then run
        # in bf16 instead of promoting to fp32 — diverging from the cluster on ~20%
        # of elements (max |Δ|~3e-2). Param-only cast mirrors fully_shard exactly.
        with torch.no_grad():
            for _p in grad_model.parameters():
                _p.data = _p.data.to(torch.bfloat16)
        _emul_bf16 = True
        # Emulate fully_shard's MixedPrecisionPolicy(cast_forward_inputs=True):
        # FSDP2 casts each wrapped module's floating-point forward INPUTS to
        # param_dtype (bf16) at the module boundary. parallelize wraps every
        # block, called as block(h, rope_cos, rope_sin) — so on the cluster the
        # fp32 rope_cos/rope_sin tables are cast to BF16 on entry to each block
        # and apply_rope runs in bf16. The param-only emul leaves the fp32 cos/sin
        # buffers fp32, so apply_rope promotes to fp32 (bf16·fp32) and diverges —
        # benign at seq 512 (100M regression passes) but bit-flipping at seq 4096
        # (1.6B). Replicate the cast with a pre-hook on each block. ALWAYS ON —
        # this is the correct emul behaviour (required for BFR), not an option.
        _cfi_dtype = torch.bfloat16

        def _cast_fwd_inputs(_m, _args):
            return tuple(
                a.to(_cfi_dtype) if (isinstance(a, torch.Tensor)
                                     and a.is_floating_point()) else a
                for a in _args
            )

        _n_cfi = 0
        for _blk in grad_model.blocks:
            _blk.register_forward_pre_hook(_cast_fwd_inputs)
            _n_cfi += 1
        LOG.info("emul: cast_forward_inputs=True emulation — bf16-casting "
                 "float forward inputs on %d blocks (matches fully_shard)", _n_cfi)
        LOG.info(
            "bf16 audit (MPS): FSDP2 ws=1 mixed precision emulated as a "
            "param-only bf16 cast + cast_forward_inputs emulation (RoPE cos/sin "
            "cast to bf16 at each block boundary, as fully_shard does) with fp32 "
            "(reduce_dtype) per-microbatch gradient accumulation."
        )
    elif _bf16_params:
        import torch.distributed as _dist

        if not _dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29577")
            _dist.init_process_group(
                backend="nccl" if dev.type == "cuda" else "gloo",
                rank=0,
                world_size=1,
            )
        os.environ["REPOP_FORCE_FSDP_WS1"] = "1"
        try:
            grad_model = build_model(cfg.model, device=dev)
            grad_model = parallelize_llama3_repop(
                grad_model,
                cfg,
                ParallelDims(dp_replicate=1, dp_shard=1, world_size=1),
            )
        finally:
            os.environ["REPOP_FORCE_FSDP_WS1"] = "0"
        LOG.info(
            "bf16 audit: per-rank grads via real FSDP2 (1-rank fully_shard, "
            "param_dtype=bf16, reduce_dtype=fp32); fold/clip/step/hash on the "
            "fp32 master."
        )
        _log_rss("grad_model_built")

    def _grad_source():
        """Model whose forward/backward produces each rank's gradient: the FSDP
        bf16 model under mixed precision, else the plain fp32 model."""
        return grad_model if grad_model is not None else model

    def _sync_grad_model() -> None:
        """Copy the canonical fp32 master (``model``) into the FSDP grad model so
        its next forward all-gathers the current weights as bf16. No-op for fp32
        runs (grad_model is None)."""
        if grad_model is None:
            return
        with torch.no_grad():
            mp = dict(model.named_parameters())
            for n, p in grad_model.named_parameters():
                loc = p.to_local() if hasattr(p, "to_local") else p.data
                loc.copy_(mp[n].data)

    def _read_rank_grads() -> dict:
        """This rank's gradients as plain CPU fp32 tensors. ``full_tensor`` pulls
        the FSDP DTensor grads to a plain tensor; the fp32 cast is a no-op when
        the grad is already fp32 (it always is — FSDP reduce_dtype=fp32)."""
        out: dict[str, torch.Tensor] = {}
        for n, p in _grad_source().named_parameters():
            if p.grad is not None:
                g = p.grad
                g = g.full_tensor() if hasattr(g, "full_tensor") else g
                out[n] = g.detach().to(device="cpu", dtype=torch.float32)
        return out

    # Seed the state-hash chain from the start checkpoint's chained_hash so the
    # canonical hash computed each replayed step (post-step, with grads) chains
    # exactly as the cluster did. The final value is compared to the target
    # checkpoint's state_hash.txt. From init the chain starts at None — the init
    # hash is standalone (not chained into the periodic step hashes), so the
    # first replayed step-N hash chains from None exactly as the cluster's did.
    audit_chained_hash = None if from_init else meta_obj.get("chained_hash")
    # The hash the TARGET checkpoint's state_hash.txt stores. The loop stamps a
    # hash on EVERY checkpoint it saves (loop.py `_hash_due or _ckpt_due`),
    # chained from the running chain — but only a hash-due step advances the
    # chain itself. For an on-cadence target the two coincide; for an
    # off-cadence one (a checkpoint between hash-due steps) this "side-link"
    # H(state@target, prev=chain) is what --expect-hash must be compared to,
    # NOT the (older) chain value.
    target_ckpt_digest: str | None = None
    # Target-step gradients held for the handoff sidecar. Captured at
    # the canonical hash point — before zero_grad — and written to
    # ``gradients.safetensors`` only after the hash-match gate passes. ``None``
    # unless the run hashed gradients (include_grads) and a handoff is requested.
    held_grads: dict[str, torch.Tensor | None] | None = None
    # LR re-warm anchor: recorded in meta by runs that resumed with a ramp
    # (reset-optimizer or rewarm_on_resume). -1 / absent = no ramp. From-init
    # replays predate any resume, so no ramp applies.
    _rewarm_anchor = -1 if from_init else int(meta_obj.get("rewarm_anchor_tokens", -1))
    # Rank-0 loss records: one per replayed step, matching the row rank 0 wrote
    # to logs/metrics.jsonl (same step / consumed_tokens / loss_ce / loss_zloss
    # semantics), so a verifier can cross-check the replay against the run's
    # logged losses. Rewritten to --loss-log after every step so a monitoring
    # client can compare live, and returned in the result dict.
    rank0_losses: list[dict] = []
    loss_log_path = Path(loss_log) if loss_log is not None else None
    if loss_log_path is not None:
        loss_log_path.parent.mkdir(parents=True, exist_ok=True)

    while not _done():
        M = microbatches_per_step(consumed, cfg.train)
        if M % N != 0:
            raise ValueError(f"microbatches_per_step ({M}) not divisible by N ({N})")
        accum = M // N
        # Reset the micro-batch bar to this step's M (= accum * dp_world_size).
        mb_pbar.reset(total=M)
        mb_pbar.set_description(f"  step {step + 1} microbatches")
        _lr = lr_schedule(consumed, cfg.train.total_tokens)
        # LR re-warm — mirrors loop.py exactly, anchored from the checkpoint
        # meta (rewarm_anchor_tokens; -1 = none) so any interval inside the
        # ramp window replays the same LR. Same expression shape as the loop
        # (int subtraction, int/int divide, float multiply) for bit-identical
        # scalars.
        if _rewarm_anchor >= 0 and cfg.schedule.rewarm_tokens > 0:
            _rw_elapsed = consumed - _rewarm_anchor
            if _rw_elapsed < cfg.schedule.rewarm_tokens:
                _lr = _lr * max(0.0, _rw_elapsed / cfg.schedule.rewarm_tokens)
        schedule_lr(optimizer, _lr)
        # bf16 warm-start — mirrors loop.py exactly: the QAT toggle is a pure
        # function of (step, config), set at the top of every replayed step.
        if cfg.model.qat.enabled and cfg.model.qat.enable_at_step > 0:
            from pretrain.model.modules.qat_warmstart import set_qat_active

            set_qat_active(model, step >= cfg.model.qat.enable_at_step)
            if grad_model is not None:
                set_qat_active(grad_model, step >= cfg.model.qat.enable_at_step)
        # Accumulate the step's loss on-device (summed over all virtual ranks'
        # micro-batches); one .item() per step for the bar, so no extra syncs.
        step_loss_t = torch.zeros((), device=dev)
        # Virtual rank 0's CE / z-loss sums, mirroring the cluster's rank-local
        # ``micro_loss_total`` / ``micro_zloss_total`` EXACTLY: a host fp64 sum
        # of the per-microbatch fp32 scalars (loop.py accumulates via
        # ``float(ce)``, NOT an on-device fp32 sum). Feeds the spike protocol's
        # loss argument and reproduces rank 0's logged ``loss_ce`` /
        # ``loss_zloss`` metrics values bit-for-bit — the verifiable rank-0
        # loss record below. Costs one device sync per rank-0 microbatch
        # (``accum`` of the step's N*accum), negligible next to the mb compute.
        ce_r0_total = 0.0
        zl_r0_total = 0.0

        # Emulate the cluster's reduction structure exactly. dp_rank r maps to
        # mesh coord (replicate = r // dp_shard, shard = r % dp_shard). The
        # gradient folds in two levels to match FSDP2:
        #   inner (per replicate): ascending sum over its shards, /dp_shard
        #                          — the DeterministicReduceScatter (AVG).
        #   outer (over replicates): the cluster's recursive_doubling order, then
        #          /dp_replicate — the DeterministicReplicate AllReduce, replayed
        #          as the balanced adjacent-pair tree (_DiskTreeFold). For pure
        #          FSDP (dp_replicate=1) it collapses to a single push.
        # Process virtual ranks in mesh order (rep-major, shard-inner, i.e.
        # r = rep*dp_shard + shard ascending). CPU accumulators stay bounded to
        # ~2 gradients: one inner sum being built, plus the fold's working pair
        # (the tree's deeper partials spill to disk via _DiskTreeFold). Holding
        # all dp_replicate at once would OOM at the real seq_len + large N.
        inner: dict[str, torch.Tensor] | None = None
        fold = _DiskTreeFold(spill_dir)
        # Sync the FSDP grad model from the current fp32 master so its forward
        # all-gathers the up-to-date weights as bf16 (no-op for fp32 runs).
        _sync_grad_model()
        # Master is now mirrored into grad_model (bf16) and idle until the step;
        # spill it to disk so it doesn't occupy the unified pool through the
        # micro-batch/fold peak. Only on the emul path (grad_model present) — when
        # grad_model is None the master IS the compute model and can't be freed.
        _master_spilled = False
        # Everything except the loss transient is now placed, so this is the
        # last and best-informed point to decide the master spill. Done once
        # (the first step), then held for the rest of the replay.
        if not _master_plan_done:
            offload_master = _plan_master_offload(
                dev, cfg, requested=offload_master,
                has_grad_model=grad_model is not None,
                n_params=_n_master_params,
            )
            _master_plan_done = True
            if offload_master and master_dir is None:
                master_dir = Path(tempfile.mkdtemp(prefix="audit_master_"))
                _master_dir_owned = True
        if offload_master and grad_model is not None and master_dir is not None:
            master_paths = _spill_master_to_disk(model, master_dir)
            _master_spilled = True
            # Return the freed master to the OS. The spill only drops the master's
            # device storage into the MPS allocator cache; with watermark=0.0 that
            # cache is never auto-returned, so without this the ~6 GB stays counted
            # in the footprint (offload is then a no-op on memory). Clearing it here
            # also resets fragmentation so the micro-batch phase grows the pool from
            # a clean base to a LOWER high-water — measured: mb0 peak footprint
            # 46.9→42.7 GB, reserved 33.9→29.4 GB vs not clearing. Once per step
            # (before the loop), NOT the per-microbatch thrash the loop avoids.
            # On CUDA this matters for the same reason: the spilled master's
            # blocks are free but still reserved in ~340 param-sized pieces, and
            # the loss's single [micro_batch*seq_len, vocab] block cannot be
            # served from between them. Once per step, never per micro-batch —
            # empty_cache syncs the device.
            _empty_cache(dev)
            if step == meta_obj.get("step", step):
                LOG.info("fp32-master offload ON: %d params spilled to %s for the "
                         "micro-batch/fold phase.", len(master_paths), master_dir)
        elif offload_master and grad_model is None:
            LOG.warning("--offload-master is a no-op without a bf16 grad_model "
                        "(the master is the compute model on this path).")
        if not _grad_plan_done:
            # Deliberately after the spill: the master's bytes are freed back to
            # the card's pool by now, and planning before that would read a free
            # figure 1x model size too small and offload gradients a card can hold.
            offload_grads = _plan_grad_offload(
                dev, cfg, requested=offload_grads,
                has_grad_model=grad_model is not None,
                n_params=_n_master_params,
            )
            _grad_plan_done = True
        _gsrc = _grad_source()
        _stream_head = os.environ.get("PRETRAIN_AUDIT_STREAM_HEAD") == "1"
        if _stream_head:
            from pretrain.model.modules.embedding_repop import UntiedEmbeddingRepop
            from pretrain.model.streaming_head_loss import streaming_head_loss

            if dev.type != "mps" or grad_model is not None or not isinstance(
                _gsrc.embedding, UntiedEmbeddingRepop
            ) or _gsrc.embedding.output.bias is not None or any(
                p.dtype != torch.float32 for p in _gsrc.parameters()
            ):
                raise ValueError("Streamed head requires the untied FP32 MPS audit path")
            # The streamed backward's wgrad chain needs backend.mm_accumulate
            # (see streaming_head_loss.py). Forward never calls it, so a repop
            # build that predates it would otherwise run a full accum-loop
            # forward — potentially the bulk of a multi-hour step — before
            # dying mid-backward with an AttributeError. Fail before that.
            from repop.backend import metal as _metal_backend

            if not hasattr(_metal_backend, "mm_accumulate"):
                raise ValueError(
                    "PRETRAIN_AUDIT_STREAM_HEAD=1 requires a repop build with "
                    "repop.backend.metal.mm_accumulate; this repop build does "
                    "not have it."
                )
            _head_chunk_rows = int(os.environ.get("PRETRAIN_AUDIT_HEAD_CHUNK_ROWS", "256"))
        _mps_fp32_grad_offload = (
            os.environ.get("PRETRAIN_AUDIT_MPS_OFFLOAD_GRADIENTS") == "1"
        )
        if _mps_fp32_grad_offload:
            if dev.type != "mps" or grad_model is not None or any(
                p.dtype != torch.float32 for p in _gsrc.parameters()
            ):
                raise ValueError("MPS gradient offload requires the FP32 MPS audit path")
            from pretrain.train.mps_gradient_accumulation import (
                accumulate_mps_fp32_gradients,
            )
        for r in range(N):
            shard = r % dp_shard
            _gsrc.zero_grad(set_to_none=True)
            # MPS bf16 emulation: accumulate the per-microbatch grads in fp32
            # ourselves rather than letting autograd sum them on the bf16 leaf.
            # This reproduces FSDP2's reduce_dtype=fp32 unsharded-grad accumulation
            # (verified byte-exact vs fully_shard at accum=8 on CUDA — see the
            # _emul_bf16 note above). For CUDA/CPU (real FSDP2) it stays None.
            # Host-side fp32 gradient accumulation. Always on for the MPS
            # emulation (its bf16 leaf cannot accumulate correctly); on CUDA
            # (or an explicit --offload-grads on any device) it is the memory
            # plan's call, buying back FSDP2's unsharded no-sync accumulator.
            # --mps-offload-grads is the other opt-in that lands here: it also
            # stores totals on CPU, but performs each old-plus-new addition on
            # MPS to preserve autograd's float semantics (see the branch below).
            _emul_accum: dict[str, torch.Tensor] | None = (
                {} if (_emul_bf16 or offload_grads or _mps_fp32_grad_offload) else None
            )
            for _k in range(accum):
                # Match the cluster's grad-sync gating: FSDP accumulates the
                # unsharded grad across micro-batches and reduces (→fp32) only on
                # the last one. At ws=1 the reduce is trivial, but mirroring the
                # gating keeps byte-equality at accum>1.
                if grad_model is not None and hasattr(grad_model, "set_requires_gradient_sync"):
                    # Draining to the host needs p.grad populated every
                    # micro-batch, so the reduce must run every micro-batch. At
                    # ws=1 the reduce-scatter is the identity, so reducing each
                    # micro-batch and summing on the host is the same fp32
                    # additions in the same order as accumulating unsharded and
                    # reducing once — what the gating below preserves at ws>1.
                    grad_model.set_requires_gradient_sync(
                        True if _emul_accum is not None else (_k == accum - 1)
                    )
                batch = next(iters[r])
                if batch_hashers is not None:
                    # Hash the CPU loader output before the device copy — same
                    # point and same per-rank chain the cluster used.
                    batch_hashers[r].update(batch)
                ids = batch["input_ids"].to(dev)
                labels = batch["labels"].to(dev)
                if not _pre_forward_logged:
                    # Everything the forward will build on is placed by now, so
                    # this is the line that says how much room the logits and
                    # grad-logits actually have. Once per replay.
                    _log_rss("pre_forward_mb0")
                    _pre_forward_logged = True
                with _autocast_ctx(dev.type == "cuda", mixed_precision=cfg.run.mixed_precision):
                    # Fused CE+z-loss — one shared softmax, row-chunked, BFR.
                    # Already memory-frugal, so MPS uses it directly too.
                    z_coeff = (
                        cfg.model.z_loss.coeff if cfg.model.z_loss.enabled else 0.0
                    )
                    if _stream_head:
                        hidden = _gsrc.forward_hidden(ids)
                        ce, zloss = streaming_head_loss(
                            hidden, _gsrc.embedding.output.weight, labels, z_coeff,
                            chunk_rows=_head_chunk_rows,
                        )
                    else:
                        out = _gsrc(ids)
                        ce, zloss = fused_ce_z_loss(out.logits, labels, z_coeff)
                    # × host reciprocal, NOT ``/ accum`` — must match loop.py
                    # exactly. ``tensor / scalar`` is true division on CPU/MPS but
                    # reciprocal-multiply on CUDA, so for non-power-of-2 accum the
                    # 1/accum grad scaling diverges last-bit across devices. See
                    # the matching comment in pretrain.train.loop.
                    loss = (ce + zloss) * (1.0 / accum)
                loss.backward()
                step_loss_t += loss.detach()
                if r == 0:
                    # Same guard shape as loop.py: zloss is a tensor unless the
                    # fused loss short-circuits z_coeff=0 to a plain float.
                    ce_r0_total += float(ce.detach())
                    zl_r0_total += float(zloss.detach()) if isinstance(zloss, torch.Tensor) else 0.0
                if _mps_fp32_grad_offload:
                    assert _emul_accum is not None
                    accumulate_mps_fp32_gradients(_gsrc, _emul_accum)
                elif _emul_accum is not None:
                    # Upcast this microbatch's bf16 grad to fp32 and sum in fp32 on
                    # CPU (bf16→fp32 is exact; fp32 add is per-element IEEE-754, so
                    # CPU-accumulation is bit-identical to on-device), then clear the
                    # leaf. Accumulating on CPU keeps the fp32 grad dict (~one model)
                    # out of the unified MPS pool during the accum loop.
                    for n, p in _gsrc.named_parameters():
                        if p.grad is not None:
                            g = p.grad
                            # Real FSDP2 hands back a DTensor; the MPS emulation
                            # a plain tensor. full_tensor() is what _read_rank_grads
                            # uses for the same reason.
                            g = g.full_tensor() if hasattr(g, "full_tensor") else g
                            g = g.detach().to(device="cpu", dtype=torch.float32)
                            _emul_accum[n] = g if n not in _emul_accum else _emul_accum[n] + g
                    _gsrc.zero_grad(set_to_none=True)
                # Drop the micro-batch's autograd graph + logits so their MPS
                # buffers return to the allocator's cache. We do NOT empty_cache
                # here: every micro-batch allocates the SAME shapes, so the cached
                # buffers are reused by the next one — evicting per micro-batch
                # would instead force a full device sync + a fresh OS allocation
                # (page-fault + zero) each time, which dominated the runtime.
                if _stream_head:
                    del hidden
                else:
                    del out
                del ce, zloss, loss
                mb_pbar.update(1)  # one micro-batch done (N*accum = M per step)
                # Per-micro-batch cache trim (gated, A/B): return this micro-batch's
                # freed MPS buffers to the OS before the next backward, to curb the
                # allocator fragmentation that otherwise creeps across micro-batches
                # (measured: footprint 42.5 GB at mb0 → 51.4 GB at mb1). The loop
                # deliberately avoids this by default (every mb reuses the same
                # shapes, so evicting forces a fresh OS allocation + zero each mb —
                # the runtime-dominating thrash noted at the `del` above). Bitwise-
                # neutral (frees only unreferenced buffers).
                if os.environ.get("PRETRAIN_AUDIT_EMPTY_CACHE_PER_MB") == "1":
                    _empty_cache(dev)
            # Read this rank's gradient as plain CPU fp32 (full_tensor pulls the
            # FSDP DTensor grads to plain tensors) and fold on CPU — the add/div
            # are per-element IEEE-754, bitwise-identical CPU↔GPU, matching the
            # cluster's reduce-scatter (AVG over shards) + all-reduce. The emul path
            # already accumulated on CPU, so reuse those tensors directly.
            rg = (_emul_accum if _emul_accum is not None else _read_rank_grads())
            # Once per rank (not per micro-batch): trim the allocator's cache so
            # fragmentation can't creep across the step's N ranks. Bitwise-neutral
            # (frees only unreferenced buffers). ~N calls/step instead of ~N·accum·2.
            # On CUDA this is a real device sync (unlike the MPS branch, which was
            # the only one this call did anything on before --offload-grads), so
            # only pay for it when some offload is actually tightening the memory
            # budget; a card with headroom keeps the old zero-sync behaviour.
            if offload_master or offload_optimizer or offload_grads:
                _empty_cache(dev)
            if shard == 0:
                inner = rg
            else:
                for n in rg:
                    inner[n].add_(rg[n])  # ascending over shard
            _log_rss(f"step{step}_rank{r}_postbwd")
            if shard == dp_shard - 1:
                for n in inner:
                    inner[n].mul_(1.0 / dp_shard)  # reduce-scatter AVG
                fold.push(inner)  # balanced tree over replicates (spills to disk)
                inner = None
        outer = fold.result()  # binary-blocks tree → combined total
        # The fold is done and grad_model's bf16 weights are no longer needed
        # until the next step's sync — reload the fp32 master for grad assignment,
        # clip, the optimizer step, LSQ refresh, and the state hash.
        if _master_spilled:
            _reload_master_from_disk(model, master_paths, dev)
            _master_spilled = False
        # Assign the folded fp32 grad onto the plain fp32 master. ``model`` is
        # always fp32 here (the FSDP grad_model holds the bf16 compute copy), so
        # the optimizer step, grad-norm/clip, and the final state hash all run on
        # the fp32 master — matching the cluster's fp32 sharded params.
        # AVG by multiplying with a host-computed reciprocal (× const), NOT a
        # tensor ÷ N. fp32 division dispatches a backend-specific reciprocal whose
        # last-bit rounding differs across CUDA / CPU / MPS for non-power-of-2 N
        # (e.g. /6 differs on ~1/3 of elements; /2,/4,/8 are exact so it never bit
        # on power-of-2 meshes — which is why the dp_replicate=1 regression missed
        # it). CUDA's `x / N` IS `x * (1/N)`, and `x * (1/N)` is bit-identical on
        # every backend, so this matches the cluster's GPU AVG while staying
        # reproducible on a CPU/MPS audit. Mirrors the × const fix in repop's LSQ
        # weight-scale refresh (qat/lsq.py). Same reasoning for the dp_shard AVG above.
        for name, p in model.named_parameters():
            if name in outer:
                p.grad = outer[name].mul_(1.0 / dp_replicate).to(dev)  # all-reduce AVG

        # Folded gradients are now owned by model parameters. Release the CPU
        # dictionaries (including aliases in the fold stack) before clipping,
        # optimizer execution and hashing. On CPU, p.grad retains the same
        # tensors; on accelerators, the completed copies retain their values.
        del outer, rg, fold, _emul_accum

        # Deterministic global-norm clip, reconstructed locally (no collective).
        # Its global norm replays the cluster's ascending-shard fold by
        # re-slicing the reconstructed full gradient into dp_shard Shard(0)
        # shards (memory-frugal: one shard view at a time, dot is fused — no
        # full t**2 temporaries). The replicate axis is already collapsed into
        # ``outer``, so only dp_shard is folded. The clip is applied in place
        # (per-element multiply ⇒ bitwise-identical to clipping the cluster's
        # shards); it returns the pre-clip global norm for the spike decision
        # below.
        from pretrain.train import global_clip

        gn = float(
            global_clip.clip_audit(model, cfg.train.grad_clip, dp_shard, dev)
        )
        # Staged QK wake-up — mirrors loop.py exactly: zero the q_norm gain
        # grads AFTER the clip, BEFORE the spike decision / step / hash.
        if step < cfg.train.qk_freeze_q_gains_until_step:
            from pretrain.train.qk_gain_control import zero_q_gain_grads

            zero_q_gain_grads(model)
        # Rank-local mean losses — the SAME fold as loop.py (fp64 host sum of
        # the per-microbatch fp32 scalars, then true division by ``accum``), so
        # ``ce_avg`` / ``zl_avg`` equal rank 0's logged ``loss_ce`` /
        # ``loss_zloss`` bit-for-bit on an honest replay.
        ce_avg = ce_r0_total / accum
        zl_avg = zl_r0_total / accum
        if os.environ.get("REPOP_DBG_GN", "0") == "1":
            LOG.info("AUDIT_GN step=%d grad_norm=%.10f ce_avg=%.10f", step, gn, ce_avg)

        # Spike decision — identical inputs to the cluster: the deterministic
        # global grad norm and virtual-rank-0's mean CE. Skip ⇒ drop grads, no
        # optimizer step, advance the cooldown counter (matches loop.py).
        skip = spike.should_skip(grad_norm=gn, loss=ce_avg, step=step)
        _halted = False
        if skip:
            try:
                spike.record_skip(step)
            except Halt as h:
                # The cluster halted here too; record this step's hash first (it
                # computed the hash before halting), then stop.
                LOG.warning("spike protocol halt at step %d during audit: %s", step, h)
                _halted = True
        else:
            if offload_optimizer:
                _offload_optimizer_step(optimizer, dev, moment_paths)
            else:
                optimizer.step()
        _completed_step = step + 1

        # ---- LSQ weight-scale refresh (post-step, pre-hash) — mirrors loop.py
        # exactly (see pretrain.train.loop). Re-pins each LSQ layer's weight_scale
        # to the post-step weights so the audit reproduces the cluster's
        # refreshed scale before hashing. The reduction is over the un-sharded K
        # axis, so the fp32 master here matches the cluster's per-rank shards
        # byte-for-byte on the same arch. Gate + ordering MUST stay identical to
        # the loop (runs regardless of spike-skip; no-op unless method="lsq" and
        # the knob is > 0), or refresh-enabled runs fail their hash check.
        _sr = cfg.model.qat.scale_refresh_every_n_steps
        if _sr > 0 and _completed_step % _sr == 0:
            from repop.qat.lsq import refresh_lsq_weight_scales

            refresh_lsq_weight_scales(model)

        # ---- QK-gain clamp (post-step, pre-hash) — mirrors loop.py exactly:
        # same gate, same ordering (after the LSQ refresh, before the hash),
        # runs regardless of a spike-skip. Elementwise clamp_ on the fp32
        # master is byte-equal to the cluster's per-shard clamp.
        if cfg.train.qk_gain_clamp > 0.0:
            from pretrain.train.qk_gain_control import clamp_qk_gains

            clamp_qk_gains(model, cfg.train.qk_gain_clamp)

        # ---- Canonical state hash (post-step, grads still live) — identical to
        # loop.py: H(weights + optimizer state + gradients + running batch digest,
        # chained). Computed BEFORE zero_grad so the step's gradients (the folded
        # grad still on model.grad) are hashed, chained from the start
        # checkpoint's chained_hash. This is what state_hash.txt now stores.
        # Mirrors loop.py's `_hash_due or _ckpt_due` gate exactly: a hash-due
        # step advances the chain, and the step a checkpoint lands on ALSO
        # gets a hash chained from the running chain (computed once and shared
        # when the step is both). The audit's target step plays the cluster's
        # ``_ckpt_due``: --expect-hash targets a checkpoint's state_hash.txt,
        # which on an off-cadence checkpoint stores that side-link.
        _hash_due = sh.every_n_steps > 0 and _completed_step % sh.every_n_steps == 0
        # This step's post-step position: step -> _completed_step, and consumed
        # gains this step's tokens (both are applied at the bottom of the loop).
        _target_reached = _reached(
            _completed_step, consumed + M * mb * cfg.train.seq_len
        )
        if sh.every_n_steps > 0 and (_hash_due or _target_reached):
            _bd = (
                combine_batch_digests([h.local_digest() for h in batch_hashers])
                if batch_hashers is not None else None
            )
            # Offload: _offload_optimizer_step left the moments on disk with
            # optimizer.state[p]["exp_avg"]/["exp_avg_sq"] == None, but the state
            # hash MUST include them (the cluster hashes the post-step moments;
            # local_shard_state_digest silently skips non-Tensor state, so without
            # this the digest would omit exp_avg/exp_avg_sq and never match — the
            # offload-audit divergence root cause). Reload the (post-step) moments
            # from disk for the digest, then drop them again so the next step's
            # fold/fwd-bwd keeps the moments off RAM (the disk copies stay current
            # — _offload_optimizer_step wrote them post-step). The per-step disk
            # reload is the known cost; optimize later (stream per-param in the
            # digest) — correctness first.
            if offload_optimizer:
                _materialize_moments(optimizer, moment_paths)
            # Sharded reconstruction: slice the full fp32 master into the
            # cluster's dp_shard Shard(0) shards, digest each, combine over N
            # virtual DP ranks in rep-major order — the single-device equivalent
            # of the cluster's per-rank to_local() digest + dp_group all_gather.
            _shard_state = audit_shard_state_digest(
                model,
                dp_shard,
                N,
                optimizer=optimizer,
                include_grads=sh.include_grads,
            )
            _h = finalize_state_hash(
                prev_hash=audit_chained_hash,
                shard_state_digest=_shard_state,
                optimizer=optimizer,
                batch_digest=_bd,
            )
            if _target_reached:
                target_ckpt_digest = _h
                # Capture the target step's final gradients for the handoff
                # sidecar while they are STILL LIVE (zero_grad below clears
                # them). Take each device gradient's host copy, then drop the
                # device reference immediately to avoid a full host/device
                # duplicate. Host copies stay resident until publication starts;
                # the save helper writes/releases them before DCP. Safe because the
                # hash is already computed above and ``zero_grad(set_to_none=True)``
                # clears these on the next line anyway. Only when the run hashed
                # gradients (include_grads) and a handoff is being produced.
                if save_checkpoint_dir is not None and sh.include_grads:
                    held_grads = {}
                    for _n, _p in model.named_parameters():
                        held_grads[_n] = (
                            None
                            if _p.grad is None
                            else _p.grad.detach().contiguous().cpu()
                        )
                        _p.grad = None  # release the device grad now
            if _hash_due:
                audit_chained_hash = _h
            if offload_optimizer:
                for _p in moment_paths:
                    _st = optimizer.state[_p]
                    _st["exp_avg"] = None
                    _st["exp_avg_sq"] = None
                    if "max_exp_avg_sq" in _st:
                        _st["max_exp_avg_sq"] = None
        optimizer.zero_grad(set_to_none=True)
        if _halted:
            step += 1
            break
        # ``model`` is the canonical fp32 master, stepped in place; the FSDP
        # grad_model is re-synced from it at the top of the next step.
        _log_rss(f"step{step}_done")

        consumed += M * mb * cfg.train.seq_len
        step += 1
        # Rank-0 loss record — mirrors the metrics.jsonl row rank 0 logged at
        # this (post-increment) step. A halted step writes no record: the
        # cluster's Halt path logs a metrics row without loss keys (loop.py
        # ``except Halt``), so there is nothing to compare against.
        rank0_losses.append({
            "step": step,
            "consumed_tokens": consumed,
            "loss_ce": ce_avg,
            "loss_zloss": zl_avg,
        })
        if loss_log_path is not None:
            _write_loss_log(loss_log_path, rank0_losses)
        # Mean loss over the step = Σ(ce+zloss) / M; step_loss_t already summed
        # loss=(ce+zloss)/accum over all N·accum=M micro-batches, so divide by N.
        pbar.update(1)  # bar count == absolute optimizer step (started at `initial`)
        pbar.set_postfix(loss=f"{step_loss_t.item() / N:.4f}", tok=f"{consumed / 1e9:.2f}B")
    mb_pbar.close()
    pbar.close()

    # Combine the per-rank batch chains in rank order — the single-device
    # equivalent of RunningBatchHasher.global_digest's all_gather + ordered
    # blake2b over the DP group.
    batch_digest = (
        combine_batch_digests([h.local_digest() for h in batch_hashers])
        if batch_hashers is not None else None
    )

    # Reclaim the fold spill dir. Files are unlinked as they're consumed, so
    # this just removes the (now-empty) auto-created temp dir; best-effort.
    if _spill_owned and spill_dir is not None:
        shutil.rmtree(spill_dir, ignore_errors=True)

    # Bring the offloaded moments back into the optimizer (on CPU) so the hash
    # can read them — the fold is freed by now, so ~64 GB (8B) has room.
    if offload_optimizer:
        _materialize_moments(optimizer, moment_paths)
        if _optim_dir_owned and optim_dir is not None:
            shutil.rmtree(optim_dir, ignore_errors=True)

    # The master was reloaded for the final step's hash, so it's resident now;
    # just drop the (stale) spill files.
    if offload_master and _master_dir_owned and master_dir is not None:
        shutil.rmtree(master_dir, ignore_errors=True)

    # Debug: tensor-level diff against a reference checkpoint to localize a hash
    # mismatch. REPOP_AUDIT_DIFF_CKPT=<dir> loads that checkpoint into a fresh
    # model+optim and reports the params/moments that differ most from the
    # replayed state. Purely diagnostic; no effect on the returned digest.
    _diff_ckpt = os.environ.get("REPOP_AUDIT_DIFF_CKPT", "")
    if _diff_ckpt:
        LOG.info("REPOP_AUDIT_DIFF_CKPT set — diffing replayed state vs %s", _diff_ckpt)
        ref_model = build_model(cfg.model, device=dev)
        ref_model = parallelize_llama3_repop(
            ref_model, cfg, ParallelDims(dp_replicate=1, dp_shard=1, world_size=1)
        )
        ref_opt = build_optimizer(ref_model, cfg.optim)
        Checkpointer(os.path.dirname(_diff_ckpt) or ".").load(_diff_ckpt, ref_model, ref_opt)
        rows = []
        ref_params = dict(ref_model.named_parameters())
        for n, p in model.named_parameters():
            if n in ref_params:
                d = (p.detach().float() - ref_params[n].detach().float()).abs().max().item()
                rows.append((d, "param", n))
        # optimizer moments (exp_avg / exp_avg_sq), matched by param order
        a_params = list(model.parameters())
        r_params = list(ref_model.parameters())
        for i, (ap, rp) in enumerate(zip(a_params, r_params)):
            a_st = optimizer.state.get(ap, {})
            r_st = ref_opt.state.get(rp, {})
            for key in ("exp_avg", "exp_avg_sq", "step"):
                if key in a_st and key in r_st and torch.is_tensor(a_st[key]):
                    d = (a_st[key].detach().float() - r_st[key].detach().float()).abs().max().item()
                    rows.append((d, f"optim.{key}", f"param[{i}]"))
        rows.sort(reverse=True)
        nz = [r for r in rows if r[0] > 0]
        LOG.info("DIFF: %d/%d tensors differ; top 15 by max|Δ|:", len(nz), len(rows))
        for d, kind, name in rows[:15]:
            LOG.info("  Δ=%.6e  %s  %s", d, kind, name)

    # The reproduced digest is what the TARGET checkpoint's state_hash.txt
    # stores: the target-step hash chained from the running chain (== the
    # chain value itself when the target is hash-due; the side-link when the
    # target checkpoint sits between hash-due steps). A halt before the target
    # reports the running chain instead (the cluster halted on that step too).
    # Fall back to a plain end-of-run hash only when periodic hashing is off
    # entirely (no chain exists) so the mismatch surfaces loudly.
    if target_ckpt_digest is not None:
        digest = target_ckpt_digest
    elif audit_chained_hash is not None:
        digest = audit_chained_hash
    else:
        # No hash-due step in the interval (misconfigured cadence). Fall back to
        # a standalone end-of-run hash via the SAME sharded scheme the cluster
        # uses, so the mismatch (if any) is apples-to-apples rather than a v2/v3
        # schema artifact. Mirrors the pre-v3 fallback's args (include_grads=False,
        # prev_hash=None) — this is a loud error-surfacing path, not a faithful
        # reproduction (a correctly-configured audit always lands on a hash-due
        # step and returns the chained value above).
        _shard_state = audit_shard_state_digest(
            model, dp_shard, N, optimizer=optimizer, include_grads=False
        )
        digest = finalize_state_hash(
            prev_hash=None,
            shard_state_digest=_shard_state,
            optimizer=optimizer,
            batch_digest=batch_digest,
        )
    result = {
        "step": step,
        "consumed_tokens": consumed,
        "state_hash": digest,
        "repop": repop_provenance,
        "device": dev.type,
        "rank0_losses": rank0_losses,
    }
    if loss_log_path is not None:
        # Final (possibly empty) write so the file exists and is complete even
        # for a zero-step replay or a halt before the first completed step.
        _write_loss_log(loss_log_path, rank0_losses)
        result["loss_log"] = str(loss_log_path)

    if expected_digest is not None:
        want = expected_digest
        result["expected"] = want
        result["match"] = (want == digest)
        LOG.info("state_hash=%s expected=%s MATCH=%s", digest[:16], want[:16], result["match"])
        if not result["match"] and expect_hash is not None and Path(expect_hash).is_file():
            # Before the writer-side hash fix, FINAL and EMERGENCY saves
            # (the run's off-cadence checkpoints) recomputed their
            # state_hash.txt after zero_grad and after the chain had advanced —
            # a double-linked "grad_none" digest that NO replay can reproduce.
            # The 1B run's step-80957 checkpoint is the known published case.
            # The replayed state may still be bit-perfect; verify against the
            # references the loop actually logged.
            note = (
                "--expect-hash was read from a file. If that file is a "
                "checkpoint state_hash.txt written by a FINAL or EMERGENCY "
                "save from before the writer-side hash fix (an off-cadence "
                "step; e.g. the "
                "1B run's step_000080957), it stores an unreproducible "
                "post-hoc digest — the replay may be bit-perfect anyway. "
                "Verify against the run's logs/state_hashes.jsonl entry for "
                "this step, the checkpoint meta.json's chained_hash, or "
                "reference_state_hashes.jsonl instead."
            )
            result["expect_hash_note"] = note
            LOG.warning("%s", note)
    else:
        LOG.info("state_hash=%s (step=%d consumed_tokens=%d)", digest[:16], step, consumed)

    # ---- Chained-audit checkpoint save ------------------------------------
    # Persist a fresh, fully-loadable checkpoint so the NEXT interval can be
    # audited starting from here (the user re-uses THIS audit's verified output
    # instead of re-auditing this interval). Written only after the replay, from
    # the post-step state that produced ``digest``.
    if save_checkpoint_dir is not None:
        # Guard 1: the digest must be a real target-step checkpoint hash.
        # ``target_ckpt_digest`` is None only when the replay never reached the
        # target with periodic hashing on (spike-halt short of the target, or
        # every_n_steps == 0 — the loud standalone fallback above). Neither
        # yields the per-checkpoint hash the loop stamps on every save, so
        # there is nothing valid to hand off.
        if target_ckpt_digest is None:
            raise ValueError(
                "--save-checkpoint-dir requires the replay to reach the target "
                "step with periodic state hashing enabled "
                "(state_hash.every_n_steps > 0); this replay produced no "
                "target-step checkpoint hash to hand off."
            )
        # Guard 2: never persist an unreproduced state. If --expect-hash was given
        # and did not match, the replayed state diverged from the canonical chain;
        # chaining the next audit off it would propagate the divergence.
        if expected_digest is not None and not result["match"]:
            # This guard runs inside audit_replay(), so on the chained flow
            # (--save-checkpoint-dir, which the audit CLI always passes) it
            # raises before main() can report. Same message shape either way;
            # see mismatch_message for why it is not truncated.
            raise ValueError(
                mismatch_message(digest, expected_digest)
                + "; refusing to save a checkpoint that would chain the next "
                  "audit off an unreproduced state."
            )
        if expected_digest is None:
            LOG.warning(
                "saving a chained-audit checkpoint WITHOUT --expect-hash: the "
                "state_hash was not verified against a canonical value, so the "
                "next audit chains off an unverified state. Pass --expect-hash to "
                "confirm this interval before handing off."
            )

        # views[0].state() is the GLOBAL stream position at the target step (every
        # view length-walks the full window sequence in lockstep), i.e. the state
        # to resume the NEXT interval from — exactly what the loop persists.
        saved_dir = _save_chained_audit_checkpoint(
            save_dir=save_checkpoint_dir,
            step=step,
            consumed=consumed,
            digest=digest,
            chained_hash_meta=audit_chained_hash,
            meta_obj=meta_obj,
            stream_state=views[0].state(),
            model=model,
            optimizer=optimizer,
            spike_state=spike.state_dict(),
            batch_hashers=batch_hashers,
            batch_digest=batch_digest,
            N=N,
            gradients=held_grads,
        )
        if held_grads is None:
            LOG.warning(
                "saving a chained-audit checkpoint WITHOUT a gradient sidecar: "
                "this run hashed with state_hash.include_grads=false, so the "
                "target-step gradients were never hashed and cannot be exported."
            )
        LOG.info(
            "saved chained-audit checkpoint → %s (step=%d consumed_tokens=%d "
            "chained_hash=%s); audit the next interval with "
            "--checkpoint %s", saved_dir, step, consumed, digest[:16], saved_dir,
        )
        result["saved_checkpoint"] = str(saved_dir)

    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    args = _parse_args()
    thread_settings = None
    if args.device == "cpu":
        from pretrain.cli.cpu_threads import configure_cpu_threads

        thread_settings = configure_cpu_threads(args.cpu_threads)
        LOG.info("CPU thread selection: %s", json.dumps(thread_settings, sort_keys=True))
    elif args.cpu_threads != "auto":
        raise SystemExit("--cpu-threads requires --device cpu")
    res = audit_replay(
        args.checkpoint,
        config_name=args.config_name,
        device=args.device,
        from_init=args.from_init,
        until_step=args.until_step,
        expect_hash=args.expect_hash,
        data_root=args.data_root,
        gcs_root=args.gcs_root,
        fetch_dest=args.fetch_dest,
        fold_spill_dir=args.fold_spill_dir,
        no_fold_spill=args.no_fold_spill,
        offload_optimizer=args.offload_optimizer,
        optimizer_offload_dir=args.optimizer_offload_dir,
        offload_master=args.offload_master,
        offload_grads=args.offload_grads,
        master_offload_dir=args.master_offload_dir,
        save_checkpoint_dir=args.save_checkpoint_dir,
        descriptor_checkpoint=args.descriptor_checkpoint,
        loss_log=args.loss_log,
    )
    if res.get("match") is False:
        msg = mismatch_message(res["state_hash"], res["expected"])
        if res.get("expect_hash_note"):
            msg += f"\n{res['expect_hash_note']}"
        raise SystemExit(msg)
    if thread_settings is not None:
        res["cpu_threads"] = thread_settings
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
