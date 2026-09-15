"""Multi-rank gloo test: DeterministicReduceScatter is a correct, fixed-order
reduce-scatter.

Runs on CPU/gloo (no GPUs). Validates that the custom comm's output equals an
ascending-rank-sum / world_size reference bitwise — the property that makes the
gradient (and every checkpoint) topology-invariant. The N=4 case is the one that
matters: at N=2 the cross-rank sum is a single commutative add. The headline
bitwise-vs-single-device equivalence on the real model is covered separately (the
GPU validation in scripts/repro/validate_deterministic_rs.py).
"""

from __future__ import annotations

import functools
import importlib.util
import os
import tempfile
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Load the comm module by file path so we don't trigger pretrain.parallel's
# __init__ (which imports the repop-backed model stack, absent in CPU CI).
_SPEC = importlib.util.spec_from_file_location(
    "det_rs",
    Path(__file__).resolve().parents[2] / "src" / "pretrain" / "parallel" / "deterministic_reduce.py",
)
_det = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_det)

# Only the reduce-scatter tests need FSDP2's pluggable reduce-scatter (torch>=
# ~2.8); the recursive-doubling replicate all-reduce uses plain p2p collectives
# available on older torch, so it is NOT gated and runs in local CPU CI.
_needs_custom_rs = pytest.mark.skipif(
    not _det._HAVE_CUSTOM_REDUCE_SCATTER,
    reason="FSDP2 pluggable reduce-scatter requires torch>=~2.8 (cluster NGC image)",
)
DeterministicReduceScatter = _det.DeterministicReduceScatter
DeterministicReplicateAllReduce = _det.DeterministicReplicateAllReduce
tree_reduce_sum = _det.tree_reduce_sum


def _setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group(backend="gloo")


def _body(rank: int, world_size: int) -> None:
    group = dist.group.WORLD
    k = 7  # elements per output shard
    # Each rank's reduce-scatter input: ws*k flat, laid out as ws chunks (one per
    # destination rank). Deterministic per rank, with values varied enough that
    # summation order would matter at N>=4 if it weren't fixed.
    gen = torch.Generator().manual_seed(100 + rank)
    inp = torch.randn(world_size * k, generator=gen, dtype=torch.float32)

    comm = DeterministicReduceScatter()
    out = torch.empty(k, dtype=torch.float32)
    comm(out, inp, group, dist.ReduceOp.AVG)

    # Gather every rank's input and output to verify against the ascending-sum
    # reference computed independently.
    all_inp = [torch.empty_like(inp) for _ in range(world_size)]
    all_out = [torch.empty_like(out) for _ in range(world_size)]
    dist.all_gather(all_inp, inp, group=group)
    dist.all_gather(all_out, out, group=group)

    # Reference: output for dst rank d = (ascending-sum over src of
    # src's chunk d) / world_size.
    for d in range(world_size):
        chunks = [all_inp[s][d * k : (d + 1) * k] for s in range(world_size)]
        ref = functools.reduce(torch.add, chunks) / world_size
        assert torch.equal(all_out[d], ref), f"rank {d} shard mismatch"


def _worker(rank: int, world_size: int, port: int, status_path: str) -> None:
    try:
        _setup(rank, world_size, port)
        _body(rank, world_size)
        dist.destroy_process_group()
        Path(status_path).write_text(f"rank{rank}:OK")
    except Exception:
        Path(status_path).write_text(f"rank{rank}:FAIL\n{traceback.format_exc()}")
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _spawn(world_size: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        status = [os.path.join(tmp, f"s{r}.txt") for r in range(world_size)]
        port = 29700 + (os.getpid() % 800)
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_worker, args=(r, world_size, port, status[r]))
            for r in range(world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
        for r, sp in enumerate(status):
            txt = Path(sp).read_text() if Path(sp).exists() else f"rank{r}:NO_STATUS"
            assert txt.startswith(f"rank{r}:OK"), txt


@_needs_custom_rs
@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 4])
def test_deterministic_reduce_scatter_matches_ascending_reference(world_size):
    _spawn(world_size)


# --------------------------------------------------------------------------- #
# DeterministicReplicateAllReduce: recursive-doubling cross-replica all-reduce.
# The cluster hook must match the single-device audit's balanced-tree fold
# (tree_reduce_sum) bitwise — see cli/audit_replay.py and the GPU-free
# end-to-end check in scripts/repro/validate_recursive_doubling.py.
# --------------------------------------------------------------------------- #


def _body_replicate(rank: int, world_size: int) -> None:
    group = dist.group.WORLD
    hook = DeterministicReplicateAllReduce(group)
    # Several shard lengths incl. non-power-of-2 — the doubling form exchanges
    # the whole buffer each step, so it must be length-agnostic.
    for numel in (1, 7, 257, 4097):
        gen = torch.Generator().manual_seed(500 + rank * 17 + numel)
        x = torch.randn(numel, generator=gen, dtype=torch.float32)
        gathered = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x.clone(), group=group)
        # AVG matches the production path: × (1/ws), NOT ÷ ws. The hook and the
        # single-device audit both reciprocal-multiply (see
        # _recursive_doubling_allreduce_avg / audit_replay); for non-power-of-2
        # ws, ÷ ws (correctly-rounded) differs from × (1/ws) in the last bit, so
        # a ÷ ws reference spuriously fails ws∈{3,5,6,7,9} even though cluster and
        # audit agree bitwise.
        ref = tree_reduce_sum([g.clone() for g in gathered]) * (1.0 / world_size)
        out = x.clone()
        hook(out)  # in-place recursive-doubling all-reduce-AVG
        assert torch.equal(out, ref), f"rank {rank} numel {numel} != tree reference"


def _worker_replicate(rank: int, world_size: int, port: int, status_path: str) -> None:
    try:
        _setup(rank, world_size, port)
        _body_replicate(rank, world_size)
        dist.destroy_process_group()
        Path(status_path).write_text(f"rank{rank}:OK")
    except Exception:
        Path(status_path).write_text(f"rank{rank}:FAIL\n{traceback.format_exc()}")
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 3, 4, 5, 6, 7, 8, 9])
def test_replicate_all_reduce_matches_tree_reference(world_size):
    with tempfile.TemporaryDirectory() as tmp:
        status = [os.path.join(tmp, f"s{r}.txt") for r in range(world_size)]
        port = 28700 + (os.getpid() % 800)
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_worker_replicate, args=(r, world_size, port, status[r]))
            for r in range(world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
        for r, sp in enumerate(status):
            txt = Path(sp).read_text() if Path(sp).exists() else f"rank{r}:NO_STATUS"
            assert txt.startswith(f"rank{r}:OK"), txt


# --------------------------------------------------------------------------- #
# deterministic_scalar_sum: telemetry-grade cross-rank scalar fold (global-
# batch loss logging). Contract: all_gather (data movement only) + HOST fp64
# ascending-rank fold — bitwise equal to a plain Python left-fold over the
# per-rank partials, identical on every rank.
# --------------------------------------------------------------------------- #


def _scalar_partial(rank: int) -> list[float]:
    # Magnitude-spread values so the fp64 fold ORDER is observable in the
    # bits at ws >= 3 (a big/small cancellation pattern loses low bits
    # differently under any non-ascending grouping).
    return [((-1.0) ** rank) * 1e16 + (rank + 1) * 1e-3, 0.1 * rank - 7.0]


def _body_scalar_sum(rank: int, world_size: int) -> None:
    group = dist.group.WORLD
    partial = torch.tensor(_scalar_partial(rank), dtype=torch.float64)
    got = _det.deterministic_scalar_sum(partial, group=group)

    # Reference: independent Python left-fold in ascending rank order.
    ref = list(_scalar_partial(0))
    for r in range(1, world_size):
        for i, v in enumerate(_scalar_partial(r)):
            ref[i] += v
    assert got == ref, f"rank {rank}: {got} != {ref}"  # exact fp64 equality


def _worker_scalar_sum(rank: int, world_size: int, port: int, status_path: str) -> None:
    try:
        _setup(rank, world_size, port)
        _body_scalar_sum(rank, world_size)
        dist.destroy_process_group()
        Path(status_path).write_text(f"rank{rank}:OK")
    except Exception:
        Path(status_path).write_text(f"rank{rank}:FAIL\n{traceback.format_exc()}")
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 3, 4, 6])
def test_deterministic_scalar_sum_matches_ascending_reference(world_size):
    with tempfile.TemporaryDirectory() as tmp:
        status = [os.path.join(tmp, f"s{r}.txt") for r in range(world_size)]
        port = 27700 + (os.getpid() % 800)
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_worker_scalar_sum, args=(r, world_size, port, status[r]))
            for r in range(world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
        for r, sp in enumerate(status):
            txt = Path(sp).read_text() if Path(sp).exists() else f"rank{r}:NO_STATUS"
            assert txt.startswith(f"rank{r}:OK"), txt


def test_deterministic_scalar_sum_single_process_passthrough():
    # No process group in this (main pytest) process: the helper must return
    # the partials unchanged — the world_size==1 / uninitialized path the
    # single-GPU and Mac audit environments take.
    assert not dist.is_initialized()
    vals = [3.140625, -1e16, 0.0]
    got = _det.deterministic_scalar_sum(torch.tensor(vals, dtype=torch.float64))
    assert got == vals
