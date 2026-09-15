"""Multi-rank gloo tests for checkpoint resume's bitwise contracts.

Each test pins down one assumption the resume path relies on. They all
run on CPU with the gloo backend so they don't need GPUs.

What's covered:

  1. ``state["step"]`` (the AdamW iteration counter) survives a save +
     primer-step + load round-trip — DCP must overwrite the primer's
     value rather than leave the primed state in place.
  2. ``_init_optim_state`` creates ``exp_avg`` / ``exp_avg_sq`` as
     DTensors whose placements match the FSDP2-sharded params, so DCP
     has a properly-sharded template to load into.
  3. ``compute_state_hash`` arrives at the same digest on every rank —
     the chained-hash continuity invariant restored from rank-0's
     ``meta.json`` is only valid if all ranks would have computed that
     same value.
  4. ``Checkpointer._get_ckpt_pg()`` returns a global gloo PG containing
     every rank. A subset PG would silently drop shards (the same bug
     ``get_optimizer_state_dict`` introduced under FSDP2).

The fifth concern from the audit — spike-protocol state surviving
across a cooldown that straddles a checkpoint — is covered by the
single-process ``tests/test_spike_protocol.py::test_state_dict_roundtrip``.
"""

from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from pretrain.train.checkpoint import Checkpointer
from pretrain.train.state_hash import compute_state_hash


class _Tiny(nn.Module):
    """Two linears — small enough to step in milliseconds, big enough
    that ``fully_shard`` actually splits parameters across 2 ranks."""

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(16, 32)
        self.lin2 = nn.Linear(32, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(torch.relu(self.lin1(x)))


def _build_sharded_model(seed: int, mesh) -> _Tiny:
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(seed)
    m = _Tiny()
    # Wrap each leaf + the root so every parameter ends up as a DTensor
    # over the mesh — matches the production wrap order (per-block +
    # root in ``parallelize_llama3_repop._apply_fsdp``).
    fully_shard(m.lin1, mesh=mesh)
    fully_shard(m.lin2, mesh=mesh)
    fully_shard(m, mesh=mesh)
    return m


def _setup_dist(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group(backend="gloo")


def _worker(
    rank: int,
    world_size: int,
    port: int,
    status_path: str,
    shared_dir: str,
    target: str,
) -> None:
    """Generic worker — dispatch to the named test body by string so the
    spawned process doesn't need to pickle the function object.

    ``shared_dir`` is a single tmpdir created by the parent before
    spawning, so all ranks read/write to the same path (necessary for
    DCP save+load where rank N reads what rank 0 wrote).
    """
    try:
        _setup_dist(rank, world_size, port)
        fn = globals()[target]
        fn(rank, world_size, shared_dir)
        dist.destroy_process_group()
        Path(status_path).write_text(f"rank{rank}:OK\n")
    except Exception:
        Path(status_path).write_text(
            f"rank{rank}:FAIL\n{traceback.format_exc()}"
        )
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _spawn(world_size: int, target: str, timeout_s: int = 120) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        status_paths = [
            os.path.join(tmpdir, f"status_{r}.txt") for r in range(world_size)
        ]
        # A subdirectory all ranks can share for DCP shards / Checkpointer
        # scratch — TemporaryDirectory in the parent so it survives across
        # all child processes and is cleaned up after.
        shared_dir = os.path.join(tmpdir, "shared")
        os.makedirs(shared_dir, exist_ok=True)
        port = 29600 + (os.getpid() % 1000)
        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(world_size):
            p = ctx.Process(
                target=_worker,
                args=(
                    rank, world_size, port, status_paths[rank], shared_dir, target,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=timeout_s)
            if p.is_alive():
                p.terminate()
                raise RuntimeError(f"worker timed out after {timeout_s}s")
        statuses = [
            Path(s).read_text() if Path(s).exists() else f"rank?:NO_OUTPUT"
            for s in status_paths
        ]
        failures = [s for s in statuses if "OK" not in s.splitlines()[0]]
        if failures:
            pytest.fail(
                "one or more ranks failed:\n" + "\n---\n".join(failures)
            )


# ---------------------------------------------------------------------------
# Risk 1: DCP non-tensor scalar round-trip for state["step"].
# ---------------------------------------------------------------------------
def _w_dcp_step_roundtrip(rank: int, world_size: int, shared_dir: str) -> None:
    """Save with step=5, prime a fresh optim (step=1), load — verify the
    loaded step is 5 (NOT the primer's 1).

    This is the specific gotcha behind ``_init_optim_state``: the primer
    populates the dict so DCP has a template to load into, but DCP must
    actually overwrite the primed values with the saved ones. If DCP
    silently kept the primer's step=1, the resumed optimizer would
    compute bias correction for "step 1" on top of step-5 weights and
    diverge immediately — same failure mode as the original
    empty-state-dict bug.
    """
    from torch.distributed.checkpoint.state_dict import _init_optim_state
    from torch.distributed.device_mesh import init_device_mesh
    import torch.distributed.checkpoint as dcp

    mesh = init_device_mesh("cpu", (world_size,))
    m = _build_sharded_model(seed=0, mesh=mesh)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    # Drive 5 real steps so step counter is 5 and moments are nonzero.
    for _ in range(5):
        x = torch.randn(2, 16)
        m(x).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    p = next(p for p in m.parameters() if p.requires_grad)
    saved_step = opt.state[p]["step"]
    saved_step_val = (
        int(saved_step.item()) if isinstance(saved_step, torch.Tensor) else int(saved_step)
    )
    assert saved_step_val == 5, f"rank{rank}: setup wrong, step={saved_step_val}"
    # Capture a moment value so we can also check the tensor path round-trips.
    saved_exp_avg = opt.state[p]["exp_avg"]
    saved_exp_avg_local = (
        saved_exp_avg.to_local().clone() if hasattr(saved_exp_avg, "to_local")
        else saved_exp_avg.clone()
    )

    ckpt_path = os.path.join(shared_dir, "dcp_step_roundtrip")
    state = {"model": m.state_dict(), "optim": opt.state_dict()}
    dcp.save(state, checkpoint_id=ckpt_path)

    # Fresh model + optim — no steps taken.
    m2 = _build_sharded_model(seed=0, mesh=mesh)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    assert not opt2.state, "fresh optimizer should have empty state"

    _init_optim_state(opt2)
    p2 = next(p for p in m2.parameters() if p.requires_grad)
    primed_step = opt2.state[p2]["step"]
    primed_step_val = (
        int(primed_step.item()) if isinstance(primed_step, torch.Tensor) else int(primed_step)
    )
    assert primed_step_val == 1, (
        f"rank{rank}: primer should produce step=1, got {primed_step_val}"
    )

    state2 = {"model": m2.state_dict(), "optim": opt2.state_dict()}
    dcp.load(state2, checkpoint_id=ckpt_path)
    m2.load_state_dict(state2["model"])
    opt2.load_state_dict(state2["optim"])

    loaded_step = opt2.state[p2]["step"]
    loaded_step_val = (
        int(loaded_step.item()) if isinstance(loaded_step, torch.Tensor) else int(loaded_step)
    )
    assert loaded_step_val == 5, (
        f"rank{rank}: step expected 5 (saved), got {loaded_step_val} — "
        f"DCP failed to overwrite the primer's step=1 on load. The "
        f"resumed optimizer would warm-start from the primer's bias "
        f"correction instead of the saved iteration."
    )

    # Tensor moments must also round-trip (sanity check on the
    # in-place tensor path).
    loaded_exp_avg = opt2.state[p2]["exp_avg"]
    loaded_local = (
        loaded_exp_avg.to_local() if hasattr(loaded_exp_avg, "to_local")
        else loaded_exp_avg
    )
    assert torch.allclose(loaded_local, saved_exp_avg_local), (
        f"rank{rank}: exp_avg differs after round-trip"
    )


def test_dcp_step_roundtrip():
    _spawn(world_size=2, target="_w_dcp_step_roundtrip")


# ---------------------------------------------------------------------------
# Risk 2: _init_optim_state produces DTensor moments matching param placements.
# ---------------------------------------------------------------------------
def _w_init_optim_state_dtensor_placements(rank: int, world_size: int, shared_dir: str) -> None:  # noqa: ARG001
    """For every param that's a DTensor, the corresponding
    ``exp_avg``/``exp_avg_sq`` allocated by ``_init_optim_state`` must
    be a DTensor with the same placements and device_mesh. If
    ``torch.zeros_like(dtensor)`` collapsed to a plain Tensor (or
    landed on the wrong mesh), DCP would write a malformed shard and
    every restored run would silently train from zero moments.
    """
    from torch.distributed.checkpoint.state_dict import _init_optim_state
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    mesh = init_device_mesh("cpu", (world_size,))
    m = _build_sharded_model(seed=0, mesh=mesh)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    assert not opt.state

    _init_optim_state(opt)

    sharded_params_seen = 0
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        assert p in opt.state, f"rank{rank}: param {name} missing from opt.state"
        st = opt.state[p]
        for key in ("exp_avg", "exp_avg_sq"):
            assert key in st, f"rank{rank}: param {name} missing state[{key!r}]"
            moment = st[key]
            if isinstance(p, DTensor):
                sharded_params_seen += 1
                assert isinstance(moment, DTensor), (
                    f"rank{rank}: param {name} is DTensor but state[{key!r}] "
                    f"is {type(moment).__name__} — torch.zeros_like did not "
                    f"preserve sharding, DCP would not be able to plan a "
                    f"per-rank shard load"
                )
                assert tuple(moment.placements) == tuple(p.placements), (
                    f"rank{rank}: param {name} placements {p.placements} "
                    f"but state[{key!r}] placements {moment.placements}"
                )
                assert moment.device_mesh == p.device_mesh, (
                    f"rank{rank}: param {name} mesh mismatch on {key!r}"
                )
                assert moment.shape == p.shape, (
                    f"rank{rank}: param {name} shape {p.shape} but "
                    f"state[{key!r}] shape {moment.shape}"
                )
    assert sharded_params_seen > 0, (
        f"rank{rank}: no DTensor params found — fully_shard wrap may not "
        f"have produced sharded params on this build"
    )


def test_init_optim_state_preserves_dtensor_placements():
    _spawn(world_size=2, target="_w_init_optim_state_dtensor_placements")


# ---------------------------------------------------------------------------
# Risk 3: compute_state_hash invariant across ranks.
# ---------------------------------------------------------------------------
def _w_state_hash_invariance(rank: int, world_size: int, shared_dir: str) -> None:  # noqa: ARG001
    """Every rank computing ``compute_state_hash`` on the same sharded
    model must produce the same digest. Resume restores rank-0's
    ``chained_hash`` to every rank under the assumption this invariant
    holds; if it broke, the resumed run's chained_hash would diverge
    from a continuous run as soon as it folded the rank-0 prev_hash
    into a non-rank-0 next hash.
    """
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world_size,))
    m = _build_sharded_model(seed=0, mesh=mesh)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # One real step so optim state isn't empty and the hash exercises
    # the optimizer-moment fold in.
    x = torch.randn(2, 16)
    m(x).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    digest_hex = compute_state_hash(m, optimizer=opt)
    digest_bytes = bytes.fromhex(digest_hex)
    assert len(digest_bytes) == 32

    # Gather all ranks' digests and verify they're byte-identical.
    gathered = [torch.empty(32, dtype=torch.uint8) for _ in range(world_size)]
    src = torch.frombuffer(bytearray(digest_bytes), dtype=torch.uint8)
    dist.all_gather(gathered, src)
    for r, g in enumerate(gathered):
        other = bytes(g.numpy())
        assert other == digest_bytes, (
            f"rank{rank}: rank-{r} digest {other.hex()[:16]}… "
            f"differs from this rank's {digest_hex[:16]}…"
        )


def test_compute_state_hash_invariant_across_ranks():
    _spawn(world_size=2, target="_w_state_hash_invariance")


# ---------------------------------------------------------------------------
# Risk 4: Checkpointer._get_ckpt_pg() includes all ranks.
# ---------------------------------------------------------------------------
def _w_ckpt_pg_global(rank: int, world_size: int, shared_dir: str) -> None:  # noqa: ARG001
    """``_get_ckpt_pg`` returns a gloo PG whose world matches the global
    world. A subset PG would cause DCP to write only that subset's
    shards — same silent-shard-loss as the FSDP1/FSDP2 mix-up that bit
    us with ``get_optimizer_state_dict``.
    """
    ckpt = Checkpointer(os.path.join(shared_dir, "ckpt_pg_test"))
    pg = ckpt._get_ckpt_pg()
    assert pg is not None, f"rank{rank}: _get_ckpt_pg returned None under dist"
    ws = dist.get_world_size(group=pg)
    assert ws == world_size, (
        f"rank{rank}: ckpt PG world_size={ws}, expected {world_size}. "
        f"Subset PG would lose shards on save."
    )
    # Idempotent: repeated calls return the cached PG.
    pg2 = ckpt._get_ckpt_pg()
    assert pg is pg2, f"rank{rank}: _get_ckpt_pg created a second PG"

    # The PG must actually be usable for a collective — exercise one
    # to make sure it isn't a stale handle.
    t = torch.tensor([float(rank)])
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=pg)
    expected = float(sum(range(world_size)))
    assert t.item() == expected, (
        f"rank{rank}: all_reduce on ckpt PG gave {t.item()}, expected {expected}"
    )


def test_ckpt_pg_includes_all_ranks_and_is_usable():
    _spawn(world_size=2, target="_w_ckpt_pg_global")


# ---------------------------------------------------------------------------
# Risk 5: optimizer ``step`` MUST be a Tensor, not a Python int.
#
# This is the bug behind "loss/perplexity diverges immediately on resume"
# with the repop optimizer. ``FSDPAwareRepopAdamW`` originally stored
# ``state["step"]`` as a Python ``int``. DCP only restores *tensor* leaves
# into the resume template (it loads into the tensor's existing storage in
# place); a non-tensor int leaf is silently left at the primer's value of 1
# after ``dcp.load``. A resumed run then bias-corrects for step≈2 instead of
# step≈N — applying drastically oversized AdamW updates on the first
# post-resume step.
#
# ``test_dcp_step_roundtrip`` above did not catch this because it uses
# ``torch.optim.AdamW``, whose ``step`` is already a tensor. This test
# pins both halves of the contract using an optimizer that mirrors the
# repop state layout (step we manage by hand, float32 moments).
# ---------------------------------------------------------------------------
class _ManualAdamW(torch.optim.Optimizer):
    """AdamW whose state layout mirrors ``FSDPAwareRepopAdamW``.

    ``tensor_step`` toggles whether ``state["step"]`` is a 0-dim int64
    tensor (correct) or a Python int (the original, broken layout).
    """

    def __init__(self, params, *, tensor_step: bool, lr: float = 1e-3):
        super().__init__(params, dict(lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1))
        self._tensor_step = tensor_step

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ARG002
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((), dtype=torch.int64) if self._tensor_step else 0
                    )
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                t = (
                    int(state["step"].item())
                    if isinstance(state["step"], torch.Tensor)
                    else state["step"]
                )
                g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                pl = p.data.to_local() if hasattr(p.data, "to_local") else p.data
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                ml = m.to_local() if hasattr(m, "to_local") else m
                vl = v.to_local() if hasattr(v, "to_local") else v
                gf = g.to(torch.float32)
                ml.mul_(b1).add_(gf, alpha=1 - b1)
                vl.mul_(b2).addcmul_(gf, gf, value=1 - b2)
                denom = (vl.sqrt() / ((1 - b2 ** t) ** 0.5)).add_(group["eps"])
                pl.add_((ml / (1 - b1 ** t) / denom).to(pl.dtype), alpha=-group["lr"])
        return None


def _loaded_step_value(opt, param) -> int:
    s = opt.state[param]["step"]
    return int(s.item()) if isinstance(s, torch.Tensor) else int(s)


def _w_step_must_be_tensor(rank: int, world_size: int, shared_dir: str) -> None:
    """Tensor ``step`` survives a save→primer→load round-trip; a Python-int
    ``step`` does NOT (it is left at the primer's value of 1).
    """
    from torch.distributed.checkpoint.state_dict import _init_optim_state
    from torch.distributed.device_mesh import init_device_mesh
    import torch.distributed.checkpoint as dcp

    mesh = init_device_mesh("cpu", (world_size,))
    n_steps = 6

    def save_then_load(tensor_step: bool, name: str) -> int:
        m = _build_sharded_model(seed=0, mesh=mesh)
        opt = _ManualAdamW(m.parameters(), tensor_step=tensor_step)
        for _ in range(n_steps):
            m(torch.randn(2, 16)).sum().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        ckpt_path = os.path.join(shared_dir, f"step_kind_{name}")
        dcp.save({"model": m.state_dict(), "optim": opt.state_dict()}, checkpoint_id=ckpt_path)

        m2 = _build_sharded_model(seed=0, mesh=mesh)
        opt2 = _ManualAdamW(m2.parameters(), tensor_step=tensor_step)
        _init_optim_state(opt2)
        st = {"model": m2.state_dict(), "optim": opt2.state_dict()}
        dcp.load(st, checkpoint_id=ckpt_path)
        m2.load_state_dict(st["model"])
        opt2.load_state_dict(st["optim"])
        p2 = next(p for p in m2.parameters() if p.requires_grad)
        return _loaded_step_value(opt2, p2)

    # The fix: a tensor step restores the saved iteration count.
    tensor_loaded = save_then_load(tensor_step=True, name="tensor")
    assert tensor_loaded == n_steps, (
        f"rank{rank}: tensor step expected {n_steps}, got {tensor_loaded} — "
        f"DCP failed to restore a tensor step leaf"
    )

    # The bug: a Python-int step is silently dropped, leaving the primer's 1.
    int_loaded = save_then_load(tensor_step=False, name="int")
    assert int_loaded == 1, (
        f"rank{rank}: expected the known DCP int-drop behaviour (step left at "
        f"primer value 1), got {int_loaded}. If this changed, the regression "
        f"guard for FSDPAwareRepopAdamW's tensor step may no longer be needed."
    )


def test_optimizer_step_must_be_tensor_to_survive_dcp():
    _spawn(world_size=2, target="_w_step_must_be_tensor")


# ---------------------------------------------------------------------------
# Risk 6: per-rank loop_extras (sampler / RNG / batch_hasher) must land for
# EVERY rank, end-to-end through Checkpointer.save → Checkpointer.load.
#
# Background: the original code had every rank torch.save / write_text its
# own per-rank files. On our cluster's Filestore RWX backend at multi-pod
# scale, 3 of 8 pods' writes silently failed to appear on the PVC, and
# the failure only surfaced as a FileNotFoundError on the NEXT resume
# (when those ranks tried to read their own files). The fix is to gather
# every rank's blob to rank 0 and have rank 0 write all per-rank files
# from a single thread on pod 0 — the same mitigation already in place
# for DCP shards via ``dedup_save_to_lowest_rank=True``.
#
# These tests pin two contracts:
#
#   1. After Checkpointer.save on world=4, every rank's per-rank files
#      exist on disk + the _COMPLETE sentinel is present.
#   2. If a per-rank file is missing at load time (simulating a partial
#      write), Checkpointer.load raises FileNotFoundError naming the
#      partial-write scenario — NOT a confusing legacy-format error.
# ---------------------------------------------------------------------------
def _w_checkpoint_full_roundtrip(rank: int, world_size: int, shared_dir: str) -> None:
    """Full Checkpointer.save → Checkpointer.load round-trip on a sharded
    model. Verifies every rank's per-rank files are present after save
    + _COMPLETE sentinel + load restores the saved sampler/optim state.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from pretrain.data.mix_sampler import MixSamplerState
    from pretrain.train.checkpoint import Checkpointer, CheckpointMeta

    mesh = init_device_mesh("cpu", (world_size,))
    m = _build_sharded_model(seed=0, mesh=mesh)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    # Drive a few steps so optimizer moments are nonzero.
    for _ in range(3):
        m(torch.randn(2, 16)).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    # Per-rank sampler state — deliberately different per rank so we
    # can assert each rank loads ITS OWN file rather than rank 0's.
    sampler_state = MixSamplerState(
        consumed_documents_per_source={"src_a": 100 + rank, "src_b": 50 + rank},
        epoch_per_source={"src_a": rank, "src_b": rank},
        mix_rng_state=None,
        carry_over=[rank, rank * 10],
    )
    meta = CheckpointMeta(
        consumed_tokens=12345,
        step=3,
        git_sha="deadbeef",
        config_resolved="{}",
        tokenizer_hash="x",
        container_digest="y",
        chained_hash=None,
    )
    # Per-rank batch hasher digest — also deliberately different.
    batch_hasher_digest = bytes([rank] * 32)

    ckpt = Checkpointer(os.path.join(shared_dir, f"full_rt_{world_size}"))
    saved = ckpt.save(
        step=3, model=m, optimizer=opt, sampler_state=sampler_state,
        meta=meta, batch_hasher_digest=batch_hasher_digest,
    )

    # Wait for rank 0 to finish writing every rank's files before asserting.
    dist.barrier()

    # Every rank checks: ALL per-rank files exist (rank 0 wrote them
    # for every rank, not just for itself), plus _COMPLETE.
    for r in range(world_size):
        for name in (
            f"sampler.rank_{r}.json",
            f"rng.rank_{r}.pt",
            f"batch_hasher.rank_{r}.bin",
        ):
            p = saved / name
            assert p.exists(), (
                f"rank{rank}: per-rank file {name} missing after save — "
                f"rank-0 gather/write path is broken"
            )
            # Non-empty (rules out the phantom 0-byte file failure we
            # saw in production).
            assert p.stat().st_size > 0, (
                f"rank{rank}: per-rank file {name} is 0 bytes"
            )
    assert (saved / "_COMPLETE").exists(), (
        f"rank{rank}: _COMPLETE sentinel missing"
    )
    assert (saved / "meta.json").exists(), f"rank{rank}: meta.json missing"

    # Load on a fresh model/optimizer. Each rank reads ITS OWN per-rank
    # file — verify by checking the loaded sampler_state matches THIS
    # rank's saved values (not rank 0's).
    m2 = _build_sharded_model(seed=1, mesh=mesh)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    loaded_sampler, loaded_meta, loaded_extras = ckpt.load(saved, m2, opt2)

    assert loaded_sampler.consumed_documents_per_source == {
        "src_a": 100 + rank, "src_b": 50 + rank,
    }, (
        f"rank{rank}: loaded sampler is not this rank's snapshot — "
        f"got {loaded_sampler.consumed_documents_per_source}, expected "
        f"{{src_a: {100+rank}, src_b: {50+rank}}}. likely loaded rank-0's "
        f"file instead of rank-{rank}'s."
    )
    assert loaded_sampler.carry_over == [rank, rank * 10]
    assert loaded_meta.step == 3
    assert loaded_extras["batch_hasher_digest"] == bytes([rank] * 32), (
        f"rank{rank}: batch hasher digest not this rank's"
    )


def test_checkpoint_full_roundtrip_per_rank_writes():
    _spawn(world_size=4, target="_w_checkpoint_full_roundtrip")


def _w_load_hard_errors_on_missing_per_rank(
    rank: int, world_size: int, shared_dir: str
) -> None:
    """If a per-rank file is missing (simulating a partial multi-pod NFS
    write), Checkpointer.load must raise FileNotFoundError with a clear
    message — NOT silently fall back to a legacy format or cold seed.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from pretrain.data.mix_sampler import MixSamplerState
    from pretrain.train.checkpoint import Checkpointer, CheckpointMeta

    mesh = init_device_mesh("cpu", (world_size,))
    m = _build_sharded_model(seed=0, mesh=mesh)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m(torch.randn(2, 16)).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    sampler_state = MixSamplerState(
        consumed_documents_per_source={"src": 10},
        epoch_per_source={"src": 0},
    )
    meta = CheckpointMeta(
        consumed_tokens=1, step=1, git_sha="x",
        config_resolved="{}", tokenizer_hash="x", container_digest="y",
    )

    ckpt = Checkpointer(os.path.join(shared_dir, f"partial_{world_size}"))
    saved = ckpt.save(
        step=1, model=m, optimizer=opt,
        sampler_state=sampler_state, meta=meta,
    )

    dist.barrier()

    # Rank 0 deletes one specific per-rank sampler file to simulate the
    # production failure where pod-1's writes never landed on the PVC.
    target_rank = world_size // 2
    if rank == 0:
        (saved / f"sampler.rank_{target_rank}.json").unlink()

    dist.barrier()

    m2 = _build_sharded_model(seed=1, mesh=mesh)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)

    # Every rank participates in load() — dcp.load is collective, so a
    # rank that bails out before it would hang the others. The per-rank
    # sampler read happens AFTER dcp.load returns, so the target rank
    # gets through the collective and then raises on its missing file;
    # other ranks load successfully.
    if rank == target_rank:
        try:
            ckpt.load(saved, m2, opt2)
        except FileNotFoundError as e:
            msg = str(e)
            assert "sampler.rank_" in msg, (
                f"rank{rank}: error message does not name the missing "
                f"sampler file. got: {msg}"
            )
            assert "partial" in msg.lower() or "incomplete" in msg.lower(), (
                f"rank{rank}: error message should mention the partial / "
                f"incomplete checkpoint scenario so operators can diagnose. "
                f"got: {msg}"
            )
            # Critically, the error must NOT mention "sampler.consumed.json"
            # — that's the legacy-format path we removed, and surfacing it
            # would send operators down a wild goose chase (the original
            # failure mode this whole change fixes).
            assert "sampler.consumed.json" not in msg, (
                f"rank{rank}: error still references the legacy "
                f"sampler.consumed.json path — the legacy fallback should "
                f"be removed. got: {msg}"
            )
        else:
            raise AssertionError(
                f"rank{rank}: load did not raise despite missing per-rank "
                f"sampler file — silent fallback is back, determinism is "
                f"silently broken"
            )
    else:
        # Other ranks have their per-rank files; load must succeed.
        ckpt.load(saved, m2, opt2)


def test_load_hard_errors_on_missing_per_rank_file():
    _spawn(world_size=4, target="_w_load_hard_errors_on_missing_per_rank")


def test_repop_optimizer_uses_tensor_step():
    """The production repop optimizer must store ``step`` as a Tensor so DCP
    can restore it on resume. Runs only where ``repop`` (CUDA) is importable;
    skips on CPU CI where the kernel can't load.
    """
    pytest.importorskip("repop")
    if not torch.cuda.is_available():
        pytest.skip("repop AdamW kernel requires CUDA")
    from pretrain.optim.adamw_repop import FSDPAwareRepopAdamW

    p = torch.nn.Parameter(torch.randn(8, 8, device="cuda"))
    opt = FSDPAwareRepopAdamW([p], lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    p.grad = torch.randn_like(p)
    opt.step()
    step = opt.state[p]["step"]
    assert isinstance(step, torch.Tensor), (
        f"FSDPAwareRepopAdamW.state['step'] is {type(step).__name__}, must be a "
        f"Tensor — a Python int is silently dropped by DCP on resume and the "
        f"resumed run bias-corrects for the wrong iteration."
    )
