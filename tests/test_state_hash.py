"""CPU tests for ``compute_state_hash``.

These run without distributed init — DTensor paths are exercised in
``tests/distributed/`` (gated on multi-rank torch.distributed). Here we
verify the per-tensor byte layout, ordering, optimizer-state coverage,
gradient/batch/prev-hash plumbing, and that any single-bit mutation in
any input flips the digest.
"""

from __future__ import annotations

import copy
import os
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from pretrain.train.state_hash import RunningBatchHasher, compute_state_hash


def _tiny_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )


def test_identical_models_hash_equal():
    a = _tiny_model(0)
    b = _tiny_model(0)
    assert compute_state_hash(a) == compute_state_hash(b)


def test_one_ulp_mutation_flips_hash():
    a = _tiny_model(0)
    h0 = compute_state_hash(a)
    with torch.no_grad():
        # Flip the lowest bit of one parameter element.
        flat = next(a.parameters()).view(-1)
        flat[0] = torch.nextafter(flat[0], flat[0] + 1.0)
    h1 = compute_state_hash(a)
    assert h0 != h1


def test_prev_hash_chains():
    a = _tiny_model(0)
    h_base = compute_state_hash(a)
    h_with_prev = compute_state_hash(a, prev_hash=h_base)
    h_with_other_prev = compute_state_hash(a, prev_hash="ab" * 32)
    assert h_with_prev != h_base
    assert h_with_prev != h_with_other_prev
    # Chaining is reproducible.
    assert compute_state_hash(a, prev_hash=h_base) == h_with_prev


def test_prev_hash_accepts_bytes_and_str():
    a = _tiny_model(0)
    raw = bytes(range(32))
    h_b = compute_state_hash(a, prev_hash=raw)
    h_s = compute_state_hash(a, prev_hash=raw.hex())
    assert h_b == h_s


def test_grads_flip_hash():
    a = _tiny_model(0)
    x = torch.randn(2, 8)
    a(x).sum().backward()

    h_no_grad = compute_state_hash(a, include_grads=False)
    h_grad = compute_state_hash(a, include_grads=True)
    assert h_no_grad != h_grad

    # Mutating a gradient must show up with include_grads=True but not
    # without.
    p = next(a.parameters())
    p.grad.view(-1)[0] += 1e-6
    h_grad_mut = compute_state_hash(a, include_grads=True)
    assert h_grad_mut != h_grad
    assert compute_state_hash(a, include_grads=False) == h_no_grad


def test_grad_none_marker_distinct_from_zero_grad():
    """Missing grad vs. all-zero grad should hash differently — otherwise
    a never-stepped optimizer would alias against a fresh zero_grad."""
    a = _tiny_model(0)
    h_none = compute_state_hash(a, include_grads=True)
    for p in a.parameters():
        p.grad = torch.zeros_like(p)
    h_zero = compute_state_hash(a, include_grads=True)
    assert h_none != h_zero


def test_batch_inputs_flip_hash():
    a = _tiny_model(0)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
        "labels": torch.tensor([[2, 3, 4, 5]], dtype=torch.int64),
    }
    h_no_batch = compute_state_hash(a)
    h_with_batch = compute_state_hash(a, batch=batch)
    assert h_no_batch != h_with_batch

    batch_mut = {
        "input_ids": torch.tensor([[1, 2, 3, 5]], dtype=torch.int64),
        "labels": batch["labels"],
    }
    h_mut = compute_state_hash(a, batch=batch_mut)
    assert h_mut != h_with_batch


def test_optimizer_moments_flip_hash():
    a = _tiny_model(0)
    opt = torch.optim.AdamW(a.parameters(), lr=1e-4)

    # Pre-step: no state populated yet — still produces a stable digest.
    h_pre = compute_state_hash(a, optimizer=opt)
    assert h_pre == compute_state_hash(_tiny_model(0), optimizer=torch.optim.AdamW(
        _tiny_model(0).parameters(), lr=1e-4
    ))

    # Run one step → moments populated.
    x = torch.randn(2, 8)
    a(x).sum().backward()
    opt.step()
    h_post = compute_state_hash(a, optimizer=opt)
    assert h_post != h_pre

    # Mutate one exp_avg element; weights unchanged.
    weights_only = compute_state_hash(a)
    p = next(a.parameters())
    opt.state[p]["exp_avg"].view(-1)[0] += 1e-6
    h_post_mut = compute_state_hash(a, optimizer=opt)
    assert h_post_mut != h_post
    # Weights-only hash is unchanged by the moment mutation.
    assert compute_state_hash(a) == weights_only


def test_param_groups_metadata_flips_hash():
    """LR (and other param-group metadata) is part of the hash, so two
    runs at the same step with diverged schedules show up.
    """
    a = _tiny_model(0)
    opt1 = torch.optim.AdamW(a.parameters(), lr=1e-4)
    opt2 = torch.optim.AdamW(_tiny_model(0).parameters(), lr=2e-4)
    h1 = compute_state_hash(a, optimizer=opt1)
    h2 = compute_state_hash(_tiny_model(0), optimizer=opt2)
    assert h1 != h2


def test_order_invariance_of_parameter_construction():
    """Two models with the same named-parameter set hash equally even if
    submodules were registered in a different order."""

    class A(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.x = nn.Linear(4, 4)
            torch.manual_seed(1)
            self.y = nn.Linear(4, 4)

    class B(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1)
            self.y = nn.Linear(4, 4)
            torch.manual_seed(0)
            self.x = nn.Linear(4, 4)

    assert compute_state_hash(A()) == compute_state_hash(B())


def test_bf16_path():
    """bf16 has no numpy dtype — the byte extraction must still work."""
    a = _tiny_model(0).to(torch.bfloat16)
    h = compute_state_hash(a)
    b = copy.deepcopy(a)
    assert compute_state_hash(b) == h
    with torch.no_grad():
        next(b.parameters()).view(-1)[0] = torch.nextafter(
            next(b.parameters()).view(-1)[0],
            next(b.parameters()).view(-1)[0] + 1.0,
        )
    assert compute_state_hash(b) != h


# -----------------------------------------------------------------------------
# RunningBatchHasher + batch_digest tests
# -----------------------------------------------------------------------------


def _mkbatch(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(0, 100, (2, 8), generator=g, dtype=torch.int64),
        "labels": torch.randint(0, 100, (2, 8), generator=g, dtype=torch.int64),
    }


def test_running_batch_hasher_deterministic_chain():
    """Same sequence of updates → same local digest."""
    a, b = RunningBatchHasher(), RunningBatchHasher()
    for s in range(5):
        a.update(_mkbatch(s))
        b.update(_mkbatch(s))
    assert a.local_digest() == b.local_digest()


def test_running_batch_hasher_chain_is_order_sensitive():
    """Reordering updates changes the digest (chain, not commutative)."""
    a, b = RunningBatchHasher(), RunningBatchHasher()
    a.update(_mkbatch(0)); a.update(_mkbatch(1))
    b.update(_mkbatch(1)); b.update(_mkbatch(0))
    assert a.local_digest() != b.local_digest()


def test_running_batch_hasher_digest_does_not_finalize():
    """Reading the digest must not consume internal state."""
    h = RunningBatchHasher()
    h.update(_mkbatch(0))
    d1 = h.local_digest()
    h.update(_mkbatch(1))
    d2 = h.local_digest()
    assert d1 != d2

    # The first digest is unchanged by the subsequent update.
    h2 = RunningBatchHasher()
    h2.update(_mkbatch(0))
    assert h2.local_digest() == d1


def test_running_batch_hasher_global_digest_no_dist():
    """``global_digest(None)`` returns the local digest unchanged."""
    h = RunningBatchHasher()
    h.update(_mkbatch(0))
    assert h.global_digest(None) == h.local_digest()


def test_batch_digest_arg_flips_compute_state_hash():
    a = _tiny_model(0)
    d1 = bytes(range(32))
    d2 = bytes(reversed(range(32)))
    h0 = compute_state_hash(a)
    h1 = compute_state_hash(a, batch_digest=d1)
    h2 = compute_state_hash(a, batch_digest=d2)
    assert h0 != h1 != h2 != h0


def test_batch_digest_wrong_length_rejected():
    a = _tiny_model(0)
    with pytest.raises(ValueError):
        compute_state_hash(a, batch_digest=b"short")


def test_running_batch_hasher_resume_matches_continuous():
    """A hasher primed with a saved chain digest at step K and updated
    over batches K+1..K+N produces the same digest as a continuous
    hasher updated over batches 1..K+N. Regression test for the
    checkpoint-resume state_hash divergence we shipped a fix for.
    """
    batches = [_mkbatch(i) for i in range(10)]
    continuous = RunningBatchHasher()
    for b in batches:
        continuous.update(b)

    split = 4
    first_half = RunningBatchHasher()
    for b in batches[:split]:
        first_half.update(b)
    saved = first_half.local_digest()

    resumed = RunningBatchHasher(prev_digest=saved)
    for b in batches[split:]:
        resumed.update(b)

    assert resumed.local_digest() == continuous.local_digest()


def test_running_batch_hasher_prev_digest_wrong_length_rejected():
    with pytest.raises(ValueError):
        RunningBatchHasher(prev_digest=b"short")


# -----------------------------------------------------------------------------
# Multi-process gloo simulation: 4 ranks, each with its own batch sequence,
# verify global_digest is identical across ranks and depends on every rank's
# input.
# -----------------------------------------------------------------------------


def _gloo_worker(rank: int, world_size: int, init_file: str, out_path: str,
                 batch_offsets: tuple[int, ...]):
    """Run on a child process. Builds a gloo PG, updates a hasher with
    rank-specific batches, writes the global digest hex to ``out_path``.
    """
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "0")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        h = RunningBatchHasher()
        # Per-rank batch sequence — offsets baked from caller so we control
        # exactly what each rank sees across the two pytest sub-runs.
        for s in batch_offsets:
            h.update(_mkbatch(s + rank * 100))
        digest = h.global_digest(dist.group.WORLD).hex()
        # One file per rank so the parent can compare them.
        with open(f"{out_path}.{rank}", "w") as f:
            f.write(digest)
    finally:
        dist.destroy_process_group()


def _spawn_gloo(world_size: int, batch_offsets: tuple[int, ...]) -> list[str]:
    with tempfile.TemporaryDirectory() as td:
        init_file = os.path.join(td, "init")
        out_path = os.path.join(td, "digest")
        # ``spawn`` waits for all children; if any raises we'd see it here.
        mp.spawn(
            _gloo_worker,
            args=(world_size, init_file, out_path, batch_offsets),
            nprocs=world_size,
            join=True,
        )
        return [open(f"{out_path}.{r}").read() for r in range(world_size)]


@pytest.mark.skipif(
    not hasattr(torch.distributed, "is_gloo_available")
    or not torch.distributed.is_gloo_available(),
    reason="gloo backend unavailable",
)
def test_global_digest_all_ranks_agree():
    """4-rank gloo: every rank computes the same global digest."""
    digests = _spawn_gloo(world_size=4, batch_offsets=(0, 1, 2))
    assert len(set(digests)) == 1, f"ranks disagree: {digests}"


@pytest.mark.skipif(
    not hasattr(torch.distributed, "is_gloo_available")
    or not torch.distributed.is_gloo_available(),
    reason="gloo backend unavailable",
)
def test_global_digest_depends_on_every_rank_input():
    """Changing only one rank's data changes the shared global digest."""
    base = _spawn_gloo(world_size=4, batch_offsets=(0, 1, 2))
    perturbed = _spawn_gloo(world_size=4, batch_offsets=(0, 1, 3))
    assert base[0] != perturbed[0]
    # Still in agreement among themselves.
    assert len(set(perturbed)) == 1
