"""Tests for the sharded-local state hash (v3) and its single-device audit
reconstruction.

Two levels:
  * CPU unit tests (no dist) — the cross-shard combine ordering/wrapping, the
    rep-major rank→shard mapping, mutation sensitivity, and the documented
    topology-dependence.
  * A multi-rank gloo + ``fully_shard`` test — the gold standard: the cluster's
    per-rank ``to_local()`` digest + ``combine_shard_state`` all_gather must
    equal the single-process audit reconstruction (``audit_shard_state_digest``,
    which slices the full-tensor master with ``shard0_logical_chunk``). This is what
    guarantees a CUDA/HSDP run's hash is reproducible bitwise on a 1-device audit.
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from pretrain.parallel.deterministic_reduce import shard0_logical_chunk
from pretrain.train.state_hash import (
    _local_logical,
    audit_shard_state_digest,
    combine_shard_state,
    finalize_state_hash,
    local_shard_state_digest,
)


def _tiny_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    # Mix of divisible and non-divisible dim0 so the ceil-split tail-shard path
    # (tail shard) is exercised against world sizes 2 and 4.
    return nn.Sequential(
        nn.Linear(8, 17),   # weight dim0=17 → non-divisible by 2 and 4
        nn.GELU(),
        nn.Linear(17, 6),   # dim0=6 → divisible by 2, not 4
    )


def _manual_cluster_combine(model, *, num_shards, num_ranks, optimizer=None,
                            include_grads=False):
    """Reference: simulate each DP rank's local digest by extracting its
    Shard(0) slice from the full model, then all_gather+combine exactly as the
    cluster would (rank r holds shard r % num_shards, rep-major)."""
    per_rank = []
    for r in range(num_ranks):
        s = r % num_shards
        per_rank.append(
            local_shard_state_digest(
                model, optimizer=optimizer, include_grads=include_grads,
                extract=(lambda t, s=s: shard0_logical_chunk(t, s, num_shards)),
            )
        )
    combined = hashlib.blake2b(digest_size=32)
    for d in per_rank:
        combined.update(d)
    return combined.digest()


# -----------------------------------------------------------------------------
# CPU unit tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("num_shards,num_ranks", [(1, 1), (2, 2), (4, 4), (1, 2), (2, 4)])
def test_audit_reconstruction_matches_manual_combine(num_shards, num_ranks):
    """audit_shard_state_digest equals an independent manual per-rank combine
    for pure-FSDP (S==N), pure-replicate (S=1), and HSDP (S<N, replicate dup)."""
    m = _tiny_model(0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(4, 8)
    m(x).sum().backward()
    opt.step()

    audit = audit_shard_state_digest(
        m, num_shards, num_ranks, optimizer=opt, include_grads=True
    )
    manual = _manual_cluster_combine(
        m, num_shards=num_shards, num_ranks=num_ranks, optimizer=opt,
        include_grads=True,
    )
    assert audit == manual


def test_hsdp_replicate_duplicates_shard_digests():
    """N=4, S=2 (dp_replicate=2): the combine must see each shard digest twice,
    in rep-major order [s0, s1, s0, s1] — not [s0, s0, s1, s1]."""
    m = _tiny_model(0)
    d0 = local_shard_state_digest(m, extract=lambda t: shard0_logical_chunk(t, 0, 2))
    d1 = local_shard_state_digest(m, extract=lambda t: shard0_logical_chunk(t, 1, 2))
    expect = hashlib.blake2b(digest_size=32)
    for d in (d0, d1, d0, d1):
        expect.update(d)
    assert audit_shard_state_digest(m, 2, 4) == expect.digest()


def test_single_shard_combine_wraps_local():
    """S=1, N=1: combine is blake2b(local) (always-wrap), and matches the
    non-distributed combine_shard_state(local, None)."""
    m = _tiny_model(0)
    local = local_shard_state_digest(m, extract=lambda t: shard0_logical_chunk(t, 0, 1))
    assert audit_shard_state_digest(m, 1, 1) == combine_shard_state(local, None)
    # _local_logical on a plain tensor is identity, so the default-extract local
    # digest equals the logical-chunk(.,0,1) one.
    assert local_shard_state_digest(m, extract=_local_logical) == local


def test_mutation_flips_finalized_hash():
    m = _tiny_model(0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m(torch.randn(4, 8)).sum().backward()
    opt.step()

    def _final(model):
        ss = audit_shard_state_digest(model, 2, 2, optimizer=opt, include_grads=True)
        return finalize_state_hash(prev_hash=None, shard_state_digest=ss, optimizer=opt)

    h0 = _final(m)
    with torch.no_grad():
        flat = next(m.parameters()).view(-1)
        flat[0] = torch.nextafter(flat[0], flat[0] + 1.0)
    assert _final(m) != h0


def test_prev_hash_and_batch_digest_chain():
    m = _tiny_model(0)
    ss = audit_shard_state_digest(m, 2, 2)
    base = finalize_state_hash(prev_hash=None, shard_state_digest=ss)
    chained = finalize_state_hash(prev_hash=base, shard_state_digest=ss)
    with_batch = finalize_state_hash(
        prev_hash=base, shard_state_digest=ss, batch_digest=bytes(range(32))
    )
    assert len({base, chained, with_batch}) == 3
    # reproducible
    assert finalize_state_hash(prev_hash=base, shard_state_digest=ss) == chained


def test_topology_dependence_is_explicit():
    """Different shard counts → different shard-state digests. This documents
    the deliberate loss of cross-topology invariance for the weight term."""
    m = _tiny_model(0)
    assert audit_shard_state_digest(m, 2, 2) != audit_shard_state_digest(m, 4, 4)


def test_finalize_rejects_wrong_length():
    with pytest.raises(ValueError):
        finalize_state_hash(prev_hash=None, shard_state_digest=b"short")
    m = _tiny_model(0)
    ss = audit_shard_state_digest(m, 1, 1)
    with pytest.raises(ValueError):
        finalize_state_hash(prev_hash=None, shard_state_digest=ss, batch_digest=b"x")


# -----------------------------------------------------------------------------
# Distributed gloo + fully_shard: cluster hash == single-process audit recon
# -----------------------------------------------------------------------------


def _reconstruct_full(model, optimizer):
    """Build a plain (unsharded) model+optimizer holding the full_tensor() of
    every sharded param/grad/moment — the audit's fp32 master equivalent.
    full_tensor() is collective, so every rank must call this."""
    plain = _tiny_model(0)  # same arch; values overwritten below
    plain_opt = torch.optim.AdamW(plain.parameters(), lr=1e-3)

    def _ft(t):
        return t.full_tensor() if hasattr(t, "full_tensor") else t

    sh_params = list(model.parameters())
    pl_params = list(plain.parameters())
    with torch.no_grad():
        for (n, p), pp in zip(model.named_parameters(), pl_params):
            pp.copy_(_ft(p.data).detach())
            if p.grad is not None:
                pp.grad = _ft(p.grad).detach().clone()
    for p_sh, p_pl in zip(sh_params, pl_params):
        st = optimizer.state.get(p_sh, {})
        if not st:
            continue
        new = {}
        for k, v in st.items():
            new[k] = _ft(v).detach().clone() if torch.is_tensor(v) else v
        plain_opt.state[p_pl] = new
    return plain, plain_opt


def _dist_worker(rank, world, dp_shard, init_file, out_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        dp_replicate = world // dp_shard
        if dp_replicate == 1:
            # Pure FSDP: 1D shard mesh.
            mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
        else:
            # HSDP: 2D (replicate, shard) mesh — global rank r → (r//dp_shard,
            # r%dp_shard), the rep-major layout the audit combine assumes.
            mesh = init_device_mesh(
                "cpu", (dp_replicate, dp_shard), mesh_dim_names=("replicate", "fsdp")
            )
        model = _tiny_model(0)            # identical init on every rank
        fully_shard(model, mesh=mesh)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Same data on every rank → reduce-scatter'd grad equals the unsharded
        # grad; the hash test only needs cluster==audit on identical state.
        torch.manual_seed(100)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()

        # Cluster side: per-rank local shard digest (default _local_logical
        # extractor strips the tail pad) + all_gather combine over all DP ranks.
        local = local_shard_state_digest(model, optimizer=opt, include_grads=True)
        shard_state = combine_shard_state(local, dist.group.WORLD)
        cluster_hash = finalize_state_hash(
            prev_hash=None, shard_state_digest=shard_state, optimizer=opt
        )

        # Audit side: reconstruct the full master (collective), slice into
        # dp_shard logical shards, combine over N==world virtual ranks.
        plain, plain_opt = _reconstruct_full(model, opt)
        audit_ss = audit_shard_state_digest(
            plain, dp_shard, world, optimizer=plain_opt, include_grads=True
        )
        audit_hash = finalize_state_hash(
            prev_hash=None, shard_state_digest=audit_ss, optimizer=plain_opt
        )

        with open(f"{out_path}.{rank}", "w") as f:
            f.write(f"{cluster_hash}|{audit_hash}")
    finally:
        dist.destroy_process_group()


def _spawn(world, dp_shard):
    with tempfile.TemporaryDirectory() as td:
        init_file = os.path.join(td, "init")
        out_path = os.path.join(td, "out")
        mp.spawn(
            _dist_worker, args=(world, dp_shard, init_file, out_path),
            nprocs=world, join=True,
        )
        return [open(f"{out_path}.{r}").read() for r in range(world)]


@pytest.mark.skipif(
    not hasattr(torch.distributed, "is_gloo_available")
    or not torch.distributed.is_gloo_available(),
    reason="gloo backend unavailable",
)
@pytest.mark.parametrize(
    "world,dp_shard",
    [(2, 2), (4, 4), (4, 2)],  # pure-FSDP×2, pure-FSDP×4, HSDP 2×2
)
def test_fully_shard_cluster_hash_equals_audit_reconstruction(world, dp_shard):
    """The whole point: distributed cluster sharded hash == single-process
    audit reconstruction, using REAL fully_shard to_local() vs the logical
    chunk — across pure-FSDP (1D mesh) and HSDP (2D mesh) topologies."""
    results = _spawn(world, dp_shard)
    clusters = [r.split("|")[0] for r in results]
    audits = [r.split("|")[1] for r in results]
    # All ranks agree on the cluster hash (rank-identical after the combine).
    assert len(set(clusters)) == 1, f"ranks disagree on cluster hash: {clusters}"
    # And every rank's audit reconstruction matches it.
    assert len(set(audits)) == 1, f"ranks disagree on audit hash: {audits}"
    assert clusters[0] == audits[0], (
        f"cluster {clusters[0][:16]} != audit {audits[0][:16]}"
    )
