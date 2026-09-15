"""Unit tests for the topology-invariant grad-norm fold (clipping + spike).

The cluster<->audit *bitwise* equality is validated on the GPU cluster (the
fold's whole point is matching FSDP2's real ``Shard(0)`` ``to_local()``). Here we
pin the parts that are checkable single-process on CPU:

  * ``shard0_chunk`` matches DTensor's documented Shard(0) split (ceil chunk
    size, zero-padded tail) — the one fragile boundary in the audit replay;
  * ``audit_total_norm`` equals the explicit per-shard sum-of-squares fold and,
    in real arithmetic, the plain full-vector norm;
  * the fold value is invariant to the shard degree W (topology invariance);
  * tiny / non-divisible parameter shapes are handled.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

# Load deterministic_reduce.py directly: importing it via the package would run
# ``pretrain.parallel.__init__``, which pulls in the model/repop chain (a
# cluster-only GPU dependency). The norm helpers under test are pure torch, so
# executing the leaf module standalone keeps these tests runnable on a CPU box.
_MOD_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/pretrain/parallel/deterministic_reduce.py"
)
_spec = importlib.util.spec_from_file_location("_detreduce_under_test", _MOD_PATH)
_dr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dr)

audit_total_norm = _dr.audit_total_norm
grad_sum_of_squares = _dr.grad_sum_of_squares
shard0_chunk = _dr.shard0_chunk
shard0_full_chunk_size = _dr.shard0_full_chunk_size
audit_per_tensor_and_global_norm = _dr.audit_per_tensor_and_global_norm
deterministic_per_tensor_and_global_norm = _dr.deterministic_per_tensor_and_global_norm


def _dtensor_shard0_reference(t: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    """Reference DTensor ``Shard(0)`` split: pad dim-0 up to ``W*ceil(D/W)`` with
    zeros, then split into W equal contiguous chunks. This is the layout FSDP2
    exposes per parameter, which ``shard0_chunk`` must reproduce."""
    d0 = t.shape[0]
    fcs = (d0 + world_size - 1) // world_size
    padded_len = fcs * world_size
    if padded_len > d0:
        pad = torch.zeros((padded_len - d0, *t.shape[1:]), dtype=t.dtype)
        t = torch.cat([t, pad], dim=0)
    return [t[r * fcs : (r + 1) * fcs] for r in range(world_size)]


class _ToyModel(torch.nn.Module):
    """A few parameters of mixed shapes; grads filled by the test."""

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.empty(8, 4))   # divisible by 4
        self.b = torch.nn.Parameter(torch.empty(6, 3))   # 6 not divisible by 4
        self.c = torch.nn.Parameter(torch.empty(2))      # 1D, dim0 < W=4
        self.d = torch.nn.Parameter(torch.empty(5, 2, 2))  # 3D, odd dim0


def _fill_grads(model: torch.nn.Module, *, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g, dtype=torch.float32)


def test_shard0_full_chunk_size():
    assert shard0_full_chunk_size(8, 4) == 2
    assert shard0_full_chunk_size(6, 4) == 2     # ceil(6/4)
    assert shard0_full_chunk_size(2, 4) == 1     # ceil(2/4)
    assert shard0_full_chunk_size(5, 4) == 2     # ceil(5/4)
    assert shard0_full_chunk_size(10, 1) == 10


def test_shard0_chunk_matches_dtensor_reference():
    for shape in [(8, 4), (6, 3), (2,), (5, 2, 2), (1,), (7, 1)]:
        t = torch.randn(*shape)
        for world_size in (1, 2, 4, 8):
            ref = _dtensor_shard0_reference(t, world_size)
            for r in range(world_size):
                got = shard0_chunk(t, r, world_size)
                assert got.shape == ref[r].shape, (shape, world_size, r)
                assert torch.equal(got, ref[r]), (shape, world_size, r)


def test_shard0_chunks_reconstruct_original():
    """Concatenating the shards and trimming the pad recovers the tensor."""
    t = torch.randn(6, 3)
    W = 4
    chunks = [shard0_chunk(t, r, W) for r in range(W)]
    rebuilt = torch.cat(chunks, dim=0)[: t.shape[0]]
    assert torch.equal(rebuilt, t)
    # Padding region is exactly zeros.
    fcs = shard0_full_chunk_size(t.shape[0], W)
    assert torch.equal(torch.cat(chunks, dim=0)[t.shape[0] :], torch.zeros(fcs * W - t.shape[0], 3))


def test_grad_sum_of_squares():
    t = torch.randn(5, 7)
    expected = (t.double() ** 2).sum()  # high-precision reference
    assert torch.allclose(grad_sum_of_squares(t).double(), expected, rtol=0, atol=1e-3)
    # Empty tensor → 0.
    assert float(grad_sum_of_squares(torch.empty(0))) == 0.0


def test_audit_norm_equals_explicit_shard_fold():
    """audit_total_norm == sqrt(Σ_r Σ_params ‖shard_r(p.grad)‖²) via the reference
    split — i.e. the exact canonical contract."""
    model = _ToyModel()
    _fill_grads(model, seed=1)
    W = 4

    s = [torch.zeros(()) for _ in range(W)]
    for _, p in model.named_parameters():
        ref = _dtensor_shard0_reference(p.grad, W)
        for r in range(W):
            s[r] = s[r] + torch.dot(ref[r].reshape(-1), ref[r].reshape(-1))
    expected = torch.stack(s).sum().sqrt()

    got = audit_total_norm(model, W, torch.device("cpu"))
    assert torch.equal(got, expected)


def test_audit_norm_close_to_full_vector_norm():
    """In real arithmetic the shard fold equals the plain global L2 norm."""
    model = _ToyModel()
    _fill_grads(model, seed=2)
    flat = torch.cat([p.grad.reshape(-1) for _, p in model.named_parameters()])
    full = flat.norm(2)
    got = audit_total_norm(model, 4, torch.device("cpu"))
    assert torch.allclose(got, full, rtol=1e-5, atol=1e-5)


def test_audit_norm_invariant_to_shard_degree():
    """Topology invariance: the fold's value (real arithmetic) does not depend on
    the shard degree W."""
    model = _ToyModel()
    _fill_grads(model, seed=3)
    vals = [float(audit_total_norm(model, W, torch.device("cpu"))) for W in (1, 2, 4, 8)]
    for v in vals[1:]:
        assert abs(v - vals[0]) < 1e-4


# ---- Per-tensor fold -------------------------------------------------------- #


def test_per_tensor_norms_match_torch():
    """Each per-tensor norm equals the plain per-parameter L2 norm."""
    model = _ToyModel()
    _fill_grads(model, seed=5)
    norms, _ = audit_per_tensor_and_global_norm(model, 4, torch.device("cpu"))
    for name, p in model.named_parameters():
        assert torch.allclose(norms[name], p.grad.norm(2), rtol=1e-5, atol=1e-5), name


def test_per_tensor_global_norm_matches_total():
    """The global norm derived from the per-tensor fold agrees (real arithmetic)
    with the plain full-vector norm."""
    model = _ToyModel()
    _fill_grads(model, seed=6)
    _, gnorm = audit_per_tensor_and_global_norm(model, 4, torch.device("cpu"))
    flat = torch.cat([p.grad.reshape(-1) for _, p in model.named_parameters()])
    assert torch.allclose(gnorm, flat.norm(2), rtol=1e-5, atol=1e-5)


def test_per_tensor_fold_bitwise_cluster_vs_audit_world1():
    """Single-process (plain tensors, no shard group) the cluster fold and the
    audit re-slice (dp_shard=1) are BITWISE identical — the invariant the
    single-device audit relies on."""
    model = _ToyModel()
    _fill_grads(model, seed=7)
    dev = torch.device("cpu")
    norms_c, g_c = deterministic_per_tensor_and_global_norm(model, dev)
    norms_a, g_a = audit_per_tensor_and_global_norm(model, 1, dev)
    assert torch.equal(g_c, g_a)
    assert set(norms_c) == set(norms_a)
    for k in norms_c:
        assert torch.equal(norms_c[k], norms_a[k]), k


def test_per_tensor_fold_invariant_to_shard_degree():
    """Per-tensor + global norms (real arithmetic) don't depend on the shard
    degree the audit re-slices over."""
    model = _ToyModel()
    _fill_grads(model, seed=8)
    dev = torch.device("cpu")
    base_norms, base_g = audit_per_tensor_and_global_norm(model, 1, dev)
    for W in (2, 4, 8):
        norms, g = audit_per_tensor_and_global_norm(model, W, dev)
        assert abs(float(g) - float(base_g)) < 1e-4, W
        for k in base_norms:
            assert abs(float(norms[k]) - float(base_norms[k])) < 1e-4, (W, k)


def test_per_tensor_skips_none_grads():
    """Parameters with no gradient are omitted from the per-tensor map (and don't
    contribute to the global norm)."""
    model = _ToyModel()
    _fill_grads(model, seed=9)
    model.c.grad = None
    norms, _ = audit_per_tensor_and_global_norm(model, 2, torch.device("cpu"))
    assert "c" not in norms
    assert {"a", "b", "d"} == set(norms)
