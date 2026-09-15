"""Deterministic global clip.

Invariants:
  * the coefficient matches the BFR-proven shape and clamps ≤ 1;
  * train (fold) and audit (re-slice) paths are BITWISE identical;
  * CPU vs MPS bitwise (the cross-device contract that γ's subnormals broke);
  * subnormal posture: the coefficient cannot go subnormal for any physical
    norm, and there is no persistent state to grind down between steps.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("repop")

from pretrain.train import global_clip  # noqa: E402
from pretrain.parallel.deterministic_reduce import (  # noqa: E402
    deterministic_per_tensor_and_global_norm as _det_norm,
)

_DEV = torch.device("cpu")


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(7, 5, bias=True),
        torch.nn.Linear(5, 3, bias=False),
    )


def _fill_grads(m, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    for p in m.parameters():
        p.grad = torch.randn(p.shape, generator=g, dtype=torch.float32) * scale


def test_clip_formula_and_clamp():
    m = _model()
    _fill_grads(m, seed=1, scale=5.0)  # big → clips
    pre = {n: p.grad.clone() for n, p in m.named_parameters()}
    norms0, g0 = _det_norm(m, _DEV)
    ret_norms, ret_gn = global_clip.clip_train(m, 1.0, _DEV)
    assert torch.equal(ret_gn, g0)  # returns PRE-clip norm
    # per-tensor norms exposed for telemetry, PRE-clip, bitwise the fold's own
    for n in norms0:
        assert torch.equal(ret_norms[n], norms0[n]), n
    h = float(torch.clamp(torch.full_like(g0, 1.0).div(g0 + 1e-6), max=1.0))
    assert h < 1.0
    for n, p in m.named_parameters():
        assert torch.allclose(p.grad, pre[n] * h, atol=1e-7), n

    _fill_grads(m, seed=2, scale=1e-3)  # small → no clip (h clamped to 1)
    pre = {n: p.grad.clone() for n, p in m.named_parameters()}
    global_clip.clip_train(m, 1.0, _DEV)
    for n, p in m.named_parameters():
        assert torch.equal(p.grad, pre[n]), n


def test_train_equals_audit_bitwise():
    mA, mB = _model(), _model()
    for step in range(4):
        _fill_grads(mA, seed=100 + step, scale=3.0)
        _fill_grads(mB, seed=100 + step, scale=3.0)
        _, gA = global_clip.clip_train(mA, 1.0, _DEV)
        gB = global_clip.clip_audit(mB, 1.0, 1, _DEV)
        assert torch.equal(gA, gB), step
        for (na, pa), (_, pb) in zip(mA.named_parameters(), mB.named_parameters()):
            assert torch.equal(pa.grad, pb.grad), (step, na)


def test_coefficient_never_subnormal():
    """For any REACHABLE norm the coefficient stays out of the fp32 subnormal
    range (contrast: stateful clipper state reached 7e-44 in run 20260703
    and broke MPS audits). Reachability: an fp32 norm is sqrt(sum-of-squares),
    and the sum overflows to inf beyond norm ≈ sqrt(fp32_max) ≈ 1.84e19 — so
    finite norms cap at ~1.84e19 (h ≈ 5.4e-20, normal) and overflowed norms
    are inf (h = 1/inf = 0.0 exactly, flush-invariant on every backend). A
    subnormal h would need a finite norm ≥ ~8.5e37, which the fold cannot
    produce."""
    fp32_min_normal = 1.1754944e-38
    max_reachable_finite_norm = 1.84e19  # sqrt(fp32 max)
    for norm_val in (0.0, 1e-20, 1.0, 75.0, 1e10, max_reachable_finite_norm):
        t = torch.tensor(norm_val, dtype=torch.float32)
        h = float(torch.clamp(torch.full_like(t, 1.0).div(t + 1e-6), max=1.0))
        assert h >= fp32_min_normal, norm_val  # finite reachable ⇒ normal, nonzero
    t = torch.tensor(float("inf"), dtype=torch.float32)
    h = float(torch.clamp(torch.full_like(t, 1.0).div(t + 1e-6), max=1.0))
    assert h == 0.0  # overflowed norm ⇒ exactly zero, FTZ-invariant


def test_zero_grads_are_noop():
    m = _model()
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    _, gn = global_clip.clip_train(m, 1.0, _DEV)
    assert float(gn) == 0.0
    for p in m.parameters():
        assert float(p.grad.abs().sum()) == 0.0  # h clamps to 1; zeros stay zeros


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_clip_bitwise_cpu_vs_mps():
    """Cross-hardware BFR for the clip path — bitwise-identical clipped
    grads on CPU and MPS."""
    m_cpu, m_mps = _model(), _model()
    m_mps.load_state_dict(m_cpu.state_dict())
    m_mps = m_mps.to("mps")
    for step in range(4):
        _fill_grads(m_cpu, seed=300 + step, scale=4.0)
        for pc, pm in zip(m_cpu.parameters(), m_mps.parameters()):
            pm.grad = pc.grad.detach().clone().to("mps")
        g_cpu = global_clip.clip_audit(m_cpu, 1.0, 1, torch.device("cpu"))
        g_mps = global_clip.clip_audit(m_mps, 1.0, 1, torch.device("mps"))
        assert torch.equal(g_cpu, g_mps.cpu()), step
        for (n, pc), (_, pm) in zip(m_cpu.named_parameters(), m_mps.named_parameters()):
            assert torch.equal(pc.grad, pm.grad.cpu()), (step, n)
