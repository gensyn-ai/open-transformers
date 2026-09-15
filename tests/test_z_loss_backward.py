"""Backward correctness for ``pretrain.model.z_loss.compute_z_loss``.

z-loss used to be built from repop's forward-only kernels, so its scalar was
detached and contributed *zero* gradient — the regulariser was inert. These
tests pin the explicit ``torch.autograd.Function`` backward:

* gradient matches a pure-torch autograd reference and the closed form,
* the row-chunked backward is bitwise-invariant to ``chunk_rows``, and
* the memory-frugal audit drop-in produces a byte-identical gradient (so the
  single-device audit reproduces the cluster).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("repop")

from pretrain.config.schema import ZLossConfig  # noqa: E402
from pretrain.model.z_loss import (  # noqa: E402
    compute_z_loss,
    compute_z_loss_inert,
    z_loss_grad,
)


def _reference_z_loss(logits: torch.Tensor, coeff: float) -> torch.Tensor:
    """Pure-torch z-loss: ``coeff * mean(logsumexp(logits)**2)``. Built from
    autograd-tracked torch ops so ``.backward()`` yields the reference grad."""
    logits_f = logits.float()
    lse = torch.logsumexp(logits_f, dim=-1)
    return coeff * (lse**2).mean()


def test_forward_value_matches_reference():
    torch.manual_seed(0)
    logits = torch.randn(2, 16, 128, dtype=torch.float32)
    cfg = ZLossConfig(enabled=True, coeff=1e-4)
    z = compute_z_loss(logits, cfg)
    z_ref = _reference_z_loss(logits, cfg.coeff)
    assert torch.allclose(z, z_ref, rtol=1e-4, atol=1e-7), f"{z.item()} vs {z_ref.item()}"


def test_gradient_matches_autograd_reference():
    torch.manual_seed(1)
    coeff = 3e-4
    cfg = ZLossConfig(enabled=True, coeff=coeff)

    logits_a = torch.randn(2, 16, 64, dtype=torch.float32, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)

    compute_z_loss(logits_a, cfg).backward()
    _reference_z_loss(logits_b, coeff).backward()

    assert logits_a.grad is not None and logits_b.grad is not None
    assert torch.allclose(logits_a.grad, logits_b.grad, rtol=1e-4, atol=1e-7), (
        f"max abs diff = {(logits_a.grad - logits_b.grad).abs().max().item()}"
    )


def test_gradient_matches_closed_form():
    """grad = (2*coeff/N) * lse_n * softmax[n, v]."""
    torch.manual_seed(2)
    coeff = 1e-4
    logits = torch.randn(3, 5, 32, dtype=torch.float32, requires_grad=True)
    compute_z_loss(logits, ZLossConfig(enabled=True, coeff=coeff)).backward()

    flat = logits.detach().reshape(-1, logits.shape[-1]).float()
    N = flat.shape[0]
    lse = torch.logsumexp(flat, dim=-1)
    softmax = torch.softmax(flat, dim=-1)
    expected = ((2.0 * coeff / N) * lse).unsqueeze(-1) * softmax
    assert torch.allclose(
        logits.grad.reshape(-1, logits.shape[-1]), expected, rtol=1e-4, atol=1e-7
    )


def test_grad_chunk_size_invariant():
    """Row-chunking the backward must not change a single bit of the result."""
    torch.manual_seed(3)
    logits = torch.randn(7, 9, 48, dtype=torch.float32)
    grad_z = torch.tensor(0.5)
    full = z_loss_grad(logits, 1e-4, grad_z, chunk_rows=10_000)
    chunked = z_loss_grad(logits, 1e-4, grad_z, chunk_rows=4)
    assert torch.equal(full, chunked)


def test_bf16_gradient_returns_input_dtype():
    torch.manual_seed(5)
    logits = torch.randn(2, 8, 64, dtype=torch.bfloat16, requires_grad=True)
    compute_z_loss(logits, ZLossConfig(enabled=True, coeff=1e-4)).backward()
    assert logits.grad is not None
    assert logits.grad.dtype == torch.bfloat16
    assert torch.isfinite(logits.grad.float()).all()
    assert logits.grad.float().abs().sum() > 0


def test_inert_is_value_only_no_gradient():
    """compute_z_loss_inert (phase-1/legacy) must be detached — value only — and
    its value must match the differentiable forward bit-for-bit on the same
    input, so a single checkout audits inert-era checkpoints faithfully."""
    torch.manual_seed(8)
    cfg = ZLossConfig(enabled=True, coeff=1e-4)
    x = torch.randn(4, 16, 256, dtype=torch.float32)

    zi = compute_z_loss_inert(x, cfg)
    assert not zi.requires_grad, "inert z-loss must carry no grad_fn"

    zd = compute_z_loss(x.clone().requires_grad_(True), cfg)
    assert torch.equal(zi, zd.detach()), "inert value must equal the differentiable forward"


def test_zero_coeff_zero_gradient():
    torch.manual_seed(6)
    logits = torch.randn(2, 8, 64, dtype=torch.float32, requires_grad=True)
    compute_z_loss(logits, ZLossConfig(enabled=True, coeff=0.0)).backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0
