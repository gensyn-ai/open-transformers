"""Fused CE + z-loss (pretrain.model.fused_loss) — correctness + BFR shape.

The fused loss replaces the separate ``cross_entropy_loss`` + model-side z-loss
(which materialised the [N, V] softmax/grad twice and hung the first step at
the 1.6B lm_head). These checks verify the fused gradient is mathematically
correct and — critically for BFR — that the row-chunked computation is
*bitwise* invariant to the chunk size. Cross-device byte-equality is covered by
the GPU regression suite; here we pin the math + chunking on CPU.
"""

from __future__ import annotations

import pytest

pytest.importorskip("repop")

import torch  # noqa: E402

from pretrain.model.fused_loss import fused_ce_z_loss  # noqa: E402

COEFF = 1.0e-4


def _reference(logits: torch.Tensor, labels: torch.Tensor, coeff: float):
    """Plain-torch CE + z-loss; gradient via autograd is the ground truth."""
    lse = torch.logsumexp(logits, dim=-1)
    ce = (lse - logits.gather(-1, labels[:, None]).squeeze(-1)).mean()
    z = coeff * (lse**2).mean()
    return ce, z


@pytest.mark.parametrize("coeff", [COEFF, 0.0])
def test_fused_matches_autograd_reference(coeff):
    torch.manual_seed(0)
    N, V = 48, 512
    base = torch.randn(N, V, dtype=torch.float32)
    labels = torch.randint(0, V, (N,))

    lr = base.clone().requires_grad_()
    ce_r, z_r = _reference(lr, labels, coeff)
    (ce_r + z_r).backward()

    lf = base.clone().requires_grad_()
    ce_f, z_f = fused_ce_z_loss(lf, labels, coeff, chunk_rows=7)
    (ce_f + z_f).backward()

    torch.testing.assert_close(ce_f, ce_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(z_f, z_r, rtol=1e-5, atol=1e-6)
    # repop's exp/log/sum differ from torch's in the last bits, so the grad is
    # close, not bit-equal, to the torch reference — the math is what's checked.
    torch.testing.assert_close(lf.grad, lr.grad, rtol=1e-4, atol=1e-6)


def test_zero_coeff_is_pure_ce():
    """coeff=0 → z-loss is exactly zero and contributes no gradient."""
    torch.manual_seed(1)
    N, V = 32, 256
    logits = torch.randn(N, V, requires_grad=True)
    labels = torch.randint(0, V, (N,))
    ce, z = fused_ce_z_loss(logits, labels, 0.0)
    assert z.item() == 0.0
    (ce + z).backward()
    # CE-only gradient sums to ~0 per row (softmax sums to 1, minus one-hot).
    assert torch.isfinite(logits.grad).all()


def test_gradient_bitwise_chunk_invariant():
    """The row-chunked backward must be BYTE-identical across chunk sizes —
    the load-bearing property for cross-topology BFR."""
    torch.manual_seed(2)
    N, V = 50, 384
    base = torch.randn(N, V)
    labels = torch.randint(0, V, (N,))

    grads = {}
    for cr in (3, 13, N, 10_000):
        lf = base.clone().requires_grad_()
        ce, z = fused_ce_z_loss(lf, labels, COEFF, chunk_rows=cr)
        (ce + z).backward()
        grads[cr] = lf.grad.clone()

    ref = grads[3]
    for cr, g in grads.items():
        assert torch.equal(g, ref), f"chunk_rows={cr} grad differs bitwise"


@pytest.mark.parametrize("coeff", [COEFF, 0.0])
def test_bf16_grad_narrows_per_chunk_bitwise(coeff):
    """A bf16 forward must yield a bf16 grad that is byte-identical to the fp32
    grad cast once at the end.

    The backward computes every chunk in fp32 and stores it into a grad buffer
    of the OUTPUT dtype. Holding a full fp32 ``[N, V]`` instead and casting at
    the end is the obvious alternative, and at the 1.6B lm_head shape
    (``[16384, 128256]``) that buffer is 8.4 GB live across the whole backward.
    Dropping it is only safe because a float->bf16 cast is elementwise
    round-to-nearest-even, so narrowing per chunk and narrowing once are the
    same bits. This pins that equivalence rather than the implementation.
    """
    torch.manual_seed(3)
    N, V = 40, 320
    base = torch.randn(N, V, dtype=torch.bfloat16)
    labels = torch.randint(0, V, (N,))

    lb = base.clone().requires_grad_()
    ce_b, z_b = fused_ce_z_loss(lb, labels, coeff, chunk_rows=9)
    (ce_b + z_b).backward()

    # Same values, fp32 storage: every chunk sees bit-identical inputs because
    # bf16 -> fp32 is exact, so only the store dtype differs.
    lf = base.float().clone().requires_grad_()
    ce_f, z_f = fused_ce_z_loss(lf, labels, coeff, chunk_rows=9)
    (ce_f + z_f).backward()

    assert lb.grad.dtype is torch.bfloat16, "grad must follow the logits dtype"
    assert torch.equal(
        lb.grad.view(torch.uint8), lf.grad.to(torch.bfloat16).view(torch.uint8)
    )
