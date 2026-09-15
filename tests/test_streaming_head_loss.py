"""Streamed head/loss must preserve both losses and both gradients."""

import pytest
import torch
from repop.nn.linear import linear

from pretrain.model.fused_loss import fused_ce_z_loss
from pretrain.model.streaming_head_loss import streaming_head_loss


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("shape", [(32, 32, 64), (96, 64, 80), (17, 31, 67)])
@pytest.mark.parametrize("coeff", [0.0, 0.01])
def test_streaming_head_bytes(device, shape, coeff):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    rows, dim, vocab = shape
    gen = torch.Generator().manual_seed(851)
    hidden = torch.randn(rows, dim, generator=gen) * 0.1
    weight = torch.randn(vocab, dim, generator=gen) * 0.1
    labels = torch.randint(vocab, (rows,), generator=gen).to(device)
    results = []
    for streamed in [False, True]:
        h = hidden.to(device).detach().requires_grad_()
        w = weight.to(device).detach().requires_grad_()
        if streamed:
            ce, z = streaming_head_loss(h, w, labels, coeff, chunk_rows=32)
        else:
            ce, z = fused_ce_z_loss(linear(h, w), labels, coeff, chunk_rows=32)
        torch.autograd.backward((ce, z), (ce.new_tensor(0.7), z.new_tensor(-0.3)))
        assert h.grad is not None and w.grad is not None
        results.append(
            [x.detach().cpu().reshape(-1).view(torch.uint8) for x in [ce, z, h.grad, w.grad]]
        )
    assert all(torch.equal(a, b) for a, b in zip(*results))
