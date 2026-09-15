"""Sanity checks for the repop-runtime model.

Skipped automatically when ``repop`` is not importable, so the test
suite still runs cleanly in dev environments without the runtime built.

What we assert:
  1. Forward shapes are correct.
  2. Backward populates finite, non-trivial grads for every trainable
     parameter (catches the "graph is wired up wrong" failure mode).
"""

from __future__ import annotations

import pytest
import torch

repop = pytest.importorskip("repop")  # noqa: F841

from pretrain.config import load_config  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.model.fused_loss import fused_ce_z_loss  # noqa: E402
from pretrain.model.init import init_weights_seeded  # noqa: E402


def _build(config_name: str, seed: int = 0):
    cfg = load_config(config_name)
    model = build_model(cfg.model)
    init_weights_seeded(model, seed=seed)
    return cfg, model


def test_repop_forward_shapes():
    cfg, m = _build("100m_smoke_repop")
    B, T = 2, 64  # repop's CPU flash kernel requires T % 32 == 0
    x = torch.randint(0, cfg.model.vocab_size, (B, T))
    out = m(x)
    assert out.logits.shape == (B, T, cfg.model.vocab_size)
    assert torch.isfinite(out.logits).all()
    # The forward does not compute z-loss: it shares the CE softmax and is
    # produced by fused_ce_z_loss, the path the loop and audit drive.
    assert out.z_loss is None
    labels = torch.randint(0, cfg.model.vocab_size, (B, T))
    assert cfg.model.z_loss.enabled
    ce, z_loss = fused_ce_z_loss(out.logits, labels, cfg.model.z_loss.coeff)
    assert ce.shape == () and torch.isfinite(ce)
    assert z_loss.shape == () and torch.isfinite(z_loss)
    assert z_loss > 0  # coeff * mean(lse^2) with lse >= log(V)


def test_repop_backward_populates_grads():
    """The autograd graph through repop's CFunctions must populate
    grads for every trainable parameter. Forward + fused CE/z-loss + backward
    is exactly the path the train loop drives every step."""
    cfg, m = _build("100m_smoke_repop")
    B, T = 2, 64  # repop's CPU flash kernel requires T % 32 == 0
    x = torch.randint(0, cfg.model.vocab_size, (B, T))
    labels = torch.randint(0, cfg.model.vocab_size, (B, T))

    out = m(x)
    ce, z_loss = fused_ce_z_loss(out.logits, labels, cfg.model.z_loss.coeff)
    loss = ce + z_loss
    loss.backward()

    any_nonzero = False
    for name, p in m.named_parameters():
        assert p.requires_grad, f"{name} unexpectedly frozen"
        assert p.grad is not None, f"{name} has no grad after backward"
        assert torch.isfinite(p.grad).all(), f"{name} grad has non-finite values"
        if p.grad.abs().sum() > 0:
            any_nonzero = True
    assert any_nonzero, "all grads are zero — autograd path likely broken"
