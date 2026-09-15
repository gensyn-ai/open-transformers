"""Init determinism + flat trunc-normal properties.

Skipped automatically when ``repop`` is not importable: model construction
now goes through repop kernels, so dev machines without the runtime built
get SKIP rather than a collection ERROR.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("repop")

from pretrain.config import load_config  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.model.init import init_weights_seeded  # noqa: E402


def test_same_seed_same_init():
    cfg = load_config("100m_smoke_repop")
    a = build_model(cfg.model)
    b = build_model(cfg.model)
    init_weights_seeded(a, 42)
    init_weights_seeded(b, 42)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb), f"divergence at {na}"


def test_different_seed_diverges():
    cfg = load_config("100m_smoke_repop")
    a = build_model(cfg.model)
    b = build_model(cfg.model)
    init_weights_seeded(a, 1)
    init_weights_seeded(b, 2)
    diff = sum(
        (pa - pb).abs().sum().item()
        for pa, pb in zip(a.parameters(), b.parameters())
    )
    assert diff > 0


def test_residual_outputs_not_depth_scaled():
    """Flat OLMo 2 init: ``attn.wo`` / ``ffn.w_down`` get the SAME trunc-normal
    std as every other matrix — no ``1/sqrt(2*n_layers)`` residual-output
    shrink (arXiv:2501.00656 §3.2)."""
    cfg = load_config("100m_smoke_repop")
    std = cfg.model.init.std

    m = build_model(cfg.model)
    init_weights_seeded(m, 0)

    block = m.blocks[0]
    wo_std = block.attn.wo.weight.std().item()
    wq_std = block.attn.wq.weight.std().item()

    # wo must match the target std (not a shrunk fraction of it) and track the
    # other projections. Allow 10% slack for finite-sample noise + trunc-normal
    # variance reduction.
    assert abs(wo_std - std) / std < 0.10, f"wo std {wo_std:.5f} vs target {std}"
    assert abs(wo_std - wq_std) / wq_std < 0.10, (
        f"wo std {wo_std:.5f} should match wq std {wq_std:.5f} (no depth scaling)"
    )
    if hasattr(block.ffn, "w_down"):
        wd_std = block.ffn.w_down.weight.std().item()
        assert abs(wd_std - std) / std < 0.10, f"w_down std {wd_std:.5f} vs {std}"


def test_norm_gains_are_one():
    from pretrain.model.modules.norm import RMSNorm

    cfg = load_config("100m_smoke_repop")
    m = build_model(cfg.model)
    init_weights_seeded(m, 0)
    for module in m.modules():
        if isinstance(module, RMSNorm):
            assert torch.allclose(module.weight, torch.ones_like(module.weight))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cross-device init parity needs CUDA"
)
def test_cpu_cuda_byte_identical_init():
    """Init on CPU and CUDA from the same seed must be bit-for-bit identical.

    This is the device-independence guarantee of the repop trunc-normal init:
    repop's Philox stream + correct_rounded_erf/erfinv are byte-equal across
    backends, so every weight matches bitwise regardless of where it was built.
    >=2D weights come from the RNG and are compared bit-exactly; 1-D params
    (zeros, norm gains) are exact too. (A QAT config's LSQ ``weight_scale`` is
    torch-reduction-derived and not part of this guarantee — see _reset_lsq.)
    """
    cfg = load_config("100m_smoke_repop")
    m_cpu = build_model(cfg.model)
    m_cuda = build_model(cfg.model).cuda()

    init_weights_seeded(m_cpu, 1234)
    init_weights_seeded(m_cuda, 1234)

    mismatches = []
    for (na, pa), (nb, pb) in zip(
        m_cpu.named_parameters(), m_cuda.named_parameters()
    ):
        assert na == nb
        if not torch.equal(pa, pb.cpu()):
            mismatches.append(na)
    assert not mismatches, f"CPU/CUDA init diverged at: {mismatches}"
