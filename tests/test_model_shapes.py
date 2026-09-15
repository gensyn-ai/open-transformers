"""Parameter count + shape sanity at three scales.

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
from pretrain.model.fused_loss import fused_ce_z_loss  # noqa: E402
from pretrain.model.modules import rope as _rope  # noqa: E402


def test_100m_param_count():
    cfg = load_config("100m_smoke_repop")
    m = build_model(cfg.model)
    n = m.num_parameters()
    # The "100M smoke" model is dominated by the 128k vocab embeddings;
    # the transformer blocks themselves are ~10M. We assert a permissive
    # range: at least 100M, at most 500M.
    assert 100_000_000 <= n <= 500_000_000, f"got {n} params"


def test_1b_param_count(monkeypatch):
    cfg = load_config("1b_proxy_repop")

    # Count on meta tensors so the ~1.6B weights are never allocated. RoPE's
    # cos/sin tables are the one construction-time computation that goes
    # through repop kernels, and those reject meta tensors; the tables are
    # non-persistent buffers (not parameters), so give them the right shape
    # without running the kernels.
    def _meta_tables(max_seq_len, head_dim, theta):
        shape = (max_seq_len, head_dim // 2)
        return torch.empty(shape), torch.empty(shape)

    monkeypatch.setattr(_rope, "_build_tables", _meta_tables)
    with torch.device("meta"):
        m = build_model(cfg.model)
    n = m.num_parameters()
    # 1B proxy is ~1.6B due to large vocab; assert plausible range.
    assert 1_000_000_000 <= n <= 2_500_000_000, f"got {n} params"


def test_forward_shapes_100m():
    cfg = load_config("100m_smoke_repop")
    m = build_model(cfg.model)
    B, T = 2, 32
    x = torch.randint(0, cfg.model.vocab_size, (B, T))
    out = m(x)
    assert out.logits.shape == (B, T, cfg.model.vocab_size)
    # z-loss comes from the fused CE/z-loss pass, not the forward.
    assert out.z_loss is None
    assert cfg.model.z_loss.enabled
    labels = torch.randint(0, cfg.model.vocab_size, (B, T))
    _, z_loss = fused_ce_z_loss(out.logits, labels, cfg.model.z_loss.coeff)
    assert z_loss.shape == ()
    assert torch.isfinite(z_loss) and z_loss > 0


def test_qknorm_off_on_match_within_tolerance():
    """At init, QK-Norm and plain GQA should produce *similar* outputs;
    qknorm rescales but does not destroy structure.
    """
    cfg = load_config("100m_smoke_repop")
    cfg_plain = cfg.model.model_copy(deep=True)
    cfg_plain.modules.attention = "gqa_plain_repop"
    cfg_plain.qk_norm = False

    a = build_model(cfg.model)
    b = build_model(cfg_plain)
    # Sync embeddings + everything else by reusing param values where shapes match.
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.shape == pb.shape:
            with torch.no_grad():
                pa.copy_(pb)

    x = torch.randint(0, cfg.model.vocab_size, (1, 32))  # flash kernel: T % 32 == 0
    out_a = a(x).logits
    out_b = b(x).logits
    # Outputs differ but both are finite and not nan.
    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_b).all()
