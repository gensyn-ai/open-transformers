"""bf16 warm-start for LSQ QAT (qat.enable_at_step) — trainer-side toggle.

Contracts:
  * inactive forward is BITWISE identical to a plain (non-QAT) repop Linear
    with the same weights — the bf16 warmup phase is indistinguishable from
    having built the model unquantized;
  * active forward is BITWISE identical to repop's raw LSQQuantizedLinear —
    the subclass adds no numeric difference to the QAT path;
  * the toggle never enters the state_dict (checkpoint layout unchanged);
  * build_linear constructs the warmstart subclass, so isinstance-based repop
    machinery (scale refresh) still applies.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("repop")

from repop.nn.linear import Linear as RepopLinear  # noqa: E402
from repop.qat.lsq import LSQQuantizedLinear, refresh_lsq_weight_scales  # noqa: E402

from pretrain.config.schema import QATConfig  # noqa: E402
from pretrain.model.modules._linear import build_linear  # noqa: E402
from pretrain.model.modules.qat_warmstart import (  # noqa: E402
    WarmstartLSQLinear,
    set_qat_active,
)


def _pair(seed=0):
    torch.manual_seed(seed)
    m = WarmstartLSQLinear(16, 8, bias=True)
    ref = RepopLinear(16, 8, bias=True)
    ref.load_state_dict(
        {k: v for k, v in m.state_dict().items() if k in ("weight", "bias")}
    )
    return m, ref


def test_inactive_is_bitwise_plain_linear():
    m, ref = _pair()
    x = torch.randn(4, 16)
    assert set_qat_active(m, False) == 1
    assert torch.equal(m(x), ref(x))
    # grads flow through the bf16 path; the (unused) scale gets none
    m.zero_grad()
    m(x).sum().backward()
    assert m.weight.grad is not None
    assert m.weight_scale.grad is None


def test_active_is_bitwise_raw_lsq():
    m, _ = _pair()
    raw = LSQQuantizedLinear(16, 8, bias=True)
    raw.load_state_dict(m.state_dict())
    x = torch.randn(4, 16)
    set_qat_active(m, True)
    assert torch.equal(m(x), raw(x))
    m.zero_grad()
    m(x).sum().backward()
    assert m.weight.grad is not None  # QAT path live


def test_toggle_not_in_state_dict_and_flip_roundtrip():
    m, ref = _pair()
    assert "qat_active" not in m.state_dict()
    x = torch.randn(4, 16)
    set_qat_active(m, False)
    off1 = m(x)
    set_qat_active(m, True)
    on = m(x)
    set_qat_active(m, False)
    off2 = m(x)
    assert torch.equal(off1, off2)  # toggle is stateless
    assert not torch.equal(on, off1)  # QAT actually quantizes
    assert torch.equal(off1, ref(x))


def test_build_linear_constructs_warmstart_subclass():
    qat = QATConfig(enabled=True, method="lsq", weight_bits=8, act_bits=8)
    lin = build_linear(16, 8, bias=False, qat=qat)
    assert isinstance(lin, WarmstartLSQLinear)
    assert isinstance(lin, LSQQuantizedLinear)  # repop isinstance machinery holds
    assert refresh_lsq_weight_scales(lin) == 1  # scale refresh sees the subclass


def test_enable_at_step_validation():
    QATConfig(enable_at_step=0)
    QATConfig(enable_at_step=1000)
    with pytest.raises(ValueError):
        QATConfig(enable_at_step=-1)
