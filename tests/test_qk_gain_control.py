"""QK-norm gain controls (clamp + staged wake-up) — Option A of the 20260703
recovery. See pretrain.train.qk_gain_control for the incident background.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pretrain.train.qk_gain_control import clamp_qk_gains, zero_q_gain_grads


class _Gain(nn.Module):
    def __init__(self, dim: int, fill: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), fill))


class _Attn(nn.Module):
    def __init__(self, q_fill: float, k_fill: float) -> None:
        super().__init__()
        self.q_norm = _Gain(8, q_fill)
        self.k_norm = _Gain(8, k_fill)
        self.wq = nn.Linear(8, 8, bias=False)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Attn(1.0, 16.0), _Attn(1.0, 25.0)])
        self.other_norm = _Gain(8, 30.0)  # NOT a q_norm/k_norm — untouched


def test_clamp_binds_only_above_cap():
    m = _Model()
    n = clamp_qk_gains(m, 20.0)
    assert n == 4  # 2 blocks × (q_norm + k_norm)
    assert float(m.blocks[0].k_norm.weight.max()) == 16.0  # below cap: untouched
    assert float(m.blocks[1].k_norm.weight.max()) == 20.0  # capped
    assert float(m.blocks[0].q_norm.weight.max()) == 1.0
    assert float(m.other_norm.weight.max()) == 30.0  # non-QK gain untouched


def test_clamp_is_symmetric_and_idempotent():
    m = _Model()
    with torch.no_grad():
        m.blocks[0].k_norm.weight[0] = -42.0
    clamp_qk_gains(m, 20.0)
    assert float(m.blocks[0].k_norm.weight[0]) == -20.0
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    clamp_qk_gains(m, 20.0)  # second application: bitwise no-op
    for n, p in m.named_parameters():
        assert torch.equal(p.data, before[n]), n


def test_zero_q_gain_grads_targets_only_q():
    m = _Model()
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    n = zero_q_gain_grads(m)
    assert n == 2  # one q_norm per block
    for blk in m.blocks:
        assert float(blk.q_norm.weight.grad.abs().sum()) == 0.0
        assert float(blk.k_norm.weight.grad.abs().sum()) > 0.0  # k side flows
        assert float(blk.wq.weight.grad.abs().sum()) > 0.0  # projections flow
    assert float(m.other_norm.weight.grad.abs().sum()) > 0.0


def test_zero_q_gain_grads_tolerates_missing_grads():
    m = _Model()  # no grads at all
    assert zero_q_gain_grads(m) == 0


def test_no_qk_norm_model_is_noop():
    m = nn.Sequential(nn.Linear(4, 4))
    assert clamp_qk_gains(m, 20.0) == 0
    assert zero_q_gain_grads(m) == 0
