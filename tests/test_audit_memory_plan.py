"""The audit's automatic memory plan (pretrain.cli.audit_replay).

The planners decide from MEASURED free VRAM, so their decisions cannot be
unit-tested off a GPU. What can be pinned here, and is what actually broke, is
the arithmetic they decide on and the guarantee that they never touch a run
they have no business touching.
"""

from __future__ import annotations

import pytest

pytest.importorskip("repop")

import torch  # noqa: E402

from pretrain.cli.audit_replay import (  # noqa: E402
    _cuda_free_bytes,
    _loss_transient_bytes,
    _plan_grad_offload,
    _plan_master_offload,
    _plan_optimizer_offload,
)

CPU = torch.device("cpu")


class _Cfg:
    """Only the fields the planners read."""

    class train:
        micro_batch_size = 4
        seq_len = 4096

    class model:
        vocab_size = 128256

    class run:
        mixed_precision = True

    class optim:
        name = "adamw_repop"


def test_loss_transient_is_both_matrices_at_the_output_dtype():
    """logits + grad-logits at [micro_batch * seq_len, vocab].

    This is the term every planner shares, and understating it is what lets a
    card that cannot fit past the check. 16384 x 128256 x 2 bytes each.
    """
    assert _loss_transient_bytes(_Cfg) == 2 * 16384 * 128256 * 2
    assert _loss_transient_bytes(_Cfg) / 1024**3 == pytest.approx(7.83, abs=0.01)


def test_loss_transient_doubles_without_mixed_precision():
    class Fp32(_Cfg):
        class run:
            mixed_precision = False

    assert _loss_transient_bytes(Fp32) == 2 * _loss_transient_bytes(_Cfg)


def test_free_bytes_is_none_off_cuda():
    assert _cuda_free_bytes(CPU) is None


@pytest.mark.parametrize(
    "plan, kwargs",
    [
        (_plan_optimizer_offload, dict(from_init=False, world_size=48)),
        (_plan_master_offload, dict(has_grad_model=True, n_params=1_610_000_000)),
        (_plan_grad_offload, dict(has_grad_model=True, n_params=1_610_000_000)),
    ],
)
def test_planners_never_fire_off_cuda(plan, kwargs):
    """A CPU or MPS replay must take exactly the path it took before."""
    args = (CPU, _Cfg) if plan is not _plan_optimizer_offload else (CPU, None, _Cfg)
    assert plan(*args, requested=False, **kwargs) is False


@pytest.mark.parametrize(
    "plan, kwargs",
    [
        (_plan_optimizer_offload, dict(from_init=False, world_size=48)),
        (_plan_master_offload, dict(has_grad_model=True, n_params=1_610_000_000)),
        (_plan_grad_offload, dict(has_grad_model=True, n_params=1_610_000_000)),
    ],
)
def test_planners_never_disable_an_explicit_flag(plan, kwargs):
    """Auto only ever escalates. Passing a flag must survive any measurement."""
    args = (CPU, _Cfg) if plan is not _plan_optimizer_offload else (CPU, None, _Cfg)
    assert plan(*args, requested=True, **kwargs) is True


@pytest.mark.parametrize("plan", [_plan_master_offload, _plan_grad_offload])
def test_no_grad_model_means_nothing_to_offload(plan):
    """Without a separate grad model the master IS the compute model, and there
    is no accumulation window to drain."""
    assert plan(
        CPU, _Cfg, requested=False, has_grad_model=False, n_params=1_610_000_000
    ) is False
