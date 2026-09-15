"""Deterministic global gradient clipping.

STATELESS by construction: nothing persists, nothing is checkpointed or
hashed, nothing can absorb or grind subnormal — stateful per-tensor clippers
have failure modes in all three (absorbing freezes, regime-change crawl,
EMA state grinding into fp32 subnormals outside the repop equivalence
boundary). The coefficient is bounded in ~[0.01, 1] so products preserve
magnitude, and its exact shape has been bitwise-validated cluster↔CPU↔MPS
across audited runs:

    h = clamp( full_like(denom, max_norm) / (‖g‖_2 + eps), max=1 );  ĝ = h·g

BFR notes:
  * the coefficient is a *tensor ÷ tensor* divide — a python-scalar numerator
    stays double-precision and is NOT guaranteed byte-identical across
    backends (verified: ``lam/t != full_like(lam)/t``);
  * the norm comes from the deterministic ascending-shard fold
    (pretrain.parallel.deterministic_reduce), identical cluster↔audit;
  * ``mul_`` by the same scalar is byte-equal on a rank-local shard (cluster)
    and the reconstructed full tensor (audit).

Subnormal posture: the coefficient can't be subnormal (max_norm/(‖g‖+1e-6)
with ‖g‖ bounded by the spike protocol), and fp32 norms can't be nonzero
subnormals (sqrt of the smallest positive fp32 ≈ 2.6e-23). See the
subnormal-domain regression in tests/test_global_clip.py.
"""

from __future__ import annotations

import torch

from pretrain.parallel.deterministic_reduce import (
    audit_per_tensor_and_global_norm,
    deterministic_per_tensor_and_global_norm,
)

# Denominator guard — torch's clip_grad_norm_ convention, applied identically
# on cluster and audit.
_EPS = 1e-6


def _to_local(t: torch.Tensor) -> torch.Tensor:
    """Rank-local shard of a DTensor (shares storage); pass-through otherwise."""
    return t.to_local() if hasattr(t, "to_local") else t


def _apply(model: torch.nn.Module, max_norm: float, global_norm: torch.Tensor) -> None:
    denom = global_norm + _EPS
    h = torch.clamp(torch.full_like(denom, max_norm).div(denom), max=1.0)
    with torch.no_grad():
        for _name, p in model.named_parameters():
            if p.grad is None:
                continue
            local = _to_local(p.grad)
            local.mul_(h.to(local.dtype))


def clip_train(
    model: torch.nn.Module, max_norm: float, device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Cluster path: deterministic fold → per-tensor + global norms, clip in
    place, return (per_tensor_norms, pre_clip_global_norm). The per-tensor
    norms are PRE-clip and come for free from the fold's single all_gather —
    the per-param grad-norm telemetry consumes them instead of issuing ~one
    tiny collective per parameter per step (the storm that dominated
    phase_clip_ms)."""
    norms, global_norm = deterministic_per_tensor_and_global_norm(model, device)
    _apply(model, max_norm, global_norm)
    return norms, global_norm


def clip_audit(
    model: torch.nn.Module, max_norm: float, dp_shard: int, device: torch.device
) -> torch.Tensor:
    """Single-device audit path: re-slice the reconstructed full gradient into
    the same shards, fold locally, then the SAME clip as the cluster."""
    _norms, global_norm = audit_per_tensor_and_global_norm(model, dp_shard, device)
    _apply(model, max_norm, global_norm)
    return global_norm
