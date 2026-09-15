"""Mixed-precision policy.

A single helper produces both the FSDP2 ``MixedPrecisionPolicy`` and the
per-module precision overrides. Other code asks the helper rather than
hardcoding dtypes — see plan/04 §5.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrecisionPolicy:
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype = torch.bfloat16
    norm_compute_dtype: torch.dtype = torch.float32
    loss_compute_dtype: torch.dtype = torch.float32


class MixedPrecisionPolicyFactory:
    """Constructs FSDP2's ``MixedPrecisionPolicy`` if available; otherwise
    returns ``None`` (used in tests on CPU where FSDP2 isn't applied).
    """

    @staticmethod
    def build_fsdp_policy(policy: PrecisionPolicy):
        try:
            # FSDP2 (PyTorch >= 2.4 release-quality)
            from torch.distributed.fsdp import MixedPrecisionPolicy
        except ImportError:
            try:
                # NGC pytorch:25.01 ships a pre-release torch 2.6 snapshotted
                # before MixedPrecisionPolicy's public re-export landed; the
                # class still exists at its old _composable home. Drop this
                # branch once we're back on a container with a release torch
                # 2.6+ (parallels the fully_shard fallback in
                # parallelize_llama3_repop.py).
                from torch.distributed._composable.fsdp import MixedPrecisionPolicy
            except ImportError:
                return None
        return MixedPrecisionPolicy(
            param_dtype=policy.param_dtype,
            reduce_dtype=policy.reduce_dtype,
            output_dtype=policy.output_dtype,
        )

    @staticmethod
    def autocast_dtype(policy: PrecisionPolicy) -> torch.dtype:
        return policy.param_dtype


DEFAULT_POLICY = PrecisionPolicy()
FP32_POLICY = PrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    output_dtype=torch.float32,
)
