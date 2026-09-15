"""RMSNorm base class.

Kept as the base for ``RMSNormRepop`` so that
``init._set_norm_gains_to_one``'s ``isinstance`` check matches. The
``rmsnorm_repop`` registry entry is provided by ``norm_repop`` — this
module does not register anything.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True
    ) -> None:
        super().__init__()
        self.eps = eps
        # Gain-free mode (elementwise_affine=False): no learnable γ. Adopted
        # for QK-norm after the 20260703 incidents — the γq⊙γk product is a
        # learned attention temperature with no restoring force, and its
        # unbounded growth drove an entropy collapse. weight=None keeps
        # named_parameters/optimizer/DCP layouts clean (no frozen tensor to
        # carry) and repop's rms_norm accepts weight=None natively.
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xf = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = (xf * rms).to(in_dtype)
        return out if self.weight is None else out * self.weight
