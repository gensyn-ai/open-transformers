"""RMSNorm using the repop reproducible runtime.

Subclasses the torch RMSNorm so ``init._set_norm_gains_to_one``'s
``isinstance(module, RMSNorm)`` check still matches and writes ones into
``self.weight``. Storage layout is identical to the torch version
(``self.weight = nn.Parameter(torch.ones(dim))``) so checkpoints, FSDP
sharding, and named_parameters iteration are unchanged.
"""

from __future__ import annotations

import torch

from pretrain.model.modules.norm import RMSNorm
from pretrain.model.registry import NORM
from repop.nn.rmsnorm import rms_norm as _rms_norm


@NORM.register("rmsnorm_repop")
class RMSNormRepop(RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)
