"""SwiGLU built end-to-end from repop primitives.

Gate and up projections are fused into a single ``w_gate_up`` GEMM
(2× intermediate width), then chunked. One kernel launch and one HBM
read of ``x`` instead of two — equivalent math, fewer round trips.

SiLU routes through ``repop.nn.activations.silu`` (autograd-backed
C++/CUDA kernel using repop's ``correct_rounded_exp``) so the activation
and its gradient stay bitwise reproducible across GPU architectures, not
just same-device deterministic. The chunked ``gate`` view is non-
contiguous; the repop kernel reads raw pointers, so we materialise with
``.contiguous()`` before the call.

DTensor handling: under TP, ``w_gate_up(x)`` returns ``DTensor[Shard(-1)]``
and chunk produces DTensor halves. SiLU is elementwise — unwrap → kernel →
rewrap is communication-free and preserves the placement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from pretrain.config.schema import QATConfig
from pretrain.model.modules._linear import build_linear
from pretrain.model.registry import FFN
from repop.nn.activations import silu as repop_silu


def _silu(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, DTensor):
        local = t.to_local().contiguous()
        out_local = repop_silu(local)
        return DTensor.from_local(out_local, t.device_mesh, t.placements)
    return repop_silu(t.contiguous())


@FFN.register("swiglu_repop")
class SwiGLURepop(nn.Module):
    def __init__(
        self,
        d_model: int,
        intermediate: int,
        qat: QATConfig | None = None,
    ) -> None:
        super().__init__()
        qat = qat or QATConfig()
        self.w_gate_up = build_linear(d_model, 2 * intermediate, bias=False, qat=qat)
        self.w_down = build_linear(intermediate, d_model, bias=False, qat=qat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(_silu(gate) * up)
