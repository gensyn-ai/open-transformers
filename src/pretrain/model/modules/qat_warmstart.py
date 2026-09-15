"""bf16 warm-start for LSQ QAT — runtime quantization toggle, trainer-side.

Motivation (20260703 post-mortem): from-scratch int8 QAT quantizes through the
most fragile phase of training — warmup, where weights are tiny and STE
gradients are computed against near-pure quantization noise (repop's own
scale-refresh docs record a grad-norm detonation at ~step 2k from exactly
this), and where the suspected init-time quantization pathology produced the
72 exactly-zero-gradient tensors that then sat frozen for 50k steps. Running
warmup in bf16 and enabling quantization at ``qat.enable_at_step`` removes the
int8 noise from that phase entirely.

Deliberately implemented WITHOUT modifying repop:
``WarmstartLSQLinear`` subclasses repop's ``LSQQuantizedLinear`` (so
``refresh_lsq_weight_scales``, FSDP wrapping, and TP isinstance checks all
keep working) and only overrides ``forward``:

  * inactive → ``repop.nn.linear.linear(input, weight, bias)`` — the exact
    kernel a non-QAT repop ``Linear`` dispatches, bitwise-identical to having
    built a plain linear with these weights;
  * active   → ``super().forward()`` — unchanged LSQ QAT.

No structural difference from the plain LSQ layer: same parameters (the
weight_scale stays registered and the trainer's every-step refresh keeps it
pinned to the current weights, so the FIRST quantized step already uses an
optimal grid — no special at-flip logic), same state_dict/DCP layout, and the
toggle is a plain attribute that never enters the checkpoint. It is a pure
function of (step, config): the loop AND the audit call
:func:`set_qat_active` at the top of every step (idempotent, state-free —
resume/replay-safe by construction), so the flip is one segment, not a
descriptor boundary.
"""

from __future__ import annotations

import torch
from repop.nn.linear import linear as _repop_linear
from repop.qat.lsq import LSQQuantizedLinear


class WarmstartLSQLinear(LSQQuantizedLinear):
    qat_active: bool = True  # runtime toggle; see module docstring

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if not self.qat_active:
            return _repop_linear(input, self.weight, self.bias)
        return super().forward(input)


def set_qat_active(module: torch.nn.Module, active: bool) -> int:
    """Set the runtime QAT toggle on every :class:`WarmstartLSQLinear` under
    ``module``. Returns the number of layers touched. Call at the top of every
    step with ``active = step >= cfg.model.qat.enable_at_step`` — in the
    training loop and in audit_replay, identically."""
    n = 0
    for m in module.modules():
        if isinstance(m, WarmstartLSQLinear):
            m.qat_active = active
            n += 1
    return n
