"""Linear-layer factory shared by the repop attention / FFN modules.

Returns repop's reproducible ``Linear`` in full precision, or — when int8
QAT is enabled — one of two BFR int8 QAT layers per ``qat.method``:
  - ``"lsq"`` (default): ``repop.qat.LSQQuantizedLinear`` (learned per-channel
    step size; built for from-scratch QAT; the bench's BFR reference path).
  - ``"absmax"``: ``repop.qat.QuantizedLinear`` (fixed absmax scale).
All three are ``nn.Module``s exposing a ``.weight`` Parameter, so weight
init (``model.init``) and the optimizer param-group split are unchanged by
the choice. (LSQ adds a learnable ``weight_scale`` Parameter, picked up by
the optimizer automatically.)

Parallelism: LSQ runs under pure FSDP2 / HSDP (``fully_shard`` shards weight
and the ``[out]`` ``weight_scale`` on the same dim-0 axis; FSDP all-gathers
both before forward, and the scale gradient reduces over the un-sharded K
axis). BFR byte-equality then holds at a *fixed* ``(world_size, mesh, NCCL
config)`` rather than across device counts. QAT under tensor parallel is still
unsupported (the ``local_map`` wrapper bypasses the quant forward) and absmax
``QuantizedLinear`` stays single-device — see the guard in
``pretrain.parallel.parallelize_llama3_repop``.
"""

from __future__ import annotations

import torch.nn as nn

from pretrain.config.schema import QATConfig
from repop.nn.linear import Linear as RepopLinear


def build_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    qat: QATConfig,
) -> nn.Module:
    if qat.enabled:
        # Lazy import: only the QAT runs pull in the repop.qat surface.
        if qat.method == "lsq":
            # Trainer-side subclass of repop's LSQQuantizedLinear adding the
            # bf16 warm-start toggle (qat.enable_at_step) — structurally and
            # checkpoint-identical to the raw LSQ layer; see qat_warmstart.
            from pretrain.model.modules.qat_warmstart import WarmstartLSQLinear

            # LSQ is inherently per-channel (learnable [out_features] scale);
            # per_channel_w does not apply.
            return WarmstartLSQLinear(
                in_features,
                out_features,
                bias=bias,
                weight_bits=qat.weight_bits,
                act_bits=qat.act_bits,
            )
        from repop.qat import QuantizedLinear

        return QuantizedLinear(
            in_features,
            out_features,
            bias=bias,
            weight_bits=qat.weight_bits,
            act_bits=qat.act_bits,
            per_channel_w=qat.per_channel_w,
        )
    return RepopLinear(in_features, out_features, bias=bias)
