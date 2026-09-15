"""AdamW optimizer with parameter-group splitting (decay vs no-decay).

Why two groups: weight decay on biases / norm gains is empirically harmful
at scale (Llama 3 / Chinchilla / GPT-NeoX). The matcher pattern lives in
config so it's auditable.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn as nn

from pretrain.config.schema import OptimConfig

LOG = logging.getLogger(__name__)


def build_param_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    no_decay_substrings: list[str],
    weight_decay: float,
) -> list[dict]:
    """Split parameters by name into decay / no-decay groups.

    A param is no-decay if its name contains *any* of the substrings in
    ``no_decay_substrings``, OR if the param is 1-D (biases / norm
    gains / RoPE buffers don't get decayed regardless).
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    no_decay_names: list[str] = []

    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        is_no_decay = (
            p.dim() < 2
            or any(s in name for s in no_decay_substrings)
        )
        if is_no_decay:
            no_decay.append(p)
            no_decay_names.append(name)
        else:
            decay.append(p)

    LOG.info(
        "param groups: decay=%d (%.1fM), no_decay=%d (%.1fM); first 5 no_decay: %s",
        len(decay),
        sum(p.numel() for p in decay) / 1e6,
        len(no_decay),
        sum(p.numel() for p in no_decay) / 1e6,
        no_decay_names[:5],
    )

    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_adamw(
    model: nn.Module,
    cfg: OptimConfig,
) -> torch.optim.AdamW:
    """Construct repop's bitwise-reproducible AdamW with research-default
    hyperparameters and the param-group split.

    The update runs through ``repop.ops.adamw_kernel_step`` so the step is
    byte-identical CPU<->CUDA<->MPS — torch.optim.AdamW's native moment math
    (sqrt/div) diverges cross-device and breaks the cross-arch audit. Surface
    matches torch.optim.AdamW (decoupled decay); ``fused`` does not apply (the
    repop kernel is the update path on every backend).
    """
    from repop.optim import AdamW as RepopAdamW

    groups = build_param_groups(
        model.named_parameters(),
        no_decay_substrings=cfg.no_decay_param_names,
        weight_decay=cfg.weight_decay,
    )
    return RepopAdamW(
        groups,
        lr=cfg.peak_lr,                    # the schedule rewrites this each step
        betas=cfg.betas,
        eps=cfg.eps,
    )
