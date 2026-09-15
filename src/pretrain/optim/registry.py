"""Optimizer registry. ``adamw`` is the only optimizer wired in by default;
``muon_hybrid`` is gated by config (ADR-004).
"""

from __future__ import annotations

import torch.nn as nn

from pretrain.config.schema import OptimConfig
from pretrain.optim.adamw import build_adamw


def build_optimizer(model: nn.Module, cfg: OptimConfig):
    if cfg.name == "adamw":
        return build_adamw(model, cfg)
    if cfg.name == "adamw_repop":
        # Lazy import — only the repop runs need the repop wheel built.
        from pretrain.optim.adamw_repop import build_adamw_repop

        return build_adamw_repop(model, cfg)
    if cfg.name == "muon_hybrid":
        # Lazy import — only loads if config selects it. Keeps the
        # default path free of any Muon-related dependency surface.
        from pretrain.optim.muon import build_muon_hybrid

        return build_muon_hybrid(model, cfg)
    raise ValueError(f"unknown optimizer: {cfg.name}")
