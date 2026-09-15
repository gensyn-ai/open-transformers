"""The three registries that gate model swappability (plan/04 §7)."""

from __future__ import annotations

import torch.nn as nn

from pretrain.util.registry import Registry

ATTENTION: Registry[nn.Module] = Registry("attention")
FFN: Registry[nn.Module] = Registry("ffn")
NORM: Registry[nn.Module] = Registry("norm")
ROPE: Registry[nn.Module] = Registry("rope")
EMBEDDING: Registry[nn.Module] = Registry("embedding")
