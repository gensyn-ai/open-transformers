"""Rotary Position Embeddings.

Standard RoPE with θ=500_000 (Llama 3 / 3.1). Tables are pre-computed for
``max_seq_len`` and indexed at runtime — no per-call sin/cos compute.

YaRN extension is registered as a separate variant so the long-context
anneal (M6) can swap to it via config without touching the base path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import repop.ops as repop_ops

from pretrain.model.registry import ROPE


def _build_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def _build_tables(max_seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = _build_inv_freq(head_dim, theta)
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)            # [seq, head_dim/2]
    # repop cos/sin: torch's libm cos/sin diverge across arch/compiler; the
    # int8 forward masks the sub-ULP delta but the fp32 rope backward exposes it.
    return repop_ops.cos(freqs), repop_ops.sin(freqs)  # both [seq, head_dim/2]


@ROPE.register("rope_default")
class RoPE(nn.Module):
    """Pre-computed cos/sin tables for max_seq_len. Apply by indexing
    a leading ``[:T]`` slice and broadcasting across heads.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 500_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        cos, sin = _build_tables(max_seq_len, head_dim, theta)
        # Buffers so they ride DCP checkpoints and live on the right device.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"RoPE table built for {self.max_seq_len} but got seq_len={seq_len}"
            )
        return self.rope_cos[:seq_len], self.rope_sin[:seq_len]


@ROPE.register("rope_yarn")
class RoPEYarn(RoPE):
    """YaRN-scaled RoPE for long-context anneal. The scaling formulas
    follow `Peng et al. (2023) — YaRN`. Concretely we apply
    NTK-by-parts to the inverse frequencies and a small length-correction
    factor to the magnitude of the embeddings.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 500_000.0,
        original_max_seq_len: int = 8192,
        scale: float = 4.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> None:
        nn.Module.__init__(self)
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")

        inv_freq = _build_inv_freq(head_dim, theta)
        # NTK-by-parts: smoothly blend high-freq (kept) vs low-freq (scaled).
        wavelen = 2.0 * math.pi / inv_freq
        ratio = original_max_seq_len / wavelen   # frequency in cycles per training-context
        smooth = ((ratio - beta_slow) / (beta_fast - beta_slow)).clamp(0.0, 1.0)
        inv_freq = (1.0 - smooth) * inv_freq / scale + smooth * inv_freq

        pos = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        # YaRN length-correction (small): m_scale ≈ 0.1 * ln(scale) + 1
        m_scale = 0.1 * math.log(scale) + 1.0
        cos = repop_ops.cos(freqs) * m_scale
        sin = repop_ops.sin(freqs) * m_scale

        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding.

    ``x``: shape [B, T, H, head_dim] (or [..., T, H, head_dim] generally).
    ``cos``/``sin``: shape [T, head_dim/2] — broadcast across batch and head.
    """
    # Split last dim into pairs.
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # x1 / x2 are [..., T, H, head_dim/2]. We need cos/sin shaped as
    # [..., T, 1, head_dim/2] so they broadcast across heads.
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    # Add a leading batch dim if x has one.
    while cos.dim() < x1.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out
