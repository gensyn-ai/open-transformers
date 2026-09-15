"""GQA (with optional QK-Norm) using the repop runtime.

Q/K/V projections + the output projection use the repop ``Linear`` (or
``repop.qat.QuantizedLinear`` when int8 QAT is enabled — see
``pretrain.model.modules._linear``); the RMSNorm on Q and K uses
``RMSNormRepop``; the attention itself uses
``repop.nn.flash_attention.causal_flash_attention``, the GQA-native fused
flash kernel that also takes a sliding ``window`` argument.

Hybrid sliding-window: 4 of every ``swa_full_every`` layers run with a
``swa_window``-key sliding window and the last in each group runs full
causal (4:1 at swa_full_every=5). The window is chosen once per layer from
``layer_idx`` (see ``__init__``). The flash kernel is GQA-native, so K/V are
NOT pre-expanded — they enter at ``n_kv_heads`` and the kernel handles the
head grouping internally (no extra HBM traffic).

The flash path bakes in a ``1/sqrt(head_dim)`` softmax scale internally and
honors ``window`` regardless of ``REPOP_EXECUTION_MODE`` (it does not route
through cuDNN/SDPA on this runtime).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pretrain.config.schema import QATConfig
from pretrain.model.modules._linear import build_linear
from pretrain.model.modules.norm_repop import RMSNormRepop
from pretrain.model.modules.rope import apply_rope
from pretrain.model.registry import ATTENTION
from repop.nn.flash_attention import (
    causal_flash_attention,
    int8pv_causal_flash_attention,
)


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    int8_pv: bool,
) -> torch.Tensor:
    """GQA-native, windowed, BFR flash attention. Routes to the int8-PV
    kernel when ``int8_pv`` is set, else the bf16 FMA flash. q/k/v are
    [B, H, T, D] with K/V at the (local) KV-head count; both kernels apply
    the 1/sqrt(head_dim) softmax scale internally. Shared by the non-TP and
    TP forward paths so they stay identical.
    """
    if int8_pv:
        return int8pv_causal_flash_attention(q, k, v, window=window)
    return causal_flash_attention(q, k, v, dropout=0.0, window=window)


class _GQARepopBase(nn.Module):
    qk_norm: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-5,
        layer_idx: int = 0,
        n_layers: int = 0,
        swa_window: int = 0,
        swa_full_every: int = 1,
        attn_int8_pv: bool = False,
        qat: QATConfig | None = None,
        qk_norm_gain: bool = True,
    ) -> None:
        super().__init__()
        if d_model != n_heads * head_dim:
            raise ValueError("d_model != n_heads * head_dim")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        # int8-PV flash attention vs the bf16 FMA flash. Both are GQA-native,
        # honor ``window``, and are BFR; int8-PV moves the P@V onto int8
        # tensor cores. See ModelConfig.attn_int8_pv.
        self.attn_int8_pv = attn_int8_pv
        # Sliding-window length in keys for this layer. The last layer in each
        # group of ``swa_full_every`` is full causal (window=0); the rest use
        # the sliding window. swa_full_every=1 => every layer full causal.
        # The final layer of the model is always full causal regardless of the
        # group pattern (n_layers need not divide swa_full_every), matching the
        # Llama-3/Gemma convention of a global-attention output layer.
        is_full = (layer_idx % swa_full_every) == (swa_full_every - 1)
        is_last = n_layers > 0 and layer_idx == n_layers - 1
        self.window = 0 if (is_full or is_last) else swa_window
        qat = qat or QATConfig()
        # Separate Q / K / V projections rather than a fused wqkv. The
        # fused form is one GEMM but its output rows aren't laid out so
        # that a colwise-TP shard puts whole heads on each rank; three
        # Linears compose cleanly with ColwiseParallel-equivalent TP at
        # the cost of two extra kernel launches per block.
        self.wq = build_linear(d_model, n_heads * head_dim, bias=False, qat=qat)
        self.wk = build_linear(d_model, n_kv_heads * head_dim, bias=False, qat=qat)
        self.wv = build_linear(d_model, n_kv_heads * head_dim, bias=False, qat=qat)
        self.wo = build_linear(d_model, d_model, bias=False, qat=qat)
        if self.qk_norm:
            # qk_norm_gain=False is the gain-free (Gemma-3-style) variant:
            # unit-RMS q/k with NO learnable per-dim γ, so attention logits
            # are architecturally bounded — the entropy-collapse axis the
            # 20260703 runs failed on does not exist. Logit softcapping is
            # the pre-identified fallback if attention proves too soft.
            self.q_norm = RMSNormRepop(
                head_dim, eps=rms_norm_eps, elementwise_affine=qk_norm_gain
            )
            self.k_norm = RMSNormRepop(
                head_dim, eps=rms_norm_eps, elementwise_affine=qk_norm_gain
            )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        if self.qk_norm:
            # Flatten to 2D rather than 3D: repop's 3D rms_norm dispatches to
            # sum3d_dim2 with grid `(N, T)`, and grid.y caps at 65535. With
            # B=4, T=4096, Hq=16 the head-flattened length is 65536 — one
            # past the limit, surfacing as cudaErrorInvalidValue. The 2D path
            # (sum2d_dim1) uses grid.x only, which is effectively unbounded.
            D = self.head_dim
            q = self.q_norm(q.reshape(B * T * self.n_heads, D)).reshape(
                B, T, self.n_heads, D
            )
            k = self.k_norm(k.reshape(B * T * self.n_kv_heads, D)).reshape(
                B, T, self.n_kv_heads, D
            )
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # causal_flash_attention is GQA-native: K/V stay at n_kv_heads and the
        # kernel groups heads internally. It expects [B, H, T, D].
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Softmax scale (1/sqrt(head_dim)) is applied inside the kernel.
        # window=0 => full causal; window>0 => sliding window in keys.
        out = _attend(q, k, v, self.window, self.attn_int8_pv)
        out = out.transpose(1, 2).contiguous().view(*x.shape)
        return self.wo(out)


@ATTENTION.register("gqa_qknorm_repop")
class GQAQKNormRepop(_GQARepopBase):
    qk_norm = True


@ATTENTION.register("gqa_plain_repop")
class GQAPlainRepop(_GQARepopBase):
    qk_norm = False
