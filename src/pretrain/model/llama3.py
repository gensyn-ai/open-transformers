"""Llama 3 transformer (forked-in-spirit from torchtitan), parametrised by
the registry pattern so attention / FFN / norm / RoPE can be swapped via
config.

The model is parallelism-agnostic. TP/FSDP2 wrapping happens externally in
``pretrain.parallel.parallelize_llama3_repop``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from pretrain.config.schema import ModelConfig
from pretrain.model.registry import ATTENTION, EMBEDDING, FFN, NORM, ROPE


@dataclass
class ModelOutput:
    logits: torch.Tensor          # [B, T, vocab_size]
    z_loss: torch.Tensor | None   # scalar; None if z-loss disabled


class TransformerBlock(nn.Module):
    """Pre-norm block: ``x = x + attn(norm1(x)); x = x + ffn(norm2(x))``.

    The submodules are resolved from registries at construction time.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        attn_cls = ATTENTION.get(cfg.modules.attention)
        ffn_cls = FFN.get(cfg.modules.ffn)
        norm_cls = NORM.get(cfg.modules.norm)

        # Per-layer QAT exemption: keep the first N blocks' linears in bf16
        # (the early-layer STE blow-up locus). See ModelConfig.qat.exempt_first_n_blocks.
        block_qat = cfg.qat
        if cfg.qat.enabled and layer_idx < cfg.qat.exempt_first_n_blocks:
            block_qat = cfg.qat.model_copy(update={"enabled": False})

        self.norm1 = norm_cls(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = attn_cls(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            layer_idx=layer_idx,
            n_layers=cfg.n_layers,
            swa_window=cfg.swa_window,
            swa_full_every=cfg.swa_full_every,
            attn_int8_pv=cfg.attn_int8_pv,
            qat=block_qat,
            qk_norm_gain=cfg.qk_norm_gain,
        )
        self.norm2 = norm_cls(cfg.d_model, eps=cfg.rms_norm_eps)
        self.ffn = ffn_cls(
            d_model=cfg.d_model,
            intermediate=cfg.ffn_intermediate,
            qat=block_qat,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x


class Llama3(nn.Module):
    """The model. ``forward()`` returns logits + optional z-loss scalar.

    The cross-entropy loss is computed in the training loop, not here, so
    this class stays loss-agnostic and easy to use for eval and inference.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        emb_cls = EMBEDDING.get(cfg.modules.embedding)
        rope_cls = ROPE.get(cfg.modules.rope)
        norm_cls = NORM.get(cfg.modules.norm)

        self.embedding = emb_cls(cfg.vocab_size, cfg.d_model)
        # Optional embedding-output norm (LSQ-QAT grad-blowup fix). Bounds the
        # residual magnitude entering block 0 for the whole run. See
        # ModelConfig.emb_norm.
        self.emb_norm = (
            norm_cls(cfg.d_model, eps=cfg.rms_norm_eps) if cfg.emb_norm else None
        )
        self.rope = rope_cls(
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len_pretrain,
            theta=cfg.rope_theta,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)]
        )
        self.norm_out = norm_cls(cfg.d_model, eps=cfg.rms_norm_eps)

    @property
    def lm_head(self) -> nn.Module:
        # Convenience accessor; some external code (eval, checkpoint
        # introspection) wants direct access to the head.
        return self.embedding.output if hasattr(self.embedding, "output") else self.embedding

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the normalized hidden states before vocabulary projection."""
        B, T = input_ids.shape
        cos, sin = self.rope(T)

        h = self.embedding.encode(input_ids)
        if self.emb_norm is not None:
            h = self.emb_norm(h)
        for block in self.blocks:
            h = block(h, cos, sin)
        return self.norm_out(h)

    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        h = self.forward_hidden(input_ids)
        logits = self.embedding.project(h)

        # z-loss is NOT computed here. CE and z-loss share the same softmax over
        # the vocab, so they are fused into one pass over the logits in the
        # training loop / audit (pretrain.model.fused_loss.fused_ce_z_loss) —
        # computing z-loss here too would materialise a second [N, V] softmax +
        # gradient at the lm_head (the phase2-zloss first-step hang). The loss
        # also needs ``labels``, which the model forward doesn't have.
        return ModelOutput(logits=logits, z_loss=None)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig, device: str | torch.device | None = None) -> Llama3:
    """Construct a Llama3 instance, optionally on a specific device.

    Init logic lives in ``pretrain.model.init.init_weights`` and is invoked
    by the train loop after construction (because we want to materialise
    on meta-device for 8B and apply init only after that's safe).
    """
    model = Llama3(cfg)
    if device is not None:
        model = model.to(device)
    return model
