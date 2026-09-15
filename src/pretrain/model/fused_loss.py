"""Fused cross-entropy + z-loss in one pass — memory-frugal and BFR.

Computing CE and z-loss separately materialises the ``[N, V]`` softmax (and a
``[N, V]`` gradient) **twice** — once for CE, once for z-loss. On the 1.6B
lm_head that doubled the grad-logits (~8 GB each) and starved NCCL's comm-init
buffers into a silent first-step hang (the phase2-zloss run).

This fuses them around a single logsumexp:
  * z-loss is just ``coeff * mean(lse**2)`` — a scalar; and CE is
    ``mean(lse - label_logit)`` — another scalar. Both reduce the SAME
    per-row ``lse`` / softmax, so the forward holds only ``lse`` ``[N]`` and
    streams the ``[N, V]`` in row-chunks (one chunk resident at a time).
  * the backward recomputes the softmax in row-chunks and writes ONE combined
    grad-logits (CE + z-loss) — never two ``[N, V]`` matrices, and nothing
    ``[N, V]`` is held between forward and backward.

BFR: built from repop's bitwise-reproducible kernels (``exp``/``sum_dim``/
``log``/``mean``/``pow``); the per-row ``1/N`` is applied as a HOST reciprocal
multiply (never ``tensor / scalar`` — that is reciprocal-multiply on CUDA but
true-division on CPU/MPS, a 1-ULP cross-device drift), and ``softmax = exp/Z``
is tensor/tensor. So the gradient is byte-portable CUDA<->CPU<->MPS.

Gradient (per row n, vocab v; N = B*T rows; m_n = rowmax):
    softmax[n,v] = exp(logit[n,v] - m_n) / Z_n
    d CE     / d logit[n,v] = (1/N) (softmax[n,v] - onehot[n,v])
    d z_loss / d logit[n,v] = (2*coeff/N) * lse_n * softmax[n,v]
The ``- m`` shift drops out of the gradient (a row-wise constant).
"""

from __future__ import annotations

import torch

from pretrain.model.z_loss import (
    Z_LOSS_GRAD_CHUNK_ROWS,
    _z_loss_lse,
)


class _FusedCEZLoss(torch.autograd.Function):
    """``(ce, z_loss)`` from one shared softmax; chunked forward + backward."""

    @staticmethod
    def forward(ctx, logits, labels, coeff, chunk_rows):  # type: ignore[override]
        from repop import ops as repop_ops

        flat = logits.reshape(-1, logits.shape[-1])
        N = flat.shape[0]
        lse = torch.empty(N, dtype=torch.float32, device=logits.device)
        label_logit = torch.empty(N, dtype=torch.float32, device=logits.device)
        for s in range(0, N, chunk_rows):
            sl = slice(s, s + chunk_rows)
            chunk_f = flat[sl].float()
            _, _, lse_c = _z_loss_lse(chunk_f)
            lse[sl] = lse_c
            label_logit[sl] = chunk_f.gather(
                dim=-1, index=labels[sl].unsqueeze(-1)
            ).squeeze(-1)
        ce = repop_ops.mean(lse - label_logit)
        if coeff != 0.0:
            z_loss = coeff * repop_ops.mean(repop_ops.pow(lse, 2.0))
        else:
            z_loss = torch.zeros((), device=lse.device, dtype=lse.dtype)

        # Save only the inputs + the small per-row lse; recompute the softmax in
        # the backward so the [N, V] is never held across the step.
        ctx.save_for_backward(logits, labels, lse)
        ctx.coeff = coeff
        ctx.chunk_rows = chunk_rows
        return ce, z_loss

    @staticmethod
    def backward(ctx, grad_ce, grad_z):  # type: ignore[override]
        logits, labels, lse = ctx.saved_tensors
        flat = logits.reshape(-1, logits.shape[-1])
        N = flat.shape[0]
        inv_N = 1.0 / N  # host reciprocal — BFR (no tensor/scalar divide)
        coeff = ctx.coeff
        cr = ctx.chunk_rows
        # Accumulate straight into the OUTPUT dtype rather than fp32-then-cast.
        # Each chunk's ``g`` is still computed in fp32; only the store narrows,
        # and a float->bf16 cast is elementwise round-to-nearest-even, so doing
        # it per chunk is bit-identical to casting the whole [N, V] at the end.
        # What it removes is the fp32 buffer: at [16384, 128256] that is 8.4 GB
        # held across the entire backward, plus the 4.2 GB transient where the
        # fp32 original and its bf16 copy were both live. For an fp32 run this
        # is a no-op (the dtypes coincide).
        grad_flat = torch.empty_like(flat)
        ce_scale = grad_ce * inv_N  # 0-dim
        for s in range(0, N, cr):
            sl = slice(s, s + cr)
            chunk_f = flat[sl].float()
            exp_shifted, Z, _ = _z_loss_lse(chunk_f)
            softmax = exp_shifted / Z.unsqueeze(-1)
            g = softmax * ce_scale  # CE: softmax / N
            if coeff != 0.0:
                coef = (2.0 * coeff * inv_N) * grad_z * lse[sl]  # [chunk]
                g = g + softmax * coef.unsqueeze(-1)
            # CE one-hot: subtract grad_ce/N at each row's label column.
            rows = g.shape[0]
            sub = ce_scale.expand(rows).unsqueeze(-1).to(g.dtype)  # [chunk, 1]
            g.scatter_add_(dim=-1, index=labels[sl].unsqueeze(-1), src=-sub)
            grad_flat[sl] = g  # narrows to grad_flat's dtype on store
        return grad_flat.reshape_as(logits), None, None, None


def fused_ce_z_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coeff: float,
    *,
    chunk_rows: int = Z_LOSS_GRAD_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``(ce, z_loss)`` over ``[..., V]`` logits + matching ``labels``.

    ``coeff=0`` disables z-loss (``z_loss == 0``, no z-loss gradient)."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    return _FusedCEZLoss.apply(flat_logits, flat_labels, coeff, chunk_rows)
