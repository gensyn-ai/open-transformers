"""FP32 vocabulary projection and loss without full logits or grad-logits."""

import logging

import torch
from repop import ops
from repop.nn.linear import linear

from pretrain.model.fused_loss import fused_ce_z_loss
from pretrain.model.z_loss import _z_loss_lse

LOG = logging.getLogger("pretrain.audit")


class _StreamingHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, coeff, chunk_rows):
        from repop.backend import cpu

        backend = cpu
        if hidden.is_mps:
            from repop.backend import metal

            backend = metal
        flat = hidden.reshape(-1, hidden.shape[-1])
        count = flat.shape[0]
        labels = labels.reshape(-1)
        lse = torch.empty(count, dtype=torch.float32, device=hidden.device)
        label_logits = torch.empty_like(lse)
        for start in range(0, count, chunk_rows):
            sl = slice(start, start + chunk_rows)
            logits = backend.mm(flat[sl], weight, False, True)
            _, _, row_lse = _z_loss_lse(logits)
            lse[sl] = row_lse
            label_logits[sl] = logits.gather(1, labels[sl, None]).squeeze(1)
        ce = ops.mean(lse - label_logits)
        zloss = (
            coeff * ops.mean(ops.pow(lse, 2.0))
            if coeff != 0.0
            else torch.zeros((), dtype=lse.dtype, device=lse.device)
        )
        ctx.save_for_backward(hidden, weight, labels, lse)
        ctx.coeff = coeff
        ctx.chunk_rows = chunk_rows
        return ce, zloss

    @staticmethod
    def backward(ctx, grad_ce, grad_z):
        from repop.backend import cpu

        hidden, weight, labels, lse = ctx.saved_tensors
        backend = cpu
        if hidden.is_mps:
            from repop.backend import metal

            backend = metal
        flat = hidden.reshape(-1, hidden.shape[-1])
        count = flat.shape[0]
        inv_count = 1.0 / count
        ce_scale = grad_ce * inv_count
        grad_hidden = torch.empty_like(flat)
        grad_weight = torch.zeros(weight.shape, dtype=weight.dtype, device=weight.device)
        for start in range(0, count, ctx.chunk_rows):
            sl = slice(start, start + ctx.chunk_rows)
            logits = backend.mm(flat[sl], weight, False, True)
            exp_shifted, denominator, _ = _z_loss_lse(logits)
            softmax = exp_shifted / denominator.unsqueeze(-1)
            grad = softmax * ce_scale
            if ctx.coeff != 0.0:
                coef = (2.0 * ctx.coeff * inv_count) * grad_z * lse[sl]
                grad = grad + softmax * coef.unsqueeze(-1)
            sub = ce_scale.expand(grad.shape[0]).unsqueeze(-1).to(grad.dtype)
            grad.scatter_add_(1, labels[sl, None], -sub)
            grad_hidden[sl] = backend.mm(grad, weight, False, False)
            # Continue the original row-ordered FMA chain, not a sum of GEMMs.
            backend.mm_accumulate(grad, flat[sl], grad_weight, True, False)
        return grad_hidden.reshape_as(hidden), grad_weight, None, None, None


def streaming_head_loss(hidden, weight, labels, coeff, *, chunk_rows=256):
    """Stream aligned FP32 CPU/MPS heads; preserve the ordinary path otherwise.

    Metal's ordinary matmul pads its K loop to 16. Aligned dimensions and panel
    boundaries avoid introducing intermediate padding into the streamed chain.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    count = hidden.numel() // hidden.shape[-1]
    basic_ok = (
        hidden.device.type in ("cpu", "mps")
        and hidden.dtype == weight.dtype == torch.float32
        and count > 0
    )
    aligned = not hidden.is_mps or all(
        value % 16 == 0
        for value in (count, hidden.shape[-1], weight.shape[0], chunk_rows)
    )
    if not (basic_ok and aligned):
        # Silent otherwise: the operator who set PRETRAIN_AUDIT_STREAM_HEAD=1
        # to avoid an OOM on the full [N, V] logits gets that same OOM back,
        # with nothing in the log to say the streamed path was ever skipped.
        reason = (
            "device/dtype/shape unsupported (need cpu or mps, matching fp32, "
            "count > 0)" if not basic_ok else
            "not 16-aligned on MPS (count, hidden_dim, vocab, chunk_rows all "
            "need %16 == 0)"
        )
        LOG.warning(
            "streaming_head_loss: falling back to the full-logits path — %s "
            "(device=%s dtype=%s/%s count=%d hidden_dim=%d vocab=%d "
            "chunk_rows=%d).",
            reason, hidden.device.type, hidden.dtype, weight.dtype, count,
            hidden.shape[-1], weight.shape[0], chunk_rows,
        )
        return fused_ce_z_loss(
            linear(hidden, weight), labels, coeff, chunk_rows=chunk_rows
        )
    return _StreamingHeadLoss.apply(hidden, weight, labels, coeff, chunk_rows)
