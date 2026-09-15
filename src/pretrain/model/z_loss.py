"""Z-loss reduction: ``lse².mean()`` over (B, T) positions.

The Llama 3 / PaLM "auxiliary z-loss" keeps log-Z (the partition-function
log-sum-exp) close to zero so the softmax denominator stays well-conditioned
at fp32. Both Llama and Qwen3.5 use the same formula.

Routed through repop's bitwise-reproducible kernels: ``amax`` provides the
max shift (associative, bit-stable regardless of dispatch order), then
``exp``/``sum_dim``/``log`` form the logsumexp, and ``pow``/``mean``
finish the reduction.

Gradient. With

    lse_n        = log(sum_v exp(logits[n, v] - m_n)) + m_n
    softmax[n,v] = exp(logits[n, v] - m_n) / Z_n        (Z_n = sum_v exp(...))
    z_loss       = coeff * mean_n(lse_n ** 2)

the chain rule gives, per row ``n`` and vocab index ``v``,

    d lse_n / d logits[n, v]  = softmax[n, v]
    d z_loss / d logits[n, v] = (2 * coeff / N) * lse_n * softmax[n, v]

The ``- m`` shift is a row-wise constant and drops out of the gradient
(standard for stable-softmax implementations). repop's ``exp``/``log``/
``sum_dim``/``pow``/``mean`` kernels carry no autograd of their own, so we
wrap the reduction in a :class:`torch.autograd.Function` that emits this
gradient explicitly — otherwise z-loss is computed and logged but
contributes nothing to the backward pass and never regularises the vocab.

The gradient is **per-row independent**, so the backward is evaluated in
row-chunks: at most one chunk's ``[chunk_rows, V]`` fp32 intermediates are
resident at a time (the lm_head logits are the memory ceiling of the whole
step, which is why the CE path is chunked too). The chunked result is
bitwise-identical to a single-shot backward regardless of ``chunk_rows``,
and the same helper backs the memory-frugal audit drop-in
(:func:`pretrain.cli.audit_replay._frugal_compute_z_loss`) so the cluster
and the single-device audit produce byte-for-byte identical logit
gradients.
"""

from __future__ import annotations

import torch

from pretrain.config.schema import ZLossConfig

# Row-chunk for the backward. Bounds the resident fp32 softmax to
# ``Z_LOSS_GRAD_CHUNK_ROWS * V`` elements; does not affect the result.
Z_LOSS_GRAD_CHUNK_ROWS = 2048



def _z_loss_lse(
    logits_f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row max-shifted ``exp``, partition ``Z`` and ``lse`` over the last
    (vocab) axis, built from repop kernels. ``logits_f`` is fp32; every op
    reduces over the vocab axis only, so a row's result is independent of how
    many rows are in ``logits_f`` (row-chunking changes nothing per row)."""
    from repop import ops as repop_ops

    m = logits_f.amax(dim=-1, keepdim=True)
    exp_shifted = repop_ops.exp(logits_f - m)
    Z = repop_ops.sum_dim(exp_shifted, dim=-1)
    lse = repop_ops.log(Z) + m.squeeze(-1)
    return exp_shifted, Z, lse


def z_loss_grad(
    logits: torch.Tensor,
    coeff: float,
    grad_z: torch.Tensor,
    chunk_rows: int = Z_LOSS_GRAD_CHUNK_ROWS,
) -> torch.Tensor:
    """``d z_loss / d logits`` = ``(2*coeff/N) * lse * softmax``, row-chunked.

    ``grad_z`` is the scalar upstream gradient of the z-loss output. Returns a
    tensor shaped like ``logits`` in ``logits``' dtype. ``N`` is the global row
    count, so the chunked write produces the same per-row gradient a single
    pass would. The elementwise div/mul are plain-torch (correctly-rounded
    fp32, device-portable) — mirrors ``_VocabParallelCEZLoss``'s backward."""
    flat = logits.reshape(-1, logits.shape[-1])
    N = flat.shape[0]
    grad_flat = torch.empty_like(flat, dtype=torch.float32)
    for s in range(0, N, chunk_rows):
        chunk_f = flat[s : s + chunk_rows].float()
        exp_shifted, Z, lse = _z_loss_lse(chunk_f)
        softmax = exp_shifted / Z.unsqueeze(-1)
        coef = (2.0 * coeff / N) * lse * grad_z
        grad_flat[s : s + chunk_f.shape[0]] = softmax * coef.unsqueeze(-1)
    return grad_flat.reshape_as(logits).to(logits.dtype)


class _ZLossFunction(torch.autograd.Function):
    """``coeff * mean(lse**2)`` with an explicit, repop-built backward.

    Forward retains only the (live) input reference; backward recomputes the
    softmax/lse in row-chunks so the full ``[N, V]`` fp32 intermediates are
    never resident across the step."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, coeff: float) -> torch.Tensor:  # type: ignore[override]
        from repop import ops as repop_ops

        logits_f = logits.float()
        _, _, lse = _z_loss_lse(logits_f)
        z = coeff * repop_ops.mean(repop_ops.pow(lse, 2.0))

        ctx.save_for_backward(logits)
        ctx.coeff = coeff
        return z

    @staticmethod
    def backward(ctx, grad_z: torch.Tensor):  # type: ignore[override]
        (logits,) = ctx.saved_tensors
        if ctx.coeff == 0.0:
            return torch.zeros_like(logits), None
        return z_loss_grad(logits, ctx.coeff, grad_z), None


def compute_z_loss(logits: torch.Tensor, cfg: ZLossConfig) -> torch.Tensor:
    """Scalar z-loss, DIFFERENTIABLE w.r.t. ``logits`` (see module docstring for
    the gradient). ``logits`` may be any float dtype; reduction is fp32.

    This is the phase-2+ / cluster path. Audits of checkpoints trained BEFORE
    the backward existed must instead use :func:`compute_z_loss_inert` (the
    audit selects by config — see ``audit_replay``); both live here so one
    checkout can reproduce either regime."""
    return _ZLossFunction.apply(logits, cfg.coeff)


def compute_z_loss_inert(logits: torch.Tensor, cfg: ZLossConfig) -> torch.Tensor:
    """LEGACY z-loss — VALUE ONLY, no gradient. Verbatim the implementation that
    shipped before the backward existed: repop's autograd-free kernels make the
    result detached, so z-loss contributes nothing to backward and regularises
    nothing. Kept so a single checkout can audit checkpoints trained in that
    regime (phase 1) bit-for-bit. No spread guard — those runs had none. Same
    ``(logits, cfg)`` signature as :func:`compute_z_loss` so the audit swaps it
    in by name."""
    from repop import ops as repop_ops

    logits_f = logits.float()
    m = logits_f.amax(dim=-1, keepdim=True)
    sum_exp = repop_ops.sum_dim(repop_ops.exp(logits_f - m), dim=-1)
    lse = repop_ops.log(sum_exp) + m.squeeze(-1)
    return cfg.coeff * repop_ops.mean(repop_ops.pow(lse, 2.0))
