"""Weight initialisation.

Truncated normal (mean 0, ``cfg.init.std``) for every matrix and zeros for
1-D parameters (biases / norm gains override this with ones). No depth- or
width-dependent scaling: every parameter shares the one std, following OLMo 2
(arXiv:2501.00656 §3.2), which found this flat init more stable than the
GPT-NeoX/Zhang-2019 scaled init (later layers shrunk by 1/sqrt(2*n_layers))
that it supersedes.

Apply *before* FSDP2 wrapping. This module sets parameter tensors in-place;
no return value.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from pretrain.model.llama3 import Llama3
from pretrain.model.modules.norm import RMSNorm


def _trunc_normal_(tensor: torch.Tensor, std: float, generator: torch.Generator | None) -> None:
    """``trunc_normal_`` that tolerates a generator/tensor device mismatch.

    PyTorch requires ``generator.device == tensor.device``. In production
    the model is already on CUDA when init runs, so a CPU generator would
    fail. We sample on a generator built on the *tensor's* device.
    """
    if generator is None:
        nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)
        return
    if generator.device == tensor.device:
        nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std, generator=generator)
        return
    # Mismatched: derive a same-device generator with a deterministic seed
    # taken from the caller's generator. This keeps init reproducible at
    # the level of "same caller seed → same init" without forcing the
    # caller to know the param device.
    seed = int(generator.initial_seed())
    g_local = torch.Generator(device=tensor.device).manual_seed(seed)
    nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std, generator=g_local)


def init_weights(model: Llama3, seed: int | None = None) -> None:
    """Production init.

    With ``seed`` provided (production path), every >=2D parameter is drawn
    from repop's counter-based Philox stream via ``stable_trunc_normal`` (see
    ``_repop_trunc_normal_init``). repop's CPU and CUDA streams are bit-equal
    for a given seed, so the init is deterministic **per seed, independent of
    device-arch** — a CPU, CUDA, or (once repop grows a Metal RNG) Metal box
    all construct byte-identical weights. The walk through
    ``named_parameters()`` consumes one consistent stream, same on every rank,
    so all ranks build the same full model before ``fully_shard`` ("init once
    then shard"); the counter is also independent of thread/block layout, so
    this is stronger than the old per-``(device-arch, seed)`` cuRAND guarantee.

    Without ``seed``, falls back to the global torch RNG — which
    ``parallel.env.set_seed`` has offset by rank, so the resulting init
    is rank-divergent. Kept only for the legacy/no-distributed path.

    Note: this seeds the repop RNG engine, NOT the torch global RNG, so the
    per-rank RNG state checkpoints save/restore (torch.get_rng_state) is
    unaffected — init is a cold-start-only event and is never replayed on
    resume.
    """
    cfg = model.cfg
    std = cfg.init.std

    if seed is None:
        # Legacy/no-distributed path: global torch RNG (rank-divergent).
        for _, p in model.named_parameters():
            if p.dim() >= 2:
                _trunc_normal_(p, std=std, generator=None)
            else:
                nn.init.zeros_(p)
    else:
        _repop_trunc_normal_init(model, std=std, seed=seed)

    _set_norm_gains_to_one(model)
    _reset_lsq_weight_scales(model)


def _repop_trunc_normal_init(model: nn.Module, std: float, seed: int) -> None:
    """Device-independent trunc-normal init via repop's byte-equal RNG.

    Resets repop's engine to ``(key=seed, counter=0)`` on every available
    backend, then walks ``named_parameters()`` in order, drawing each >=2D
    weight from ``stable_trunc_normal`` (mean 0, std, bounds [-2std, 2std] —
    the same parameters as the old ``torch.nn.init.trunc_normal_``) and zeroing
    1-D params. Because all draws on one rank hit a single backend engine
    seeded identically, and the algorithm matches ``trunc_normal_`` element for
    element, a CPU run and a CUDA run produce bit-identical weights.

    Seeds repop directly (``set_cpu_seed``/``set_cuda_seed``) rather than
    ``set_reproducibility`` so the torch global RNG — what checkpoints persist —
    is left alone.
    """
    import repop.rand as repop_rand
    from repop.backend import cpu as _repop_cpu

    try:
        from repop.backend import cuda as _repop_cuda
    except ImportError:
        _repop_cuda = None

    # Reset both engines (key <- seed, counter <- 0) so the stream is identical
    # whether this rank inits on CPU or CUDA.
    _repop_cpu.set_cpu_seed(seed)
    if _repop_cuda is not None and torch.cuda.is_available():
        _repop_cuda.set_cuda_seed(seed)

    a, b = -2.0 * std, 2.0 * std
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.dim() >= 2:
                vals = repop_rand.stable_trunc_normal(
                    tuple(p.shape), std=std, a=a, b=b, is_cuda=p.is_cuda
                )
                p.copy_(vals)
            else:
                p.zero_()


def init_weights_seeded(model: Llama3, seed: int) -> None:
    """Seeded init — primarily for tests asserting `same seed → same init`.

    Routes through the same repop byte-equal RNG path as ``init_weights``, so
    `same seed → same init` now holds *across devices* (CPU vs CUDA), not just
    per-device-arch as the old ``torch.Generator`` path did.
    """
    cfg = model.cfg
    std = cfg.init.std
    _repop_trunc_normal_init(model, std=std, seed=seed)
    _set_norm_gains_to_one(model)
    _reset_lsq_weight_scales(model)


def _set_norm_gains_to_one(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, RMSNorm):
            with torch.no_grad():
                if getattr(module, "weight", None) is not None:
                    module.weight.fill_(1.0)


def _reset_lsq_weight_scales(model: nn.Module) -> None:
    """Re-derive every ``LSQQuantizedLinear.weight_scale`` from the *final*
    weights.

    Must run last: the main init loop zeros all 1-D params (which includes the
    learnable ``weight_scale``) — so the only correct place to set the
    per-channel LSQ scale is here, after the weights have reached their initial
    values. This
    reproduces the LSQ-recommended init (``2·mean(|w_row|)/√qmax``, Esser et al.)
    that ``LSQQuantizedLinear.__init__`` computes from the default ``nn.Linear``
    weights — which is both clobbered by the zero-pass and stale w.r.t. the
    trunc-normal weights. Runs before FSDP wrapping so the canonical scale is
    what gets sharded.

    No-op when repop / LSQ isn't present (CPU dev, non-QAT runs).
    """
    try:
        from repop import ops
        from repop.qat.functional import _qmax_for_bits
        from repop.qat.lsq import LSQQuantizedLinear
    except ImportError:
        return

    for module in model.modules():
        if not isinstance(module, LSQQuantizedLinear):
            continue
        w_qmax = _qmax_for_bits(module.weight_bits)
        with torch.no_grad():
            # Cross-device BFR weight_scale (this is the copy that sets the
            # PERSISTED init scale, so it must be device-independent for the
            # init audit to match across CPU/CUDA/arch). Two hazards, both fp32:
            #   1. the K-axis mean: torch.mean(dim=1) is NOT byte-identical
            #      CPU<->GPU (different reduction order, ~1.9e-9) — use repop's
            #      BFR reduction (ops.mean_dim) instead, which is byte-exact.
            #   2. ×const not ÷scalar: tensor÷python-scalar isn't byte-identical
            #      CPU<->GPU; fold 2/sqrt(qmax) into one host constant + multiply.
            # (mirrors repop's "eliminate tensor/python-scalar divides" fix;
            # the reduction swap is the additional piece needed for the init hash.)
            w_abs = module.weight.detach().abs()
            if w_abs.is_mps:
                # repop has no Metal reduction kernel; the PERSISTED init scale
                # must be the cross-device BFR value, so compute the byte-exact
                # K-axis mean on CPU (repop's reference) and copy back to the
                # mps weight_scale. Mirrors repop.ops.var_mean's HIP handling.
                init_scale = (
                    ops.mean_dim(w_abs.cpu(), dim=1) * (2.0 / math.sqrt(w_qmax))
                ).to(w_abs.device)
            else:
                init_scale = ops.mean_dim(w_abs, dim=1) * (2.0 / math.sqrt(w_qmax))
            module.weight_scale.copy_(init_scale.clamp_min(1e-6))
