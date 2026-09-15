"""AdamW backed by repop's bitwise-reproducible C++/CUDA kernel.

Drop-in for ``build_adamw`` — same param-group split (decay vs no-decay),
same hyperparameters — but the parameter update runs through
``repop.optim.AdamW`` so the optimizer step is reproducible across
identical-hardware runs. ``fused`` is ignored because the repop step is
already a single hand-rolled kernel.

Under FSDP2 ``fully_shard`` parameters become ``DTensor``-wrapped: they
report the *global* shape via ``.numel()`` but their storage holds only
the rank-local shard. The repop kernel is a plain C++ extension with no
DTensor dispatch, so it sees a tensor whose ``numel`` doesn't match its
allocated storage and crashes inside its internal ``view({size})``. We
override ``step`` to unwrap each tensor via ``.to_local()`` at the
kernel boundary; ``.to_local()`` returns the underlying local-shard
storage (not a copy), so in-place updates flow back into the DTensor.
DDP and single-GPU paths see plain tensors and the unwrap is a no-op.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from repop import ops
from repop.optim import AdamW as RepopAdamW

from pretrain.config.schema import OptimConfig
from pretrain.optim.adamw import build_param_groups

LOG = logging.getLogger(__name__)


def _to_local(t: torch.Tensor | None) -> torch.Tensor | None:
    """Unwrap a ``DTensor`` to its rank-local component; pass-through for
    plain tensors. The local tensor shares storage with the DTensor, so
    in-place writes are visible through both wrappers.
    """
    if t is None:
        return None
    return t.to_local() if hasattr(t, "to_local") else t


class FSDPAwareRepopAdamW(RepopAdamW):
    """``repop.optim.AdamW`` with DTensor-unwrapping at the kernel boundary.
    See module docstring for the motivation.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    # ``step`` MUST be a Tensor, not a Python int. DCP only
                    # restores tensor leaves into the resume template (it
                    # loads into the existing storage in place); a non-tensor
                    # int leaf is silently left at the primer's value of 1
                    # on ``dcp.load``. A resumed run would then bias-correct
                    # for step≈2 instead of step≈N and diverge immediately —
                    # see tests/distributed/test_checkpoint_resume_invariants
                    # ::test_optimizer_step_must_be_tensor_to_survive_dcp.
                    state["step"] = torch.zeros((), dtype=torch.int64)
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, dtype=torch.float32
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, dtype=torch.float32
                    )
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, dtype=torch.float32
                        )
                else:
                    # Legacy-checkpoint migration, mirrors upstream.
                    if not isinstance(state["step"], torch.Tensor):
                        state["step"] = torch.tensor(
                            int(state["step"]), dtype=torch.int64
                        )
                    if state["exp_avg"].dtype != torch.float32:
                        state["exp_avg"] = state["exp_avg"].to(torch.float32)
                    if state["exp_avg_sq"].dtype != torch.float32:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(torch.float32)
                    if (
                        "max_exp_avg_sq" in state
                        and state["max_exp_avg_sq"].dtype != torch.float32
                    ):
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                            torch.float32
                        )

                state["step"] += 1

                ops.adamw_kernel_step(
                    param=_to_local(p.data),
                    grad=_to_local(p.grad),
                    exp_avg=_to_local(state["exp_avg"]),
                    exp_avg_sq=_to_local(state["exp_avg_sq"]),
                    max_exp_avg_sq=_to_local(state.get("max_exp_avg_sq", None)),
                    step=int(state["step"].item()),
                    lr=group["lr"],
                    beta1=beta1,
                    beta2=beta2,
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    amsgrad=group["amsgrad"],
                )

        return loss


def prime_optimizer_state(optimizer: torch.optim.Optimizer) -> int:
    """Eagerly materialise zero AdamW state (step/exp_avg/exp_avg_sq) for EVERY
    param in every group. Bitwise-neutral: the kernel treats primed zero state
    exactly like its own lazy init (step 0 -> +=1, zero moments).

    Why eager: the repop step materialises state lazily, only for params whose
    grad is not None. Under the bf16 QAT warm-start (qat.enable_at_step) the
    LSQ weight_scale params receive NO grads before the flip, so a checkpoint
    saved pre-flip carries a SMALLER optim-state key set than the resume
    template primes for -> DCP load fails with "Missing key ...optim.state.N"
    (caught by the e2e regression, 2026-07-21). Priming at build time keeps
    the key set constant for the whole run. Called by the training loop AND by
    audit_replay after build_optimizer, so from-init replays hash-match the
    cluster from step 1.
    """
    n = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros((), dtype=torch.int64)
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format, dtype=torch.float32
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format, dtype=torch.float32
                )
                if group.get("amsgrad"):
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, dtype=torch.float32
                    )
                n += 1
    return n


def build_adamw_repop(model: nn.Module, cfg: OptimConfig) -> FSDPAwareRepopAdamW:
    groups = build_param_groups(
        model.named_parameters(),
        no_decay_substrings=cfg.no_decay_param_names,
        weight_decay=cfg.weight_decay,
    )
    if cfg.fused:
        LOG.info("adamw_repop: cfg.fused=true ignored — repop has its own kernel")
    return FSDPAwareRepopAdamW(
        groups,
        lr=cfg.peak_lr,
        betas=cfg.betas,
        eps=cfg.eps,
    )
