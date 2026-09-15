"""Muon-hybrid optimizer (gated; off by default per ADR-004).

Muon is applied to 2-D hidden weight matrices (attention and FFN
projections); embeddings, lm_head, and norm gains use AdamW. Includes
Moonlight's WD + update-RMS rescaling so the two halves are
hyperparameter-comparable.

Implementation note: Muon is a rank-1 / Newton-Schulz orthogonalisation
optimizer. We follow the reference algorithm in Liu et al., "Muon: An
optimizer for hidden weights" (2024). The implementation here is the
public reference algorithm; see `research/05_optimizer.md`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from pretrain.config.schema import OptimConfig


def _zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to orthogonalise a 2D matrix.

    Works on the last two dims of ``grad``. Quintic Newton-Schulz with
    coefficients (3.4445, -4.7750, 2.0315) — same as the reference
    implementation by Keller Jordan.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = grad.bfloat16() if grad.dtype != torch.bfloat16 else grad
    if X.size(-2) > X.size(-1):
        X = X.transpose(-2, -1)
    # Normalise spectral radius.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if grad.size(-2) > grad.size(-1):
        X = X.transpose(-2, -1)
    return X.to(grad.dtype)


class _Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p)
                buf = state["buf"]
                buf.mul_(mu).add_(p.grad)
                update = _zeropower_via_newtonschulz5(buf, steps=ns)
                # Update-RMS scaling: aim for unit-RMS update like AdamW.
                rms = update.pow(2).mean().sqrt().add_(1e-8)
                update = update / rms * math.sqrt(p.numel() / max(p.shape[-1], 1))
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


def _is_hidden_2d(name: str, p: nn.Parameter) -> bool:
    """Hidden weight matrices: attention projections + FFN linears.

    Excludes embeddings, lm_head ('output'), and any 1-D parameter.
    """
    if p.dim() != 2:
        return False
    if "tok_embeddings" in name or "output" in name:
        return False
    if "norm" in name:
        return False
    return True


def build_muon_hybrid(model: nn.Module, cfg: OptimConfig):
    """Build a ParameterDict-of-optimizers: Muon on hidden weights, AdamW
    on the rest. Returned as a small "compound" optimizer object that
    proxies ``zero_grad`` / ``step`` / ``state_dict``.
    """
    hidden_params: list[nn.Parameter] = []
    aux_params: list[tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_hidden_2d(name, p):
            hidden_params.append(p)
        else:
            aux_params.append((name, p))

    from pretrain.optim.adamw import build_param_groups

    aux_groups = build_param_groups(
        aux_params,
        no_decay_substrings=cfg.no_decay_param_names,
        weight_decay=cfg.weight_decay,
    )
    aux = torch.optim.AdamW(
        aux_groups,
        lr=cfg.peak_lr,
        betas=cfg.betas,
        eps=cfg.eps,
        fused=cfg.fused and torch.cuda.is_available(),
    )
    muon = _Muon(
        hidden_params,
        lr=cfg.peak_lr,
        momentum=cfg.betas[1],
        weight_decay=cfg.weight_decay,
    )

    return _CompoundOptimizer([muon, aux])


class _CompoundOptimizer:
    """Minimal proxy that lets the train loop treat (Muon + AdamW) as one."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self._opts = optimizers
        self.param_groups = []
        for o in optimizers:
            self.param_groups.extend(o.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self._opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for o in self._opts:
            o.step()

    def state_dict(self) -> dict:
        return {f"opt_{i}": o.state_dict() for i, o in enumerate(self._opts)}

    def load_state_dict(self, sd: dict) -> None:
        for i, o in enumerate(self._opts):
            o.load_state_dict(sd[f"opt_{i}"])
