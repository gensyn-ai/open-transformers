"""QK-norm gain control — clamp and staged wake-up for the attention-logit
temperature (Option A of the July 2026 incident recovery plan).

Why this exists: QK-norm bounds attention logits only while its learnable
gains stay bounded — the logit is Σᵢ γqᵢ·γkᵢ·q̂ᵢ·k̂ᵢ, so the γq⊙γk product is a
learned temperature with no restoring force (norm gains sit in the no-decay
group). A July 2026 1B QAT run demonstrated the failure end to
end: after the frozen q-side tensors woke, the gain product marched from 19.5 to 28+ at a
constant +0.5/100 steps, loss degraded while gradients *calmed* (entropy
collapse: saturated attention starves the softmax gradient), then the run
flipped into clip-throttled gradient bursts (global clip engaged on 53% of
steps at step ~52,000).

Two deterministic, elementwise controls, both mirrored verbatim in
``pretrain.cli.audit_replay`` (BFR: ``clamp_`` and ``zero_``/``mul_(0)`` are
exact ops — identical bytes on CUDA/CPU/MPS, and identical whether applied to
a rank-local shard or the audit's full tensor):

* ``clamp_qk_gains`` — post-optimizer-step hard cap on |γ| for every
  q_norm/k_norm gain. Runs regardless of a spike-skip (idempotent when weights
  are unchanged), like the LSQ scale refresh it sits next to.
* ``zero_q_gain_grads`` — staged wake-up: zero the q_norm gain gradients
  post-clip / pre-step while ``optimizer_step < until_step``, so wq re-inflates
  against a FIXED query temperature before the gains are released. Ordering
  matters: AFTER the grad clip and BEFORE the optimizer step + state hash
  (Adam moments stay ~0 during the hold — no accumulated kick at release —
  and both paths hash the zeroed grads identically).
"""

from __future__ import annotations

import torch


def _local(t: torch.Tensor) -> torch.Tensor:
    """Rank-local shard of a DTensor (shares storage); pass-through otherwise."""
    return t.to_local() if hasattr(t, "to_local") else t


def _qk_norm_weights(model: torch.nn.Module, kinds: tuple[str, ...]):
    for _name, mod in model.named_modules():
        for kind in kinds:
            norm = getattr(mod, kind, None)
            if norm is not None and getattr(norm, "weight", None) is not None:
                yield norm.weight


def clamp_qk_gains(model: torch.nn.Module, cap: float) -> int:
    """In-place clamp of every q_norm/k_norm gain to [-cap, cap]. Returns the
    number of gain tensors touched (0 ⇒ the model has no QK-norm)."""
    n = 0
    with torch.no_grad():
        for w in _qk_norm_weights(model, ("q_norm", "k_norm")):
            _local(w.data).clamp_(-cap, cap)
            n += 1
    return n


def zero_q_gain_grads(model: torch.nn.Module) -> int:
    """In-place zero of every q_norm gain GRADIENT (k side untouched). Returns
    the number of gradients zeroed. Call post-clip / pre-optimizer-step."""
    n = 0
    with torch.no_grad():
        for w in _qk_norm_weights(model, ("q_norm",)):
            if w.grad is not None:
                _local(w.grad).zero_()
                n += 1
    return n
