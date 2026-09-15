"""Loss-spike detect + skip + halt protocol (plan/05 §8).

The protocol does NOT roll back checkpoints itself — that's an operator
decision invoked from the runbook. What we automate:

  1. Detect: ``grad_norm_pre_clip > threshold`` OR loss is NaN/Inf.
  2. Skip: on a fresh detection, zero grads for the offending step AND for
     the next ``skip_steps_on_spike - 1`` steps (the "cooldown"). A spike
     rarely is a single bad gradient — the gradient distribution has a
     short tail after the trigger, so we let the model see fresh batches
     before stepping again.
  3. Halt: if more than ``K`` *fresh events* (not cooldown skips) occur in
     a ``W``-step window, surface ``Halt`` to the loop so on-call is
     paged.

Rehearse with an injected spike on the 1B proxy before relying on this
at 8B (plan/05 §8 last paragraph).
"""

from __future__ import annotations

import collections
import dataclasses
import math
from typing import Deque


@dataclasses.dataclass
class SpikeProtocolState:
    skipped_steps: int = 0
    last_skip_step: int = -1
    halted: bool = False
    halt_reason: str = ""


class Halt(Exception):
    """Raised to halt the train loop on repeated spikes."""


class SpikeProtocol:
    def __init__(
        self,
        threshold: float,
        skips_in_window_to_halt: int,
        halt_window_steps: int,
        skip_steps_on_spike: int = 1,
        start_step: int = 0,
    ) -> None:
        self.threshold = threshold
        self.skips_in_window_to_halt = skips_in_window_to_halt
        self.halt_window_steps = halt_window_steps
        self.skip_steps_on_spike = max(int(skip_steps_on_spike), 1)
        self.start_step = start_step
        # ``_recent_skip_steps`` tracks *fresh* trigger events (not the
        # cascade of forced-skip steps that follow), so the halt threshold
        # measures genuine instability — otherwise a single spike with a
        # 50-step cooldown would always trip the halt.
        self._recent_skip_steps: Deque[int] = collections.deque()
        self._cooldown_remaining: int = 0
        self.state = SpikeProtocolState()

    def should_skip(self, grad_norm: float, loss: float, step: int = 0) -> bool:
        # NaN/Inf is always pathological — surface it regardless of warmup.
        if math.isnan(loss) or math.isinf(loss):
            return True
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            return True
        # Inside the post-spike cooldown: skip regardless of the current
        # grad norm. The loop must still call ``record_skip`` so the
        # counter advances.
        if self._cooldown_remaining > 0:
            return True
        if step < self.start_step:
            return False
        return grad_norm > self.threshold

    def record_skip(self, step: int) -> None:
        self.state.skipped_steps += 1
        self.state.last_skip_step = step
        if self._cooldown_remaining > 0:
            # Still inside the cooldown from a previous fresh event — just
            # advance the counter. Do not register against the halt window.
            self._cooldown_remaining -= 1
            return
        # Fresh spike event: register against the halt window and start
        # the cooldown. ``-1`` because this call already accounts for the
        # offending step itself.
        self._recent_skip_steps.append(step)
        cutoff = step - self.halt_window_steps
        while self._recent_skip_steps and self._recent_skip_steps[0] < cutoff:
            self._recent_skip_steps.popleft()
        self._cooldown_remaining = self.skip_steps_on_spike - 1
        if len(self._recent_skip_steps) >= self.skips_in_window_to_halt:
            self.state.halted = True
            self.state.halt_reason = (
                f"{len(self._recent_skip_steps)} spike events in last "
                f"{self.halt_window_steps} steps (threshold {self.threshold})"
            )
            raise Halt(self.state.halt_reason)

    def state_dict(self) -> dict:
        """Serialise the mutable mid-run state — config fields (threshold
        etc.) are rebuilt from cfg on resume and intentionally omitted.

        A resumed run that doesn't restore this would, after a spike that
        landed inside ``skip_steps_on_spike`` of the checkpoint boundary,
        step the optimizer at the next iteration instead of continuing
        the cooldown. The halt-window deque also matters: two spikes
        straddling a checkpoint could land below the halt threshold on
        resume even when a continuous run would have halted.
        """
        return {
            "recent_skip_steps": list(self._recent_skip_steps),
            "cooldown_remaining": int(self._cooldown_remaining),
            "skipped_steps": int(self.state.skipped_steps),
            "last_skip_step": int(self.state.last_skip_step),
            "halted": bool(self.state.halted),
            "halt_reason": str(self.state.halt_reason),
        }

    def load_state_dict(self, sd: dict) -> None:
        self._recent_skip_steps = collections.deque(
            int(s) for s in sd.get("recent_skip_steps", [])
        )
        self._cooldown_remaining = int(sd.get("cooldown_remaining", 0))
        self.state = SpikeProtocolState(
            skipped_steps=int(sd.get("skipped_steps", 0)),
            last_skip_step=int(sd.get("last_skip_step", -1)),
            halted=bool(sd.get("halted", False)),
            halt_reason=str(sd.get("halt_reason", "")),
        )
