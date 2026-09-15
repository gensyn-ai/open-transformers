"""Learning-rate schedules.

Schedules are pure math operating on ``(consumed_tokens, total_tokens)``.
We do **not** track step counts — tokens are the canonical clock so the
batch schedule (which changes grad-accum, hence tokens-per-step) doesn't
desynchronise the LR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

from pretrain.config.schema import OptimConfig, ScheduleConfig, TrainConfig


class LRSchedule(Protocol):
    def __call__(self, consumed_tokens: int, total_tokens: int) -> float:
        ...


def _tokens_per_warmup_step(train: TrainConfig) -> int:
    """During warmup we are in the smallest batch phase."""
    return train.global_batch_tokens.warmup


def cosine_schedule(
    sched: ScheduleConfig,
    train: TrainConfig,
    optim: OptimConfig,
) -> LRSchedule:
    """Linear warmup → cosine decay to ``min_lr_frac * peak_lr``.

    Cycle length is the full token budget (Chinchilla §5).
    """
    peak = optim.peak_lr
    min_lr = peak * sched.min_lr_frac
    warmup_tokens = sched.warmup_steps * _tokens_per_warmup_step(train)

    def fn(consumed: int, total: int) -> float:
        if consumed < warmup_tokens:
            return peak * consumed / max(warmup_tokens, 1)
        remaining = max(total - warmup_tokens, 1)
        progress = min(max((consumed - warmup_tokens) / remaining, 0.0), 1.0)
        return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))

    return fn


def wsd_schedule(
    sched: ScheduleConfig,
    train: TrainConfig,
    optim: OptimConfig,
) -> LRSchedule:
    """Warmup → constant peak → linear decay to ``decay_to_frac * peak``."""
    peak = optim.peak_lr
    end_lr = peak * sched.wsd_decay_to_frac
    warmup_tokens = sched.warmup_steps * _tokens_per_warmup_step(train)

    def fn(consumed: int, total: int) -> float:
        if consumed < warmup_tokens:
            return peak * consumed / max(warmup_tokens, 1)
        decay_start = sched.wsd_decay_start_frac * total
        if consumed < decay_start:
            return peak
        decay_len = max(total - decay_start, 1)
        progress = min(max((consumed - decay_start) / decay_len, 0.0), 1.0)
        return peak + (end_lr - peak) * progress

    return fn


def linear_anneal_schedule(
    sched: ScheduleConfig,
    train: TrainConfig,
    optim: OptimConfig,
) -> LRSchedule:
    """Hold peak until ``anneal_start_tokens``, then decay linearly to
    ``min_lr_frac * peak`` at the end of the token budget.

    The midtraining schedule (OLMo 2, 2501.00656 §4.1): branch from a
    pretrain checkpoint at ``anneal_start_tokens`` (= the checkpoint's
    ``consumed_tokens``; the resume guard in ``pretrain.train.midtrain``
    enforces the match) and drive the LR linearly to zero over the
    midtraining tokens. Differs from WSD only in that the decay start is
    an ABSOLUTE token count, not a fraction of ``total_tokens`` — so the
    anchor stays put when the budget is edited. ``optim.peak_lr`` should
    be the pretrain LR at the branch point, and ``min_lr_frac`` 0.0 for
    the paper's anneal-to-zero. With ``anneal_start_tokens: 0`` this is
    also usable cold: linear warmup, then linear decay over the run.
    """
    peak = optim.peak_lr
    end_lr = peak * sched.min_lr_frac
    warmup_tokens = sched.warmup_steps * _tokens_per_warmup_step(train)

    def fn(consumed: int, total: int) -> float:
        if consumed < warmup_tokens:
            return peak * consumed / max(warmup_tokens, 1)
        anneal_start = max(sched.anneal_start_tokens, warmup_tokens)
        if consumed < anneal_start:
            return peak
        anneal_len = max(total - anneal_start, 1)
        progress = min(max((consumed - anneal_start) / anneal_len, 0.0), 1.0)
        return peak + (end_lr - peak) * progress

    return fn


def build_schedule(
    sched: ScheduleConfig,
    train: TrainConfig,
    optim: OptimConfig,
) -> LRSchedule:
    if sched.name == "cosine":
        return cosine_schedule(sched, train, optim)
    if sched.name == "wsd":
        return wsd_schedule(sched, train, optim)
    if sched.name == "linear_anneal":
        if train.total_tokens <= sched.anneal_start_tokens:
            raise ValueError(
                f"linear_anneal: train.total_tokens ({train.total_tokens}) must "
                f"exceed schedule.anneal_start_tokens "
                f"({sched.anneal_start_tokens}). total_tokens is ABSOLUTE "
                f"(pretrain + midtrain): set it to the branch checkpoint's "
                f"consumed_tokens plus the midtraining budget."
            )
        return linear_anneal_schedule(sched, train, optim)
    raise ValueError(f"unknown schedule: {sched.name}")


def schedule_lr(optimizer, lr: float) -> None:
    """Write ``lr`` into every param group."""
    for g in optimizer.param_groups:
        g["lr"] = lr
