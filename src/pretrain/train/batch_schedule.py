"""Three-phase batch warmup schedule (plan/05 §3).

Implementation rule: never grow the batch by changing ``micro_batch_size``
or ``seq_len`` (both are shape-fixed for ``torch.compile`` and FSDP2
prefetch). Only grad-accum varies.

Two coordinate systems live here:

  * The *legacy* per-rank view (``grad_accum_steps`` / ``tokens_per_optimizer_step``)
    derives accum from ``dp_world_size`` — so the same global-batch target
    splits into different micro-batch groupings at different world sizes. This
    is the topology coupling the audit work removes.
  * The *canonical* view (``microbatches_per_step`` and the assignment helpers
    below) defines, independent of world size, how many micro-batches the whole
    cluster consumes per optimizer step and which (virtual) rank owns each. A
    micro-batch is the atomic forward/backward unit — TP-paired ranks share one
    — so assignment is at micro-batch granularity, never split mid-batch. When
    the total isn't divisible by the world size, the lowest ranks each take one
    extra micro-batch (so every rank still triggers exactly one grad-sync per
    step; the collective stays balanced). See ``data/global_stream.py`` for the
    window stream these indices slice into.
"""

from __future__ import annotations

import dataclasses

from pretrain.config.schema import TrainConfig


def current_global_batch_tokens(consumed_tokens: int, train: TrainConfig) -> int:
    """Return the target global-batch-tokens for the *current* token count.

    The schedule is ``warmup → main → late``:
        consumed < warmup_to_main: warmup
        consumed < main_to_late: main
        else: late
    """
    if consumed_tokens < train.warmup_to_main_at_tokens:
        return train.global_batch_tokens.warmup
    if consumed_tokens < train.main_to_late_at_tokens:
        return train.global_batch_tokens.main
    return train.global_batch_tokens.late


def grad_accum_steps(
    consumed_tokens: int,
    train: TrainConfig,
    *,
    dp_world_size: int,
) -> int:
    """How many micro-batches per optimizer step at this token count.

    ``global_batch_tokens = dp_world_size * micro_batch_size * seq_len * accum``,
    rearranged. ``dp_world_size`` is the unique-token degree (excludes TP,
    and any future CP/PP) — TP-paired ranks consume the same minibatch, so
    they don't add to the global batch. We round up: a partial accumulation
    at a phase boundary is preferable to under-shooting the target batch.
    """
    target = current_global_batch_tokens(consumed_tokens, train)
    per_step_per_rank = train.micro_batch_size * train.seq_len
    per_step_global = per_step_per_rank * max(dp_world_size, 1)
    accum = max(1, target // per_step_global)
    if accum * per_step_global < target:
        accum += 1
    return accum


def tokens_per_optimizer_step(
    consumed_tokens: int,
    train: TrainConfig,
    *,
    dp_world_size: int,
) -> int:
    accum = grad_accum_steps(consumed_tokens, train, dp_world_size=dp_world_size)
    return accum * train.micro_batch_size * train.seq_len * max(dp_world_size, 1)


# --------------------------------------------------------------------------- #
# Canonical, world-size-independent view.
# --------------------------------------------------------------------------- #


def microbatches_per_step(consumed_tokens: int, train: TrainConfig) -> int:
    """Total micro-batches the whole cluster consumes in one optimizer step.

    ``M = ceil(global_batch_tokens / (micro_batch_size * seq_len))`` — the
    canonical analogue of the legacy ``dp_world_size * grad_accum_steps`` count,
    but defined with **no** world-size term. Every topology consumes exactly
    ``M`` micro-batches (hence ``M * micro_batch_size`` windows) per step, so
    the phase schedule, token accounting, and data stream stay identical across
    device counts. Rounds up for the same reason ``grad_accum_steps`` does: a
    partial top-up at a phase boundary beats under-shooting the target.
    """
    target = current_global_batch_tokens(consumed_tokens, train)
    per_microbatch = train.micro_batch_size * train.seq_len
    m = max(1, target // per_microbatch)
    if m * per_microbatch < target:
        m += 1
    return m


def windows_per_step(consumed_tokens: int, train: TrainConfig) -> int:
    """Total packed windows (sequences) in one optimizer step = ``M * mb``."""
    return microbatches_per_step(consumed_tokens, train) * train.micro_batch_size


def canonical_tokens_per_step(consumed_tokens: int, train: TrainConfig) -> int:
    """Tokens consumed per optimizer step — world-size-independent.

    ``windows_per_step * seq_len``. Replaces the legacy
    ``accum * mb * seq * dp_world_size`` so ``consumed_tokens`` (and therefore
    the warmup/main/late boundaries and the ``ckpt_every_tokens`` cadence) lands
    on the same global step at every world size and in the single-device audit.
    """
    return windows_per_step(consumed_tokens, train) * train.seq_len


def accum_for_rank(total_microbatches: int, *, world_size: int, rank: int) -> int:
    """How many micro-batches rank ``rank`` of ``world_size`` owns this step.

    Round-robin by ``m % world_size``, so the lowest ``M % world_size`` ranks
    each take one extra: ``ceil((M - rank) / world_size)``.
    """
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    n = max(world_size, 1)
    return (total_microbatches - rank + n - 1) // n if rank < total_microbatches else 0


def microbatch_indices_for_rank(
    total_microbatches: int, *, world_size: int, rank: int
) -> list[int]:
    """Global-within-step micro-batch indices owned by ``rank``, in order.

    Micro-batch ``m`` (``0 <= m < M``) is owned by ``rank = m % world_size`` at
    local accum index ``m // world_size``. The single-device audit drives all
    ``m`` in order, assigning each to virtual rank ``m % world_size``.
    """
    return [m for m in range(total_microbatches) if owner_rank(m, world_size=world_size) == rank]


def owner_rank(microbatch: int, *, world_size: int) -> int:
    """The DP rank that trains micro-batch ``microbatch`` of a step.

    The one place the ownership rule lives. The training loop, the doc map's
    ``dp_rank`` column and the single-device audit all call this, so a change to
    the assignment (contiguous blocks, say) cannot leave one of them behind.
    """
    return microbatch % max(world_size, 1)


@dataclasses.dataclass(frozen=True)
class StepPlan:
    """One optimizer step's canonical layout, independent of world size.

    ``base_window`` is the global window index (``GlobalStream`` position) of the
    step's first window; the step spans windows
    ``[base_window, base_window + microbatches * mb)``. Micro-batch ``m`` covers
    windows ``[base_window + m*mb, base_window + (m+1)*mb)``.
    """

    step: int
    consumed_tokens: int      # consumed BEFORE this step (drives the target)
    base_window: int
    microbatches: int
    tokens_this_step: int


def window_position(
    global_window: int, plan: StepPlan, *, micro_batch_size: int
) -> tuple[int, int]:
    """``(microbatch, slot)`` of a global window inside ``plan``."""
    rel = global_window - plan.base_window
    if not 0 <= rel < plan.microbatches * micro_batch_size:
        raise ValueError(f"window {global_window} is not in step {plan.step}")
    return rel // micro_batch_size, rel % micro_batch_size


def iter_step_plans(
    consumed_tokens: int,
    windows_emitted: int,
    train: TrainConfig,
    *,
    start_step: int = 0,
):
    """Yield :class:`StepPlan` for successive optimizer steps from a resume point.

    Walks the piecewise-constant schedule one step at a time (cheap integer
    math). Seed ``consumed_tokens`` / ``windows_emitted`` / ``start_step`` from
    the checkpoint so the cluster, a resumed run, and the audit all enumerate
    the identical step layout. Infinite generator — caller stops on its own
    condition (e.g. reaching the next checkpoint's token threshold).
    """
    mb = train.micro_batch_size
    step = start_step
    while True:
        m = microbatches_per_step(consumed_tokens, train)
        tokens_this_step = m * mb * train.seq_len
        yield StepPlan(
            step=step,
            consumed_tokens=consumed_tokens,
            base_window=windows_emitted,
            microbatches=m,
            tokens_this_step=tokens_this_step,
        )
        consumed_tokens += tokens_this_step
        windows_emitted += m * mb
        step += 1
