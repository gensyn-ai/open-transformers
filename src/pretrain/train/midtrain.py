"""Midtraining (OLMo 2, 2501.00656 §4): resume-time guards for anneal runs.

A midtraining run branches from a pretrain checkpoint and drives the LR
linearly to zero over a fresh high-quality mixture (here: the full
Dolma3-Dolmino annealing mixture, ``configs/data/dolma3_dolmino_mix.yaml``).
The trainer already has
every mechanism this needs — ``--resume-from``, per-source stream state
keyed by source NAME (a new mixture starts each new source at document
0), ``train.resume_reset_optimizer`` + ``schedule.rewarm_tokens`` for a
cold-moment branch — so midtraining is a config recipe, not a new loop.
See ``configs/train/1b_midtrain_math.yaml`` and ``docs/midtraining_math.md``.

What is NOT mechanically checkable from config alone is the anchor
arithmetic between the config and the checkpoint actually passed at
launch. The failure modes are silent, and they split by what the guard
can prove:

* ERRORS — ``schedule.anneal_start_tokens`` above the checkpoint's
  ``consumed_tokens`` quietly holds peak LR past the branch point
  (annealing less than configured), and ``train.total_tokens`` at or
  below the checkpoint's ``consumed_tokens`` makes the training loop
  exit immediately with no error — the classic
  forgot-to-raise-total_tokens footgun, since total_tokens is ABSOLUTE
  (pretrain + midtrain). Both raise at launch.
* WARNING ONLY — ``anneal_start_tokens`` BELOW the checkpoint's
  ``consumed_tokens`` starts the run mid-decay. This direction cannot be
  an error: it is exactly what a legitimate crash-resume of the
  midtraining run looks like (``consumed_tokens`` only grows past the
  anchor). The guard surfaces it instead — it always logs the effective
  LR at the resume point next to the peak, and warns loudly when the
  resume lands past the anchor, so a misconfigured anchor (e.g. a config
  that kept ``linear_anneal.yaml``'s ``anneal_start_tokens: 0`` default)
  is visible in the first log lines rather than discovered from the loss
  curve.

``check_linear_anneal_resume`` is called from the resume block in
``pretrain.train.loop`` for every resumed run whose schedule is
``linear_anneal`` and is a no-op otherwise.
"""

from __future__ import annotations

import logging

from pretrain.config.schema import RootConfig
from pretrain.optim.schedules import build_schedule

LOG = logging.getLogger(__name__)


def check_linear_anneal_resume(cfg: RootConfig, meta_consumed_tokens: int) -> None:
    """Validate a resumed ``linear_anneal`` run against the checkpoint meta.

    Raises ``ValueError`` on anchor/budget arithmetic the guard can prove
    wrong; logs the effective resume-point LR and warns on a past-the-anchor
    resume, which it cannot distinguish from a crash-resume (see module
    docstring). No-op for other schedules.
    """
    if cfg.schedule.name != "linear_anneal":
        return

    anneal_start = cfg.schedule.anneal_start_tokens
    if meta_consumed_tokens < anneal_start:
        raise ValueError(
            f"linear_anneal resume: checkpoint consumed_tokens "
            f"({meta_consumed_tokens}) is below schedule.anneal_start_tokens "
            f"({anneal_start}) — the run would silently hold peak LR until the "
            f"anchor instead of annealing from the branch point. Set "
            f"anneal_start_tokens to this checkpoint's consumed_tokens "
            f"(pretrain-inspect-checkpoint prints it), or use schedule=wsd if "
            f"a constant-LR extension before the decay is intended."
        )
    if cfg.train.total_tokens <= meta_consumed_tokens:
        raise ValueError(
            f"linear_anneal resume: train.total_tokens ({cfg.train.total_tokens}) "
            f"<= checkpoint consumed_tokens ({meta_consumed_tokens}) — the "
            f"training loop would exit immediately. total_tokens is ABSOLUTE "
            f"(pretrain + midtrain): set it to anneal_start_tokens plus the "
            f"midtraining token budget."
        )

    # Effective-LR visibility for the direction the guard cannot prove wrong
    # (consumed past the anchor == what a crash-resume looks like): always log
    # the LR the schedule yields at the resume point, and warn when the resume
    # lands mid-decay so a misconfigured anchor shows up in the first log
    # lines. A legitimate crash-resume prints the same expected line each time.
    lr_now = build_schedule(cfg.schedule, cfg.train, cfg.optim)(
        meta_consumed_tokens, cfg.train.total_tokens
    )
    LOG.info(
        "linear_anneal resume: lr at resume point = %.6e (peak_lr %.6e, "
        "anchor %d, total %d, consumed %d)",
        lr_now, cfg.optim.peak_lr, anneal_start,
        cfg.train.total_tokens, meta_consumed_tokens,
    )
    if meta_consumed_tokens > anneal_start:
        pct = 100.0 * lr_now / cfg.optim.peak_lr if cfg.optim.peak_lr else 0.0
        LOG.warning(
            "linear_anneal resume: checkpoint is %d tokens PAST the anneal "
            "anchor (%d) — LR starts mid-decay at %.6e (%.1f%% of peak). "
            "Expected if this is a crash-resume of the midtraining run "
            "itself; if this is a NEW branch from a pretrain checkpoint, "
            "schedule.anneal_start_tokens is misconfigured and must equal "
            "the checkpoint's consumed_tokens (%d).",
            meta_consumed_tokens - anneal_start, anneal_start, lr_now, pct,
            meta_consumed_tokens,
        )
