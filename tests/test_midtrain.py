"""Resume-time guards for midtraining anneal runs (pretrain.train.midtrain)."""

from __future__ import annotations

import pytest

from pretrain.config import load_config
from pretrain.train.midtrain import check_linear_anneal_resume

BRANCH = 400_000_000_000  # 1b_midtrain_math's anneal_start_tokens


def test_passes_at_branch_point(caplog):
    cfg = load_config("1b_midtrain_math")
    with caplog.at_level("INFO", logger="pretrain.train.midtrain"):
        check_linear_anneal_resume(cfg, BRANCH)
    # Effective LR is logged on every linear_anneal resume; a clean branch
    # (consumed == anchor) must NOT trip the past-the-anchor warning.
    assert any("lr at resume point" in r.message for r in caplog.records)
    assert not any(r.levelname == "WARNING" for r in caplog.records)


def test_passes_on_crash_resume_past_anchor(caplog):
    # A crash-resume of the midtraining run itself lands past the anchor —
    # allowed, but loudly: the mid-decay direction is indistinguishable from
    # a misconfigured anchor, so it warns instead of raising.
    cfg = load_config("1b_midtrain_math")
    with caplog.at_level("INFO", logger="pretrain.train.midtrain"):
        check_linear_anneal_resume(cfg, BRANCH + 5_000_000_000)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "PAST the anneal anchor" in warnings[0].message


def test_rejects_checkpoint_before_anchor():
    # Branching from an earlier checkpoint than the configured anchor would
    # silently hold peak LR until the anchor instead of annealing.
    cfg = load_config("1b_midtrain_math")
    with pytest.raises(ValueError, match="anneal_start_tokens"):
        check_linear_anneal_resume(cfg, BRANCH - 10_000_000_000)


def test_rejects_exhausted_budget():
    # total_tokens <= checkpoint consumed_tokens would make the training
    # loop exit immediately with no error.
    cfg = load_config("1b_midtrain_math")
    with pytest.raises(ValueError, match="total_tokens"):
        check_linear_anneal_resume(cfg, cfg.train.total_tokens)


def test_noop_for_other_schedules():
    # The guard only constrains linear_anneal runs; a cosine resume with any
    # token count passes untouched.
    cfg = load_config("1b_repop_v2")
    check_linear_anneal_resume(cfg, 0)
    check_linear_anneal_resume(cfg, cfg.train.total_tokens)
