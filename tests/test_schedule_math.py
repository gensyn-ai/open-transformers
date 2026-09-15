"""LR schedule + batch-warmup math at the documented endpoints."""

from __future__ import annotations

import math

import pytest

from pretrain.config import load_config
from pretrain.optim.schedules import build_schedule
from pretrain.train.batch_schedule import (
    current_global_batch_tokens,
    grad_accum_steps,
    tokens_per_optimizer_step,
)


def test_cosine_endpoints():
    cfg = load_config("1b_repop_v2")
    sched = build_schedule(cfg.schedule, cfg.train, cfg.optim)
    total = cfg.train.total_tokens
    peak = cfg.optim.peak_lr
    min_lr = peak * cfg.schedule.min_lr_frac

    # At step 0 lr is 0 (warmup starts at 0).
    assert sched(0, total) == 0.0
    # At end of warmup, exactly peak.
    warmup_tokens = cfg.schedule.warmup_steps * cfg.train.global_batch_tokens.warmup
    assert math.isclose(sched(warmup_tokens, total), peak, rel_tol=1e-6)
    # At end of cosine cycle, exactly min_lr.
    assert math.isclose(sched(total, total), min_lr, rel_tol=1e-6)


def test_cosine_monotone_after_warmup():
    cfg = load_config("1b_repop_v2")
    sched = build_schedule(cfg.schedule, cfg.train, cfg.optim)
    total = cfg.train.total_tokens
    warmup = cfg.schedule.warmup_steps * cfg.train.global_batch_tokens.warmup

    prev = sched(warmup, total)
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        cur = sched(warmup + int(frac * (total - warmup)), total)
        assert cur < prev + 1e-9
        prev = cur


def test_wsd_constant_then_decay():
    cfg = load_config("1b_repop_v2", overrides=["schedule=wsd"])
    sched = build_schedule(cfg.schedule, cfg.train, cfg.optim)
    total = cfg.train.total_tokens
    peak = cfg.optim.peak_lr

    warmup = cfg.schedule.warmup_steps * cfg.train.global_batch_tokens.warmup
    decay_start = int(cfg.schedule.wsd_decay_start_frac * total)

    # Constant region.
    assert math.isclose(sched(warmup, total), peak, rel_tol=1e-6)
    assert math.isclose(sched(decay_start - 1, total), peak, rel_tol=1e-6)

    # Decay tail.
    end_lr = peak * cfg.schedule.wsd_decay_to_frac
    assert math.isclose(sched(total, total), end_lr, rel_tol=1e-6)


def test_linear_anneal_endpoints():
    # Midtraining anneal (OLMo 2 §4.1): peak up to the branch point, then
    # linear to zero at the end of the (absolute) budget.
    cfg = load_config("1b_midtrain_math")
    sched = build_schedule(cfg.schedule, cfg.train, cfg.optim)
    total = cfg.train.total_tokens
    peak = cfg.optim.peak_lr
    start = cfg.schedule.anneal_start_tokens

    assert cfg.schedule.name == "linear_anneal"
    assert cfg.schedule.min_lr_frac == 0.0
    # Held at peak everywhere before the branch point (warmup_steps=0).
    assert sched(0, total) == peak
    assert sched(start - 1, total) == peak
    assert sched(start, total) == peak
    # Exactly linear across the anneal window.
    assert math.isclose(sched(start + (total - start) // 2, total), peak / 2,
                        rel_tol=1e-6)
    # Anneals to exactly zero at the end of the budget.
    assert sched(total, total) == 0.0


def test_linear_anneal_monotone():
    cfg = load_config("1b_midtrain_math")
    sched = build_schedule(cfg.schedule, cfg.train, cfg.optim)
    total = cfg.train.total_tokens
    start = cfg.schedule.anneal_start_tokens

    prev = sched(start, total)
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
        cur = sched(start + int(frac * (total - start)), total)
        assert cur <= prev + 1e-12
        prev = cur


def test_linear_anneal_cold_start_warmup():
    # anneal_start_tokens=0 degrades to linear warmup -> linear decay, so the
    # schedule is also usable outside a resume.
    cfg = load_config(
        "1b_midtrain_math",
        overrides=[
            "schedule.anneal_start_tokens=0",
            "schedule.warmup_steps=667",
            "train.total_tokens=10_000_000_000",
        ],
    )
    sched = build_schedule(cfg.schedule, cfg.train, cfg.optim)
    total = cfg.train.total_tokens
    peak = cfg.optim.peak_lr
    warmup = cfg.schedule.warmup_steps * cfg.train.global_batch_tokens.warmup

    assert sched(0, total) == 0.0
    assert math.isclose(sched(warmup, total), peak, rel_tol=1e-6)
    assert sched(total, total) == 0.0


def test_linear_anneal_rejects_budget_at_or_below_anchor():
    # total_tokens is ABSOLUTE (pretrain + midtrain); forgetting to raise it
    # past the branch point must fail at build time, not no-op at runtime.
    cfg = load_config(
        "1b_midtrain_math",
        overrides=["train.total_tokens=400_000_000_000"],
    )
    with pytest.raises(ValueError, match="anneal_start_tokens"):
        build_schedule(cfg.schedule, cfg.train, cfg.optim)


def test_batch_phases_match_plan():
    cfg = load_config("1b_repop_v2")
    # 1b_repop_v2 recipe: warmup 3.1M (< 4B), main 4.7M (< 360B), late 9.4M.
    assert current_global_batch_tokens(0, cfg.train) == 3_145_728
    assert current_global_batch_tokens(5_000_000_000, cfg.train) == 4_718_592
    assert current_global_batch_tokens(380_000_000_000, cfg.train) == 9_437_184


def test_grad_accum_at_8x_h100():
    cfg = load_config("1b_repop_v2")
    # Probe at dp_world_size=8 (the recipe's parallelism layout is irrelevant
    # to grad_accum_steps' math; it only depends on the global-batch target
    # and the DP-unique-token degree). 1b_repop_v2: 3.1M/4.7M/9.4M targets over
    # mb=4 * seq=4096 * dp=8 = 131_072 tokens/step -> accum 24/36/72.
    assert grad_accum_steps(0, cfg.train, dp_world_size=8) == 24
    assert grad_accum_steps(5_000_000_000, cfg.train, dp_world_size=8) == 36
    assert grad_accum_steps(380_000_000_000, cfg.train, dp_world_size=8) == 72


def test_grad_accum_scales_with_dp_world_size():
    # At dp_world_size=2 (warmup phase) we need 4× the accum we'd need on 8
    # ranks to reach the same global-batch target (24 * 4 == 96).
    cfg = load_config("1b_repop_v2")
    assert grad_accum_steps(0, cfg.train, dp_world_size=2) == 96


def test_tokens_per_optimizer_step_matches_target():
    cfg = load_config("1b_repop_v2")
    main_phase_consumed = 5_000_000_000
    tokens = tokens_per_optimizer_step(
        main_phase_consumed, cfg.train, dp_world_size=8
    )
    # Should be a multiple of micro_batch * seq_len * dp_world_size that
    # hits at least the main-phase target.
    assert tokens >= cfg.train.global_batch_tokens.main
    assert tokens % (cfg.train.micro_batch_size * cfg.train.seq_len * 8) == 0
