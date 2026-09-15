"""Spike detect / skip / halt protocol."""

from __future__ import annotations

import math

import pytest

from pretrain.train.spike_protocol import Halt, SpikeProtocol


def test_threshold_skip():
    p = SpikeProtocol(threshold=5.0, skips_in_window_to_halt=10, halt_window_steps=100)
    assert p.should_skip(grad_norm=4.0, loss=1.0) is False
    assert p.should_skip(grad_norm=5.5, loss=1.0) is True


def test_nan_skips():
    p = SpikeProtocol(threshold=5.0, skips_in_window_to_halt=10, halt_window_steps=100)
    assert p.should_skip(grad_norm=1.0, loss=math.nan) is True
    assert p.should_skip(grad_norm=math.inf, loss=1.0) is True


def test_halt_after_window():
    p = SpikeProtocol(threshold=5.0, skips_in_window_to_halt=3, halt_window_steps=10)
    p.record_skip(0)
    p.record_skip(2)
    with pytest.raises(Halt):
        p.record_skip(5)


def test_old_skips_drop_out_of_window():
    """Skips that fall outside the rolling window must not count toward halt."""
    p = SpikeProtocol(threshold=5.0, skips_in_window_to_halt=3, halt_window_steps=10)
    p.record_skip(0)
    p.record_skip(2)
    # Step 100: window is [90, 100], so steps 0 and 2 are out. Only this
    # one skip is in-window — far from the halt threshold of 3.
    p.record_skip(100)
    # State should reflect 1 in-window skip even though total is 3.
    assert len(p._recent_skip_steps) == 1
    # Two more in-window skips do trigger halt.
    p.record_skip(101)
    with pytest.raises(Halt):
        p.record_skip(102)


def test_start_step_suppresses_threshold_during_warmup():
    """Pre-warmup grad spikes don't trip the threshold check."""
    p = SpikeProtocol(
        threshold=5.0, skips_in_window_to_halt=10, halt_window_steps=100, start_step=2000,
    )
    assert p.should_skip(grad_norm=50.0, loss=1.0, step=10) is False
    assert p.should_skip(grad_norm=50.0, loss=1.0, step=2000) is True


def test_start_step_does_not_suppress_nan():
    """NaN/Inf is always pathological — start_step must not gate it."""
    p = SpikeProtocol(
        threshold=5.0, skips_in_window_to_halt=10, halt_window_steps=100, start_step=2000,
    )
    assert p.should_skip(grad_norm=1.0, loss=math.nan, step=10) is True
    assert p.should_skip(grad_norm=math.inf, loss=1.0, step=10) is True


def test_cooldown_skips_subsequent_steps_after_spike():
    """A fresh spike opens a cooldown window: the next K-1 steps are
    force-skipped even with a healthy grad norm. After the cooldown
    ends, normal threshold checks resume.
    """
    p = SpikeProtocol(
        threshold=5.0,
        skip_steps_on_spike=50,
        skips_in_window_to_halt=10,
        halt_window_steps=200,
    )
    # Fresh spike at step 100.
    assert p.should_skip(grad_norm=10.0, loss=1.0, step=100) is True
    p.record_skip(100)

    # Steps 101..149 (49 more) are still inside the cooldown — even
    # healthy grad norms must be force-skipped.
    for s in range(101, 150):
        assert p.should_skip(grad_norm=0.1, loss=1.0, step=s) is True
        p.record_skip(s)

    # Cooldown is exhausted: a healthy step is no longer skipped.
    assert p.should_skip(grad_norm=0.1, loss=1.0, step=150) is False


def test_cooldown_skips_do_not_count_toward_halt():
    """Halt counts spike *events*, not the forced-skip cascade. A single
    spike with a 50-step cooldown must not on its own trip the halt.
    """
    p = SpikeProtocol(
        threshold=5.0,
        skip_steps_on_spike=50,
        skips_in_window_to_halt=3,
        halt_window_steps=200,
    )
    # One fresh spike + 49 cooldown skips. Halt threshold is 3 events.
    assert p.should_skip(grad_norm=10.0, loss=1.0, step=10) is True
    p.record_skip(10)
    for s in range(11, 60):
        p.record_skip(s)
    # Only the original event counts toward the halt window.
    assert len(p._recent_skip_steps) == 1


def test_cooldown_does_not_swallow_nan():
    """NaN inside a cooldown still surfaces as a skip (it always does)."""
    p = SpikeProtocol(
        threshold=5.0,
        skip_steps_on_spike=50,
        skips_in_window_to_halt=10,
        halt_window_steps=200,
    )
    p.record_skip(10)        # opens cooldown
    assert p.should_skip(grad_norm=math.nan, loss=1.0, step=11) is True


def test_state_dict_roundtrip_preserves_cooldown_and_window():
    """A spike whose cooldown straddles a checkpoint must keep skipping
    on resume; the halt-window deque and counters must also survive.
    """
    src = SpikeProtocol(
        threshold=5.0,
        skip_steps_on_spike=20,         # 1 fresh event + 19 cooldown skips
        skips_in_window_to_halt=3,
        halt_window_steps=1000,
    )
    # First fresh event opens a 19-step cooldown.
    src.record_skip(10)
    # Halfway through the cooldown — this is the case we care about:
    # checkpoint mid-cooldown.
    for s in range(11, 20):
        src.record_skip(s)
    assert src._cooldown_remaining == 10
    assert list(src._recent_skip_steps) == [10]

    sd = src.state_dict()
    dst = SpikeProtocol(
        threshold=5.0,
        skip_steps_on_spike=20,
        skips_in_window_to_halt=3,
        halt_window_steps=1000,
    )
    dst.load_state_dict(sd)

    # Mid-cooldown remaining count copies through — without this the
    # post-resume run would prematurely re-step the optimizer.
    assert dst._cooldown_remaining == 10
    # Fresh-event deque copies through.
    assert list(dst._recent_skip_steps) == [10]
    # Counters survive so metrics stay continuous.
    assert dst.state.skipped_steps == src.state.skipped_steps == 10
    assert dst.state.last_skip_step == src.state.last_skip_step == 19

    # Behavioural check: dst still treats subsequent steps as cooldown
    # (forces a skip even when grad_norm is healthy).
    assert dst.should_skip(grad_norm=0.1, loss=1.0, step=20) is True


def _halt_step_of_diverged_run(p: SpikeProtocol, max_steps: int = 5000):
    """Run a permanently-diverged loop through ``p``; return the halt step.

    Every step's grad norm is over threshold, i.e. the worst case the halt
    exists to page on. Returns ``None`` if the protocol never halts.
    """
    for step in range(max_steps):
        if p.should_skip(grad_norm=1.0e9, loss=1.0, step=step):
            try:
                p.record_skip(step)
            except Halt:
                return step
    return None


def test_unreachable_ratio_never_halts_on_a_diverged_run():
    """The config-schema predicate is not arithmetic pedantry: 50/5/50 lets a
    permanently-diverged run skip forever without ever paging on-call.
    """
    p = SpikeProtocol(
        threshold=5.0,
        skip_steps_on_spike=50,
        skips_in_window_to_halt=5,
        halt_window_steps=50,
    )
    assert _halt_step_of_diverged_run(p) is None
    # Every one of the 5000 steps was skipped — the optimizer never stepped.
    assert p.state.skipped_steps == 5000
    # The deque never gets past 2 fresh events, against the 5 required.
    assert len(p._recent_skip_steps) <= 2


def test_schema_default_spike_ratio_halts_on_a_diverged_run():
    """The shipped SpikeConfig defaults must actually be able to halt."""
    from pretrain.config import SpikeConfig

    cfg = SpikeConfig()
    p = SpikeProtocol(
        threshold=cfg.grad_norm_threshold,
        skip_steps_on_spike=cfg.skip_steps_on_spike,
        skips_in_window_to_halt=cfg.skips_in_window_to_halt,
        halt_window_steps=cfg.halt_window_steps,
    )
    halt_step = _halt_step_of_diverged_run(p)
    assert halt_step is not None
    # (skips_in_window_to_halt - 1) fresh events spaced skip_steps_on_spike
    # apart, so the halt lands on a bounded, predictable step.
    expected = (cfg.skips_in_window_to_halt - 1) * cfg.skip_steps_on_spike
    assert halt_step == expected
    assert p.state.halted is True
