"""Config load + validation."""

from __future__ import annotations

import pytest

from pretrain.config import SpikeConfig, load_config
from pretrain.config.load import _default_configs_dir

# Configs the loader is expected to reject, with the reason. Anything in
# configs/train/ that is not listed here must load cleanly.
KNOWN_BROKEN_TRAIN_CONFIGS: set[str] = set()


def test_load_all_train_configs():
    names = sorted(p.stem for p in (_default_configs_dir() / "train").glob("*.yaml"))
    assert names, "no train configs found"
    stale_skips = KNOWN_BROKEN_TRAIN_CONFIGS - set(names)
    assert not stale_skips, f"skip-list entries no longer exist: {stale_skips}"
    for name in names:
        if name in KNOWN_BROKEN_TRAIN_CONFIGS:
            with pytest.raises(Exception):
                load_config(name)
            continue
        cfg = load_config(name)
        assert cfg.model.name
        assert cfg.train.total_tokens > 0


def test_typo_in_lr_rejected():
    """Pydantic should reject string-where-float typos."""
    with pytest.raises(Exception):
        load_config("100m_smoke_repop", overrides=["optim.peak_lr=zero"])


def test_extra_field_rejected():
    """Schema is ``extra=forbid``; unknown keys should error."""
    with pytest.raises(Exception):
        load_config("100m_smoke_repop", overrides=["+optim.bogus_field=1"])


def test_d_model_consistency_validated():
    """The schema validator catches d_model != n_heads * head_dim."""
    with pytest.raises(Exception):
        load_config("100m_smoke_repop", overrides=["model.head_dim=63"])


def test_vocab_size_multiple_of_128():
    with pytest.raises(Exception):
        load_config("100m_smoke_repop", overrides=["model.vocab_size=128255"])


def test_data_weights_normalised():
    cfg = load_config("1b_repop_v2")
    weights = cfg.data.normalised_weights()
    assert abs(sum(weights) - 1.0) < 1e-9
    # recipe_v1_proportional: dclm_baseline dominates at 0.6671.
    assert weights[0] == max(weights)
    assert abs(weights[0] - 0.6671) < 1e-9


def test_seq_len_mismatch_rejected():
    """data.seq_len and train.seq_len must agree (loader vs token accountant)."""
    with pytest.raises(Exception):
        load_config("100m_smoke_repop", overrides=["train.seq_len=2048"])


def test_seq_len_exceeds_rope_table_rejected():
    """data.seq_len > model.max_seq_len_pretrain breaks the pre-built RoPE table."""
    with pytest.raises(Exception):
        load_config(
            "100m_smoke_repop",
            overrides=["data.seq_len=4096", "train.seq_len=4096"],
        )


def test_unreachable_spike_halt_rejected():
    """A spike block whose halt can never fire must not load.

    50/5/50 needs 4 gaps of 50 steps between fresh events — 200 steps — inside
    a 50-step window. The run would skip forever instead of paging.
    """
    with pytest.raises(Exception, match="mathematically impossible"):
        load_config(
            "100m_smoke_repop",
            overrides=[
                "train.spike.skip_steps_on_spike=50",
                "train.spike.halt_window_steps=50",
                "train.spike.skips_in_window_to_halt=5",
            ],
        )


def test_spike_halt_window_boundary_accepted():
    """Exactly-fits is reachable: (5-1)*25 == 100, so 100 must load."""
    cfg = load_config(
        "100m_smoke_repop",
        overrides=[
            "train.spike.skip_steps_on_spike=25",
            "train.spike.halt_window_steps=100",
            "train.spike.skips_in_window_to_halt=5",
        ],
    )
    assert cfg.train.spike.halt_window_steps == 100


def test_spike_defaults_can_halt():
    """A bare SpikeConfig() must satisfy its own validator.

    Otherwise any config with a partial ``spike:`` block fails at load.
    """
    spike = SpikeConfig()
    gaps = (spike.skips_in_window_to_halt - 1) * spike.skip_steps_on_spike
    assert gaps <= spike.halt_window_steps
