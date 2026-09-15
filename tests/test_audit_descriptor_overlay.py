"""Resume-boundary descriptor overlay for audit_replay.

A mid-run fork that changes a segment-scoped descriptor field (repop_env,
config, clip algorithm) splits the
audit chain: the boundary interval's starting STATE lives in the old segment's
last checkpoint while its compute was shaped by the new segment's descriptor.
``_overlay_descriptor_meta`` stitches that one interval: segment-scoped keys
come from the descriptor checkpoint's meta, position keys (step,
consumed_tokens, windows_emitted, chained_hash) and the run-scoped seed stay
with the starting checkpoint.
"""

from __future__ import annotations

import pytest

from pretrain.cli.audit_replay import _SEGMENT_META_KEYS, _overlay_descriptor_meta


def _old_meta() -> dict:
    """The pre-fork segment's meta: no re-warm anchor, Hadamard on."""
    return {
        "step": 50200,
        "consumed_tokens": 234_872_635_392,
        "windows_emitted": 111,
        "chained_hash": "aaaa",
        "seed": 42,
        "config_resolved": '{"old": true}',
        "repop_env": {"REPOP_LSQ_STE_BWD_HADAMARD": "1"},
        "reduction_mode": "deterministic_allgather",
        "dp_world_size": 48,
        "dp_replicate": 6,
        "dp_shard": 8,
        "replicate_reduce_algo": "recursive_doubling",
        "clip_algo": "global",
    }


def _new_meta() -> dict:
    """The forked segment's first checkpoint: new env, position advanced."""
    m = _old_meta()
    m.update(
        step=50300,
        consumed_tokens=235_000_000_000,
        windows_emitted=222,
        chained_hash="bbbb",
        config_resolved='{"old": false}',
        repop_env={"REPOP_LSQ_STE_BWD_HADAMARD": "0"},
        rewarm_anchor_tokens=234_872_635_392,
    )
    return m


def test_overlay_takes_segment_keys_and_keeps_position_keys():
    meta = _old_meta()
    changed = _overlay_descriptor_meta(meta, _new_meta())

    # Segment-scoped: descriptor wins — including the re-warm anchor, which is
    # a token count but describes the SEGMENT's fork point (the Option-A
    # boundary-audit regression: without it the first post-fork step replays
    # at full LR instead of the live run's lr = 0).
    assert meta["repop_env"] == {"REPOP_LSQ_STE_BWD_HADAMARD": "0"}
    assert meta["config_resolved"] == '{"old": false}'
    assert meta["rewarm_anchor_tokens"] == 234_872_635_392
    assert set(changed) == {"config_resolved", "repop_env", "rewarm_anchor_tokens"}

    # Position / chain / run-scoped: starting checkpoint wins.
    assert meta["step"] == 50200
    assert meta["consumed_tokens"] == 234_872_635_392
    assert meta["windows_emitted"] == 111
    assert meta["chained_hash"] == "aaaa"
    assert meta["seed"] == 42


def test_overlay_removes_keys_absent_from_descriptor():
    """Auditing TOWARD an older segment (descriptor predates the field): the
    key must be REMOVED so the legacy default applies — rewarm_anchor_tokens
    absent ⇒ -1 ⇒ no re-warm ramp, byte-exact."""
    meta = _old_meta()
    meta["rewarm_anchor_tokens"] = 234_872_635_392  # start meta HAS the field
    desc = _old_meta()  # descriptor does not
    changed = _overlay_descriptor_meta(meta, desc)
    assert "rewarm_anchor_tokens" not in meta
    assert "rewarm_anchor_tokens" in changed


def test_overlay_identical_metas_is_a_noop():
    meta = _old_meta()
    before = dict(meta)
    assert _overlay_descriptor_meta(meta, _old_meta()) == []
    assert meta == before


def test_overlay_rejects_seed_mismatch():
    desc = _new_meta()
    desc["seed"] = 43
    with pytest.raises(ValueError, match="different run"):
        _overlay_descriptor_meta(_old_meta(), desc)


def test_segment_keys_cover_the_clipper_and_env():
    """Guard the key list against accidental trimming: every field the audit
    uses to reconstruct the clipper/kernels must be segment-scoped, and no
    position field may ever be."""
    for k in (
        "repop_env",
        "config_resolved",
        "clip_algo",
        "grad_norm_algo",
        "replicate_reduce_algo",
        "reduction_mode",
        "rewarm_anchor_tokens",
    ):
        assert k in _SEGMENT_META_KEYS
    for k in ("step", "consumed_tokens", "windows_emitted", "chained_hash", "seed"):
        assert k not in _SEGMENT_META_KEYS
