"""Implausible topology/position integers in meta.json must be refused early.

meta.json is untrusted input, and the audit's memory and wall-clock scale
linearly with ``dp_world_size`` (one virtual-rank loader per dp rank): a
corrupt or doctored value would exhaust memory long before any other check
fails. The bounds gate runs in the preflight section, before the model build /
DCP load, so these tests only need a meta.json fixture.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("repop")

from pretrain.cli.audit_replay import audit_replay  # noqa: E402
from pretrain.config import load_config  # noqa: E402


def _tiny_cfg():
    cfg = load_config("100m_smoke_repop")
    cfg.model.d_model = 64
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.head_dim = 16
    cfg.model.ffn_intermediate = 64
    cfg.model.n_layers = 2
    cfg.run.seed = 1234
    return cfg


def _write_meta_only_ckpt(tmp_path, cfg, **meta_overrides):
    d = tmp_path / "checkpoints" / "step_000000010"
    d.mkdir(parents=True)
    meta = {
        "seed": cfg.run.seed,
        "step": 10,
        "consumed_tokens": 10 * 1024,
        "chained_hash": "ab" * 32,
        "reduction_mode": "deterministic_allgather",
        "dp_world_size": 2,
        "dp_replicate": 1,
        "dp_shard": 2,
        "config_resolved": cfg.model_dump_json(),
        "repop_env": {},
    }
    meta.update(meta_overrides)
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def test_huge_dp_world_size_is_refused(tmp_path):
    cfg = _tiny_cfg()
    d = _write_meta_only_ckpt(
        tmp_path, cfg, dp_world_size=10**6, dp_replicate=1, dp_shard=10**6
    )
    with pytest.raises(ValueError, match="implausible dp_world_size"):
        audit_replay(str(d), device="cpu")


def test_dp_world_size_just_over_cluster_is_refused(tmp_path):
    """One past the current cluster bound (48) is refused — the boundary the
    reviewer asked to pin to the real cluster size."""
    from pretrain.cli.audit_replay import _MAX_AUDIT_WORLD

    cfg = _tiny_cfg()
    d = _write_meta_only_ckpt(
        tmp_path,
        cfg,
        dp_world_size=_MAX_AUDIT_WORLD + 1,
        dp_replicate=1,
        dp_shard=_MAX_AUDIT_WORLD + 1,
    )
    with pytest.raises(ValueError, match="implausible dp_world_size"):
        audit_replay(str(d), device="cpu")


def test_negative_mesh_factor_is_refused(tmp_path):
    cfg = _tiny_cfg()
    d = _write_meta_only_ckpt(
        tmp_path, cfg, dp_world_size=2, dp_replicate=-1, dp_shard=-2
    )
    with pytest.raises(ValueError, match="implausible mesh"):
        audit_replay(str(d), device="cpu")


def test_mesh_product_mismatch_is_refused(tmp_path):
    cfg = _tiny_cfg()
    d = _write_meta_only_ckpt(
        tmp_path, cfg, dp_world_size=4, dp_replicate=2, dp_shard=3
    )
    with pytest.raises(ValueError, match="mesh mismatch"):
        audit_replay(str(d), device="cpu")


def test_negative_step_is_refused(tmp_path):
    cfg = _tiny_cfg()
    d = _write_meta_only_ckpt(tmp_path, cfg, step=-5)
    with pytest.raises(ValueError, match="negative position"):
        audit_replay(str(d), device="cpu")


def test_negative_consumed_tokens_is_refused(tmp_path):
    cfg = _tiny_cfg()
    d = _write_meta_only_ckpt(tmp_path, cfg, consumed_tokens=-1)
    with pytest.raises(ValueError, match="negative position"):
        audit_replay(str(d), device="cpu")


def test_plausible_mesh_passes_the_gate(tmp_path):
    """A sane mesh must get PAST the bounds gate (it then fails later, at the
    data/DCP stage, because this fixture has no shards or weights). DCP wraps
    its failure in CheckpointException, which derives from BaseException."""
    cfg = _tiny_cfg()
    d = _write_meta_only_ckpt(tmp_path, cfg)
    with pytest.raises(BaseException) as ei:
        audit_replay(str(d), device="cpu")
    msg = str(ei.value)
    assert "implausible" not in msg and "mesh mismatch" not in msg
    assert "negative position" not in msg
