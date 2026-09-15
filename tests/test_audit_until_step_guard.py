"""``audit_replay --until-step`` at or below the start step must be refused.

With ``until_step <= meta.step`` the replay loop runs zero iterations, the
"reproduced" digest degenerates to the start checkpoint's own ``chained_hash``,
and an ``--expect-hash`` pointed at that same checkpoint's ``state_hash.txt``
trivially "matches" — a PASS that verified nothing. The guard runs before the
model build / DCP load, so these tests only need a meta.json fixture.
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


def _write_meta_only_ckpt(tmp_path, cfg, step: int = 10):
    """A meta.json-only checkpoint dir: enough for every preflight validation
    that must fire BEFORE the expensive model build / DCP load."""
    d = tmp_path / "checkpoints" / f"step_{step:09d}"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "seed": cfg.run.seed,
                "step": step,
                "consumed_tokens": step * 1024,
                "chained_hash": "ab" * 32,
                "reduction_mode": "deterministic_allgather",
                "dp_world_size": 2,
                "dp_replicate": 1,
                "dp_shard": 2,
                "config_resolved": cfg.model_dump_json(),
                "repop_env": {},
            }
        )
    )
    return d


@pytest.mark.parametrize("until_step", [10, 9, 0, -1])
def test_until_step_at_or_below_start_is_refused(tmp_path, until_step):
    cfg = _tiny_cfg()
    d = _write_meta_only_ckpt(tmp_path, cfg, step=10)
    with pytest.raises(ValueError, match="not beyond the checkpoint"):
        audit_replay(str(d), until_step=until_step, device="cpu")


def test_from_init_target_below_checkpoint_step_is_allowed(tmp_path):
    """--from-init replays 0 -> target; a target below the checkpoint's step is
    legitimate there (and target 0 is the init-only mode). Guard must not fire.
    This from-init run fails LATER (no recorded init hash to verify against is
    fine — it proceeds; here it stops at data loading), proving the guard let
    it through."""
    cfg = _tiny_cfg()
    d = _write_meta_only_ckpt(tmp_path, cfg, step=10)
    try:
        res = audit_replay(str(d), from_init=True, until_step=0, device="cpu")
    except ValueError as e:
        assert "not beyond the checkpoint" not in str(e)
    else:
        # init-only mode: returns the init digest without replaying.
        assert res["mode"] == "init"
