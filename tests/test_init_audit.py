"""Init-audit round-trip: the loop's step-0 (at_init) state hash must be
reproducible by ``audit_replay --from-init`` on a single device.

The loop records a standalone (prev_hash=None) hash of the freshly initialized
model plus its primed optimizer at step 0; the audit rebuilds ``build_model +
init_weights(seed) + build_optimizer + prime_optimizer_state`` and must reproduce
it bit-for-bit. Since repop's trunc_normal init is
device-independent, this holds on any device — exercised here on CPU.

Skipped when ``repop`` isn't importable (model construction goes through repop
kernels). AC-off config so ``parallelize`` is a no-op on CPU (no FQN prefixes),
matching what the loop hashed.
"""

from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("repop")

from pretrain.cli.audit_replay import audit_replay  # noqa: E402
from pretrain.config import load_config  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.model.init import init_weights  # noqa: E402
from pretrain.parallel.parallel_dims import ParallelDims  # noqa: E402
from pretrain.parallel.parallelize_llama3_repop import (  # noqa: E402
    parallelize_llama3_repop,
)
from pretrain.train.state_hash import compute_state_hash  # noqa: E402


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


def _loop_init_hash(cfg) -> str:
    """Reproduce the loop's ``at_init`` computation exactly: build_model ->
    init_weights(seed) -> parallelize(ws=1) -> build_optimizer ->
    prime_optimizer_state -> standalone hash.

    The loop primes zero AdamW state for every param right after
    build_optimizer (train/loop.py) and audit_replay does the same, so the
    step-0 hash covers the primed zero moments, not an empty optimizer."""
    from pretrain.optim.adamw_repop import prime_optimizer_state
    from pretrain.optim.registry import build_optimizer

    model = build_model(cfg.model)
    init_weights(model, seed=cfg.run.seed)
    model = parallelize_llama3_repop(model, cfg, ParallelDims(1, 1, 1, 1))
    optimizer = build_optimizer(model, cfg.optim)
    prime_optimizer_state(optimizer)
    return compute_state_hash(
        model, optimizer=optimizer, include_grads=False, prev_hash=None
    )


def _write_step0(tmp_path, cfg, state_hash: str):
    """Minimal step-0 checkpoint dir: the from-init audit reads only meta.json
    (seed/config/repop_env) and state_hash.txt — no DCP weight load."""
    d = tmp_path / "checkpoints" / "step_000000000"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "seed": cfg.run.seed,
                "config_resolved": cfg.model_dump_json(),
                "repop_env": {},
            }
        )
    )
    (d / "state_hash.txt").write_text(state_hash + "\n")
    return d


def test_from_init_audit_matches_loop_hash(tmp_path):
    cfg = _tiny_cfg()
    init_hash = _loop_init_hash(cfg)  # what the loop records at step 0
    d = _write_step0(tmp_path, cfg, init_hash)
    res = audit_replay(str(d), from_init=True, device="cpu")
    assert res["mode"] == "init"
    assert res["state_hash"] == init_hash
    assert res["match"] is True, f"audit {res['state_hash']} != loop {init_hash}"


def test_from_init_audit_is_deterministic(tmp_path):
    """Two independent inits at the same seed produce the same init hash."""
    cfg = _tiny_cfg()
    assert _loop_init_hash(cfg) == _loop_init_hash(cfg)


def test_from_init_audit_detects_divergent_init(tmp_path):
    """A wrong recorded init hash must fail the audit (not silently pass)."""
    cfg = _tiny_cfg()
    d = _write_step0(tmp_path, cfg, "deadbeef" * 8)
    res = audit_replay(str(d), from_init=True, device="cpu")
    assert res["match"] is False


def test_config_only_init_audit_no_checkpoint(tmp_path, monkeypatch):
    """A config-only init audit (no --checkpoint) regenerates init from the
    config's seed and compares against --expect-hash. The tiny cfg is applied
    via a monkeypatched load_config so audit_replay's own --config-name load
    (config_name="100m_smoke_repop") picks up the same shrunk architecture."""
    import pretrain.cli.audit_replay as ar

    cfg = _tiny_cfg()
    init_hash = _loop_init_hash(cfg)
    monkeypatch.setattr(ar, "load_config", lambda name: _tiny_cfg())

    # Matching expect-hash → pass.
    res = audit_replay(
        None,
        from_init=True,
        until_step=0,
        config_name="100m_smoke_repop",
        device="cpu",
        expect_hash=init_hash,
    )
    assert res["mode"] == "init"
    assert res["state_hash"] == init_hash
    assert res["match"] is True

    # No expect-hash → reports the digest with no comparison.
    res2 = audit_replay(
        None, from_init=True, until_step=0, config_name="100m_smoke_repop", device="cpu"
    )
    assert res2["state_hash"] == init_hash
    assert "match" not in res2


def test_config_only_requires_config_name():
    """No checkpoint and no config-name is a usage error, not a crash."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="config-name"):
        audit_replay(None, from_init=True, until_step=0, device="cpu")


def test_config_only_rejects_replay():
    """Without a checkpoint there's nothing to replay toward; until_step>0 errors."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="only init can be verified"):
        audit_replay(
            None, from_init=True, until_step=10, config_name="x", device="cpu"
        )


def test_from_init_audit_with_real_checkpoint_save(tmp_path):
    """Exercise the real step-0 save path the loop uses: Checkpointer.save with
    an UNSTEPPED but primed optimizer (zero moments for every param, no .step()
    yet), then verify audit --from-init reproduces the hash from the written
    artifact."""
    from pretrain.data.mix_sampler import MixSamplerState
    from pretrain.optim.adamw_repop import prime_optimizer_state
    from pretrain.optim.registry import build_optimizer
    from pretrain.parallel.deterministic_reduce import REPLICATE_REDUCE_ALGO
    from pretrain.train.checkpoint import Checkpointer, CheckpointMeta

    cfg = _tiny_cfg()
    model = build_model(cfg.model)
    init_weights(model, seed=cfg.run.seed)
    model = parallelize_llama3_repop(model, cfg, ParallelDims(1, 1, 1, 1))
    # Unstepped optimizer, primed as the loop primes it at build time.
    optim = build_optimizer(model, cfg.optim)
    prime_optimizer_state(optim)
    init_hash = compute_state_hash(
        model, optimizer=optim, include_grads=False, prev_hash=None
    )

    ckpt = Checkpointer(tmp_path / "checkpoints")
    meta = CheckpointMeta(
        consumed_tokens=0,
        step=0,
        git_sha="",
        config_resolved=cfg.model_dump_json(),
        tokenizer_hash="",
        container_digest="",
        seed=cfg.run.seed,
        dp_world_size=1,
        # The loop stamps every checkpoint of an auditable run with these
        # (train/loop.py _save_checkpoint); audit_replay refuses any other
        # replicate_reduce_algo. CheckpointMeta's defaults describe a legacy
        # nccl run and are not what a cold-start step-0 save carries.
        reduction_mode=cfg.run.reduction_mode,
        replicate_reduce_algo=REPLICATE_REDUCE_ALGO,
    )
    saved = ckpt.save(
        0,
        model,
        optim,
        MixSamplerState(consumed_documents_per_source={}, epoch_per_source={}),
        meta,
        state_hash=init_hash,
    )
    res = audit_replay(str(saved), from_init=True, device="cpu")
    assert res["match"] is True, f"audit {res['state_hash']} != saved {init_hash}"
