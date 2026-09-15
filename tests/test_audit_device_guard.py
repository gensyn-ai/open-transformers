"""``--device cuda/mps`` must hard-error when the backend is unavailable.

The old behaviour silently downgraded to CPU, which is bitwise-equivalent but
intractable at real model sizes (a 1.6B replay that takes ~12 h on MPS never
finishes on CPU) — the audit just looked hung. Backend availability is
monkeypatched so these tests are deterministic on any host.
"""

from __future__ import annotations

import json

import pytest
import torch

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


def _write_meta_only_ckpt(tmp_path, cfg):
    d = tmp_path / "checkpoints" / "step_000000010"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
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
        )
    )
    return d


def test_cuda_unavailable_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = _write_meta_only_ckpt(tmp_path, _tiny_cfg())
    with pytest.raises(RuntimeError, match="cuda requested but CUDA is not"):
        audit_replay(str(d), device="cuda")


def test_mps_unavailable_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    d = _write_meta_only_ckpt(tmp_path, _tiny_cfg())
    with pytest.raises(RuntimeError, match="mps requested but MPS is not"):
        audit_replay(str(d), device="mps")


def test_indexed_cuda_string_is_gated(tmp_path, monkeypatch):
    """An indexed device string (programmatic callers pass "cuda:0") must hit
    the same guard — the old bare-string equality missed it."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = _write_meta_only_ckpt(tmp_path, _tiny_cfg())
    with pytest.raises(RuntimeError, match="CUDA is not"):
        audit_replay(str(d), device="cuda:0")


def test_indexed_mps_string_is_gated(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    d = _write_meta_only_ckpt(tmp_path, _tiny_cfg())
    with pytest.raises(RuntimeError, match="MPS is not"):
        audit_replay(str(d), device="mps:0")


def test_cpu_is_always_accepted(tmp_path):
    """--device cpu passes the gate (the meta-only fixture then fails later,
    at the data/DCP stage — proving the guard is device-specific)."""
    d = _write_meta_only_ckpt(tmp_path, _tiny_cfg())
    with pytest.raises(BaseException) as ei:
        audit_replay(str(d), device="cpu")
    assert "requested but" not in str(ei.value)
