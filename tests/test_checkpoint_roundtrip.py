"""Save → load round-trip on a tiny CPU model. The DCP path requires
distributed init, so on CPU we exercise the fallback ``torch.save`` path
of ``Checkpointer`` directly.

Skipped automatically when ``repop`` is not importable: model construction
now goes through repop kernels, so dev machines without the runtime built
get SKIP rather than a collection ERROR.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("repop")

from pretrain.config import load_config  # noqa: E402
from pretrain.data.mix_sampler import MixSamplerState  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.model.init import init_weights_seeded  # noqa: E402
from pretrain.train.checkpoint import Checkpointer, CheckpointMeta  # noqa: E402


def test_checkpoint_save_load_cpu(tmp_path):
    cfg = load_config("100m_smoke_repop")
    cfg.model.d_model = 64
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.head_dim = 16
    cfg.model.ffn_intermediate = 64
    cfg.model.n_layers = 2

    model = build_model(cfg.model)
    init_weights_seeded(model, 0)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Run one step so optim has state to save.
    x = torch.randint(0, cfg.model.vocab_size, (1, 32))
    out = model(x)
    out.logits.sum().backward()
    optim.step()
    optim.zero_grad()

    ckpt = Checkpointer(tmp_path / "ckpts")
    sampler_state = MixSamplerState(
        consumed_documents_per_source={"a": 100, "b": 50},
        epoch_per_source={"a": 0, "b": 0},
    )
    meta = CheckpointMeta(
        consumed_tokens=12345,
        step=10,
        git_sha="abcdef",
        config_resolved="{}",
        tokenizer_hash="x",
        container_digest="y",
    )
    saved_dir = ckpt.save(10, model, optim, sampler_state, meta)

    # Capture original weights, mutate the live model, then reload.
    snap = {n: p.clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    sampler_state2, meta2, extras2 = ckpt.load(saved_dir, model, optim)

    for n, p in model.named_parameters():
        assert torch.allclose(p, snap[n]), f"param {n} not restored"
    assert sampler_state2.consumed_documents_per_source == {"a": 100, "b": 50}
    assert meta2.step == 10
    assert meta2.consumed_tokens == 12345


def test_checkpoint_writes_complete_sentinel(tmp_path):
    """Every successful save must drop a ``_COMPLETE`` marker as its
    final action. ``latest()`` filters on this to skip partial saves.
    """
    cfg = load_config("100m_smoke_repop")
    cfg.model.d_model = 64
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.head_dim = 16
    cfg.model.ffn_intermediate = 64
    cfg.model.n_layers = 2

    model = build_model(cfg.model)
    init_weights_seeded(model, 0)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt = Checkpointer(tmp_path / "ckpts")
    sampler_state = MixSamplerState(
        consumed_documents_per_source={"a": 1}, epoch_per_source={"a": 0},
    )
    meta = CheckpointMeta(
        consumed_tokens=1, step=1, git_sha="x",
        config_resolved="{}", tokenizer_hash="x", container_digest="y",
    )
    saved = ckpt.save(1, model, optim, sampler_state, meta)
    assert (saved / "_COMPLETE").exists()


def test_latest_skips_incomplete_dirs(tmp_path):
    """A step dir without ``_COMPLETE`` is treated as if it doesn't exist —
    represents a crashed-mid-write save. ``latest()`` falls back to the
    previous complete step rather than handing back a corrupt path.
    """
    ckpt_root = tmp_path / "c"
    ckpt_root.mkdir()

    # Two complete + one in-flight.
    for s in [1, 2, 3]:
        d = ckpt_root / f"step_{s:09d}"
        d.mkdir()
        if s != 3:
            (d / "_COMPLETE").write_text("")

    ckpt = Checkpointer(ckpt_root)
    latest = ckpt.latest()
    assert latest is not None
    assert latest.name == "step_000000002", (
        f"expected step_2 (last complete), got {latest.name}. step_3 "
        f"lacks _COMPLETE and must be skipped."
    )

    # No complete dirs → None (rather than handing back step_3).
    (ckpt_root / "step_000000001" / "_COMPLETE").unlink()
    (ckpt_root / "step_000000002" / "_COMPLETE").unlink()
    assert ckpt.latest() is None


def test_load_hard_errors_on_missing_per_rank_file(tmp_path):
    """Single-process version of the partial-write hard-error test —
    proves the error message names the actual problem (partial NFS
    write) instead of the removed legacy-format path.
    """
    cfg = load_config("100m_smoke_repop")
    cfg.model.d_model = 64
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.head_dim = 16
    cfg.model.ffn_intermediate = 64
    cfg.model.n_layers = 2

    model = build_model(cfg.model)
    init_weights_seeded(model, 0)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, cfg.model.vocab_size, (1, 32))
    model(x).logits.sum().backward()
    optim.step()
    optim.zero_grad()

    ckpt = Checkpointer(tmp_path / "ckpts")
    sampler_state = MixSamplerState(
        consumed_documents_per_source={"a": 1}, epoch_per_source={"a": 0},
    )
    meta = CheckpointMeta(
        consumed_tokens=1, step=1, git_sha="x",
        config_resolved="{}", tokenizer_hash="x", container_digest="y",
    )
    saved = ckpt.save(1, model, optim, sampler_state, meta)

    # Simulate partial multi-pod NFS write: per-rank sampler file missing.
    (saved / "sampler.rank_0.json").unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        ckpt.load(saved, model, optim)

    msg = str(exc_info.value)
    assert "sampler.rank_0.json" in msg
    assert "partial" in msg.lower() or "incomplete" in msg.lower()
    # Legacy fallback path is gone — message must not name it (it would
    # send operators down a wild goose chase, the original failure mode).
    assert "sampler.consumed.json" not in msg


def test_checkpoint_garbage_collect(tmp_path):
    """``garbage_collect`` keeps the last N + every K-th step."""
    ckpt_root = tmp_path / "c"
    ckpt_root.mkdir()
    # Create dummy step dirs.
    for s in [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 22, 23, 24]:
        (ckpt_root / f"step_{s:09d}").mkdir()

    ckpt = Checkpointer(ckpt_root)
    ckpt.garbage_collect(keep_last=4, keep_every=10)

    surviving = sorted(p.name for p in ckpt_root.iterdir())
    # Should keep the last 4 (21,22,23,24) + every 10th (10, 20).
    expected = sorted(
        f"step_{s:09d}" for s in [10, 20, 21, 22, 23, 24]
    )
    assert surviving == expected
