"""Checkpoint round-trip for the canonical-stream (auditable) path (Part D).

Single-process / CPU, no repop: a plain model exercises Checkpointer's
torch.save fallback. Verifies that a GlobalStreamState is written once as
global_stream.json (not per-rank), restored bit-exact on load, and that the
run-descriptor fields on CheckpointMeta survive the round-trip.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pretrain.data.global_stream import GlobalStreamState
from pretrain.train.checkpoint import Checkpointer, CheckpointMeta


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16)

    def forward(self, x):
        return self.lin(x)


def _stepped_model():
    m = _Tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    (m(torch.randn(4, 16)) ** 2).sum().backward()
    opt.step()
    opt.zero_grad()
    return m, opt


def test_global_stream_checkpoint_roundtrip(tmp_path):
    m, opt = _stepped_model()
    ckpt = Checkpointer(tmp_path / "ckpts")
    gss = GlobalStreamState(
        consumed_documents_per_source={"a": 123, "b": 45},
        epoch_per_source={"a": 0, "b": 1},
        mix_rng_state={"bit_generator": "PCG64"},
        carry_over=[7, 8, 9],
        windows_emitted=256,
    )
    meta = CheckpointMeta(
        consumed_tokens=999, step=42, git_sha="abc", config_resolved="{}",
        tokenizer_hash="t", container_digest="c",
        reduction_mode="deterministic_allgather", dp_world_size=8, seed=42,
        torch_version=torch.__version__, numpy_version="1.26", windows_emitted=256,
    )
    saved = ckpt.save(42, m, opt, None, meta, global_stream_state=gss)

    # One global file, no per-rank sampler files.
    assert (saved / "global_stream.json").exists()
    assert not (saved / "sampler.rank_0.json").exists()

    state, meta2, _ = ckpt.load(saved, m, opt)
    assert isinstance(state, GlobalStreamState)
    assert state.consumed_documents_per_source == {"a": 123, "b": 45}
    assert state.epoch_per_source == {"a": 0, "b": 1}
    assert state.carry_over == [7, 8, 9]
    assert state.windows_emitted == 256
    # Run-descriptor survives.
    assert meta2.reduction_mode == "deterministic_allgather"
    assert meta2.dp_world_size == 8
    assert meta2.windows_emitted == 256


def test_legacy_meta_without_descriptor_still_loads(tmp_path):
    """Old checkpoints (no descriptor fields) load with benign defaults."""
    import dataclasses
    import json

    d = tmp_path / "ckpts" / "step_000000001"
    d.mkdir(parents=True)
    old = {
        "consumed_tokens": 1, "step": 1, "git_sha": "x", "config_resolved": "{}",
        "tokenizer_hash": "t", "container_digest": "c", "restart_count": 0,
        "chained_hash": None,
    }
    (d / "meta.json").write_text(json.dumps(old))
    meta = CheckpointMeta(**json.loads((d / "meta.json").read_text()))
    assert meta.reduction_mode == "nccl"  # default
    assert meta.dp_world_size == 0
    # round-trips back out
    assert "reduction_mode" in dataclasses.asdict(meta)
