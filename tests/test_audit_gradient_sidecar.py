"""Contract tests for the audit-handoff gradient sidecar.

The chained-audit handoff (``--save-checkpoint-dir``) exports the target step's
final gradients as ``gradients.safetensors`` alongside the checkpoint, so a
recipient can recompute the logged v3 state hash without replaying the preceding
interval. These tests round-trip the full handoff (checkpoint + sidecar) through
the production export/load helpers and assert the v3 hash is preserved bitwise,
including the ``grad is None`` vs a zero-valued gradient distinction.

Runs on CPU with a plain ``nn.Module`` + ``torch.optim.AdamW``, no distributed
init. The full-handoff round-trip requires repop installed: the actual production
v3 hash helpers import the parallel package, which imports the native runtime.
Run that regression in the audit-kit/training environment; ``uv sync --extra dev``
alone is not sufficient. The serialization/publication tests do not need repop. The
sidecar is written directly into the unpublished checkpoint directory after the
hash-match gate, then published with the checkpoint; ``_load_sidecar`` is the test-side counterpart of
the verify-side loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pretrain.cli.audit_replay import (
    _GRAD_SIDECAR_FILENAME,
    _save_chained_audit_checkpoint,
    _write_gradient_sidecar,
)
from pretrain.data.global_stream import GlobalStreamState
from pretrain.train.checkpoint import CheckpointMeta, Checkpointer
from pretrain.train.spike_protocol import SpikeProtocol
from pretrain.train.state_hash import (
    RunningBatchHasher,
    audit_shard_state_digest,
    combine_batch_digests,
    finalize_state_hash,
)


class _GradModel(nn.Module):
    """Three layers with nonzero, zero and None gradients at the hash point.

    ``c`` is not in ``forward``, so it never accumulates a gradient during the
    target step (its ``grad`` stays None). The test primes ``c`` through the
    optimizer once so the saved checkpoint carries a *complete* optimizer state
    — a partial state (a param with no moments) is an unsupported checkpoint
    shape, not something the export path special-cases.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(8, 8)  # nonzero grads
        self.b = nn.Linear(8, 8)  # zeroed grads
        self.c = nn.Linear(4, 4)  # grad None (not in forward)

    def forward(self, x):
        return self.a(x) + self.b(x)


def _meta_obj(N: int, dp_shard: int) -> dict:
    """A meta.json dict as an auditable run's loop would have written it."""
    import dataclasses

    meta = CheckpointMeta(
        consumed_tokens=1000,
        step=100,
        git_sha="deadbeef",
        config_resolved='{"model":"tiny"}',
        tokenizer_hash="tok",
        container_digest="img",
        chained_hash="a" * 32,
        reduction_mode="deterministic_allgather",
        dp_world_size=N,
        dp_replicate=N // dp_shard,
        dp_shard=dp_shard,
        replicate_reduce_algo="recursive_doubling",
        grad_norm_algo="deterministic_per_tensor_sos_v1",
        clip_algo="global",
        seed=42,
        windows_emitted=500,
        repop_env={"REPOP_EXECUTION_MODE": "cross_device_reproducible"},
    )
    return json.loads(json.dumps(dataclasses.asdict(meta)))


def _capture_grads(model) -> dict[str, torch.Tensor | None]:
    """Mirror the audit's hash-point capture: name → detached CPU grad | None."""
    return {
        n: (None if p.grad is None else p.grad.detach().contiguous().cpu())
        for n, p in model.named_parameters()
    }


def _load_sidecar(path: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Read the exported tensors and explicit None-gradient names."""
    import safetensors

    with safetensors.safe_open(str(path), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        md = f.metadata() or {}
        none_names = (
            json.loads(md["none_grad_names"]) if md.get("none_grad_names") else []
        )
    return tensors, none_names


@pytest.mark.parametrize("N,dp_shard", [(1, 1), (4, 2), (6, 3)])
def test_gradient_sidecar_full_handoff_roundtrip(tmp_path, N, dp_shard):
    model = _GradModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Prime the optimizer so the saved state is complete: ``c`` must have
    # moments even though it receives no gradient in the target step below.
    (model(torch.randn(4, 8)).sum() + model.c(torch.randn(4, 4)).sum()).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    # Live gradients: nonzero on a, explicit zeros on b, None on c.
    model(torch.randn(4, 8)).sum().backward()
    for p in model.b.parameters():
        p.grad.zero_()
    opt.step()  # real moment state; grads stay live (AdamW does not clear them)

    # Known preceding chain commitment + per-rank batch-hasher state.
    prev_commitment = "deadbeef" * 8
    hashers = [RunningBatchHasher() for _ in range(N)]
    for r, h in enumerate(hashers):
        h.update({"input_ids": torch.full((1, 4), r + 1), "labels": torch.zeros(1, 4)})
    bd = combine_batch_digests([h.local_digest() for h in hashers])

    # Pre-export target hash (v3, gradients live).
    target = finalize_state_hash(
        prev_hash=prev_commitment,
        shard_state_digest=audit_shard_state_digest(
            model, dp_shard, N, optimizer=opt, include_grads=True
        ),
        optimizer=opt,
        batch_digest=bd,
    )

    # Production export path: capture the gradients while live, clear the
    # model's, save the handoff checkpoint, then write the sidecar into it.
    held = _capture_grads(model)
    opt.zero_grad(set_to_none=True)
    saved = _save_chained_audit_checkpoint(
        save_dir=str(tmp_path / "handoff"),
        step=101,
        consumed=1010,
        digest=target,
        chained_hash_meta=target,
        meta_obj=_meta_obj(N, dp_shard),
        stream_state=GlobalStreamState(
            consumed_documents_per_source={"web": 123},
            epoch_per_source={"web": 0},
            windows_emitted=512,
        ),
        model=model,
        optimizer=opt,
        spike_state=SpikeProtocol(
            threshold=100.0,
            skips_in_window_to_halt=3,
            halt_window_steps=50,
            skip_steps_on_spike=2,
            start_step=0,
        ).state_dict(),
        batch_hashers=hashers,
        batch_digest=bd,
        N=N,
        gradients=held,
    )
    assert held == {}  # host gradients are released before the DCP save
    sidecar = saved / _GRAD_SIDECAR_FILENAME
    assert sidecar.exists()
    assert (saved / "_COMPLETE").exists()
    assert Checkpointer(tmp_path / "handoff").latest() == saved

    # Load the actual exported checkpoint, not a second serialization of the
    # original live objects. Gradients must come exclusively from the sidecar.
    model2 = _GradModel()
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(10)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.9)
    _, loaded_meta, _ = Checkpointer(tmp_path / "handoff").load(saved, model2, opt2)
    N, dp_shard = loaded_meta.dp_world_size, loaded_meta.dp_shard
    assert model2.c.weight in opt2.state  # complete optimizer state round-trips

    # Reattach gradients with dtype/shape/None semantics preserved.
    tensors, none_names = _load_sidecar(sidecar)
    param_names = {n for n, _ in model2.named_parameters()}
    assert set(tensors) | set(none_names) == param_names
    assert set(tensors) & set(none_names) == set()
    assert set(tensors) == {"a.weight", "a.bias", "b.weight", "b.bias"}
    assert set(none_names) == {"c.weight", "c.bias"}
    for n, p in model2.named_parameters():
        if n in tensors:
            p.grad = tensors[n]
        else:
            p.grad = None

    # Restore the per-rank batch digests from the handoff.
    hashers2 = [
        RunningBatchHasher(
            prev_digest=(saved / f"batch_hasher.rank_{r}.bin").read_bytes()
        )
        for r in range(N)
    ]
    bd2 = combine_batch_digests([h.local_digest() for h in hashers2])

    # Reconstruct the v3 hash from the reloaded state: exact equality.
    rebuilt = finalize_state_hash(
        prev_hash=prev_commitment,
        shard_state_digest=audit_shard_state_digest(
            model2, dp_shard, N, optimizer=opt2, include_grads=True
        ),
        optimizer=opt2,
        batch_digest=bd2,
    )
    assert rebuilt == target == (saved / "state_hash.txt").read_text().strip()
    assert all(p.grad is None for p in model.parameters())

    # None is not interchangeable with a zero tensor in the commitment.
    model2.c.weight.grad = torch.zeros_like(model2.c.weight)
    assert finalize_state_hash(
        prev_hash=prev_commitment,
        shard_state_digest=audit_shard_state_digest(
            model2, dp_shard, N, optimizer=opt2, include_grads=True
        ),
        optimizer=opt2, batch_digest=bd2,
    ) != target
    model2.c.weight.grad = None

    # Mutating a restored gradient must change the digest.
    tensors["a.weight"].view(-1)[0] += 1.0
    mutated = finalize_state_hash(
        prev_hash=prev_commitment,
        shard_state_digest=audit_shard_state_digest(
            model2, dp_shard, N, optimizer=opt2, include_grads=True
        ),
        optimizer=opt2,
        batch_digest=bd2,
    )
    assert mutated != target


def test_gradient_sidecar_bf16_and_none_roundtrip(tmp_path):
    """bf16 gradients (no numpy equivalent) round-trip bit-exactly, and the
    None-vs-tensor split is recorded unambiguously."""
    torch.manual_seed(0)
    m = nn.Module()
    m.lin = nn.Linear(4, 4).to(torch.bfloat16)
    m.unused = nn.Linear(2, 2).to(torch.bfloat16)  # grad stays None
    m.lin.weight.grad = torch.randn(4, 4, dtype=torch.bfloat16)
    m.lin.bias.grad = torch.randn(4, dtype=torch.bfloat16)

    p = tmp_path / "g.safetensors"
    _write_gradient_sidecar(_capture_grads(m), p)
    tensors, none_names = _load_sidecar(p)

    assert set(tensors) == {"lin.weight", "lin.bias"}
    assert set(none_names) == {"unused.weight", "unused.bias"}
    assert tensors["lin.weight"].dtype == torch.bfloat16
    assert tensors["lin.weight"].shape == (4, 4)
    assert torch.equal(tensors["lin.weight"], m.lin.weight.grad.detach().cpu())
    assert torch.equal(tensors["lin.bias"], m.lin.bias.grad.detach().cpu())


def _save_tiny_handoff(root, gradients=None):
    model = _GradModel()
    return _save_chained_audit_checkpoint(
        save_dir=str(root), step=101, consumed=1010, digest="b" * 64,
        chained_hash_meta="b" * 64, meta_obj=_meta_obj(1, 1),
        stream_state=GlobalStreamState(
            consumed_documents_per_source={"web": 1},
            epoch_per_source={"web": 0}, windows_emitted=1,
        ),
        model=model, optimizer=torch.optim.AdamW(model.parameters()),
        spike_state={}, batch_hashers=None, batch_digest=None, N=1,
        gradients=gradients,
    )


def test_existing_handoff_is_not_overwritten(tmp_path):
    root = tmp_path / "handoff"
    saved = _save_tiny_handoff(root, _capture_grads(_GradModel()))
    original = (saved / _GRAD_SIDECAR_FILENAME).read_bytes()
    # A re-run into the same dir refuses rather than silently overwriting, and
    # leaves the existing handoff (checkpoint + sidecar) untouched.
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _save_tiny_handoff(root)
    assert (saved / _GRAD_SIDECAR_FILENAME).read_bytes() == original
    assert Checkpointer(root).latest() == saved


def test_sidecar_written_and_released_before_checkpoint_save(tmp_path, monkeypatch):
    root = tmp_path / "handoff"
    model = _GradModel()
    model(torch.ones(1, 8)).sum().backward()
    held = _capture_grads(model)
    original_save = Checkpointer.save
    calls = []

    def inspect_save(checkpointer, step, *args, **kwargs):
        # The only sidecar is already on the destination filesystem. Its host
        # tensors have been released before DCP starts allocating save buffers.
        sidecar = checkpointer.root / f"step_{step:09d}" / _GRAD_SIDECAR_FILENAME
        assert sidecar.is_file()
        assert held == {}
        assert sidecar.stat().st_dev == root.stat().st_dev
        assert Checkpointer(root).latest() is None
        saved = original_save(checkpointer, step, *args, **kwargs)
        assert (saved / "_COMPLETE").exists()
        assert Checkpointer(root).latest() is None
        calls.append(saved)
        return saved

    monkeypatch.setattr(Checkpointer, "save", inspect_save)
    saved = _save_tiny_handoff(root, held)
    assert len(calls) == 1
    assert Checkpointer(root).latest() == saved
    assert sorted(root.iterdir()) == [saved]
    tensors, none_names = _load_sidecar(saved / _GRAD_SIDECAR_FILENAME)
    assert torch.equal(tensors["a.weight"], model.a.weight.grad)
    assert set(none_names) == {"c.weight", "c.bias"}


@pytest.mark.parametrize("failure_phase", ["sidecar", "checkpoint", "publish"])
@pytest.mark.parametrize("error", [OSError("disk full"), KeyboardInterrupt()])
def test_failed_handoff_never_publishes_complete(tmp_path, monkeypatch, failure_phase, error):
    import safetensors.torch as sft

    root = tmp_path / "handoff"
    held = _capture_grads(_GradModel())
    original_save = Checkpointer.save
    original_rename = Path.rename

    def fail_sidecar(tensors, filename, **kwargs):
        Path(filename).write_bytes(b"partial safetensors")
        assert Checkpointer(root).latest() is None
        raise error

    def fail_checkpoint(checkpointer, *args, **kwargs):
        saved = original_save(checkpointer, *args, **kwargs)
        assert (saved / "_COMPLETE").exists()
        assert Checkpointer(root).latest() is None
        raise error

    def fail_publish(path, destination):
        if destination == root / "step_000000101":
            assert (path / "_COMPLETE").exists()
            assert (path / _GRAD_SIDECAR_FILENAME).exists()
            assert Checkpointer(root).latest() is None
            raise error
        return original_rename(path, destination)

    if failure_phase == "sidecar":
        monkeypatch.setattr(sft, "save_file", fail_sidecar)
    elif failure_phase == "checkpoint":
        monkeypatch.setattr(Checkpointer, "save", fail_checkpoint)
    else:
        monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(type(error)):
        _save_tiny_handoff(root, held)
    assert Checkpointer(root).latest() is None
    assert list(root.iterdir()) == []
