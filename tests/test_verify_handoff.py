"""Disk-only verification of a hand-off against the published state hash.

Client A's export path (``_save_chained_audit_checkpoint`` + the gradient
sidecar) writes a real hand-off; client B loads it into fresh objects
and reproduces the run's published v3 hash with no forward, no backward and no
data. Every way that reconstruction can be wrong — a mutated weight, moment or
gradient, an explicit zero where the run had ``None``, a missing per-rank batch
chain, the wrong preceding commitment — has to end in a refusal, not a pass.

Runs on CPU with repop installed (the real parallel/hash import path requires
it). The integration test also invokes the public CLI in a fresh process with
a tiny real Llama/repop model, not a replacement model builder or hash stub.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytest.importorskip("repop")

from pretrain.cli.audit_replay import (
    _GRAD_SIDECAR_FILENAME,
    _save_chained_audit_checkpoint,
    _write_gradient_sidecar,
)
from pretrain.cli.verify_handoff import (
    HandoffError,
    attach_gradients,
    read_batch_digest,
    verify_handoff,
    verify_loaded_state,
)
from pretrain.config import load_config
from pretrain.data.global_stream import GlobalStreamState
from pretrain.train.checkpoint import CheckpointMeta, Checkpointer
from pretrain.train.state_hash import (
    RunningBatchHasher,
    audit_shard_state_digest,
    combine_batch_digests,
    finalize_state_hash,
)

PREV = "deadbeef" * 8


class _GradModel(nn.Module):
    """Three layers with nonzero, zero and None gradients at the hash point."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)   # zeroed grads
        self.c = nn.Linear(4, 4)   # grad None (not in forward)

    def forward(self, x):
        return self.a(x) + self.b(x)


def _config_resolved(every_n_steps: int) -> str:
    """A real resolved config, as the loop serialises one into meta.json."""
    cfg = load_config("100m_smoke_repop")
    doc = json.loads(cfg.model_dump_json())
    doc["train"]["state_hash"]["every_n_steps"] = every_n_steps
    return json.dumps(doc)


def _meta_obj(N: int, dp_shard: int, chained: str, *, every_n_steps: int = 1) -> dict:
    meta = CheckpointMeta(
        consumed_tokens=1000,
        step=100,
        git_sha="deadbeef",
        config_resolved=_config_resolved(every_n_steps),
        tokenizer_hash="tok",
        container_digest="img",
        chained_hash=chained,
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


def _publish(tmp_path, N=1, dp_shard=1, step=101, every_n_steps=1):
    """Client A: train a step, hash it, export the complete hand-off.

    Returns ``(saved_dir, published_hash)`` — the digest is what the run's
    ``state_hashes.jsonl`` would carry for this step.
    """
    model = _GradModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # As in the production dense model, all parameters have optimizer moments;
    # c took part previously but has no gradient at the target step.
    (model(torch.randn(4, 8)).sum() + model.c(torch.randn(4, 4)).sum()).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    model(torch.randn(4, 8)).sum().backward()
    for p in model.b.parameters():
        p.grad.zero_()
    opt.step()

    hashers = [RunningBatchHasher() for _ in range(N)]
    for r, h in enumerate(hashers):
        h.update({"input_ids": torch.full((1, 4), r + 1), "labels": torch.zeros(1, 4)})
    bd = combine_batch_digests([h.local_digest() for h in hashers])

    published = finalize_state_hash(
        prev_hash=PREV,
        shard_state_digest=audit_shard_state_digest(
            model, dp_shard, N, optimizer=opt, include_grads=True),
        optimizer=opt, batch_digest=bd,
    )

    held = {n: None if p.grad is None else p.grad.detach().cpu()
            for n, p in model.named_parameters()}
    opt.zero_grad(set_to_none=True)
    saved = _save_chained_audit_checkpoint(
        save_dir=str(tmp_path / "handoff"), step=step, consumed=1010,
        digest=published, chained_hash_meta=published,
        meta_obj=_meta_obj(N, dp_shard, published, every_n_steps=every_n_steps),
        stream_state=GlobalStreamState(
            consumed_documents_per_source={"web": 123},
            epoch_per_source={"web": 0}, windows_emitted=512),
        model=model, optimizer=opt, spike_state={},
        batch_hashers=hashers, batch_digest=bd, N=N,
        gradients=held,
    )
    return saved, published


def _receive(saved: Path):
    """Client B: fresh objects, state loaded only from the downloaded hand-off."""
    model = _GradModel()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(10)
    opt = torch.optim.AdamW(model.parameters(), lr=0.9)
    Checkpointer(saved.parent).load(saved, model, opt)
    return model, opt


def _verify(saved, model, opt, published, *, N=1, dp_shard=1, prev=PREV):
    return verify_loaded_state(
        model, opt, saved, expect_hash=published, prev_hash=prev,
        dp_world_size=N, dp_shard=dp_shard,
        include_grads=True, include_batch=True,
    )


# ── the happy path ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("N,dp_shard", [(1, 1), (4, 2), (6, 3)])
def test_a_downloaded_handoff_reproduces_the_published_hash(tmp_path, N, dp_shard):
    saved, published = _publish(tmp_path, N, dp_shard)
    model, opt = _receive(saved)

    result = _verify(saved, model, opt, published, N=N, dp_shard=dp_shard)

    assert result["verified"]
    assert result["reconstructed_hash"] == published
    assert result["gradients_attached"] == 4          # a.*, b.* — c.* were None
    # The verification gradients are spent, not inherited by the continuation.
    assert all(p.grad is None for p in model.parameters())


def test_gradients_and_batch_chains_actually_travel(tmp_path):
    """The hand-off carries what the recipient needs, not just what loads."""
    saved, _ = _publish(tmp_path, N=4, dp_shard=2)
    assert (saved / _GRAD_SIDECAR_FILENAME).is_file()
    assert all((saved / f"batch_hasher.rank_{r}.bin").is_file() for r in range(4))
    assert (saved / "meta.json").is_file() and (saved / "_COMPLETE").is_file()


# ── every way it must fail ───────────────────────────────────────────────────

@pytest.mark.parametrize("mutate", ["weight", "moment", "gradient"])
def test_altered_state_fails_the_commitment(tmp_path, mutate):
    saved, published = _publish(tmp_path)
    model, opt = _receive(saved)

    if mutate == "weight":
        with torch.no_grad():
            model.a.weight.view(-1)[0] += 1.0
    elif mutate == "moment":
        opt.state[model.a.weight]["exp_avg"].view(-1)[0] += 1.0
    else:
        sidecar = saved / _GRAD_SIDECAR_FILENAME
        raw = bytearray(sidecar.read_bytes())
        raw[-1] ^= 0xFF          # last tensor byte, header untouched
        sidecar.write_bytes(bytes(raw))

    assert not _verify(saved, model, opt, published)["verified"]


def test_a_missing_sidecar_is_not_read_as_every_grad_none(tmp_path):
    saved, published = _publish(tmp_path)
    model, opt = _receive(saved)
    (saved / _GRAD_SIDECAR_FILENAME).unlink()

    with pytest.raises(HandoffError, match="never be read as"):
        _verify(saved, model, opt, published)


def test_an_explicit_zero_is_not_a_none_gradient(tmp_path):
    """The distinction the sidecar's none_grad_names exists to preserve."""
    saved, published = _publish(tmp_path)
    model, opt = _receive(saved)
    assert _verify(saved, model, opt, published)["verified"]

    import safetensors
    import safetensors.torch as sft
    with safetensors.safe_open(
            str(saved / _GRAD_SIDECAR_FILENAME), framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        none_names = json.loads(f.metadata()["none_grad_names"])
    # Promote both None gradients to explicit zeros — same "no gradient
    # signal", a different commitment.
    for name in none_names:
        tensors[name] = torch.zeros_like(dict(model.named_parameters())[name])
    sft.save_file(tensors, str(saved / _GRAD_SIDECAR_FILENAME), metadata={
        "format": "pretrain-audit-gradients", "format_version": "1",
        "none_grad_names": json.dumps([])})

    assert not _verify(saved, model, opt, published)["verified"]


def test_an_incomplete_sidecar_is_refused_not_defaulted(tmp_path):
    saved, published = _publish(tmp_path)
    model, opt = _receive(saved)

    import safetensors
    import safetensors.torch as sft
    with safetensors.safe_open(
            str(saved / _GRAD_SIDECAR_FILENAME), framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys() if k != "a.bias"}
    sft.save_file(tensors, str(saved / _GRAD_SIDECAR_FILENAME), metadata={
        "format": "pretrain-audit-gradients", "format_version": "1",
        "none_grad_names": json.dumps(["c.weight", "c.bias"])})

    with pytest.raises(HandoffError, match="accounts for no gradient"):
        _verify(saved, model, opt, published)


def test_a_sidecar_for_a_different_model_is_refused(tmp_path):
    other = nn.Linear(3, 3)
    other.weight.grad = torch.zeros_like(other.weight)
    other.bias.grad = torch.zeros_like(other.bias)
    p = tmp_path / _GRAD_SIDECAR_FILENAME
    _write_gradient_sidecar({n: p.grad for n, p in other.named_parameters()}, p)

    with pytest.raises(HandoffError, match="does not have"):
        attach_gradients(_GradModel(), p)


def test_a_missing_rank_chain_fails_before_hashing(tmp_path):
    saved, published = _publish(tmp_path, N=4, dp_shard=2)
    model, opt = _receive(saved)
    (saved / "batch_hasher.rank_3.bin").unlink()

    with pytest.raises(HandoffError, match="batch_hasher.rank_3.bin is missing"):
        _verify(saved, model, opt, published, N=4, dp_shard=2)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_batch_combine_preserves_wire_format(ranks):
    digests = [bytes([r]) * 32 for r in range(ranks)]
    expected = (digests[0] if ranks == 1 else
                hashlib.blake2b(b"".join(digests), digest_size=32).digest())
    assert combine_batch_digests(digests) == expected


def test_rank_chain_boundaries_cannot_be_repartitioned(tmp_path):
    saved, _ = _publish(tmp_path, N=4, dp_shard=2)
    first = saved / "batch_hasher.rank_0.bin"
    second = saved / "batch_hasher.rank_1.bin"
    # Concatenation is unchanged, but neither file is a valid 32-byte chain.
    second.write_bytes(first.read_bytes() + second.read_bytes())
    first.write_bytes(b"")
    with pytest.raises(HandoffError, match="32 bytes"):
        read_batch_digest(saved, 4)


def test_only_rank_zeros_chain_cannot_stand_in_for_the_run(tmp_path):
    """A single rank's digest is the global one only when the run had one rank."""
    saved, published = _publish(tmp_path, N=4, dp_shard=2)
    assert read_batch_digest(saved, 4) != (saved / "batch_hasher.rank_0.bin").read_bytes()


def test_the_wrong_preceding_commitment_fails(tmp_path):
    saved, published = _publish(tmp_path)
    model, opt = _receive(saved)
    assert not _verify(saved, model, opt, published, prev="0" * 64)["verified"]


def test_the_wrong_target_commitment_fails(tmp_path):
    saved, _ = _publish(tmp_path)
    model, opt = _receive(saved)
    assert not _verify(saved, model, opt, "b" * 64)["verified"]


def test_the_wrong_topology_fails(tmp_path):
    """The v3 digest is shard-layout dependent; guessing it is not an option."""
    saved, published = _publish(tmp_path, N=4, dp_shard=2)
    model, opt = _receive(saved)
    assert not _verify(saved, model, opt, published, N=4, dp_shard=4)["verified"]


# ── the descriptor gates, before anything is loaded ──────────────────────────

def _write_bare_handoff(root: Path, meta: dict, *, stamped: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps(meta))
    (root / "_COMPLETE").write_text("")
    if stamped is not None:
        (root / "state_hash.txt").write_text(stamped + "\n")
    return root


def test_an_incomplete_download_is_refused(tmp_path):
    d = _write_bare_handoff(tmp_path / "h", _meta_obj(1, 1, "a" * 64))
    (d / "_COMPLETE").unlink()
    with pytest.raises(HandoffError, match="_COMPLETE"):
        verify_handoff(d, expect_hash="a" * 64, prev_hash=PREV)


def test_an_artifact_claiming_another_state_is_refused(tmp_path):
    """The artifact's own state_hash.txt is not the target — and disagreeing
    with the published log is grounds to stop, not to prefer one."""
    d = _write_bare_handoff(tmp_path / "h", _meta_obj(1, 1, "c" * 64), stamped="c" * 64)
    with pytest.raises(HandoffError, match="the run published"):
        verify_handoff(d, expect_hash="d" * 64, prev_hash=PREV)


def test_an_off_cadence_handoff_is_rejected_not_reinterpreted(tmp_path):
    """Its state_hash.txt is a side-link the published chain never folds in."""
    meta = _meta_obj(1, 1, "a" * 64, every_n_steps=8)
    meta["step"] = 101
    d = _write_bare_handoff(tmp_path / "h", meta)
    with pytest.raises(HandoffError, match="off the run's hash cadence"):
        verify_handoff(d, expect_hash="a" * 64, prev_hash=PREV)


def test_a_side_link_disagreeing_with_the_running_chain_is_rejected(tmp_path):
    meta = _meta_obj(1, 1, "a" * 64)
    d = _write_bare_handoff(tmp_path / "h", meta, stamped="b" * 64)
    with pytest.raises(HandoffError, match="running chain"):
        verify_handoff(d, expect_hash="b" * 64, prev_hash=PREV)


@pytest.mark.parametrize("bad", [
    {"dp_world_size": 4096, "dp_replicate": 1, "dp_shard": 4096},
    {"dp_world_size": 4, "dp_replicate": 3, "dp_shard": 2},
])
def test_an_implausible_topology_is_refused(tmp_path, bad):
    meta = _meta_obj(1, 1, "a" * 64) | bad
    d = _write_bare_handoff(tmp_path / "h", meta, stamped="a" * 64)
    with pytest.raises(HandoffError, match="topology"):
        verify_handoff(d, expect_hash="a" * 64, prev_hash=PREV)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="requires MPS")),
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA")),
])
def test_attach_gradients_uses_parameter_device(tmp_path, device):
    model = nn.Linear(4, 4, device=device, dtype=torch.bfloat16)
    expected = {n: torch.ones(p.shape, dtype=p.dtype) for n, p in model.named_parameters()}
    path = tmp_path / _GRAD_SIDECAR_FILENAME
    _write_gradient_sidecar(expected, path)
    assert attach_gradients(model, path) == len(expected)
    for name, p in model.named_parameters():
        assert p.grad.device == p.device
        assert p.grad.dtype == p.dtype
        assert torch.equal(p.grad.cpu(), expected[name])


@pytest.mark.parametrize("metadata", [
    {},
    {"format": "other", "format_version": "1", "none_grad_names": "[]"},
    {"format": "pretrain-audit-gradients", "format_version": "2", "none_grad_names": "[]"},
    {"format": "pretrain-audit-gradients", "format_version": "1", "none_grad_names": "null"},
    {"format": "pretrain-audit-gradients", "format_version": "1", "none_grad_names": '[123]'},
])
def test_invalid_sidecar_metadata_fails(tmp_path, metadata):
    from safetensors.torch import save_file

    path = tmp_path / _GRAD_SIDECAR_FILENAME
    save_file({}, str(path), metadata=metadata)
    with pytest.raises(HandoffError):
        attach_gradients(nn.Linear(2, 2), path)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="requires MPS")),
])
def test_real_verifier_cli_roundtrip_and_tampering(tmp_path, device):
    """No model-builder, DCP-loader, native-optimizer or verifier mocks.

    Export a tiny real Llama/repop checkpoint, then let the installed CLI build
    its own model/optimizer from meta.json in a separate process. Synthetic
    gradients keep this a hash/load test, not a training benchmark.
    """
    entry = Path(sys.executable).with_name("pretrain-audit-verify-handoff")
    if not entry.is_file():
        pytest.skip("install this checkout to test its console entrypoint")

    from pretrain.config import parse_config_resolved
    from pretrain.model import build_model
    from pretrain.optim.adamw_repop import prime_optimizer_state
    from pretrain.optim.registry import build_optimizer
    from pretrain.parallel.parallel_dims import ParallelDims
    from pretrain.parallel.parallelize_llama3_repop import parallelize_llama3_repop

    doc = json.loads(_config_resolved(1))
    doc["model"].update(n_layers=1, d_model=32, n_heads=2, n_kv_heads=1,
                        head_dim=16, ffn_intermediate=64, vocab_size=128,
                        max_seq_len_pretrain=16)
    doc["train"]["seq_len"] = doc["data"]["seq_len"] = 8
    cfg = parse_config_resolved(json.dumps(doc))
    model = parallelize_llama3_repop(
        build_model(cfg.model, device="cpu"), cfg,
        ParallelDims(dp_replicate=1, dp_shard=1, world_size=1),
    )
    optimizer = build_optimizer(model, cfg.optim)
    prime_optimizer_state(optimizer)
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.125)
    optimizer.step()
    hashers = [RunningBatchHasher() for _ in range(4)]
    for r, h in enumerate(hashers):
        h.update({"input_ids": torch.full((1, 4), r, dtype=torch.int64)})
    bd = combine_batch_digests([h.local_digest() for h in hashers])
    published = finalize_state_hash(
        prev_hash=PREV,
        shard_state_digest=audit_shard_state_digest(
            model, 2, 4, optimizer=optimizer, include_grads=True),
        optimizer=optimizer, batch_digest=bd,
    )
    held = {n: p.grad.detach().cpu() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    meta = _meta_obj(4, 2, published)
    meta["config_resolved"] = cfg.model_dump_json()
    saved = _save_chained_audit_checkpoint(
        save_dir=str(tmp_path / "handoff"), step=101, consumed=1010,
        digest=published, chained_hash_meta=published, meta_obj=meta,
        stream_state=GlobalStreamState(
            consumed_documents_per_source={"web": 1},
            epoch_per_source={"web": 0}, windows_emitted=1),
        model=model, optimizer=optimizer, spike_state={},
        batch_hashers=hashers, batch_digest=bd, N=4, gradients=held,
    )
    result_path = tmp_path / "verdict.json"
    command = [str(entry), "--checkpoint", str(saved), "--expect-hash", published,
               "--prev-hash", PREV, "--device", device, "--json", str(result_path)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result == json.loads(result_path.read_text())
    assert result["verified"] is True
    assert result["reconstructed_hash"] == published
    assert result["dp_world_size"] == 4 and result["dp_shard"] == 2

    # Leave the author's state_hash.txt untouched; only alter tensor bytes.
    from safetensors import safe_open
    from safetensors.torch import save_file

    path = saved / _GRAD_SIDECAR_FILENAME
    with safe_open(str(path), framework="pt") as f:
        tensors = {n: f.get_tensor(n) for n in f.keys()}
        metadata = f.metadata()
    tensors[sorted(tensors)[0]].view(-1)[0] += 1
    save_file(tensors, str(path), metadata=metadata)
    proc = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert proc.returncode != 0
    assert json.loads(proc.stdout)["verified"] is False


@pytest.mark.parametrize("fields", [
    {}, {"step": None}, {"step": 0}, {"step": -1},
    {"step": True}, {"step": "101"}, {"step": 1.5},
])
def test_missing_or_invalid_step_is_refused(tmp_path, fields):
    meta = _meta_obj(1, 1, "a" * 64)
    meta.pop("step")
    meta.update(fields)
    saved = _write_bare_handoff(tmp_path / "handoff", meta)
    with pytest.raises(HandoffError, match="step must be a positive integer"):
        verify_handoff(saved, expect_hash="a" * 64, prev_hash=PREV)


@pytest.mark.parametrize("error_type", [
    HandoffError, ValueError, KeyError, OSError, RuntimeError,
])
def test_cli_refusal_overwrites_json_result(tmp_path, monkeypatch, capsys, caplog, error_type):
    import pretrain.cli.verify_handoff as verifier

    output = tmp_path / "result.json"
    output.write_text('{"verified": true}')

    def fail(*args, **kwargs):
        raise error_type("refused")

    monkeypatch.setattr(verifier, "verify_handoff", fail)
    assert verifier.main([
        "--checkpoint", str(tmp_path), "--expect-hash", "a" * 64,
        "--prev-hash", PREV, "--json", str(output),
    ]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["verified"] is False
    assert "refused" in result["error"]
    assert json.loads(output.read_text()) == result
    traces = [r for r in caplog.records if r.exc_info]
    if error_type is HandoffError:
        assert result["error"] == "refused"
        assert not traces
    else:
        assert result["error"].startswith(error_type.__name__ + ": ")
        assert len(traces) == 1
        assert traces[0].exc_info[0] is error_type


def test_cli_dcp_failure_produces_json(tmp_path, monkeypatch, capsys, caplog):
    import pretrain.cli.verify_handoff as verifier
    from torch.distributed.checkpoint.api import CheckpointException

    failure = CheckpointException("load failed", {})

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(verifier, "verify_handoff", fail)
    output = tmp_path / "result.json"
    assert verifier.main([
        "--checkpoint", str(tmp_path), "--expect-hash", "a" * 64,
        "--prev-hash", PREV, "--json", str(output),
    ]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["verified"] is False
    assert result["error"] == f"CheckpointException: {failure}"
    assert json.loads(output.read_text()) == result
    assert any(r.exc_info and r.exc_info[1] is failure for r in caplog.records)


def test_cli_json_write_error_fails_closed(tmp_path, monkeypatch, capsys):
    import pretrain.cli.verify_handoff as verifier

    monkeypatch.setattr(verifier, "verify_handoff", lambda *a, **kw: {"verified": True})
    # A directory cannot be overwritten with the requested JSON result.
    assert verifier.main([
        "--checkpoint", str(tmp_path), "--expect-hash", "a" * 64,
        "--prev-hash", PREV, "--json", str(tmp_path),
    ]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["verified"] is False
    assert "cannot write" in result["error"]


@pytest.mark.parametrize("flag", ["--expect-hash", "--prev-hash"])
@pytest.mark.parametrize("value", ["abcd", "file"])
def test_commitment_inputs_are_literals_and_name_the_bad_flag(tmp_path, capsys, flag, value):
    from pretrain.cli.verify_handoff import main

    claim = tmp_path / "state_hash.txt"
    claim.write_text("a" * 64)
    bad = str(claim) if value == "file" else value
    args = {"--expect-hash": "a" * 64, "--prev-hash": PREV}
    args[flag] = bad
    assert main(["--checkpoint", str(tmp_path),
                 *[item for pair in args.items() for item in pair]]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["verified"] is False
    assert result["error"].startswith(flag + " must be a full literal")


@pytest.mark.parametrize("gate", ["cadence", "topology", "claim", "cuda", "mps"])
def test_cheap_refusals_do_not_apply_checkpoint_environment(tmp_path, monkeypatch, gate):
    import os

    monkeypatch.setenv("REPOP_EXECUTION_MODE", "original")
    meta = _meta_obj(1, 1, "a" * 64, every_n_steps=8 if gate == "cadence" else 1)
    if gate == "topology":
        meta["dp_world_size"] = 4096
    device = gate if gate in ("cuda", "mps") else "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    stamped = "b" * 64 if gate == "claim" else "a" * 64
    saved = _write_bare_handoff(tmp_path, meta, stamped=stamped)
    reason = {"cadence": "off the run's hash cadence", "topology": "topology",
              "claim": "running chain", "cuda": "CUDA is unavailable",
              "mps": "MPS is unavailable"}[gate]
    with pytest.raises(HandoffError, match=reason):
        verify_handoff(saved, expect_hash="a" * 64, prev_hash=PREV, device=device)
    assert os.environ["REPOP_EXECUTION_MODE"] == "original"


def test_checkpoint_cannot_force_single_rank_fsdp(monkeypatch):
    import os

    from pretrain.cli.audit_replay import _apply_repop_env

    monkeypatch.setenv("REPOP_EXECUTION_MODE", "original")
    monkeypatch.delenv("REPOP_FORCE_FSDP_WS1", raising=False)
    with pytest.raises(ValueError, match="REPOP_FORCE_FSDP_WS1"):
        _apply_repop_env({"REPOP_EXECUTION_MODE": "fast", "REPOP_FORCE_FSDP_WS1": "1"})
    assert os.environ["REPOP_EXECUTION_MODE"] == "original"
    assert "REPOP_FORCE_FSDP_WS1" not in os.environ
