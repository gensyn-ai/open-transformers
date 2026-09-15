"""``Checkpointer.load`` must not unpickle arbitrary objects.

A checkpoint directory is untrusted input: ``audit_replay`` is meant to be
pointed at a dir another user produced (``--save-checkpoint-dir``) or one
pulled from a ``--gcs-root`` mirror. Both ``torch.load`` call sites in
``Checkpointer.load`` therefore pass ``weights_only=True``, so a doctored
``rng.rank_0.pt`` / ``fallback.pt`` fails to load instead of executing the
checkpoint author's code in the auditor's process.

Deliberately repop-free (a plain ``nn.Module``, no ``build_model``) so this
runs on any CPU dev box — see ``test_checkpoint_roundtrip.py`` for the
model-shaped round-trip.
"""

from __future__ import annotations

import os
import pickle

import pytest
import torch

from pretrain.data.mix_sampler import MixSamplerState
from pretrain.train.checkpoint import Checkpointer, CheckpointMeta


class _Payload:
    """Stands in for a malicious ``__reduce__``: touches a file when unpickled."""

    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return (os.system, (f"touch {self.path!r}",))


def _tiny_model():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(8, 16)
            self.lin = torch.nn.Linear(16, 16)

        def forward(self, idx):
            return self.lin(self.emb(idx))

    return Net()


def _save_ckpt(tmp_path):
    """A real single-process checkpoint (DCP works without dist init)."""
    model = _tiny_model()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(torch.randint(0, 8, (1, 4))).sum().backward()
    optim.step()
    optim.zero_grad()

    meta = CheckpointMeta(
        consumed_tokens=1234,
        step=7,
        git_sha="deadbeef",
        config_resolved="{}",
        tokenizer_hash="x",
        container_digest="y",
    )
    ckptr = Checkpointer(tmp_path / "ckpts")
    ckpt_dir = ckptr.save(
        7,
        model,
        optim,
        MixSamplerState(
            consumed_documents_per_source={"a": 1},
            epoch_per_source={"a": 0},
        ),
        meta,
    )
    return ckptr, ckpt_dir, model


def _fresh_pair():
    model = _tiny_model()
    return model, torch.optim.AdamW(model.parameters(), lr=1e-4)


def test_benign_checkpoint_still_loads(tmp_path):
    """The allowlisted unpickler must accept the RNG blob we actually write."""
    ckptr, ckpt_dir, model = _save_ckpt(tmp_path)
    assert (ckpt_dir / "rng.rank_0.pt").is_file()

    model2, optim2 = _fresh_pair()
    _, meta2, _ = ckptr.load(ckpt_dir, model2, optim2)
    assert (meta2.step, meta2.consumed_tokens) == (7, 1234)
    for (n, a), (_, b) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        assert torch.equal(a, b), f"param {n} not restored"


def test_doctored_rng_blob_does_not_execute_code(tmp_path):
    """The RNG blob is read on every resume, so it needs no other corruption."""
    ckptr, ckpt_dir, _ = _save_ckpt(tmp_path)
    canary = tmp_path / "rng-payload-ran"
    (ckpt_dir / "rng.rank_0.pt").write_bytes(pickle.dumps(_Payload(canary)))

    model2, optim2 = _fresh_pair()
    with pytest.raises(pickle.UnpicklingError):
        ckptr.load(ckpt_dir, model2, optim2)
    assert not canary.exists(), (
        "rng.rank_0.pt payload executed: Checkpointer.load unpickled untrusted "
        "objects from a checkpoint directory"
    )


def _break_dcp_load(monkeypatch):
    """Send ``Checkpointer.load`` down its ``fallback.pt`` branch.

    Corrupting ``dcp/`` does not work: DCP wraps read failures in
    ``CheckpointException``, which derives from ``BaseException`` and so
    escapes the ``except Exception`` around the DCP read. A plain exception
    from ``dcp.load`` is what that handler is written for.
    """
    import torch.distributed.checkpoint as dcp

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated DCP read failure")

    monkeypatch.setattr(dcp, "load", _raise)


def test_benign_fallback_still_loads(tmp_path, monkeypatch):
    ckptr, ckpt_dir, model = _save_ckpt(tmp_path)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.save(
        {"model": model.state_dict(), "optim": optim.state_dict()},
        ckpt_dir / "fallback.pt",
    )

    _break_dcp_load(monkeypatch)
    model2, optim2 = _fresh_pair()
    _, meta2, _ = ckptr.load(ckpt_dir, model2, optim2)
    assert meta2.step == 7
    for (n, a), (_, b) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        assert torch.equal(a, b), f"param {n} not restored from fallback.pt"


def test_doctored_fallback_does_not_execute_code(tmp_path, monkeypatch):
    ckptr, ckpt_dir, _ = _save_ckpt(tmp_path)
    canary = tmp_path / "fallback-payload-ran"
    (ckpt_dir / "fallback.pt").write_bytes(
        pickle.dumps({"model": _Payload(canary)})
    )

    _break_dcp_load(monkeypatch)
    model2, optim2 = _fresh_pair()
    with pytest.raises(pickle.UnpicklingError):
        ckptr.load(ckpt_dir, model2, optim2)
    assert not canary.exists(), (
        "fallback.pt payload executed: Checkpointer.load unpickled untrusted "
        "objects from a checkpoint directory"
    )


def test_doctored_dcp_metadata_does_not_execute_code(tmp_path):
    """``dcp.load`` reads ``dcp/.metadata`` with a plain ``pickle.load`` inside
    torch (FileSystemReader.read_metadata), which ``weights_only=True`` cannot
    reach. ``Checkpointer.load`` must reject a doctored metadata BEFORE handing
    the directory to DCP — and must NOT fall through to the fallback.pt branch
    (the validation runs outside the try/except around the DCP read)."""
    ckptr, ckpt_dir, _ = _save_ckpt(tmp_path)
    assert (ckpt_dir / "dcp" / ".metadata").is_file(), "fixture: no DCP metadata"
    canary = tmp_path / "dcp-metadata-payload-ran"
    (ckpt_dir / "dcp" / ".metadata").write_bytes(pickle.dumps(_Payload(canary)))
    # A benign fallback.pt is present: a silent fall-through would "succeed".
    model, optim = _fresh_pair()
    torch.save(
        {"model": model.state_dict(), "optim": optim.state_dict()},
        ckpt_dir / "fallback.pt",
    )

    model2, optim2 = _fresh_pair()
    with pytest.raises(pickle.UnpicklingError):
        ckptr.load(ckpt_dir, model2, optim2)
    assert not canary.exists(), (
        "dcp/.metadata payload executed: torch's pickle-based metadata reader "
        "ran untrusted code before Checkpointer.load validated it"
    )


def test_benign_dcp_metadata_passes_validation(tmp_path):
    """The allowlist must admit every global a genuine DCP metadata contains
    (regression guard for torch upgrades widening the metadata schema)."""
    from pretrain.train.checkpoint import _validate_dcp_metadata

    _, ckpt_dir, _ = _save_ckpt(tmp_path)
    _validate_dcp_metadata(ckpt_dir / "dcp")  # must not raise


def _stack_global_reduce(module: str, name: str, arg: str) -> bytes:
    """A proto-4 pickle: STACK_GLOBAL(module, name) then REDUCE(arg), i.e.
    ``(module.name)(arg)`` on load. STACK_GLOBAL (not the proto-0 GLOBAL
    opcode) is what a real DCP ``.metadata`` uses, and it is the opcode whose
    find_class dot-walks ``name`` — so this is the faithful shape of the
    attribute-walk bypass."""
    def su(s: str) -> bytes:
        b = s.encode()
        assert len(b) < 256
        return b"\x8c" + bytes([len(b)]) + b  # SHORT_BINUNICODE
    return (
        b"\x80\x04"                              # PROTO 4
        + su(module) + su(name) + b"\x93"        # STACK_GLOBAL -> module.name
        + su(arg) + b"\x85" + b"R"               # TUPLE1(arg), REDUCE
        + b"."                                    # STOP
    )


def test_attack_payload_is_not_vacuous(tmp_path):
    """Sanity: the STACK_GLOBAL/os.system payload really DOES execute on a
    permissive unpickler (else the rejection tests below prove nothing)."""
    import io

    canary = tmp_path / "permissive-ran"
    payload = _stack_global_reduce(
        "torch.distributed.checkpoint.format_utils", "os.system", f"touch {canary}"
    )
    pickle.Unpickler(io.BytesIO(payload)).load()
    assert canary.exists(), "payload did not execute — test would be vacuous"


def test_dotted_name_attribute_walk_is_rejected(tmp_path):
    """The critical bypass: a submodule under torch.distributed.checkpoint
    re-exports os at module scope, so STACK_GLOBAL
    `torch.distributed.checkpoint.format_utils` / `os.system` resolves to
    os.system via CPython's attribute walk. The dotted-name rule must reject it
    BEFORE it resolves — no canary file created."""
    from pretrain.train.checkpoint import _DCPMetadataUnpickler
    import io

    canary = tmp_path / "attr-walk-ran"
    payload = _stack_global_reduce(
        "torch.distributed.checkpoint.format_utils", "os.system", f"touch {canary}"
    )
    with pytest.raises(pickle.UnpicklingError):
        _DCPMetadataUnpickler(io.BytesIO(payload)).load()
    assert not canary.exists()


def test_module_scope_function_in_dcp_namespace_is_rejected(tmp_path):
    """A bare (non-dotted) callable in the DCP namespace — e.g.
    format_utils.dcp_to_torch_save — is a function, not a metadata class, so
    the class-identity rule must reject it even though the module prefix
    matches."""
    from pretrain.train.checkpoint import _DCPMetadataUnpickler
    import io

    payload = _stack_global_reduce(
        "torch.distributed.checkpoint.format_utils", "dcp_to_torch_save", "x"
    )
    with pytest.raises(pickle.UnpicklingError):
        _DCPMetadataUnpickler(io.BytesIO(payload)).load()


def test_reexported_foreign_class_in_dcp_namespace_is_rejected(tmp_path):
    """A dangerous CLASS re-exported into the DCP namespace (its __module__ is
    NOT the DCP namespace) must be rejected — class-ness alone is insufficient,
    the defining module must also be in the namespace."""
    from pretrain.train.checkpoint import _DCPMetadataUnpickler
    import io
    import subprocess
    import torch.distributed.checkpoint.metadata as meta_mod

    # Simulate a torch version that re-exported subprocess.Popen at module scope.
    monkey_had = hasattr(meta_mod, "Popen")
    meta_mod.Popen = subprocess.Popen
    try:
        payload = _stack_global_reduce(
            "torch.distributed.checkpoint.metadata", "Popen", "echo hi"
        )
        with pytest.raises(pickle.UnpicklingError):
            _DCPMetadataUnpickler(io.BytesIO(payload)).load()
    finally:
        if not monkey_had:
            del meta_mod.Popen
