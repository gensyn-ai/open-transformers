"""Round-trip tests for the DCP <-> safetensors converter.

Pure-torch: builds nested state dicts directly (no repop / model build), so
these run on any dev machine. The DCP side uses the same offline no-dist
read/write path the CLI uses.

Comparison note: torch's offline DCP reconstruction (``set_element``)
rebuilds int-keyed dicts (optimizer ``state``) as lists and tuples as lists,
so round-trip equality is asserted between two states that BOTH went through
a DCP read — that is the representation the converter contracts to preserve
(the dotted FQNs, which are identical either way).
"""

from __future__ import annotations

import json
import pickle

import pytest
import torch

from pretrain.cli.dcp_safetensors import (
    bitwise_tensor_equal,
    dcp_to_safetensors,
    flatten_state,
    load_dcp_offline,
    safetensors_to_dcp,
    save_dcp_offline,
    state_equal,
    unflatten_state,
)


def _sample_state() -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        "model": {
            "tok_embeddings.weight": torch.randn(8, 4, generator=g).bfloat16(),
            "layers.0.attn.wq.weight": torch.randn(4, 4, generator=g),
            "norm.weight": torch.randn(4, generator=g, dtype=torch.float64),
        },
        "optim": {
            "state": {
                0: {
                    "exp_avg": torch.randn(8, 4, generator=g),
                    "exp_avg_sq": torch.randn(8, 4, generator=g).abs(),
                    "step": torch.tensor(41.0),  # 0-dim tensor
                },
            },
            "param_groups": [
                {
                    "lr": 3e-4,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                    "params": [0],
                    "fused": None,
                }
            ],
        },
    }


def test_flatten_unflatten_is_lossless():
    state = _sample_state()
    # Add leaf types DCP never emits but the skeleton claims to support.
    state["extras"] = {
        "inf": float("inf"),
        "nan": float("nan"),
        "blob": b"\x00\xffbytes",
        "flag": True,
        "nothing": None,
    }
    skeleton, tensors = flatten_state(state)
    rebuilt = unflatten_state(json.loads(json.dumps(skeleton)), tensors)
    assert state_equal(rebuilt["model"], state["model"])
    assert rebuilt["optim"]["state"][0].keys() == state["optim"]["state"][0].keys()
    # Typed dict keys and tuples survive (no DCP leg here, so exact types).
    assert list(rebuilt["optim"]["state"].keys()) == [0]
    assert rebuilt["optim"]["param_groups"][0]["betas"] == (0.9, 0.95)
    assert rebuilt["extras"]["inf"] == float("inf")
    assert rebuilt["extras"]["nan"] != rebuilt["extras"]["nan"]  # NaN
    assert rebuilt["extras"]["blob"] == b"\x00\xffbytes"
    assert rebuilt["extras"]["flag"] is True
    assert rebuilt["extras"]["nothing"] is None


def test_dcp_safetensors_dcp_roundtrip(tmp_path):
    dcp1 = tmp_path / "dcp1"
    save_dcp_offline(_sample_state(), dcp1)

    st = tmp_path / "ckpt.safetensors"
    stats = dcp_to_safetensors(dcp1, st, verify=True)
    assert stats["top_level_keys"] == ["model", "optim"]

    dcp2 = tmp_path / "dcp2"
    safetensors_to_dcp(st, dcp2, verify=True)

    a, b = load_dcp_offline(dcp1), load_dcp_offline(dcp2)
    assert state_equal(a, b)
    # Spot-check bit-exactness across dtypes incl. bf16 explicitly.
    for k in ("tok_embeddings.weight", "layers.0.attn.wq.weight", "norm.weight"):
        assert bitwise_tensor_equal(a["model"][k], b["model"][k])
        assert a["model"][k].dtype == b["model"][k].dtype


def test_to_safetensors_accepts_ckpt_dir_and_filters_keys(tmp_path):
    ckpt = tmp_path / "step_000000010"
    save_dcp_offline(_sample_state(), ckpt / "dcp")

    st = tmp_path / "model_only.safetensors"
    stats = dcp_to_safetensors(ckpt, st, keys=["model"], verify=True)
    assert stats["top_level_keys"] == ["model"]

    dcp2 = tmp_path / "dcp2"
    safetensors_to_dcp(st, dcp2, verify=True)
    reloaded = load_dcp_offline(dcp2)
    assert set(reloaded) == {"model"}
    assert state_equal(reloaded["model"], load_dcp_offline(ckpt / "dcp")["model"])


def test_to_safetensors_unknown_key_fails(tmp_path):
    dcp1 = tmp_path / "dcp1"
    save_dcp_offline(_sample_state(), dcp1)
    with pytest.raises(SystemExit, match="available top-level keys"):
        dcp_to_safetensors(dcp1, tmp_path / "x.safetensors", keys=["nope"])


def test_foreign_safetensors_top_level_key(tmp_path):
    from safetensors.torch import save_file

    w = {"layers.0.weight": torch.randn(3, 3), "head.weight": torch.randn(2, 3)}
    foreign = tmp_path / "foreign.safetensors"
    save_file(w, str(foreign))  # no pretrain skeleton metadata

    dcp = tmp_path / "dcp"
    safetensors_to_dcp(foreign, dcp, top_level_key="model", verify=True)
    reloaded = load_dcp_offline(dcp)
    assert set(reloaded) == {"model"}
    for k, t in w.items():
        assert bitwise_tensor_equal(reloaded["model"][k], t)


def test_top_level_key_rejected_for_skeleton_files(tmp_path):
    dcp1 = tmp_path / "dcp1"
    save_dcp_offline(_sample_state(), dcp1)
    st = tmp_path / "ckpt.safetensors"
    dcp_to_safetensors(dcp1, st)
    with pytest.raises(SystemExit, match="only for foreign safetensors"):
        safetensors_to_dcp(st, tmp_path / "dcp2", top_level_key="model")


def test_refuses_nonempty_output_dir(tmp_path):
    dcp1 = tmp_path / "dcp1"
    save_dcp_offline(_sample_state(), dcp1)
    st = tmp_path / "ckpt.safetensors"
    dcp_to_safetensors(dcp1, st)
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "existing.distcp").write_text("do not clobber")
    with pytest.raises(SystemExit, match="non-empty"):
        safetensors_to_dcp(st, out)


def _sample_ckpt_dir(tmp_path, name="step_000000010"):
    """A full checkpoint dir in the Checkpointer layout: dcp/ + sidecar."""
    ckpt = tmp_path / name
    save_dcp_offline(_sample_state(), ckpt / "dcp")
    (ckpt / "meta.json").write_text(json.dumps({"step": 10, "seed": 1234}, indent=2))
    (ckpt / "global_stream.json").write_text(
        json.dumps({"consumed_documents_per_source": {"a": 7}, "epoch_per_source": {"a": 0}})
    )
    (ckpt / "spike_protocol.json").write_text(json.dumps({"halted": False}))
    (ckpt / "state_hash.txt").write_text("ab" * 32 + "\n")
    (ckpt / "sampler.rank_1.json").write_text(json.dumps({"legacy": True}))
    for rank in (0, 1):
        torch.save(
            {"cpu": torch.get_rng_state(), "cuda": [torch.randint(0, 256, (16,), dtype=torch.uint8)] if rank else None},
            ckpt / f"rng.rank_{rank}.pt",
        )
        (ckpt / f"batch_hasher.rank_{rank}.bin").write_bytes(bytes([rank]) * 32)
    # Chained-audit hand-offs additionally carry the target step's gradients
    # so a recipient can recompute the logged state hash.
    from safetensors.torch import save_file

    save_file(
        {"w.grad": torch.randn(8, 4).bfloat16()},
        str(ckpt / "gradients.safetensors"),
        metadata={
            "format": "pretrain-audit-gradients",
            "format_version": "1",
            "none_grad_names": json.dumps(["b"]),
        },
    )
    (ckpt / "_COMPLETE").write_text("")
    return ckpt


def test_full_checkpoint_dir_roundtrip(tmp_path):
    """A step_N/ dir round-trips COMPLETELY: dcp payload bitwise, sidecar
    text/bin files byte-identical, RNG blobs semantically identical, and the
    unpacked dir has the _COMPLETE sentinel — i.e. it is a resume point, not
    just a weight-transport artifact (the hand-off requirement)."""
    ckpt = _sample_ckpt_dir(tmp_path)
    st = tmp_path / "ckpt.safetensors"
    stats = dcp_to_safetensors(ckpt, st, verify=True)
    # meta/global_stream/spike_protocol/state_hash + sampler.rank_1 + 2 bins
    assert stats["sidecar_files"] == 7

    out = tmp_path / "unpacked"
    stats2 = safetensors_to_dcp(st, out, verify=True)
    assert stats2["sidecar_files"] == stats["sidecar_files"]

    # dcp payload bitwise.
    assert state_equal(load_dcp_offline(ckpt / "dcp"), load_dcp_offline(out / "dcp"))
    # Text + binary sidecar files byte-for-byte.
    for name in (
        "meta.json", "global_stream.json", "spike_protocol.json",
        "state_hash.txt", "sampler.rank_1.json",
        "batch_hasher.rank_0.bin", "batch_hasher.rank_1.bin",
    ):
        assert (out / name).read_bytes() == (ckpt / name).read_bytes(), name
    # RNG blobs: re-serialised by torch.save, so compare decoded content.
    for rank in (0, 1):
        a = torch.load(ckpt / f"rng.rank_{rank}.pt", weights_only=True)
        b = torch.load(out / f"rng.rank_{rank}.pt", weights_only=True)
        assert bitwise_tensor_equal(a["cpu"], b["cpu"])
        assert (a["cuda"] is None) == (b["cuda"] is None)
        if a["cuda"] is not None:
            assert all(bitwise_tensor_equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))
    # Gradient sidecar: tensors bitwise, header metadata verbatim.
    from safetensors import safe_open

    with safe_open(str(ckpt / "gradients.safetensors"), framework="pt") as fa, \
         safe_open(str(out / "gradients.safetensors"), framework="pt") as fb:
        assert set(fa.keys()) == set(fb.keys())
        assert fa.metadata() == fb.metadata()
        for k in fa.keys():
            assert bitwise_tensor_equal(fa.get_tensor(k), fb.get_tensor(k)), k
    assert (out / "_COMPLETE").exists()


def test_keys_filter_skips_sidecar(tmp_path):
    """--keys produces a tensors-only file; unpacking it must NOT fabricate a
    checkpoint-looking dir (no meta.json, no _COMPLETE)."""
    ckpt = _sample_ckpt_dir(tmp_path)
    st = tmp_path / "model_only.safetensors"
    stats = dcp_to_safetensors(ckpt, st, keys=["model"])
    assert stats["sidecar_files"] is None

    out = tmp_path / "unpacked"
    safetensors_to_dcp(st, out, verify=True)
    assert (out / ".metadata").is_file()  # bare DCP dir layout
    assert not (out / "meta.json").exists()
    assert not (out / "_COMPLETE").exists()


def test_malicious_sidecar_filename_rejected(tmp_path):
    """Sidecar filenames come from an untrusted header and become paths on
    unpack — anything outside the exact allowlist must be refused before any
    write (path traversal / clobbering guard)."""
    from safetensors.torch import save_file

    from pretrain.cli.dcp_safetensors import _SIDECAR_KEY, _SKELETON_KEY

    state = {"model": {"w": torch.randn(2, 2)}}
    skeleton, tensors = flatten_state(state)
    evil_sidecar = {"files": {"../evil.txt": "pwned"}, "rng": {}, "grads": None}
    sc_skeleton, tensors = flatten_state(evil_sidecar, ("__sidecar__",), tensors)
    st = tmp_path / "evil.safetensors"
    save_file(tensors, str(st), metadata={
        _SKELETON_KEY: json.dumps(skeleton),
        _SIDECAR_KEY: json.dumps(sc_skeleton),
    })
    with pytest.raises(SystemExit, match="not an allowed checkpoint entry"):
        safetensors_to_dcp(st, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "out").exists()


def test_doctored_dcp_metadata_is_rejected(tmp_path):
    """The untrusted-input gate must fire before any DCP read: a .metadata
    referencing a global outside the DCP schema is refused, not executed."""
    import os

    dcp1 = tmp_path / "dcp1"
    save_dcp_offline(_sample_state(), dcp1)
    (dcp1 / ".metadata").write_bytes(pickle.dumps(os.system))
    with pytest.raises(pickle.UnpicklingError, match="outside the allowlist"):
        dcp_to_safetensors(dcp1, tmp_path / "x.safetensors")
