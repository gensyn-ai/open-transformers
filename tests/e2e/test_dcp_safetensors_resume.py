"""The decisive resumability test for the DCP <-> safetensors converter
(hand-off relay): pack a real checkpoint, unpack it, replay the next
interval with ``audit_replay``, and assert the v3 chained state hash matches
the run's published one.

Bitwise tensor equality and sidecar byte-fidelity (tests/test_dcp_safetensors.py)
are necessary but do not establish that an unpacked hand-off actually RESUMES:
that additionally needs the per-rank RNG and batch-hasher state, the run meta,
and the global-stream position to survive the trip — miss any of them and the
replay reports a NO MATCH indistinguishable from genuine divergence.

Drives the REAL ``pretrain.train.loop.train`` on CPU (tiny model, synthetic
shards), reusing the harness from test_final_emergency_ckpt_hash. Skipped when
``repop`` isn't importable.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("repop")

from pretrain.cli.audit_replay import audit_replay  # noqa: E402
from pretrain.cli.dcp_safetensors import (  # noqa: E402
    dcp_to_safetensors,
    safetensors_to_dcp,
)

from .test_final_emergency_ckpt_hash import (  # noqa: E402
    _jsonl_hashes,
    _run_dir,
    _scrub_local_repop_env,
    _tiny_cfg,
)


def test_unpacked_handoff_resumes_and_audits(tmp_path, monkeypatch):
    """ckpt_every_steps=5, 7 total steps. Pack the step-5 checkpoint into a
    safetensors hand-off, unpack it elsewhere, and audit-replay 5->7 from the
    UNPACKED directory against the run's step-7 state_hash.txt. A PASS here
    means the hand-off carried everything a resume needs."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _scrub_local_repop_env(monkeypatch)
    from pretrain.train.loop import train

    cfg = _tiny_cfg(tmp_path, run_id="handoff5", total_steps=7, ckpt_every_steps=5)
    train(cfg)

    run_dir = _run_dir(cfg)
    step5 = run_dir / "checkpoints" / "step_000000005"
    step7 = run_dir / "checkpoints" / "step_000000007"
    assert step5.is_dir() and step7.is_dir()
    hashes = _jsonl_hashes(run_dir)

    # Pack -> unpack. The unpacked dir keeps the step_N naming purely for
    # legibility; audit_replay reads the step from meta.json.
    handoff = tmp_path / "handoff.safetensors"
    stats = dcp_to_safetensors(step5, handoff, verify=True)
    assert stats["sidecar_files"], "step-5 checkpoint produced no sidecar"

    unpacked = tmp_path / "relay" / "step_000000005"
    stats2 = safetensors_to_dcp(handoff, unpacked, verify=True)
    assert stats2["sidecar_files"] == stats["sidecar_files"]

    # The auditor's documented flow, pointed at the unpacked hand-off:
    # --expect-hash from the run's own step-7 checkpoint.
    res = audit_replay(
        str(unpacked),
        until_step=7,
        expect_hash=str(step7 / "state_hash.txt"),
        device="cpu",
    )
    assert res["state_hash"] == hashes[7], (
        f"replay from the unpacked hand-off reproduced "
        f"{res['state_hash'][:16]}…, run log says {hashes[7][:16]}… — "
        f"resume state was lost in the safetensors round trip"
    )
    assert res["match"] is True


def test_two_user_chained_audit_relay_through_safetensors(tmp_path, monkeypatch):
    """The hand-off relay end-to-end, with the safetensors conversion in the
    middle: user A audits 5->6 and saves a chained-audit hand-off
    (--save-checkpoint-dir, incl. the gradients.safetensors export);
    the hand-off travels as ONE safetensors file; user B unpacks it and audits
    6->7 from the unpacked dir against the run's step-7 hash. Also pins that
    the gradient export survives the trip content-identically — dropping it
    would break the recipient's hash recomputation even though the replay
    itself would still pass."""
    from safetensors import safe_open

    monkeypatch.setenv("WANDB_MODE", "disabled")
    _scrub_local_repop_env(monkeypatch)
    from pretrain.train.loop import train

    cfg = _tiny_cfg(tmp_path, run_id="relay", total_steps=7, ckpt_every_steps=5)
    train(cfg)

    run_dir = _run_dir(cfg)
    step5 = run_dir / "checkpoints" / "step_000000005"
    step7 = run_dir / "checkpoints" / "step_000000007"
    hashes = _jsonl_hashes(run_dir)

    # ---- user A: audit 5->6, save the chained hand-off ------------------------
    res_a = audit_replay(
        str(step5),
        until_step=6,
        expect_hash=hashes[6],
        save_checkpoint_dir=str(tmp_path / "a_out"),
        device="cpu",
    )
    assert res_a["match"] is True
    handoff_dir = tmp_path / "a_out" / "step_000000006"
    assert (handoff_dir / "gradients.safetensors").is_file(), (
        "chained-audit save produced no gradient export — harness drift"
    )

    # ---- the wire: one safetensors file ---------------------------------------
    wire = tmp_path / "handoff.safetensors"
    dcp_to_safetensors(handoff_dir, wire, verify=True)

    # ---- user B: unpack, audit 6->7 from the unpacked hand-off ----------------
    unpacked = tmp_path / "b_in" / "step_000000006"
    safetensors_to_dcp(wire, unpacked, verify=True)

    with safe_open(str(handoff_dir / "gradients.safetensors"), framework="pt") as fa, \
         safe_open(str(unpacked / "gradients.safetensors"), framework="pt") as fb:
        assert set(fa.keys()) == set(fb.keys()) and fa.metadata() == fb.metadata()
        for k in fa.keys():
            ga, gb = fa.get_tensor(k), fb.get_tensor(k)
            assert ga.dtype == gb.dtype and torch.equal(
                ga.view(-1).view(torch.uint8), gb.view(-1).view(torch.uint8)
            ), f"gradient {k} differs after the round trip"

    res_b = audit_replay(
        str(unpacked),
        until_step=7,
        expect_hash=str(step7 / "state_hash.txt"),
        device="cpu",
    )
    assert res_b["state_hash"] == hashes[7], (
        f"user B's replay from the round-tripped hand-off reproduced "
        f"{res_b['state_hash'][:16]}…, run log says {hashes[7][:16]}…"
    )
    assert res_b["match"] is True
