"""Regression: final / emergency checkpoints stamp the at-step hash.

The 1B run's final checkpoint (step 80957 — the run's only off-cadence save)
stored a ``state_hash.txt`` that matched NO log and that no replay could
reproduce: the save fired OUTSIDE the step loop, after ``zero_grad`` and after
the running chain had already advanced to the step's own hash, so the
post-hoc recompute both double-linked the chain (H(state, prev=itself)) and
hashed the cleared grads as "grad_none". The same defect was latent in the
emergency (SIGTERM) save path.

These tests drive the REAL ``pretrain.train.loop.train`` on CPU (tiny model,
synthetic shards) and pin the contract:

  1. A run whose final step is OFF the checkpoint cadence (ckpt every 5,
     total tokens end at step 7) writes a final checkpoint whose
     state_hash.txt equals that step's ``logs/state_hashes.jsonl`` entry, and
     an ``audit_replay`` of the last interval taking --expect-hash from that
     very file PASSES.
  2. Same for a forced emergency (SIGTERM-flag) save landing mid-run on a
     step with no cadence checkpoint at all.

Skipped when ``repop`` isn't importable (model construction goes through
repop kernels).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("repop")

from pretrain.cli.audit_replay import audit_replay  # noqa: E402
from pretrain.config import load_config  # noqa: E402
from pretrain.data.indexed_dataset import IndexedDatasetWriter  # noqa: E402
from pretrain.data.manifest import ShardInfo, SourceManifest  # noqa: E402

_SEQ_LEN = 64          # % 32 == 0 (repop flash pad contract)
_MB_SIZE = 2
_GLOBAL_BATCH = 256    # -> 2 microbatches/step at mb 2 x seq 64
_TOKENS_PER_STEP = _GLOBAL_BATCH


def _write_synthetic_source(base: Path, name: str, n_docs: int) -> None:
    """Same layout as ``prepare_data synthetic`` (IndexedDataset + manifest)."""
    src_dir = base / name
    src_dir.mkdir(parents=True, exist_ok=True)
    prefix = src_dir / f"{name}_00000"
    rng = np.random.default_rng(42)
    doc_len = 96
    with IndexedDatasetWriter(prefix, dtype=np.uint32) as w:
        for _ in range(n_docs):
            w.add_document(rng.integers(1, 4096, size=doc_len, dtype=np.uint32))
    SourceManifest(
        name=name,
        tokenizer_hash="synthetic-no-tokenizer",
        dtype="uint32",
        shards=[
            ShardInfo(
                prefix=f"{name}_00000",
                num_documents=n_docs,
                token_count=n_docs * doc_len,
            )
        ],
    ).save(src_dir / "manifest.yaml")


def _scrub_local_repop_env(monkeypatch) -> None:
    """repop's import sets REPOP_METAL_SHADER_DIR on macOS. A genuine cluster
    run never records it, and audit_replay's repop_env allowlist refuses metas
    that do — drop it so the loop's ``_capture_repop_env`` matches a real run
    (the CPU-only test never dispatches to Metal)."""
    monkeypatch.delenv("REPOP_METAL_SHADER_DIR", raising=False)


def _tiny_cfg(tmp_path: Path, *, run_id: str, total_steps: int, ckpt_every_steps: int):
    """The 100m smoke config shrunk to seconds on CPU. Auditable mode
    (deterministic_allgather + every-step hashing) is kept — it is the mode
    under test. QAT/AC/bf16 are switched off: orthogonal to the save/hash
    threading this file pins, and each adds CPU time."""
    cfg = load_config("100m_smoke_repop")
    cfg.model.d_model = 64
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.head_dim = 16
    cfg.model.ffn_intermediate = 64
    cfg.model.n_layers = 2
    cfg.model.vocab_size = 4096
    cfg.model.max_seq_len_pretrain = 128
    cfg.model.qat.enabled = False
    cfg.model.qat.scale_refresh_every_n_steps = 0
    cfg.model.qat.enable_at_step = 0

    cfg.train.seq_len = _SEQ_LEN
    cfg.data.seq_len = _SEQ_LEN
    cfg.train.micro_batch_size = _MB_SIZE
    cfg.train.global_batch_tokens.warmup = _GLOBAL_BATCH
    cfg.train.global_batch_tokens.main = _GLOBAL_BATCH
    cfg.train.global_batch_tokens.late = _GLOBAL_BATCH
    cfg.train.total_tokens = total_steps * _TOKENS_PER_STEP
    cfg.train.ckpt_every_steps = ckpt_every_steps
    cfg.train.state_hash.every_n_steps = 1
    cfg.train.state_hash.at_init = True
    cfg.train.state_hash.include_grads = True

    cfg.run.run_id = run_id
    cfg.run.output_dir = str(tmp_path / "runs")
    cfg.run.mixed_precision = False
    cfg.run.activation_checkpoint = False

    data_dir = tmp_path / "shards"
    _write_synthetic_source(data_dir, "synth_a", 100)
    _write_synthetic_source(data_dir, "synth_b", 100)
    cfg.data.sources[0].path = str(data_dir / "synth_a")
    cfg.data.sources[1].path = str(data_dir / "synth_b")
    return cfg


def _run_dir(cfg) -> Path:
    return Path(cfg.run.output_dir) / cfg.run.run_id


def _jsonl_hashes(run_dir: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in (run_dir / "logs" / "state_hashes.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("kind") != "init":
            out[rec["step"]] = rec["state_hash"]
    return out


def test_final_off_cadence_ckpt_hash_matches_log_and_audits(tmp_path, monkeypatch):
    """ckpt_every_steps=5, run ends at step 7: the final save is off-cadence.
    Its state_hash.txt must equal logs/state_hashes.jsonl[7] (every step is
    hash-due), meta.chained_hash must agree (no double-link), and the 5->7
    interval must audit PASS against the checkpoint's own state_hash.txt —
    exactly the --expect-hash source the 80957 artifact broke."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _scrub_local_repop_env(monkeypatch)
    from pretrain.train.loop import train

    cfg = _tiny_cfg(tmp_path, run_id="final7", total_steps=7, ckpt_every_steps=5)
    train(cfg)

    run_dir = _run_dir(cfg)
    ckpts = run_dir / "checkpoints"
    step5, step7 = ckpts / "step_000000005", ckpts / "step_000000007"
    assert step5.is_dir(), "cadence checkpoint at step 5 missing"
    assert step7.is_dir(), "final off-cadence checkpoint at step 7 missing"

    hashes = _jsonl_hashes(run_dir)
    assert 7 in hashes, "step 7 missing from state_hashes.jsonl"
    stamped = (step7 / "state_hash.txt").read_text().strip()
    assert stamped == hashes[7], (
        f"final checkpoint state_hash.txt {stamped[:16]}… != logged step-7 "
        f"hash {hashes[7][:16]}… — the final save recomputed instead of "
        f"threading the at-step hash"
    )
    # No double-link: the chain advanced to step 7's hash BEFORE the save, and
    # both files must carry that same value (step 7 is hash-due).
    meta = json.loads((step7 / "meta.json").read_text())
    assert meta["chained_hash"] == hashes[7]

    # The auditor's documented flow: --expect-hash from the target
    # checkpoint's state_hash.txt. A bit-perfect replay of 5->7 must PASS.
    res = audit_replay(
        str(step5),
        until_step=7,
        expect_hash=str(step7 / "state_hash.txt"),
        device="cpu",
    )
    assert res["state_hash"] == hashes[7], (
        f"replay reproduced {res['state_hash'][:16]}…, log says {hashes[7][:16]}…"
    )
    assert res["match"] is True


def test_emergency_save_ckpt_hash_matches_log_and_audits(tmp_path, monkeypatch):
    """Force the SIGTERM flag mid-step-3 of a run with no cadence checkpoints
    (ckpt_every_steps=100). The emergency save must land AT step 3 with that
    step's logged hash, and audit --from-init (init -> 3 replay) must PASS
    against the emergency checkpoint's own state_hash.txt."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _scrub_local_repop_env(monkeypatch)
    import pretrain.train.loop as loop_mod
    from pretrain.train.spike_protocol import SpikeProtocol

    # Ensure the module flag starts clean and is restored after the test.
    monkeypatch.setattr(loop_mod, "_emergency_exit_requested", False)

    # Trip the flag while step 3 is executing: ``should_skip`` runs mid-step
    # with the pre-increment counter (2), so the post-step emergency poll sees
    # the flag at optimizer_step == 3 — grads live, chain not yet consulted.
    orig_should_skip = SpikeProtocol.should_skip

    def tripping_should_skip(self, **kw):
        if kw.get("step") == 2:
            loop_mod._emergency_exit_requested = True
        return orig_should_skip(self, **kw)

    monkeypatch.setattr(SpikeProtocol, "should_skip", tripping_should_skip)

    cfg = _tiny_cfg(tmp_path, run_id="sigterm3", total_steps=50, ckpt_every_steps=100)
    with pytest.raises(SystemExit) as exc:
        loop_mod.train(cfg)
    assert exc.value.code == 0

    run_dir = _run_dir(cfg)
    ckpts = sorted(p.name for p in (run_dir / "checkpoints").iterdir() if p.is_dir())
    assert ckpts == ["step_000000003"], (
        f"expected exactly the emergency checkpoint at step 3, got {ckpts}"
    )
    step3 = run_dir / "checkpoints" / "step_000000003"

    hashes = _jsonl_hashes(run_dir)
    assert 3 in hashes, "step 3 missing from state_hashes.jsonl"
    stamped = (step3 / "state_hash.txt").read_text().strip()
    assert stamped == hashes[3], (
        f"emergency checkpoint state_hash.txt {stamped[:16]}… != logged "
        f"step-3 hash {hashes[3][:16]}… (latent emergency-path defect)"
    )
    meta = json.loads((step3 / "meta.json").read_text())
    assert meta["chained_hash"] == hashes[3]

    # Verify by replay from init (the only earlier trusted state — there is
    # deliberately no cadence checkpoint before the emergency save).
    res = audit_replay(str(step3), from_init=True, device="cpu")
    assert res["state_hash"] == hashes[3]
    assert res["match"] is True
