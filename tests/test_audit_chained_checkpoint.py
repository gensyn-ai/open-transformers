"""Contract test for the chained-audit checkpoint save
(``audit_replay._save_chained_audit_checkpoint`` / ``--save-checkpoint-dir``).

Validates the file-layout handoff between a user who audits interval T→T+1 and
saves, and a second user who audits T+1→T+2 by loading that save: the saved dir
must be a fully-formed checkpoint that ``Checkpointer.load`` restores, with the
metadata advanced correctly and every virtual rank's batch-hasher chain present.

Uses a plain ``nn.Module`` (not ``build_model``) so it runs without a built
``repop`` backend — the save/load path under test is repop-independent; only the
model *kernels* the full audit replays need repop.
"""

from __future__ import annotations

import json

import torch

from pretrain.cli.audit_replay import _save_chained_audit_checkpoint
from pretrain.data.global_stream import GlobalStreamState
from pretrain.train.checkpoint import CheckpointMeta, Checkpointer
from pretrain.train.spike_protocol import SpikeProtocol
from pretrain.train.state_hash import RunningBatchHasher


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))


def _orig_meta_obj(N: int, dp_shard: int) -> dict:
    """A meta.json dict as an auditable run's loop would have written it."""
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
    # Round-trip through json exactly like reading the file back.
    import dataclasses

    return json.loads(json.dumps(dataclasses.asdict(meta)))


def test_chained_audit_checkpoint_roundtrip(tmp_path):
    N, dp_shard = 4, 2
    meta_obj = _orig_meta_obj(N, dp_shard)

    model = _tiny_model()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # One step so the optimizer carries real moment state to persist.
    model(torch.randn(2, 8)).sum().backward()
    optim.step()
    optim.zero_grad(set_to_none=True)

    # Per-virtual-rank batch hashers with DISTINCT chains, so a rank mix-up in the
    # save (e.g. writing rank 0's digest to every file) would be caught.
    hashers = [RunningBatchHasher() for _ in range(N)]
    for r, h in enumerate(hashers):
        h.update({"input_ids": torch.full((1, 4), r + 1), "labels": torch.zeros(1, 4)})
    rank_digests = [h.local_digest() for h in hashers]
    assert len(set(rank_digests)) == N  # all distinct

    stream_state = GlobalStreamState(
        consumed_documents_per_source={"web": 123},
        epoch_per_source={"web": 0},
        windows_emitted=512,
    )
    spike = SpikeProtocol(
        threshold=100.0,
        skips_in_window_to_halt=3,
        halt_window_steps=50,
        skip_steps_on_spike=2,
        start_step=0,
    )

    target_step, target_consumed = 101, 1010
    digest = "b" * 32  # the target-step hash the audit verified
    # Hash-due target: the running chain advanced to the same value, so meta
    # and state_hash.txt coincide (the off-cadence split has its own test).
    saved = _save_chained_audit_checkpoint(
        save_dir=str(tmp_path / "handoff"),
        step=target_step,
        consumed=target_consumed,
        digest=digest,
        chained_hash_meta=digest,
        meta_obj=meta_obj,
        stream_state=stream_state,
        model=model,
        optimizer=optim,
        spike_state=spike.state_dict(),
        batch_hashers=hashers,
        batch_digest=b"\x00" * 32,
        N=N,
    )

    # ---- file-layout contract -------------------------------------------------
    assert saved.name == f"step_{target_step:09d}"
    assert (saved / "_COMPLETE").exists()
    assert (saved / "meta.json").exists()
    assert (saved / "global_stream.json").exists()
    assert (saved / "spike_protocol.json").exists()
    assert (saved / "state_hash.txt").read_text().strip() == digest
    assert (saved / "rng.rank_0.pt").exists()
    # Every virtual rank's batch-hasher chain must be present (the next audit
    # primes r in range(N)) and byte-equal to that rank's digest.
    for r in range(N):
        p = saved / f"batch_hasher.rank_{r}.bin"
        assert p.exists(), f"missing {p.name}"
        assert p.read_bytes() == rank_digests[r]

    # ---- the second user loads it --------------------------------------------
    model2 = _tiny_model()
    # Perturb so a failed load would be visible.
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(1.0)
    optim2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    loaded_stream, loaded_meta, _extras = Checkpointer(str(tmp_path / "handoff")).load(
        saved, model2, optim2
    )

    # Metadata: advanced fields updated, run descriptor preserved verbatim.
    assert loaded_meta.step == target_step
    assert loaded_meta.consumed_tokens == target_consumed
    assert loaded_meta.chained_hash == digest
    assert loaded_meta.windows_emitted == 512
    assert loaded_meta.seed == 42
    assert loaded_meta.reduction_mode == "deterministic_allgather"
    assert loaded_meta.dp_world_size == N and loaded_meta.dp_shard == dp_shard
    assert loaded_meta.clip_algo == "global"
    assert loaded_meta.repop_env["REPOP_EXECUTION_MODE"] == "cross_device_reproducible"

    # Global stream position round-trips (resumes the NEXT interval).
    assert isinstance(loaded_stream, GlobalStreamState)
    assert loaded_stream.windows_emitted == 512
    assert loaded_stream.consumed_documents_per_source == {"web": 123}

    # Model weights restored bit-exactly from the handoff.
    for (n, p1), (_, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1.detach(), p2.detach()), f"param {n} differs after load"


def test_off_cadence_save_splits_chain_and_target_hash(tmp_path):
    """An off-cadence target (a checkpoint between hash-due steps) hands off
    TWO distinct values, mirroring loop._save_checkpoint exactly: meta.json's
    chained_hash carries the RUNNING chain (what the next interval's hashes
    chain from), while state_hash.txt stores the target-step hash chained from
    it (the value the loop stamps on every checkpoint it saves). Writing the
    target hash into meta too would fork the chain at every off-cadence hop."""
    N, dp_shard = 2, 2
    meta_obj = _orig_meta_obj(N, dp_shard)

    model = _tiny_model()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 8)).sum().backward()
    optim.step()
    optim.zero_grad(set_to_none=True)

    running_chain = "c" * 32   # chain value at the last hash-due step
    target_hash = "d" * 32     # H(state@target, prev=running_chain)

    saved = _save_chained_audit_checkpoint(
        save_dir=str(tmp_path / "handoff-off-cadence"),
        step=150,
        consumed=1500,
        digest=target_hash,
        chained_hash_meta=running_chain,
        meta_obj=meta_obj,
        stream_state=GlobalStreamState(
            consumed_documents_per_source={"web": 200},
            epoch_per_source={"web": 0},
            windows_emitted=600,
        ),
        model=model,
        optimizer=optim,
        spike_state=SpikeProtocol(
            threshold=100.0,
            skips_in_window_to_halt=3,
            halt_window_steps=50,
            skip_steps_on_spike=2,
            start_step=0,
        ).state_dict(),
        batch_hashers=None,
        batch_digest=None,
        N=N,
    )

    assert (saved / "state_hash.txt").read_text().strip() == target_hash
    meta = json.loads((saved / "meta.json").read_text())
    assert meta["chained_hash"] == running_chain

    # The next audit seeds its chain from meta (not from state_hash.txt).
    model2 = _tiny_model()
    optim2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    _, loaded_meta, _ = Checkpointer(str(tmp_path / "handoff-off-cadence")).load(
        saved, model2, optim2
    )
    assert loaded_meta.chained_hash == running_chain
    assert loaded_meta.step == 150
