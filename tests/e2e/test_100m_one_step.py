"""End-to-end smoke: tokenise tiny synthetic corpus, build model,
run a few train steps, save / resume, run more steps. The M0 gate
(plan/02 §1, plan/07 §1.2) reduced to something a CPU-only laptop
can run in seconds.

Skipped automatically when ``repop`` is not importable: model construction
now goes through repop kernels, so dev machines without the runtime built
get SKIP rather than a collection ERROR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("repop")

from pretrain.config import load_config  # noqa: E402
from pretrain.data.indexed_dataset import IndexedDatasetWriter  # noqa: E402
from pretrain.data.loader import build_loader  # noqa: E402
from pretrain.data.manifest import ShardInfo, SourceManifest  # noqa: E402
from pretrain.data.mix_sampler import MixSampler  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.model.init import init_weights_seeded  # noqa: E402
from pretrain.optim.adamw import build_adamw  # noqa: E402
from pretrain.optim.schedules import build_schedule, schedule_lr  # noqa: E402
from pretrain.train.batch_schedule import grad_accum_steps  # noqa: E402


def _write_synthetic_source(
    base: Path, name: str, n_docs: int, vocab: int = 4096, doc_len: int = 64
) -> tuple[SourceManifest, str]:
    base = base / name
    base.mkdir(parents=True, exist_ok=True)
    prefix = base / f"{name}_00000"
    rng = np.random.default_rng(42)
    with IndexedDatasetWriter(prefix, dtype=np.uint32) as w:
        for _ in range(n_docs):
            w.add_document(rng.integers(0, vocab, size=doc_len, dtype=np.uint32))
    manifest = SourceManifest(
        name=name,
        tokenizer_hash="testhash",
        dtype="uint32",
        shards=[
            ShardInfo(
                prefix=f"{name}_00000",
                num_documents=n_docs,
                token_count=n_docs * doc_len,
            )
        ],
    )
    manifest.save(base / "manifest.yaml")
    return manifest, str(base)


def test_e2e_100m_smoke(tmp_path):
    cfg = load_config("100m_smoke_repop")
    # Override seq_len smaller so test is fast on CPU.
    cfg.train.seq_len = 64
    cfg.train.micro_batch_size = 2
    cfg.model.max_seq_len_pretrain = 128
    cfg.model.vocab_size = 4096
    cfg.model.d_model = 128
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.head_dim = 32
    cfg.model.ffn_intermediate = 256
    cfg.model.n_layers = 2

    # Synthetic data
    a, da = _write_synthetic_source(tmp_path, "a", 100)
    b, db = _write_synthetic_source(tmp_path, "b", 100)
    sampler = MixSampler(
        manifests=[a, b],
        manifest_dirs=[da, db],
        weights=[0.7, 0.3],
        seq_len=cfg.train.seq_len,
        seed=0,
        eos_id=0,
    )

    # Model + optimizer.
    model = build_model(cfg.model, device="cpu")
    init_weights_seeded(model, 1)
    optimizer = build_adamw(model, cfg.optim)
    sched = build_schedule(cfg.schedule, cfg.train, cfg.optim)

    losses_before_resume: list[float] = []
    it = iter(sampler)

    def run_step(step_idx: int, consumed_tokens: int) -> tuple[float, int]:
        accum = grad_accum_steps(consumed_tokens, cfg.train, dp_world_size=1)
        accum = min(accum, 2)  # cap so test is fast
        lr = sched(consumed_tokens, cfg.train.total_tokens) or 1e-4
        schedule_lr(optimizer, lr if lr > 0 else 1e-4)

        ce_total = 0.0
        for _ in range(accum):
            chunks = [next(it) for _ in range(cfg.train.micro_batch_size)]
            batch = torch.from_numpy(np.stack(chunks)).long()
            inp = batch[:, :-1]
            lab = batch[:, 1:]
            out = model(inp)
            ce = torch.nn.functional.cross_entropy(
                out.logits.float().view(-1, out.logits.size(-1)),
                lab.reshape(-1),
            )
            zloss = out.z_loss if out.z_loss is not None else torch.zeros(())
            loss = (ce + zloss) / accum
            loss.backward()
            ce_total += float(ce.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        consumed_tokens += accum * cfg.train.micro_batch_size * cfg.train.seq_len
        return ce_total / accum, consumed_tokens

    consumed = 0
    for step in range(5):
        loss, consumed = run_step(step, consumed)
        losses_before_resume.append(loss)
        assert torch.isfinite(torch.tensor(loss))

    # Save: weights + optimizer state + sampler state.
    blob = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "sampler": sampler.state(),
        "consumed": consumed,
    }
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(blob, ckpt_path)

    # Reload into a fresh model + optimizer.
    model2 = build_model(cfg.model, device="cpu")
    optimizer2 = build_adamw(model2, cfg.optim)
    state = torch.load(ckpt_path, weights_only=False)
    model2.load_state_dict(state["model"])
    optimizer2.load_state_dict(state["optim"])
    sampler2 = MixSampler(
        manifests=[a, b],
        manifest_dirs=[da, db],
        weights=[0.7, 0.3],
        seq_len=cfg.train.seq_len,
        seed=0,
        eos_id=0,
        state=state["sampler"],
    )

    # Continue training for a few more steps. We only assert that the
    # resumed model continues to produce finite losses; bit-exact replay
    # is covered by the dedicated checkpoint test.
    it2 = iter(sampler2)
    consumed2 = state["consumed"]
    for step in range(5, 8):
        accum = 2
        lr = sched(consumed2, cfg.train.total_tokens) or 1e-4
        schedule_lr(optimizer2, lr if lr > 0 else 1e-4)
        ce_total = 0.0
        for _ in range(accum):
            chunks = [next(it2) for _ in range(cfg.train.micro_batch_size)]
            batch = torch.from_numpy(np.stack(chunks)).long()
            out = model2(batch[:, :-1])
            ce = torch.nn.functional.cross_entropy(
                out.logits.float().view(-1, out.logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )
            zloss = out.z_loss if out.z_loss is not None else torch.zeros(())
            loss = (ce + zloss) / accum
            loss.backward()
            ce_total += float(ce.detach())
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        optimizer2.step()
        optimizer2.zero_grad(set_to_none=True)
        consumed2 += accum * cfg.train.micro_batch_size * cfg.train.seq_len
        assert torch.isfinite(torch.tensor(ce_total / accum))
