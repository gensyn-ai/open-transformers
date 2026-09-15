"""Smoke test for pretrain.eval.olmes_runner against the installed olmes.

Runs one full OLMES task pair (arc_easy MCF + CF, curated 5-shot, limit=2)
through a tiny random model on CPU. Validates the whole chain: tokenizer
wrap, HFLM_Verbose adapter init, oe_eval task loading + fewshot sources,
batched loglikelihood, per-task metrics, the mc_or_rc aggregate, and
summary.json. Scores are meaningless (random weights); what's asserted is
structure. Re-run this whenever the olmes pin moves.

Usage:  python scripts/repro/olmes_smoke.py [out_dir]
Needs network for the first run (HF datasets: ai2_arc).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

from pretrain.config.schema import ModelConfig
from pretrain.data.tokenizer import Tokenizer, train_tokenizer
from pretrain.eval.olmes_runner import run_olmes
from pretrain.model import build_model


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="olmes_smoke_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tiny tokenizer with the production special tokens, so the wrap path
    # (eos/bos/pad registration) is exercised faithfully.
    tok_path = out_dir / "tokenizer.json"
    if not tok_path.exists():
        sample = ["The quick brown fox jumps over the lazy dog. " * 20] * 200
        train_tokenizer(sample, tok_path, vocab_size=1000, min_frequency=1)
    # Model vocab rounds up to the config's multiple-of-128 constraint;
    # rows past the tokenizer vocab are dead logits, as in production.
    vocab_size = -(-Tokenizer(tok_path).vocab_size // 128) * 128

    cfg = ModelConfig(
        name="tiny_olmes_smoke",
        n_layers=2,
        d_model=128,
        n_heads=2,
        n_kv_heads=1,
        head_dim=64,  # the repop CPU flash kernel's preferred head_dim
        ffn_intermediate=128,
        vocab_size=vocab_size,
        max_seq_len_pretrain=512,
        swa_window=32,
        swa_full_every=2,
        attn_int8_pv=False,
    )
    torch.manual_seed(0)
    model = build_model(cfg, device="cpu")
    model.eval()

    summary = run_olmes(
        model=model,
        tokenizer_path=tok_path,
        out_dir=out_dir,
        tasks=["arc_easy::olmes"],
        batch_size=2,
        max_length=512,
        limit=2,
        device="cpu",
        model_label="tiny-smoke",
    )
    print(json.dumps(summary, indent=2))

    aggs = summary["aggregates"]
    assert "arc_easy::olmes" in aggs, f"missing mc_or_rc aggregate: {aggs}"
    assert summary["num_leaf_tasks"] == 2, summary
    assert (out_dir / "metrics-all.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    print(f"OLMES smoke PASS (out_dir={out_dir})")


if __name__ == "__main__":
    main()
