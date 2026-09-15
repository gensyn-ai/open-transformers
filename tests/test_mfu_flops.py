"""MFU FLOP accounting: dense + attention + LM-head terms, and the all-chips
(TP-inclusive) denominator.

Loads ``obs/metrics.py`` by file path so the test doesn't pull the package
``__init__`` chain (wandb / repop), absent in CPU CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "metrics_under_test",
    Path(__file__).resolve().parent.parent / "src" / "pretrain" / "obs" / "metrics.py",
)
_m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_m)
transformer_flops_per_token = _m.transformer_flops_per_token
mfu = _m.mfu


def test_flops_per_token_terms():
    # L=2, d=4, s=8, V=16 → dense 72·L·d²=2304, attn 12·L·s·d=768, head 6·V·d=384.
    f = transformer_flops_per_token(n_layers=2, d_model=4, seq_len=8, vocab_size=16)
    assert f == 2304.0 + 768.0 + 384.0


def test_attention_and_head_are_included():
    """The fix must add the O(seq) attention term + LM head over the bare 6N
    dense count — at long context / large vocab these are far from negligible."""
    L, d, s, V = 32, 4096, 4096, 128_000
    dense_only = 72.0 * L * d**2
    full = transformer_flops_per_token(L, d, s, V)
    attention = 12.0 * L * s * d
    lm_head = 6.0 * V * d
    assert full == dense_only + attention + lm_head
    # Attention alone is a double-digit % of dense at s≈d — not "a few percent".
    assert attention / dense_only > 0.10


def test_mfu_basic_arithmetic():
    f = transformer_flops_per_token(n_layers=2, d_model=4, seq_len=8, vocab_size=16)
    got = mfu(flops_per_token=f, tokens_per_second=10.0, n_gpus=2, peak_flops_per_gpu=1e15)
    assert got == f * 10.0 / (2 * 1e15)


def test_mfu_invariant_under_perfect_strong_scaling():
    """With a fixed global batch, perfect strong scaling means tokens/s ∝ G, so
    MFU is invariant in the GPU count G — the property that makes it a 'true'
    efficiency metric (and the reason inverse-scaling MFU signals bad scaling,
    not a metric artifact)."""
    f = transformer_flops_per_token(n_layers=4, d_model=256, seq_len=1024, vocab_size=4096)
    base_tps = 5_000.0  # tokens/s on 1 GPU
    ref = mfu(flops_per_token=f, tokens_per_second=base_tps, n_gpus=1)
    for g in (2, 4, 8, 16):
        scaled = mfu(flops_per_token=f, tokens_per_second=base_tps * g, n_gpus=g)
        assert abs(scaled - ref) < 1e-18, f"MFU drifted at G={g}: {scaled} vs {ref}"
