"""Mix sampler determinism + distribution tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pretrain.data.indexed_dataset import IndexedDatasetWriter
from pretrain.data.manifest import ShardInfo, SourceManifest
from pretrain.data.mix_sampler import MixSampler, MixSamplerState


def _make_source(
    base_dir: Path,
    name: str,
    n_docs: int,
    base: int,
    doc_len: int = 10,
) -> tuple[SourceManifest, str]:
    src_dir = base_dir / name
    src_dir.mkdir(parents=True)
    prefix = src_dir / f"{name}_00000"
    with IndexedDatasetWriter(prefix, dtype=np.uint32) as w:
        for i in range(n_docs):
            w.add_document(
                np.arange(base + i * doc_len, base + (i + 1) * doc_len, dtype=np.uint32)
            )
    manifest = SourceManifest(
        name=name,
        tokenizer_hash="deadbeef",
        dtype="uint32",
        shards=[
            ShardInfo(
                prefix=f"{name}_00000",
                num_documents=n_docs,
                token_count=n_docs * doc_len,
            )
        ],
    )
    manifest.save(src_dir / "manifest.yaml")
    return manifest, str(src_dir)


def test_determinism(tmp_path):
    a, da = _make_source(tmp_path, "a", 50, 0)
    b, db = _make_source(tmp_path, "b", 30, 1000)

    s1 = MixSampler([a, b], [da, db], [0.7, 0.3], seq_len=64, seed=42, eos_id=999)
    s2 = MixSampler([a, b], [da, db], [0.7, 0.3], seq_len=64, seed=42, eos_id=999)

    chunks_1 = [next(iter(s1)) for _ in range(5)]
    chunks_2 = [next(iter(s2)) for _ in range(5)]
    for c1, c2 in zip(chunks_1, chunks_2):
        np.testing.assert_array_equal(c1, c2)


def test_different_seeds_diverge(tmp_path):
    a, da = _make_source(tmp_path, "a", 50, 0)
    b, db = _make_source(tmp_path, "b", 30, 1000)
    s1 = MixSampler([a, b], [da, db], [0.7, 0.3], seq_len=64, seed=1, eos_id=999)
    s2 = MixSampler([a, b], [da, db], [0.7, 0.3], seq_len=64, seed=2, eos_id=999)
    c1 = next(iter(s1))
    c2 = next(iter(s2))
    assert not np.array_equal(c1, c2)


def test_distribution_matches_weights(tmp_path):
    # Source A: tokens in [0, 1000), Source B: tokens in [10_000, 11_000), EOS=999.
    a, da = _make_source(tmp_path, "a", 100, 0, doc_len=10)
    b, db = _make_source(tmp_path, "b", 100, 10_000, doc_len=10)
    weights = [0.7, 0.3]
    s = MixSampler([a, b], [da, db], weights, seq_len=20_000, seed=0, eos_id=999)
    big = next(iter(s))
    is_a = int(np.sum((big >= 0) & (big < 1000) & (big != 999)))
    is_b = int(np.sum((big >= 10_000) & (big < 11_000)))
    total = is_a + is_b
    a_frac = is_a / total
    # 1000+ docs sampled; deviation should be < 5pp.
    assert abs(a_frac - 0.7) < 0.05, f"A fraction {a_frac:.3f} too far from 0.7"


def test_rank_aware_disjoint(tmp_path):
    a, da = _make_source(tmp_path, "a", 200, 0, doc_len=10)
    # World size 4 → each rank should see roughly 1/4 of the documents
    # *across an epoch*. We test that two different ranks emit different
    # token streams (the strongest cheap check).
    s_r0 = MixSampler([a], [da], [1.0], seq_len=200, seed=42, rank=0, world_size=4, eos_id=999)
    s_r1 = MixSampler([a], [da], [1.0], seq_len=200, seed=42, rank=1, world_size=4, eos_id=999)
    c0 = next(iter(s_r0))
    c1 = next(iter(s_r1))
    assert not np.array_equal(c0, c1)


def test_resume_from_state(tmp_path):
    a, da = _make_source(tmp_path, "a", 50, 0)
    b, db = _make_source(tmp_path, "b", 30, 1000)

    s1 = MixSampler([a, b], [da, db], [0.7, 0.3], seq_len=64, seed=7, eos_id=999)
    it1 = iter(s1)
    chunks_pre = [next(it1) for _ in range(3)]
    state = s1.state()
    chunks_post = [next(it1) for _ in range(3)]

    # Restart with the saved state — should match chunks_post exactly.
    s2 = MixSampler(
        [a, b], [da, db], [0.7, 0.3], seq_len=64, seed=7, eos_id=999, state=state
    )
    it2 = iter(s2)
    chunks_resumed = [next(it2) for _ in range(3)]
    for i, (cp, cr) in enumerate(zip(chunks_post, chunks_resumed)):
        np.testing.assert_array_equal(
            cp, cr, err_msg=f"chunk {i} not bit-exact after resume"
        )


def test_token_share_weights_via_build_loader(tmp_path, monkeypatch):
    """Verify build_loader converts token-share weights → doc-probabilities
    that yield the requested token-mix empirically.

    Setup mimics the recipe footgun: two sources with very different avg
    tokens-per-document. A naive per-doc weight of [0.5, 0.5] would give
    a token-mix skewed toward the longer-doc source; with token-share
    weights of [0.5, 0.5] the loader should produce ~50/50 by tokens.
    """
    # short_src: 100 docs × 5 tok = 500 tokens
    # long_src:  100 docs × 50 tok = 5000 tokens
    short_m, short_dir = _make_source(tmp_path, "short_src", 100, 0,    doc_len=5)
    long_m,  long_dir  = _make_source(tmp_path, "long_src",  100, 1000, doc_len=50)

    from pretrain.config.schema import DataConfig, DataSourceConfig
    from pretrain.data.loader import build_loader

    cfg = DataConfig(
        sources=[
            DataSourceConfig(name="short_src", path=short_dir, weight=0.5),
            DataSourceConfig(name="long_src",  path=long_dir,  weight=0.5),
        ],
        seq_len=64,
        document_separator_id=999,
        weights_are_token_shares=True,
    )

    _, sampler = build_loader(
        cfg, micro_batch_size=1, rank=0, world_size=1, seed=0, eos_id=999,
        pin_memory=False,
    )

    # Doc-probabilities must satisfy p_i ∝ τ_i / L_i. With τ=[0.5,0.5] and
    # L=[5,50] → p ∝ [0.1, 0.01], normalize → [10/11, 1/11].
    np.testing.assert_allclose(sampler._weights, [10 / 11, 1 / 11], rtol=1e-9)

    # Empirical token-mix on a long-enough draw should be ~50/50. Each
    # source's contribution = count_of_draws × avg_tok_per_doc. We tag
    # each source's tokens via the disjoint id ranges _make_source uses.
    rng_state = sampler._rng.bit_generator.state
    n_refills = 20_000
    from_short = from_long = 0
    for _ in range(n_refills):
        # Draw a single source pick and one doc; bypass packing.
        src = int(sampler._rng.choice(2, p=sampler._weights))
        doc = sampler._walkers[src].next_document()
        # short_src's token IDs are in [0, 500); long_src's are in [1000, 6000).
        if doc[0] < 500:
            from_short += len(doc)
        else:
            from_long += len(doc)
    total = from_short + from_long
    short_share = from_short / total
    # Tolerance: ~50/50 with sampling noise; should land within ±2pp at 20k draws.
    assert 0.48 < short_share < 0.52, (
        f"expected ~50/50 token-mix, got short={short_share:.3f} long={1-short_share:.3f}"
    )
