# 03 — Data Pipeline

*Translates `research/01_pretraining_corpus.md` into the engineering
machinery that produces deterministic, rank-aware, multi-source token
streams. The data pipeline is where most pretraining incidents start;
treat it that way.*

---

## 1. The corpus mix (recipe_v1)

| Source | Weight | License | Raw token count (source-tokenizer) | Where to fetch |
|---|---|---|---|---|
| DCLM-Baseline 1.0 | **75 %** | CC-BY-4.0 | 4 T (GPT-NeoX) | `mlfoundations/dclm-baseline-1.0` (HF) |
| FineWeb-Edu (`int_score≥3`) | **12 %** | ODC-By | 1.3 T (GPT-2) | `HuggingFaceFW/fineweb-edu` (HF) |
| The Stack v2, dedup, permissive sub-pool | **10 %** | various permissive | ~ 900 B (StarCoder tok.) | `bigcode/the-stack-v2-dedup` (HF) — filter to permissive licenses on ingest |
| Proof-Pile-2 | **3 %** | various permissive | ~ 55 B (Llama tok.) | `EleutherAI/proof-pile-2` (HF) |

Weights are **token weights after re-tokenisation** with our 128 k tokenizer
(see §3). Counts on the dataset cards are not directly usable.

## 2. Tokenizer

**Decision: train a fresh byte-level BPE, vocab 128 256 (multiple of 128
for tensor-core friendliness), on a representative sample of the
pretraining mix.**

Trained once, then frozen. The trained `tokenizer.json` is committed in
artifact storage with a content hash; the hash is recorded in every
checkpoint.

### Sample for tokenizer training

- ~ 50 GB sampled stratified across the four sources at the recipe weights
- English + ≥ 10 other languages (so the byte-level encoder sees enough
  multilingual byte coverage; we do not pretrain multilingual but want
  later flexibility)
- Code from Stack v2 with explicit indent/newline normalisation off

Trainer: HuggingFace `tokenizers` library, `ByteLevel` pre-tokeniser, BPE
trainer, `add_prefix_space=False`, `byte_fallback=True`, explicit digit
splitting per Llama 3. Trained in one job, single CPU node, ~ 6–8 hours.

### Why we train rather than reuse Llama 3's

- Reproducibility. The Llama 3 tokenizer ships under a license that
  complicates downstream reuse for some consumers; ours is fully
  open-licensed.
- We control the sample distribution.
- Risk: a fresh tokenizer is one of the few things that breaks if you get
  wrong (bad tokenization → wasted FLOPs forever). Mitigation: a unit
  test computes bytes-per-token on a held-out 1 GB sample and asserts
  within 5 % of Llama 3 tokenizer's value before we accept the tokenizer.

## 3. Re-tokenisation discipline (load-bearing)

Token counts on dataset cards are GPT-2 / GPT-NeoX. **Mixing weights and
token-budget targets are computed against our 128 k tokenizer's counts,
not the source counts.** This is `01_…` §6's most concrete debt-prevention
note and we wire it into the pipeline rather than rely on convention.

The data preparation pipeline:

```
parquet shards → text streams → our tokenizer →
indexed binary shards (.bin / .idx, Megatron format) →
shuffled shard list per source →
mix sampler →
DataLoader
```

Implemented in `src/pretrain/data/prepare.py`. Run once per source, on
CPU nodes, output written to NVMe. **Output token counts per source are
the source of truth for mix weights** — a manifest YAML records them and
the mix sampler reads from it.

## 4. Indexed binary format

**Decision: Megatron-style `.bin` + `.idx` indexed binary**, vendored as
~ 200 LOC inside `src/pretrain/data/indexed_dataset.py`. We do not depend
on the Megatron package itself.

### Why this format

- O(1) random access by document index → trivial deterministic shuffling
  with a seed.
- mmap-backed → effectively zero-copy reads, no JSON / parquet decode in
  the hot path.
- Battle-tested at every NeMo / Megatron pretrain we know of.
- Format spec is short and stable; vendoring is safer than depending on
  Megatron-Core's evolving Python API.

### Why not WebDataset / Mosaic StreamingDataset / HF datasets

- WebDataset: tar-based, sequential; needs a shuffle buffer for
  randomness. Slower for our pattern.
- Mosaic StreamingDataset: excellent for cloud-streaming setups but our
  data is local NVMe, where its compression layer adds latency for no
  gain.
- HF `datasets`: parquet is fine for prep / inspection, terrible in the
  pretrain hot path (Arrow decode CPU cost is non-trivial at 2 M
  tokens/step).

### Sharding parameters

- 1 GB per `.bin` shard (matches NeMo's default and gives ~thousands of
  shards per source).
- One `.idx` per shard.
- Manifest YAML lists shards + cumulative token counts per source.

## 5. Mix sampler (the multi-source weighting)

**Decision: deterministic per-step weighted sampling across sources, with
shard-level shuffling per epoch and document-level shuffling within
shards.**

### Algorithm

For each global step:
1. Each rank consumes `micro_batch_size · seq_len` tokens.
2. The mix sampler treats the four sources as a categorical distribution
   `p = (0.75, 0.12, 0.10, 0.03)`; for each *document* (or document
   chunk) it samples a source.
3. Within a source, draw the next document from a deterministic
   shard-and-document iterator.

The sampler is rank-aware: rank `r` of `R` reads only documents whose
`document_id mod R == r`. This is the simplest correct way to keep the
data loader DDP/FSDP-clean today and unchanged at multi-node tomorrow.

### Sequence packing

We use **document concatenation with EOS separators** rather than
strict-document packing. This is the approach used by Llama 3 / DCLM /
FineWeb training reports. Attention masks default to causal (no
intra-document boundary mask) since cross-document attention is the
common practice in published recipes. (FlashAttention `varlen` packing is
available as a config flag if we want strict isolation in the long-ctx
anneal phase.)

### Determinism

- Seed propagates into sampler, shuffle, and torch RNG.
- Resume-from-checkpoint records the consumed-token count per source;
  the sampler fast-forwards using the seed + count.
- Unit-tested: same seed + same step → same documents.

## 6. The dataloader

```python
loader = build_loader(
    sources=manifest.sources,
    weights=cfg.data.weights,
    seq_len=cfg.train.seq_len,
    micro_batch_size=cfg.train.micro_batch_size,
    rank=dist.get_rank(),
    world_size=dist.get_world_size(),
    seed=cfg.run.seed,
    consumed_tokens_per_source=resume_state.consumed,
)
```

- `num_workers=4` per rank; mmap reads make this almost free.
- `pin_memory=True`.
- `prefetch_factor=2`.
- No fancy dynamic-batching: fixed `seq_len` (4 096) and fixed
  `micro_batch_size`; the global batch grows by adjusting gradient
  accumulation, not sequence shape (`05_training.md` §3).

## 7. Tooling for incidents

The data pipeline ships with three diagnostic CLIs because we WILL
need them:

- `python -m pretrain.cli.prepare_data --inspect <shard.bin>` — dumps
  N decoded documents from a shard to verify tokenization / encoding.
- `python -m pretrain.cli.prepare_data --token-stats` — bytes/token,
  per-source, against the manifest.
- `python -m pretrain.cli.prepare_data --replay-step --step N --seed S`
  — emits exactly the documents the dataloader would produce at step N
  with seed S. Critical for "did the loader give us a bad batch at
  step 12 345?" investigations.

## 8. Data-side tech debt risks (and what we did about each)

| Risk | Mitigation |
|---|---|
| Mix weights silently wrong because someone used source token counts | Manifest YAML is the single source of truth; CI test asserts mix sampler reads from manifest, never from per-source `.idx`. |
| Tokenizer change mid-project requires re-tokenising everything | Tokenizer hash recorded in every checkpoint and shard manifest; loader refuses to consume shards whose hash doesn't match the model's. |
| Stack v2 license filtering is fiddly | Apply during `prepare.py`, not in the loader. The shards on disk are already compliant. Manifest records the filter version. |
| FineWeb-Edu / DCLM share underlying CC documents → near-duplicates | Cross-deduplicate the FineWeb-Edu shards against DCLM-Baseline using a 5-gram MinHash (Zyda-2-style) during prep. Recorded as an explicit prep step, not "we'll do it later". |
| Resume from checkpoint replays already-seen data | Per-source `consumed_tokens` in checkpoint state; sampler fast-forwards. Tested. |
| Long-context anneal needs different seq_len without re-prep | Indexed binary is sequence-length-agnostic; only the loader packs to a different `seq_len`. No re-prep. |
| Shard list shuffle is rank-correlated and we double-train some docs | Shard-level shuffle uses `(epoch_seed, source_id)`; document-level uses `(epoch_seed, source_id, shard_id)`. Tested across `world_size ∈ {1, 4, 8}`. |
