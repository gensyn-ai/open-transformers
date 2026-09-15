# Pretraining-Run Design — 7B Dense LLM, Single-Machine Reproducible

*Synthesis of the five topic reports (`01_…` through `05_…`) into a single, citation-backed pretraining recipe. Last updated: 2026-04-27.*

This document consolidates the recommendations. **All numerical claims trace back to one of the five detail reports** in this directory; consult those for primary citations.

---

## TL;DR — the recipe

| Dimension | Recommendation | Anchor |
|---|---|---|
| Total parameters | **~8.0 B** dense decoder-only (Llama 3 8B / Qwen3-8B class) | `03_architecture.md` |
| Tokenizer | Byte-level BPE, **vocab 128,256** (Llama-3 tiktoken-style) | `03_architecture.md` |
| Token budget | **150 B tokens primary target** (Chinchilla-optimal); stretch goal 300–500 B | `02_scaling_laws_token_budget.md` |
| Corpus | **DCLM-Baseline 75 % + FineWeb-Edu 12 % + The Stack v2 10 % + Proof-Pile-2 3 %**, retokenised with our 128k tokenizer before mixing | `01_pretraining_corpus.md` |
| Context length (pretrain) | **8,192**, RoPE θ = 500,000; extend to ≥32k via YaRN in a continued-pretraining phase | `03_architecture.md` |
| Optimizer | **AdamW**, β1=0.9, β2=0.95, ε=1e-8, weight decay 0.1, grad clip 1.0 | `05_optimizer.md` |
| Peak LR / schedule | **3e-4 peak**, 2 000-step linear warmup, cosine decay to 3e-5 (10 %) | `05_optimizer.md` |
| Batch size | Warm up 1 M → **2 M tokens** main → 4 M tokens late phase | `04_batch_size.md` |
| Precision | bf16 mixed-precision activations, fp32 optimizer master weights | `05_optimizer.md` |
| Compute | ≈ 5.9 × 10²¹ FLOPs ⇒ **~5–7 k H100-hours** (~3–4 weeks on a single 8×H100 node at ~50 % MFU) | `02_scaling_laws_token_budget.md`, `04_batch_size.md` |

---

## 1. The model — 8 B dense decoder-only

The closest published reference points (Llama 3 8B, Qwen3-8B, Mistral 7B) **converge on essentially the same architecture**, which is the strongest possible scientific signal: independent labs reaching the same answer. We adopt that consensus.

| Component | Value | Reasoning |
|---|---|---|
| `n_layers` | 32 | Llama 3 8B / Qwen3-8B / Mistral 7B all use 32 |
| `d_model` | 4 096 | Same |
| `n_heads` | 32 (head_dim 128) | Same |
| `n_kv_heads` | 8 (GQA 4:1) | Llama 3 / Qwen3 / Mistral all use 4:1 GQA — Ainslie et al. 2023 |
| FFN | **SwiGLU**, `intermediate_size = 14,336` | Shazeer 2020; same multiplier as Llama 3 8B |
| Normalisation | **Pre-LN RMSNorm** (eps 1e-5) **+ QK-Norm** | RMSNorm = Zhang & Sennrich 2019; QK-Norm from ViT-22B / Qwen3 for cheap stability |
| Position encoding | **RoPE**, θ = 500 000 | Su et al. 2021; θ=500k matches Llama 3 to permit later YaRN extension |
| Tokenizer / vocab | byte-level BPE, **128 256** | Llama 3 default; ~24 % better compression than 32k SP tokenizers |
| Embeddings | **untied** | Llama 3 default; small param overhead, modest quality gain |
| Init | trunc-Normal σ = 0.02; output projections (`o_proj`, `down_proj`) scaled by `1/√(2·n_layers)` | GPT-NeoX-style scaled init |
| Pretraining seq length | **8 192** | Match Llama 3; cheaper than 32k-from-scratch |
| Z-loss | optional, coefficient **1e-5** (Chameleon) — 1e-4 is PaLM's original | PaLM §5 verbatim: `z_loss = 10⁻⁴ · log²(Z)`; Chameleon uses 10⁻⁵ which is more appropriate when QK-Norm is already in place |
| Logit soft-cap | **off** | Gemma-2-only; not adopted by Llama/Qwen |
| MoE | **No** — dense only | Reproducibility + single-machine simplicity |

**Total parameters ≈ 8.0 B** (matches Llama 3 8B / Qwen3-8B class).

This is essentially "Llama 3 8B + QK-Norm" — the intersection of the two best-documented recent SOTA recipes.

---

## 2. The data — DCLM-led mix, retokenised

### Why this mix

* **DCLM-Baseline** is the strongest cited single-source web corpus at the 7B / 1–3 T-token scale: a 7B / 2.6 T-token DCLM run reaches **MMLU 63.7 / CORE 57.1**, beating Llama-2 7B and OLMo-1.7 7B at a fraction of Llama-3's compute (Li et al. 2024). License is **CC-BY-4.0** — the most permissive of the top-tier corpora.
* **FineWeb-Edu** complements DCLM on educational/reasoning benchmarks; cross-deduplication against DCLM (à la Zyda-2) is a known-good combination.
* **The Stack v2** for code (~900 B tokens, permissive-only sub-pool) — needed for code/reasoning capability per the OLMo-2 / SmolLM3 / Llemma recipes.
* **Proof-Pile-2** for math (~55 B tokens) — used by Llemma; cheap to mix in 2–3 %.

### Concrete mix

| Source | Weight | Notes |
|---|---|---|
| DCLM-Baseline | **75 %** | CC-BY-4.0; web backbone |
| FineWeb-Edu (or `…-score-2`) | **12 %** | ODC-By; education-quality boost |
| The Stack v2 (dedup, permissive) | **10 %** | Code |
| Proof-Pile-2 | **3 %** | Math |

**Retokenisation discipline.** Token counts on dataset cards are usually GPT-2 or GPT-NeoX. With our 128k Llama-3 tokenizer the same byte stream produces ~15–25 % fewer tokens. Therefore: (a) state the tokenizer when reporting our budget, (b) re-tokenise every source with the *training* tokenizer before computing mix weights, otherwise weights silently drift (`01_pretraining_corpus.md`, §6).

### Token budget — Chinchilla-optimal vs. over-trained

Chinchilla (Hoffmann et al. 2022) gives 20 tokens/parameter as the compute-optimal ratio. Besiroglu et al. 2024 re-fit the parametric form and the 20:1 rule survives. For 8 B params:

> **8 × 10⁹ × 20 ≈ 160 B tokens (compute-optimal)** ([Hoffmann 2022](https://arxiv.org/abs/2203.15556); [Besiroglu 2024](https://arxiv.org/abs/2404.10102)).

Modern frontier models *over-train* (Llama-3 8B was trained on ~15 T tokens, ~1875:1) for inference-cost reasons (Sardana et al. "Beyond Chinchilla-Optimal"). For a single-machine reproducible run we recommend:

* **Primary target: 150 B tokens** — Chinchilla-optimal, ≈ 5.9 × 10²¹ FLOPs, ~3 weeks on one 8×H100 node.
* **Stretch goal: 300–500 B tokens** — moves the recipe toward the Llama-2 over-training regime, doubling-to-tripling wall-clock but giving meaningfully better downstream metrics.

We deliberately **don't** chase Llama-3-style 1–15 T tokens on a single node — that's tens-to-hundreds of thousand H100-hours and breaks the "single-machine" frame.

---

## 3. The training schedule

### 3.1 Optimizer

**AdamW** is the only optimizer with strong, multi-lab, multi-billion-parameter dense-LLM reproductions. Lion / Sophia / Muon all show promise but lack public 7 B+ dense reproductions outside the originating lab as of 2026-04. Distributed Shampoo won AlgoPerf 2024 External-Tuning but lacks a turnkey single-node 7B recipe. The MLCommons AlgoPerf v0.6 leaderboard has not moved since 2025-03-24 — Distributed Shampoo (0.6244) still leads NadamW (0.4590) in External-Tuning; no Muon, SOAP, or Schedule-Free entries have been added.

| Hyperparameter | Value | Notes |
|---|---|---|
| β1 | 0.9 | GPT-3 / Llama / Chinchilla consensus |
| β2 | **0.95** | Not 0.999 — the modern LLM-pretraining convention |
| ε | 1e-8 | Standard |
| Weight decay (decoupled) | 0.1 | Loshchilov & Hutter 2019; Llama default |
| Gradient clip (global L2) | 1.0 | Llama / GPT-3 default |
| Peak LR | **3e-4** | Llama-1 7B and Llama-3 8B value |
| Min LR | **3e-5** (10 % of peak) | Cosine floor |
| Warmup | **2 000 steps**, linear | ~0.5–1 % of total |
| Schedule | **Cosine decay** to min LR over total tokens | Universal default |
| Precision | **bf16 mixed**, fp32 optimizer master | Standard for 8×H100 |

**Backup plan.** If extending the run mid-training is likely (e.g. swapping in higher-quality data for a "midtraining" anneal), switch the schedule to **Warmup-Stable-Decay (WSD)** (MiniCPM, DeepSeek-V2/V3): the constant "stable" phase makes branching multiple decay tails from a single checkpoint cheap.

**Future upgrade path — re-evaluated.** **Muon** (with Moonlight's WD + update-RMS-rescaling fixes) is the most promising AdamW replacement, but **all public >10B Muon-trained models are MoE** (Kimi K2, GLM-4.5, INTELLECT-3) — there is **no public dense 7B+ Muon reproduction** outside Moonshot AI as of 2026-04. Worse, the Stanford/Marin "Fantastic Pretraining Optimizers" benchmark ([arXiv:2509.02046](https://arxiv.org/abs/2509.02046)) shows the matrix-preconditioner advantage *shrinks with scale* — from ~1.4× at 0.1 B to ~1.1× at 1.2 B. Extrapolating to 8 B suggests the realistic gain is 1.0–1.1×, not 2×. The bake-off is still worth running on the 1 B proxy, but **expectations should be calibrated downward** and AdamW remains the responsible default.

### 3.2 Batch size

Critical-batch-size theory (McCandlish et al. 2018) and the modern fit B*(D) ≈ 22.91 · D⁰·⁴⁷ (Zhang 2024) say batch should grow as training proceeds. Llama-3, PaLM, GPT-3, DeepSeek-V3 all use a warmup-then-step schedule. The cleanest size-matched anchor is **GPT-3 6.7B**, which targeted **2 M tokens** per Brown et al. 2020 Table 2.1 — directly supporting the 2 M main-phase choice below. The Llama-3 8B and Qwen3 8B specific batch numbers are *not publicly disclosed* (only the 405B 4 M → 8 M → 16 M schedule is). For our 8 B / 150 B-token target on 8×H100:

| Phase | Tokens | Global batch | Grad-accum (micro_bs=4, seq=4k, DP=8) |
|---|---|---|---|
| Warm-up | 0 → 4 B | 1 M tokens (256 seq) | 8 |
| Main | 4 B → 140 B | **2 M tokens (512 seq)** | 16 |
| Late | 140 B → 150 B | 4 M tokens (1024 seq) | 32 |
| Optional long-ctx anneal | +20 B at 8k | 4 M tokens | 32 |

Sanity check: a 4 M-token step is ≈ 1.7 × 10¹⁷ FLOPs ≈ 42 s on 8×H100 at ~50 % bf16 MFU, consistent with MosaicML's published MPT-7B numbers (440 × A100-40 GB × 9.5 d ≈ 1 T tokens).

### 3.3 Compute estimate

* FLOPs (6ND approximation, Kaplan 2020 / Hoffmann 2022): **6 · 8 × 10⁹ · 150 × 10⁹ ≈ 5.9 × 10²¹ FLOPs.**
* H100 BF16 peak ≈ 1.0 PFLOP/s; at 50 % MFU → 0.5 PFLOP/s sustained.
* On 8×H100: ~5–7 k GPU-hours ≈ **3–4 weeks wall-clock** for the 150 B-token run.
* Linear scaling implies the 300 B stretch is ~6–8 weeks; 500 B is ~10–13 weeks.

(Cross-checked against Llama-2-7B model card: 184 320 A100-hours / 2 T tokens; MPT-7B blog: 440 A100s × 9.5 d / 1 T tokens.)

---

## 4. Validation plan (for "scientifically supported")

A pretraining run is only as defensible as its evaluation. We adopt two cross-cutting validation gates.

1. **Smaller-scale proxy first.** Before committing 3+ weeks of single-node compute, run a **1 B-parameter, 20 B-token** proxy with the *exact* recipe (architecture, optimizer, schedule, mix, tokenizer). Verify (a) loss curves match published 1 B baselines (e.g. SmolLM-1.7B, Pythia-1B), (b) MMLU/ARC trends are positive, (c) no spikes / NaNs. This is also where Muon / WSD-vs-cosine bake-offs belong if we want to consider them.
2. **Standardised eval suite at 7B.** Use the DCLM CORE/EXTENDED benchmark (53 tasks, scales 412 M → 7 B) as the primary downstream gate (Li et al. 2024) — this gives published comparison points for Llama-2 7B, OLMo-1.7 7B, MAP-Neo 7B, DCLM-Baseline 7B. Eval at 5–10 checkpoints across the run; confirm monotone improvement on the macro-average.

---

## 5. Risks and open questions

| Risk | Mitigation |
|---|---|
| 150 B tokens may be visibly under-trained vs. Llama-3 8B (~15 T tokens). | Position the run as Chinchilla-optimal, not frontier-class; benchmark vs. Llama-2 7B / OLMo-1.7 7B / DCLM-7B-2.6T which trained at comparable budgets. |
| FineWeb / DCLM token counts are GPT-2-tokenized; Llama-3 128k tokenizer compresses ~15–25 % more efficiently. | Re-tokenise all sources with the training tokenizer before computing mix weights and budget targets. |
| Single-machine 8×H100 may not fit `seq=8192 × micro_bs=4` for 8 B with full activation checkpointing. | Drop to seq=4096 main, with an 8k anneal at the end (this matches the schedule above). FSDP-3 + activation checkpointing required. |
| AdamW is the safe choice but the *expected* Muon speedup at 8 B may be ~1.0–1.1×, not 2× (Stanford/Marin 2025). | Run the Muon bake-off on the 1 B proxy with realistic expectations; only swap if the gain is clearly material. |
| DCLM ablation deltas now verified (DCLM Table 9: 7B-2.6T → CORE 57.1 / MMLU 63.7 / EXTENDED 45.4); FineWeb's per-benchmark ablation is rendered as a figure (Fig. 10) and is **not extractable as numbers** from the public PDF. | Two pinpoint FineWeb numbers are available (FineWeb 33 % MMLU / 46 % ARC; FineWeb-Edu 37 % / 57 % at 1.82 B params, 350 B tokens); for full per-benchmark deltas, source the original ablation figure directly. |
| **Nemotron-CC license is the NVIDIA Data Access Agreement for Model Training** — *not* CC-BY and *not* Common Crawl ToU. More restrictive than DCLM-Baseline (CC-BY-4.0) or FineWeb-Edu (ODC-By). | Stick with DCLM + FineWeb-Edu unless the Nemotron-CC license is acceptable for the specific use case; if Nemotron-CC is acceptable, the +5.6 MMLU at 8B/1T (Nemotron-CC-HQ 59.0 vs DCLM 53.4) makes it a serious upgrade candidate. |
| Zyda-2 (5.07 T, ODC-By) is an alternative turnkey mix already cross-deduped against DCLM + FineWeb-Edu. | Swap in if you'd rather not run the cross-dedup pipeline yourself. |

---

## 6. Where to look in the detail reports

| Question | File |
|---|---|
| Which corpus, what license, what mix ratios, why? | `01_pretraining_corpus.md` |
| Why 150 B tokens? Chinchilla derivation, FLOPs, GPU-hours? | `02_scaling_laws_token_budget.md` |
| Architecture choices (RoPE θ, GQA ratio, vocab, init, …)? | `03_architecture.md` |
| Batch-size schedule, critical batch, single-node sizing? | `04_batch_size.md` |
| Optimizer, LR, weight decay, schedule, alternatives? | `05_optimizer.md` |

---

## 7. Sequencing — how to actually run this

1. **Week 0 — tokenizer + data prep.** Train (or adopt) the 128 k Llama-3-style tokenizer. Re-tokenise DCLM-Baseline (75 %), FineWeb-Edu (12 %), Stack v2 dedup (10 %), Proof-Pile-2 (3 %). Pre-shuffle into ~1 GB shards.
2. **Week 1 — 1 B proxy run.** Same architecture downscaled (n_layers=24, d_model=2048, n_heads=16, n_kv_heads=4, intermediate=5632), 20 B tokens, full recipe. Validate loss curve, MMLU, no spikes.
3. **Week 2 — 8 B kickoff.** 2 000-step warmup, batch warm-up to 1 M then 2 M tokens. Eval every 5 B tokens on DCLM-CORE.
4. **Weeks 3–5 — main run** to 140 B tokens at 2 M-token batch.
5. **Week 5 — late phase** to 150 B tokens at 4 M-token batch.
6. **Week 6 (optional)** — long-context anneal (+20 B tokens at seq=8192) and YaRN-extended evaluation at 32 k.
7. **Continuously** — checkpoint every 5 B tokens; run DCLM-CORE eval; archive logs.

This gives a defensible, reproducible 7-8 B pretraining run grounded in published recipes from independent labs, with every numerical choice traceable to a citation in the detail reports.

---

## Appendix A — Follow-up verifications (2026-04-27)

Three follow-up agents ran with WebSearch enabled to fill the caveats flagged in v1 of this synthesis. The full quoted text and per-table numbers live in the appended `## 8.` / `## 10.` / `## 11.` sections of `01_…`, `04_…`, `05_…`, and `03_…` respectively. Headline updates only:

* **DCLM ablation table fully extracted** (Tables 4, 5, 9 of arXiv:2406.11794): 7B-2.6T DCLM-Baseline beats Llama-2 7B by +7.9 CORE / +17.9 MMLU; model-based filtering (fastText OH-2.5+ELI5) beats heuristic-only RefinedWeb at 1B-1x by +2.7 CORE.
* **FineWeb tech-report ablation is figure-only** (Fig. 10 / Appendix Fig. 15) — qualitative ordering preserved, but per-benchmark MMLU/ARC deltas across all 8 corpora are not extractable as numbers from the public PDF. Two pinpoint numbers are quotable.
* **Nemotron-CC license = NVIDIA Data Access Agreement for Model Training** (not CC-BY, not Common Crawl ToU). Headline +5.6 MMLU vs DCLM at 8B / 1T tokens confirmed (Table 5: 59.0 vs 53.4); long-horizon Table 6 confirmed (Nemotron-CC 8B 70.3 MMLU vs Llama-3.1 8B 65.3).
* **GPT-3 batch warmup**: canonical "32 k → 3.2 M tokens over 4–12 B tokens" schedule traced to GPT-3 175B; **GPT-3 6.7B target was 2 M tokens** (Table 2.1) — directly supports our 2 M main-phase choice. Llama-3 8B and Qwen3 8B specific batches remain undisclosed by Meta and Alibaba.
* **PaLM z-loss verbatim**: §5 of arXiv:2204.02311 — `z_loss = 10⁻⁴ · log²(Z)`. Chameleon §2.3 (arXiv:2405.09818) uses **10⁻⁵**; with QK-Norm already specified, the lighter Chameleon coefficient is the better default.
* **Optimizer landscape unchanged**: AlgoPerf v0.6 leaderboard frozen since 2025-03-24; Distributed Shampoo still leads External-Tuning (0.6244) over NadamW (0.4590). **No public dense 7B+ Muon reproduction outside Moonshot AI**, and the matrix-preconditioner advantage shrinks from ~1.4× (0.1 B) to ~1.1× (1.2 B) per Stanford/Marin's "Fantastic Pretraining Optimizers" (arXiv:2509.02046). AdamW recommendation stands; Muon expectations recalibrated downward.

**Net effect on the recipe**: no recommendation reversed, but three small refinements applied — z-loss coefficient lowered to 1e-5 (Chameleon convention), Muon-bake-off expectations recalibrated to 1.0–1.1× rather than 2×, and Nemotron-CC re-classified as license-restricted relative to DCLM and FineWeb-Edu.
