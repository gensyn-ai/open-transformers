# 07 — Validation, Eval, and Acceptance Gates

*The 8 B run is too expensive to validate post-hoc. This document is the
test pyramid, the proxy run, the eval harness, and the explicit gates
between each phase.*

---

## 1. Test pyramid

### 1.1 Unit tests (`tests/`)

Run on every PR, < 5 min on a single GPU.

- `test_model_shapes.py` — instantiate 100 M, 1 B, 8 B (8 B on `meta`
  device); assert parameter count and shapes match the config.
- `test_indexed_dataset.py` — round-trip a synthetic corpus through
  `prepare.py` → indexed binary → loader; assert exact byte equality.
- `test_mix_sampler.py` — assert (a) determinism: same seed + step →
  same documents; (b) source distribution converges to configured
  weights within ε over 1 M samples.
- `test_schedule_math.py` — at every batch-phase boundary and the
  cosine warmup/decay endpoints, LR is exactly the documented value.
- `test_checkpoint_roundtrip.py` — save → reload → 100 more steps
  with identical RNG must match losses bit-for-bit (within fp32 tol)
  vs. an uninterrupted reference.
- `test_init.py` — same seed → same init across world sizes ∈ {1, 4, 8}.
- `test_qknorm_off_on_match.py` — at scale 100 M, QK-Norm config produces
  forward outputs within tolerance of plain GQA at init (sanity).

### 1.2 End-to-end smoke (`tests/e2e/`)

Run on a 1-GPU box, < 30 min.

- `test_100m_one_step.py` — full pipeline: tokenize 1 GB of synthetic
  text, prepare shards, build model, run 10 steps, save checkpoint,
  resume, run 10 more, save again. Asserts:
  - No NaN / Inf at any step.
  - Loss decreases.
  - Checkpoint resume matches an uninterrupted reference for the next 10
    steps.
  - Throughput within 50 % of the documented baseline (catches
    regressions cheaply).

This test is the **M0 gate** and runs in CI on every PR.

### 1.3 1 B proxy run (the M2 gate)

The most important test in this document.

**Recipe**: identical to the 8 B recipe in every dimension that
*can* be identical, scaled only on architecture:

| Field | 1 B proxy | 8 B main |
|---|---|---|
| n_layers | 24 | 32 |
| d_model | 2048 | 4096 |
| n_heads | 16 | 32 |
| n_kv_heads | 4 | 8 |
| ffn_intermediate | 5632 | 14336 |
| Tokens | **20 B** | 150 B |
| Batch tokens (main) | 1 M | 2 M |
| LR / β1 / β2 / WD / clip | identical | identical |
| Schedule | cosine, 2k warmup | cosine, 2k warmup |
| Mix recipe | recipe_v1 | recipe_v1 |
| Tokenizer | identical | identical |
| Optimizer | AdamW | AdamW |
| QK-Norm, z-loss, RoPE θ | identical | identical |

**Acceptance criteria for advancing past M2:**

1. Loss curve qualitatively matches published 1 B baselines
   (Pythia-1B, SmolLM-1.7B early-tokens regime). We chart against a
   fixed reference curve.
2. **No unrecovered loss spikes** (we DO want to see at least one and
   confirm the spike protocol works; rehearsal target: inject one
   artificial spike at step ~ 500 and verify protocol recovers).
3. MMLU 5-shot trends positive over training (we expect ≥ 25 % at end
   of 20 B tokens — random is 25; this is a noise floor check, not a
   quality bar).
4. DCLM-CORE macro-average improves monotonically across at least 3 of
   the 4 mid-training evals.
5. MFU ≥ 45 %.
6. **No `torch.compile` recompilations after warmup.**
7. Checkpoint round-trip test (full save → restart on a second machine
   if available, otherwise same machine) matches loss for 100 steps.
8. Resume-from-spike-rollback rehearsal completed.

If any of these fails, we do not proceed to M3 / 8 B. Period.

This is the line a distinguished fellow draws: 3 weeks of single-node
compute is not a debugging surface.

### 1.4 Optional: Muon bake-off on 1 B proxy

Done **only if** an engineer has explicit owner-time for it. Two 1 B
runs side-by-side: AdamW vs. Muon-hybrid, identical data and schedule.
Decision rule: adopt Muon for 8 B only if downstream eval at 20 B tokens
is within 0.5 pts of AdamW *and* wall-clock is ≥ 1.15× faster.

The research recalibrated expectation is that Muon's 1.4× win at 0.1 B
shrinks to ~ 1.1× at 1.2 B — so we set the bar at 1.15× to make sure the
gain is decisive. Default behaviour is to skip this entirely for the
first project iteration.

## 2. The DCLM-CORE eval harness

**Decision: vendor the DCLM eval scripts (~ 53 tasks across 8 categories,
documented in DCLM repo + `research/01_pretraining_corpus.md`) as
`src/pretrain/eval/dclm_core.py`.**

### What it runs

- DCLM CORE (~22 tasks): the headline downstream comparison.
- DCLM EXTENDED (~31 more tasks): for full picture; lighter weighting
  in our gating logic but logged.

These are the tasks that have published baselines for Llama 2 7B,
OLMo-1.7 7B, MAP-Neo 7B, DCLM-Baseline 7B (see `research/00_synthesis.md`
§4) — i.e., the comparators we will be measured against.

### How it runs

- **Async, on a separate process per node-rank-0**, using a
  fully-materialised eval-only copy of the model (gathered from FSDP2
  shards once per eval).
- Eval batch size is independent of training batch.
- Total time per pass on 8 × H100: ~ 30 minutes (rough estimate;
  measured on the 1 B proxy).
- Triggers: every 5 B tokens during M2 / M4 / M5; ad-hoc on-demand.

### Outputs

- W&B custom panels (one per task category + macro-average).
- A per-eval JSON file in `runs/<run_id>/evals/<step>.json` for offline
  comparison.

### `lm-eval-harness` adapter

For non-DCLM benchmarks people will ask about (HellaSwag, ARC, etc.)
we expose a thin `lm_eval_adapter.py` that wraps our model in the
`lm-eval` HF interface. We do not optimise this path; if anyone wants
deep eval, they can do it post-hoc on a checkpoint.

## 3. Held-out perplexity

A fixed held-out set (250 M tokens; 80 % web, 10 % code, 5 % math,
5 % held-out FineWeb-Edu) is built during data prep and **not used in
training**. Its hash is in the manifest.

PPL on this set is logged every 1 B tokens. Used for:
- Cosine-decay sanity (PPL should decrease monotonically barring micro-
  noise).
- A trivial sanity check that we are not training on our eval set
  (PPL identical to a random initialisation would be a smoking gun).

## 4. Acceptance gates between phases

Hard gates, not "let's discuss":

| Transition | Gate |
|---|---|
| Pre-M0 → M0 (start) | Repo + container build green; smoke test passes on a single GPU |
| M0 → M1 | Smoke test stays green over a week of CI runs |
| M1 → M2 | Token-count manifest matches expected within 2 %; held-out set hash recorded; tokenizer hash committed |
| M2 → M3 | All 1 B proxy acceptance criteria (§1.3) pass |
| M3 → M4 | First 100 steps at 8 B match expected loss and grad-norm bands; MFU ≥ 50 %; no compile recompilations; spike protocol re-rehearsed at 8 B scale (one artificial spike) |
| M4 → M5 | DCLM-CORE macro-average improving monotonically across the last 4 evals |
| M5 → M6 (long ctx) | Optional; gates same as M4→M5 plus long-context 1 B-proxy validation passed |

## 5. What this validation plan prevents

- **Wasted compute on a known-broken pipeline.** The M0/M1/M2 ladder
  catches every category of bug we can catch cheaply.
- **Silent regressions during the run.** Continuous DCLM-CORE eval +
  monotone-improvement gate catches data-quality or LR-schedule bugs
  within 5–10 B tokens of them happening.
- **Surprises at the end.** Held-out PPL plus published-baseline
  comparison on DCLM means we know whether we hit the recipe target by
  step ~ 30 B, not at step 150 B.
- **"Retries" turning into mystery investigations.** Spike protocol +
  rollback recipe is rehearsed at 1 B *before* it is needed at 8 B.

## 6. What this validation plan intentionally does NOT do

- **No automatic gating on absolute MMLU thresholds.** The recipe is
  Chinchilla-optimal at 150 B, not frontier-class at 15 T. We compare
  to Llama 2 7B and DCLM-Baseline 7B / 2.6 T, not Llama 3 8B / 15 T.
  Setting an MMLU threshold would tempt someone to optimise for it.
- **No live A/B test between training runs.** One run, well validated.
- **No human-eval loop.** Out of scope for pretraining; consumers do
  their own SFT/RLHF.
- **No "checkpoint averaging" as quality insurance.** We commit to the
  final checkpoint; if it is bad we dig into root cause, not paper
  over.
