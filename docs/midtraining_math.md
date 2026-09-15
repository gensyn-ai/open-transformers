# Midtraining on the Dolma3-Dolmino annealing mixture

Implements the mid-training stage of OLMo 2 ([2501.00656](https://arxiv.org/abs/2501.00656) §4):
branch from a pretrain checkpoint, switch the data to a fresh high-quality
mixture, and drive the LR **linearly to zero** over the midtraining token
budget. The data source is `allenai/dolma3_dolmino_mix-100B-1125`
(ingredient1) — the full ~100B-token annealing pool AI2 built for OLMo 3 32B
stage-2 (23 sources: math, code, QA, reasoning, instruction-tuning, natural
web/PDF), not a math-only subset.

> This branch originally restricted to just the **math subset** of the
> older `allenai/dolmino-mix-1124` (OLMo 2 §4.4) — see git history for that
> recipe if you need to reproduce it. It was replaced because the two
> datasets don't share a layout (no `data/math/<component>/` directory,
> no published per-component doc counts) and because the newer, larger mix
> is the one actually worth annealing on for this run.

Most of the machinery already existed (`--resume-from`, per-source stream
state keyed by source name, `resume_reset_optimizer`/`rewarm_tokens`). This
branch adds the pieces that didn't:

| Piece | Where |
| --- | --- |
| `dolma3_dolmino_mix` puller (round-robin across ~146 source dirs, no reweighting needed — it's already AI2's pre-built mixture) | `scripts/build_corpus.py` (`pull_dolma3_dolmino_mix`) |
| `linear_anneal` LR schedule (decay start at an ABSOLUTE token count) | `src/pretrain/optim/schedules.py`, `configs/schedule/linear_anneal.yaml` |
| Resume guard for anchor/budget arithmetic | `src/pretrain/train/midtrain.py` (called from `loop.py`'s resume block) |
| Data recipe | `configs/data/dolma3_dolmino_mix.yaml` |
| Run config (1B v2) | `configs/train/1b_midtrain_math.yaml` |
| Cluster prep job | not tracked here — wraps `scripts/build_corpus.py --only dolma3_dolmino_mix` |

## The recipe, mapped to the paper

- **Data.** `allenai/dolma3_dolmino_mix-100B-1125`, ingredient1 (the
  complete one — ingredient2 is missing the `tinymath-pot` source and has
  a couple of inconsistently-named directories). ~100B tokens across 23
  sources laid out as one directory per source
  (`data/ingredient1-<source>/*.jsonl.zst`); the repo publishes no exact
  per-source doc/token count. Unlike the old math-only subset (7
  hand-weighted components reconstructed from the OLMo 2 paper's table),
  this needs no reweighting — it IS AI2's own pre-proportioned mixture, so
  pulling every file whole reproduces it directly.
- **LR (§4.1).** Linear anneal from the pretrain LR at the branch point to 0.
  Paper finding: the peak-LR choice barely matters; annealing to zero is the
  load-bearing part.
- **Optimizer.** Continuation semantics — moments are carried over
  (`resume_reset_optimizer: false`). Same model/optim config as pretrain, so
  the checkpoint loads clean.
- **Cycling.** At the current default 10B-token midtraining budget, this is
  ~10% of the full ~100B-token mixture — no cycling within one pass. (The
  old math-only subset was ~10.7B tokens total, so a 10B budget there was
  ~1 epoch and up to ~2x repetition was deliberately fine per §4.4.2 —
  raise `MIDTRAIN_TOKENS` well past ~100B before that trade-off is relevant
  again here.)
- **Souping (§4.5).** Train 3 anneals differing only in `run.seed` (data
  order) and average the final weights; equal-or-better than the best single
  run in all six paper mixes. Not automated here — a checkpoint-averaging CLI
  is the natural follow-up.

## Runbook (local / single-node)

```bash
python scripts/build_corpus.py --shard-suffix '' \
    --only dolma3_dolmino_mix --docs-dolma3 999999999999
pretrain-inspect-checkpoint runs/<run>/checkpoints/step_NNNNNN   # -> consumed_tokens
RESUME_FROM=runs/<run>/checkpoints/step_NNNNNN \
    ./scripts/launch_single_node.sh 1b_midtrain_math \
    schedule.anneal_start_tokens=<consumed> \
    train.total_tokens=<consumed + budget> \
    optim.peak_lr=<pretrain LR at consumed>
```

## Component blocking and the raw-level shuffle

Observed on the first anneal run (20260807, on the older math-only subset)
as a square-wave `loss_ce` with smooth within-plateau decay, ~57-optimizer-
step quantum. Three facts compose into it, and they generalize directly to
the current full-mixture source:

1. Multi-source HF datasets ship components in separate directory trees —
   not homogenized upstream.
2. The old `pull_dolmino_math` wrote its 7 components sequentially into one
   JSONL, and `write_shards` cuts that stream in order — so each ~1 GiB
   shard was component-pure. `pull_dolma3_dolmino_mix` improves on this
   (round-robins across all ~146 source directories one file per source per
   round, instead of exhausting each in turn — see its docstring for the
   fairness caveat on small pull targets) but a coarser block-order risk
   remains at the whole-round-robin-cycle granularity.
3. The sampler's per-epoch permutation is **shard-granular**
   (`mix_sampler._build_global_perm`: shard-level shuffle, then within-shard
   shuffle). Fine for the pretrain corpus (each source internally
   homogeneous; the mix RNG interleaves *sources* per document) — but a
   single multi-component source keeps its component blocks at shard size.

Fix: `build_corpus.py` globally shuffles the raw JSONL between pull and shard
(`shuffle_jsonl` — deterministic two-pass bucket shuffle, seeded via
`--shuffle-seed`, recorded in a `.shuffled` sidecar that also serves as the
idempotency gate). Every shard then draws uniformly from the mixture, and the
shard-granular sampler permutation becomes harmless. `_SHUFFLE_BUCKETS` was
raised from 64 to 512 for the ~10x larger raw volume, to keep peak
per-bucket memory in the same ballpark. A raw file pulled before this step
existed converges on the next prep run (shuffled in place, no re-pull) —
but shards built from unshuffled raw are NOT rebuilt while their
`manifest.yaml` exists; delete the shard dir (and optionally the raw) to
regenerate.

## Footguns this branch guards against (and two it can't)

Guarded:

- `train.total_tokens` left at the pretrain value → loop would exit
  immediately with no error. Now fails at launch (`check_linear_anneal_resume`).
- `anneal_start_tokens` not matching the checkpoint → LR silently held at
  peak (or starting mid-decay). Now fails at launch.
- Midtraining source leaking into pretraining preset builds →
  `dolma3_dolmino_mix` lives in `MIDTRAIN_SOURCES`, runs only under `--only`.
- Eval-contaminating splits (e.g. `dolmino_math_gsm8k_socratic_test_0...`)
  embed the split name IN the filename rather than as a path segment —
  `_is_eval_split` matches on a token boundary across the whole path, not
  just whole path segments, so this doesn't silently slip through.

Not mechanically guarded — read before running:

- **Tokenizer skew.** Nothing on the loader path validates `tokenizer_hash`
  across sources. Shard `dolma3_dolmino_mix` with the *same*
  `data/tokenizer.json` as the pretrain corpus or the run trains on garbage
  token ids.
- **Audit continuity.** Swapping the data mixture at resume rebuilds the
  stream with new source names; `state_hash`/`windows_emitted` cross-checks
  against the *pretrain* stream no longer apply from the branch point on.
  The midtrain run starts its own chain (batch-hasher digest carries over).
