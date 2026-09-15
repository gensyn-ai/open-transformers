# Implementation Plan — 8B Dense LLM Pretraining

*Companion to `research/00_synthesis.md`. This document set translates the
research recipe into engineering choices that will hold up at implementation
time. Last updated 2026-04-27.*

---

## 1. What we are building

A reproducible single-machine (8 × H100 / DGX-class) pretraining run of an
**8 B-parameter dense decoder-only LLM** ("Llama 3 8B + QK-Norm"), targeting
**150 B tokens** as the primary horizon and 300–500 B as a stretch.

The implementation is a **PyTorch-native, FSDP2-first** codebase that scales
unchanged to multi-node DDP/FSDP2 when we need it, and follows NVIDIA's
playbook (TransformerEngine, FlashAttention, NCCL tuning, NSight profiling,
NGC container) at every level where they offer a measurable win without
locking us into a heavyweight framework.

## 2. Engineering principles (the tech-debt-prevention contract)

These are the principles every other doc in this plan refers back to. If a
choice violates one of these without an ADR explaining why, it should be
rejected at review.

1. **Boring, upstream, battle-tested over clever and bespoke.** No custom
   transformer kernels, no custom optimizer, no custom dataloader format
   that does not already have a published reference. Every layer in this
   stack is something multiple labs are running in production today.
2. **Single-machine today, multi-node-shaped from day one.** All
   parallelism is wrapped behind `torch.distributed` even when world size
   is 1; checkpoints are distributed-format from step 0; the dataloader is
   rank-aware. The single-node→multi-node transition is a config flip, not
   a refactor.
3. **One config, no hidden defaults.** Every hyperparameter, every
   architectural switch, every dataset weight lives in a typed YAML config
   resolved by Hydra/Pydantic. Code reads from the resolved config; it
   does not have its own defaults. (Exception: numerical constants
   internal to a kernel, e.g. `eps`.)
4. **Swappability is a first-class feature.** Layers (attention, FFN,
   norm), optimizer, LR schedule, and tokenizer are all selected by name
   from a registry. Adding "QK-Norm off" or "Muon for hidden weights" is
   a config edit + a registered class, never a fork.
5. **Reproducibility is a hard requirement, not an aspiration.** Pinned
   container, pinned deps, deterministic dataloader (indexed binary +
   seeded shuffle), git-SHA + config-hash recorded in every checkpoint,
   fixed eval prompts.
6. **Observability is non-negotiable.** Per-layer gradient/parameter
   norms, attention-logit statistics, MFU, throughput, loss spikes, all
   streamed to W&B from step 0. If we cannot see it, we cannot debug it.
7. **Validation gates the run.** Nothing 8 B-scale until the 1 B proxy
   has cleared its acceptance criteria; nothing past 30 B tokens at 8 B
   without a green DCLM-CORE eval. These gates are wired into the
   pipeline, not relying on someone remembering.
8. **Commit and document.** Every non-trivial choice has an ADR in
   `09_decisions.md` with the alternatives we rejected, the rejection
   reason, and the revisit trigger. We do not relitigate decisions in
   chat.

## 3. Where each topic lives

| File | Topic | What you go here for |
|---|---|---|
| `01_stack.md` | Framework + libraries + container | Why FSDP2-native PyTorch, not NeMo or Lightning; full pinned dependency set |
| `02_repo_layout.md` | Project structure + config | The directory tree, the config schema, the registry pattern |
| `03_data.md` | Tokenizer + corpus + dataloader | Retokenisation pipeline, indexed binary format, mix sampler, determinism |
| `04_model.md` | Model architecture | Module-by-module mapping of `research/03_…` to code; init; swap points |
| `05_training.md` | Training loop | Optimizer, schedule, batch warmup, checkpointing, loss-spike protocol |
| `06_perf.md` | Performance + Nvidia playbook | TE, FlashAttention, torch.compile, NCCL, MFU targets, profiling cadence |
| `07_validation.md` | Eval + acceptance gates | 1 B proxy plan, smoke-test pyramid, DCLM-CORE harness, monotone-improvement contract |
| `08_ops.md` | Reproducibility + run book | Container build, env, restart procedure, dashboards, on-call |
| `09_decisions.md` | ADRs + tech-debt register | The decision log, the rejection log, the revisit triggers |

## 4. Milestone sequence

Mirrors `research/00_synthesis.md` §7 but with an engineering view of who
needs what built before each phase can start.

| Phase | Calendar | Engineering deliverables required to start |
|---|---|---|
| **M0 — Repo + container** | Week 0 | Repo skeleton (§02), pinned NGC container (§01), end-to-end "100 M model on 1 GPU for 1 step" smoke test green (§07) |
| **M1 — Data** | Week 0–1 | Tokenizer trained, all four sources retokenised into indexed binary shards (§03), mix sampler unit-tested |
| **M2 — 1 B proxy** | Week 1–2 | Full architecture (§04) compiled and FSDP2-wrapped on 1 node, optimizer + WSD/cosine schedule (§05), W&B + loss-spike detector live (§08), DCLM-CORE eval-harness wired in (§07). Run 20 B tokens. |
| **M3 — 8 B kickoff** | Week 2 | 1 B proxy passed acceptance gates. Resume-from-checkpoint verified. Batch-warmup schedule wired (§05). |
| **M4 — Main run** | Weeks 3–5 | 4 → 140 B tokens at 2 M-token batch. Eval every 5 B tokens. Spike protocol exercised (rehearsed before run). |
| **M5 — Late phase** | Week 5 | 140 → 150 B at 4 M-token batch. |
| **M6 — Long-context anneal (optional)** | Week 6 | YaRN config + 8 k-context anneal recipe pre-validated on 1 B proxy. |

## 5. Out of scope (explicit non-goals)

State these so they don't accrete as silent assumptions.

- **MoE.** Dense only. Re-evaluated only if a future project needs it.
- **Multi-node training.** Designed for it, not exercised in this run.
- **Custom CUDA kernels.** Use TE + FlashAttention via SDPA; do not write
  our own.
- **Post-training (SFT / DPO / RLHF).** Out of scope for this plan;
  artefacts (tokenizer, base checkpoint) are produced in a form that
  downstream teams can pick up.
- **Frontier-class token budget (1 T+).** Not feasible on a single node;
  the plan calls this out explicitly so we do not silently retarget.
- **Closed-source data.** Every corpus source is permissively licensed;
  Nemotron-CC remains rejected on license grounds (see ADR-007).
