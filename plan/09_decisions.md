# 09 — Decision Log & Tech-Debt Register

*The one place we record non-trivial choices, the alternatives we
rejected, and the conditions under which we'd revisit. Append-only;
treat as the project's institutional memory.*

---

## ADR template

```
### ADR-NNN: <title>
- **Status**: Accepted | Superseded by ADR-XXX
- **Date**: YYYY-MM-DD
- **Decision**: <one sentence>
- **Alternatives rejected**: <bullet list with rationale>
- **Consequences**: <what this commits us to>
- **Revisit trigger**: <observable condition that would force re-evaluation>
```

---

## ADR-001: PyTorch-native FSDP2 over NeMo / Megatron / Lightning

- **Status**: Accepted
- **Date**: 2026-04-27
- **Decision**: Build directly on PyTorch ≥ 2.6 + FSDP2, with TE / FA /
  DCP as upstream components. No higher-level training framework.
- **Alternatives rejected**:
  - **NeMo / Megatron-LM**: opinionated launcher + spec system adds a
    second framework on top of PyTorch; TP/PP machinery unnecessary at
    8 B on 8 × H100. We borrow TE, the indexed-binary format, and
    NSight recipes — not the framework.
  - **Lightning Fabric**: wrapper layer adds no value at single-node
    scale; forfeits direct FSDP2 lifecycle control.
  - **HF `Trainer`**: not designed for pretraining-scale control over
    FSDP2 lifecycle, custom batch schedules, and custom losses.
  - **Bespoke from scratch (nanoGPT-style)**: rewriting an audited
    transformer is unjustified debt.
- **Consequences**: We fork `torchtitan/llama3` as the model module
  starting point; everything else is in our repo. Multi-node move is a
  config flip plus exercising the parallelism mesh.
- **Revisit trigger**: target model exceeds ~ 30 B params, OR we
  outgrow single-node and need TP/PP — at which point we port to
  Megatron-Core's parallelism primitives (still PyTorch-native).

## ADR-002: Megatron-style indexed binary data format (vendored)

- **Status**: Accepted
- **Date**: 2026-04-27
- **Decision**: Use Megatron's `.bin` + `.idx` indexed binary format,
  vendored as ~ 200 LOC in `src/pretrain/data/indexed_dataset.py`.
- **Alternatives rejected**:
  - **WebDataset**: tar-based sequential; needs shuffle buffer;
    slower for our access pattern.
  - **Mosaic StreamingDataset**: optimised for cloud streaming;
    compression layer is wasted overhead on local NVMe.
  - **HF datasets / parquet in the hot path**: Arrow decode CPU cost
    non-trivial at 2 M tokens/step.
  - **Depending on Megatron-Core directly**: ~ 1 GB transitive deps for
    a 200 LOC file.
- **Consequences**: We own the format reader. The format itself is
  stable; vendoring is safer than a moving Megatron API.
- **Revisit trigger**: another format demonstrates ≥ 1.5× throughput at
  matched memory; OR an upstream change to indexed binary breaks our
  reader.

## ADR-003: Cosine schedule, not WSD, for the primary run

- **Status**: Accepted
- **Date**: 2026-04-27
- **Decision**: Linear warmup → cosine decay to 10 % of peak LR over
  150 B tokens.
- **Alternatives rejected**:
  - **Warmup-Stable-Decay (WSD)**: better if we plan to branch
    multiple decay tails for data-mix experiments. We have no such
    plan.
  - **Constant LR + step decay (DeepSeek-V2)**: less established for
    7–8 B; Llama 3 / Chinchilla used cosine.
- **Consequences**: We commit to a single 150 B token horizon. Mid-run
  schedule changes are not supported; if we want a stretch run, we
  start a new run from a checkpoint with a freshly-configured cosine
  for the new horizon.
- **Revisit trigger**: we decide to plan mid-run data-mix experiments
  *before* the 1 B proxy starts (i.e., before M2 — switching is
  expensive after).

## ADR-004: AdamW (fused) for primary run; Muon as 1 B-proxy bake-off only

- **Status**: Accepted
- **Date**: 2026-04-27
- **Decision**: AdamW with research-default hyperparameters
  (β1=0.9, β2=0.95, eps=1e-8, WD=0.1, clip=1.0, peak LR=3e-4) for the
  8 B run. Muon-hybrid implemented but evaluated only at 1 B-proxy
  scale.
- **Alternatives rejected**:
  - **Muon (default)**: no public dense ≥ 7 B reproduction outside
    Moonshot AI; expected gain at 8 B is 1.0–1.1× per Stanford/Marin
    bake-off; calibration risk too high for one-shot run.
  - **Sophia, Lion**: under-validated at this scale on language
    modelling.
  - **Distributed Shampoo**: AlgoPerf winner but no turnkey single-node
    7 B recipe.
- **Consequences**: We accept AdamW's known performance; we miss
  Muon's possible 1.05× speedup unless the 1 B bake-off is decisive.
- **Revisit trigger**: 1 B bake-off shows ≥ 1.15× wall-clock gain at
  matched downstream eval, **AND** there is owner-time to handle Muon's
  calibration risk during the main run.

## ADR-005: bf16 mixed-precision; fp8 deferred

- **Status**: Accepted
- **Date**: 2026-04-27
- **Decision**: Activations / matmul in bf16, fp32 master + optimizer
  state, fp32 reductions in RMSNorm and loss. fp8 (TE delayed-scaling)
  not enabled in the primary 150 B run.
- **Alternatives rejected**:
  - **fp8 from day 1**: meaningful MFU win but needs delayed-scaling
    calibration; debugging fp8 numeric drift across a 3-week run is
    out of scope.
  - **fp32 throughout**: too slow; no one trains 8 B models in fp32.
- **Consequences**: We leave a likely 1.1–1.3× MFU on the table.
- **Revisit trigger**: we plan a 300 B+ stretch run after 150 B target
  lands cleanly. Validate fp8 on the 1 B proxy before adopting.

## ADR-006: Fresh 128 k tokenizer, not Llama 3's

- **Status**: Accepted
- **Date**: 2026-04-27
- **Decision**: Train our own byte-level BPE, vocab 128 256, on a
  representative sample of the pretraining mix. Frozen and
  content-hashed.
- **Alternatives rejected**:
  - **Reuse Llama 3 tokenizer**: license complications for some
    downstream consumers; we want a fully-open-licensed artefact.
  - **Reuse Qwen3 / Mistral tokenizer**: same license concern; also
    mix shouldn't bias us toward another model's training distribution.
  - **GPT-2 / NeoX 50 k tokenizer**: too small; ~ 25 % more tokens per
    document; wastes compute.
- **Consequences**: One extra prep step (tokenizer training) and a hard
  unit-test gate on bytes-per-token within 5 % of Llama 3.
- **Revisit trigger**: bytes-per-token diverges > 10 % from Llama 3 on
  the held-out validation sample (signals the training sample was
  unrepresentative); OR a downstream consumer needs an existing
  open-licensed tokenizer.

## ADR-007: Reject Nemotron-CC despite its quality lift

- **Status**: Accepted
- **Date**: 2026-04-27
- **Decision**: Stick with DCLM-Baseline + FineWeb-Edu mix; do not
  include Nemotron-CC.
- **Alternatives rejected**:
  - **Add Nemotron-CC at a partial weight**: license is the NVIDIA
    Data Access Agreement for Model Training, which is more
    restrictive than CC-BY-4.0 / ODC-By and risks downstream consumer
    constraints.
- **Consequences**: We forfeit the +5.6 MMLU advantage Nemotron-CC
  shows at 8 B / 1 T tokens; our run is positioned as
  Chinchilla-optimal vs. Llama-2 7B / DCLM-7B-2.6T baselines, not as
  Llama-3-class.
- **Revisit trigger**: legal/business sign-off on the NVIDIA
  Data Access Agreement; OR Nemotron-CC re-released under CC-BY.

## ADR-008: Z-loss coefficient 1e-5 (Chameleon), not 1e-4 (PaLM)

- **Status**: Accepted
- **Date**: 2026-04-27
- **Decision**: Z-loss enabled with coefficient 1e-5.
- **Alternatives rejected**:
  - **PaLM's 1e-4**: pre-dates QK-Norm; with QK-Norm in place the
    lighter coefficient is sufficient and reduces interference with
    cross-entropy.
  - **Off**: cheap insurance; not worth turning off.
- **Consequences**: One extra scalar in the loss (logged separately).
- **Revisit trigger**: late-training output-logit drift observed
  during a run.

## ADR-009: dist.checkpoint (DCP) with async save

- **Status**: Accepted
- **Date**: 2026-04-27
- **Decision**: Use `torch.distributed.checkpoint` for sharded
  checkpointing; async save process so the main loop never blocks.
- **Alternatives rejected**:
  - **`torch.save` with rank-0 gather**: memory pressure at 8 B; loses
    cross-shape resume.
  - **`safetensors` per rank, manual orchestration**: re-implements
    DCP poorly.
- **Consequences**: Cross-shape resume works (hedges against changing
  GPU count); resume tests are bit-reproducible (within float tol).
- **Revisit trigger**: DCP API breaking changes upstream that pinning
  PyTorch doesn't shield us from.

## ADR-010: Hydra + Pydantic for config

- **Status**: Accepted
- **Date**: 2026-04-27
- **Decision**: Hydra for composition / CLI override; Pydantic for type
  validation. Configs are the single source of truth for every
  hyperparameter.
- **Alternatives rejected**:
  - **Hydra alone**: silent typo / type-error failures.
  - **Pure Pydantic + argparse**: composition is painful; reproducing
    a Hydra-style override CLI is significant work.
  - **YAML + dataclasses by hand**: re-implements Hydra poorly.
- **Consequences**: Two libraries instead of one; learning curve for
  Hydra's composition syntax.
- **Revisit trigger**: structured-config support in Hydra changes
  semantics again.

## ADR-011: Dense-only; MoE explicitly out of scope

- **Status**: Accepted
- **Date**: 2026-04-27
- **Decision**: Dense decoder-only model; no MoE.
- **Alternatives rejected**:
  - **MoE (DeepSeek-V3 / Kimi K2-style)**: load balancing,
    auxiliary-loss-free balancing, expert sharding, all-to-all
    communication add substantial engineering surface area; outside
    "single-machine reproducible" framing.
- **Consequences**: We forfeit the FLOPs/param efficiency of MoE.
- **Revisit trigger**: a future project explicitly chartered for MoE.

---

## Tech-debt watch list

Items that are not bad now but could become debt under the wrong
conditions. Monitor; add an ADR when one becomes a real decision.

| Area | What to watch | Trigger to act |
|---|---|---|
| FSDP2 API churn | PyTorch 2.7+ deprecating any of `fully_shard`, `MixedPrecisionPolicy`, `set_requires_gradient_sync` | Any breaking change → write an ADR for the migration plan before adopting |
| `torch.compile` perf vs cost | Recompilation events, compile-time, NaN bugs traceable to compile | If compile costs > 30 min on 8 B startup, gate compile behind an explicit warm-cache step |
| TE / FlashAttention pinning | Container updates that bring in new TE/FA versions | Re-validate with full smoke test before adopting; pin digest for run |
| Eval drift | DCLM repo updating eval task definitions mid-project | Vendor the eval scripts at a fixed SHA; do not auto-update |
| Manifest hashing | Manifest schema changes between prep tool versions | Schema versioned; loader refuses unknown schema version |
| W&B as single source of truth for metrics | W&B outage, account migration | All metrics also written locally in JSONL; replay tool maintained |
| Single-rank-0 eval bottleneck | If eval pass starts to dominate wall-clock late in the run | Move to FSDP-sharded eval; we can do this without breaking the harness because we already gather a full model copy explicitly |
| `seq_len` change for long-ctx anneal | Recompilation, FA varlen mode, RoPE table size | Validated on 1 B proxy before M6 |

---

## How we use this file

1. Every PR that introduces a non-trivial choice **must** add an ADR
   here, or reference an existing one.
2. Reviewers can require an ADR before approving.
3. We do not relitigate decisions in chat. If someone wants to change
   one, they open a PR superseding the ADR with a new ADR.
4. The first time a "watch list" item triggers, we promote it to an
   ADR with the resolution.
