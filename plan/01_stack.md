# 01 — Stack & Framework

*The most consequential tech-debt-prevention decision in this plan. Every
later document assumes the choices in §1.*

---

## 1. Framework — PyTorch-native, FSDP2-first

**Decision: build directly on PyTorch ≥ 2.6 with FSDP2, NVIDIA
TransformerEngine layers where they pay off, FlashAttention via
`torch.nn.functional.scaled_dot_product_attention` (SDPA), and
`torch.distributed.checkpoint` for state.** No higher-level training
framework wrapping the loop.

### Why not NeMo / Megatron-LM

- NeMo is the closest thing to "Nvidia's official 7 B+ pretraining playbook"
  and is a serious option. We *borrow* from it heavily — TransformerEngine,
  the Megatron-style indexed binary data format, the NSight profiling
  recipes — but **we do not adopt the full framework** because:
  - NeMo's config / launcher / Hydra layer is opinionated to the point of
    being a second framework on top of PyTorch; "swap a layer" means
    learning Megatron-Core's spec system first.
  - Megatron-LM's tensor-parallel + pipeline-parallel machinery is genuinely
    useful at 70 B+. At 8 B on 8 × H100 it is unnecessary complexity:
    FSDP2 alone fits.
  - The user goal explicitly mentions "PyTorch distributed or Lightning" —
    i.e., a PyTorch-native path, not a turnkey framework.
- **Revisit trigger**: if/when we go multi-node and need TP/PP for a >30 B
  parameter model, port to Megatron-Core or `torchtitan`'s parallelism
  primitives — but those are PyTorch-native too.

### Why not Lightning / Lightning Fabric

- Adds a wrapper layer around `torch.distributed` whose value at 8 × H100
  scale is mostly cosmetic.
- We forfeit direct control over FSDP2 lifecycle (`fully_shard`, mixed
  precision policies, prefetch, async checkpoint) which we want for perf
  tuning.
- **Revisit trigger**: never, at this scale. Re-evaluate at >32 nodes.

### Why not HuggingFace Transformers + Accelerate

- `transformers` is the right thing for inference/eval and we will use it
  there, but its `Trainer` is not designed for the level of control we
  want at pretraining scale (FSDP2 lifecycle, custom batch schedules,
  custom loss reductions, etc.).
- We will reuse HF *components* — the tokenizer trainer, the eval harness
  — without adopting `Trainer`.

### Why not roll-your-own (nanoGPT-style)

- Our model is the Llama 3 architecture. Reference implementations exist
  (Meta's `llama3`, `torchtitan`, HF) and have been audited at scale. A
  bespoke transformer is debt we don't need.
- We will fork `torchtitan`'s `llama3` model module as the starting point
  (it is a clean, FSDP2-native, ~600-line implementation Meta uses for
  benchmarking) and bolt on QK-Norm. See `04_model.md`.

### Reference points we are tracking

- **`torchtitan`** ([github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)) —
  the PyTorch team's reference Llama 3 pretraining stack on FSDP2 + TP +
  PP. This is the closest thing to a turnkey starting point and matches
  our principles. We pull from it but maintain our own fork because we
  need full control over data and eval.
- **Megatron-Core / NeMo** — for indexed-binary data tooling and TE
  integration patterns.
- **MosaicML LLM Foundry** — for streaming dataset patterns and the MFU
  calculation formula.

## 2. Container

**Decision: NGC PyTorch container (`nvcr.io/nvidia/pytorch:25.xx-py3`),
pinned by digest in our `Dockerfile`.**

Rationale:
- Bundles a tested combination of CUDA, cuDNN, NCCL, PyTorch, TE,
  FlashAttention, and Apex. Version drift between these is the #1 source
  of pretraining repro failures.
- Includes NSight Systems / Compute, which we use for profiling
  (§06).
- Updated monthly by Nvidia; we pin a specific tag for the run lifetime
  and do not bump mid-run.

The `Dockerfile` adds (and only adds):
- Our pinned `pyproject.toml` (uv-managed, `uv.lock` committed).
- `lm-eval-harness` and the DCLM eval scripts.
- W&B agent for observability.

We do **not** add anything that overrides containerised CUDA / NCCL / TE.
If we need to change one of those, we change the base tag.

## 3. Dependencies — pin everything

`pyproject.toml` + `uv.lock` for Python deps; container digest for system
deps.

**Hard-pinned, not "≥":**

| Package | Version pin | Why pinned |
|---|---|---|
| `torch` | container default (≥ 2.6) | FSDP2 API stable here; do not bump mid-run |
| `transformer-engine` | container default | TE ABI tied to CUDA/cuDNN versions |
| `flash-attn` | container default | Provides SDPA backend for the FA kernel |
| `tokenizers` (HF) | exact | tokenizer.json reproducibility |
| `datasets` (HF) | exact | for staging ingestion only; not in the hot path |
| `numpy`, `pyarrow` | exact | data prep |
| `hydra-core`, `pydantic` | exact | config |
| `wandb` | exact | logging |
| `lm-eval` | exact | downstream eval |

We use **uv** (not `pip` / `poetry`) because `uv.lock` is reproducible across
machines and gives us bit-identical environments outside the container, for
data prep and eval jobs that don't need GPUs.

## 4. The "Nvidia playbook" components we adopt

This is the answer to the user's "focus on Nvidia's playbooks" requirement:

| Nvidia playbook component | What we use | Where covered |
|---|---|---|
| TransformerEngine `Linear`/`LayerNormLinear`/fused attention | bf16 today, fp8 evaluated for late phase only | `04_model.md`, `06_perf.md` |
| FlashAttention 2/3 (FA-3 if present in container) | via SDPA backend selection | `06_perf.md` |
| Megatron-style indexed binary data format (`.bin` + `.idx`) | adopt the format and the `IndexedDataset` reader | `03_data.md` |
| NCCL env tuning (`NCCL_AVOID_RECORD_STREAMS`, `NCCL_NVLS_ENABLE`, etc.) | applied via `08_ops.md` env file | `06_perf.md`, `08_ops.md` |
| NSight Systems profiling cadence | profile on day 1 of each phase, archive `.nsys-rep` | `06_perf.md` |
| Distributed Checkpointing (`torch.distributed.checkpoint`) | with async save | `05_training.md` |
| MFU calculation (PaLM-style 6 N D) | dashboard panel from step 0 | `06_perf.md` |

What we **do not** adopt from the Nvidia playbook (with reasons):
- **Megatron-LM full launcher** — too opinionated; we want PyTorch-native.
- **Apex fused optimizer** — `torch.optim.AdamW` with `fused=True` matches
  Apex's perf in PyTorch 2.6+ and is one fewer dep.
- **TE `TransformerLayer` monolithic block** — we want explicit module
  boundaries for swappability (§04). We use TE primitives, not the layer
  wrapper.

## 5. Hardware assumptions

- 1 × DGX-class node, 8 × H100 80 GB SXM5, NVLink + NVSwitch (900 GB/s
  per-GPU bisection).
- Local NVMe scratch ≥ 2 TB for data shards + checkpoints.
- bf16 BF16 peak ≈ 1 PFLOP/s/GPU; we target ≥ 50 % MFU sustained
  (`06_perf.md`).
- DGX Spark (GB10) is also a target. The codebase must run there
  unmodified — same FSDP2 path, the only difference is fewer/more GPUs
  per node and the eventual NVLink topology. No Spark-specific code
  paths.

## 6. What this stack costs in tech debt (honest accounting)

- **FSDP2 API churn risk.** FSDP2 is younger than FSDP1 and the API has
  shifted between PyTorch 2.4 → 2.6. Mitigation: pin PyTorch by container
  digest; don't take feature flags that aren't in 2.6 stable.
- **TE bf16 path subtleties.** TE's `Linear` does the right thing, but
  the cast policy needs to match `MixedPrecisionPolicy` on the FSDP2 wrap.
  Mitigation: a single helper in `model/precision.py` configures both;
  unit-tested on a 100 M model.
- **Indexed binary data format is Megatron-flavoured.** Documented but
  not standard outside Megatron/NeMo. Mitigation: vendor the reader
  (~200 LOC) instead of depending on the full Megatron package; this
  also removes a ~1 GB transitive dep.
- **`torch.compile` regressions.** Real risk on a complex model. We will
  default to `compile=True` with `mode="default"` (not `max-autotune`) on
  the 1 B proxy and will allow `compile=False` as a one-line escape valve
  if it bites us at 8 B.

These costs are smaller than the alternative (taking on NeMo or
Lightning), and each has a documented mitigation rather than a "we'll
deal with it later" note.
