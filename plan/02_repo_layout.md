# 02 — Repo Layout & Configuration

*The shape of the codebase. Optimised for "someone new can find anything in
one click" and "swap a layer or hyperparameter without forking".*

---

## 1. Directory tree

```
pretrain/
├── pyproject.toml             # uv-managed, deps pinned in uv.lock
├── uv.lock
├── Dockerfile                 # NGC base image, digest-pinned
├── README.md                  # quickstart only; design lives in plan/
│
├── configs/                   # Hydra config tree — see §3
│   ├── train/
│   │   ├── 100m_smoke.yaml
│   │   ├── 1b_proxy.yaml
│   │   ├── 8b_main.yaml
│   │   └── 8b_long_ctx_anneal.yaml
│   ├── model/
│   │   ├── llama3_8b.yaml
│   │   ├── llama3_8b_qknorm.yaml      # the recipe we use
│   │   └── llama3_1b_proxy.yaml
│   ├── data/
│   │   ├── recipe_v1.yaml             # 75/12/10/3 mix
│   │   └── proxy.yaml                 # smaller mix for 1B proxy
│   ├── optim/
│   │   ├── adamw_default.yaml
│   │   └── muon_hidden_adamw_aux.yaml # bake-off only, off by default
│   ├── schedule/
│   │   ├── cosine.yaml                # default
│   │   └── wsd.yaml                   # available; not used unless we plan a mid-run anneal
│   └── run/
│       ├── single_node_8xh100.yaml
│       └── multi_node_template.yaml   # written but unexercised
│
├── src/pretrain/
│   ├── __init__.py
│   ├── cli/
│   │   ├── train.py                   # `python -m pretrain.cli.train ...`
│   │   ├── prepare_data.py            # tokenise + retokenise + shard
│   │   ├── eval.py                    # standalone eval
│   │   └── inspect_checkpoint.py
│   │
│   ├── model/
│   │   ├── llama3.py                  # the model. Forked from torchtitan.
│   │   ├── modules/
│   │   │   ├── attention.py           # MHA / GQA + QK-Norm
│   │   │   ├── ffn.py                 # SwiGLU
│   │   │   ├── norm.py                # RMSNorm
│   │   │   ├── rope.py                # RoPE θ=500k
│   │   │   └── embedding.py
│   │   ├── precision.py               # bf16 mixed-precision policy + TE casts
│   │   ├── init.py                    # truncated normal + scaled output init
│   │   └── registry.py                # name→class map for swap
│   │
│   ├── data/
│   │   ├── tokenizer.py               # train + load
│   │   ├── indexed_dataset.py         # Megatron-style .bin/.idx reader
│   │   ├── prepare.py                 # parquet → indexed binary shards
│   │   ├── mix_sampler.py             # weighted multi-source sampler
│   │   └── loader.py                  # DataLoader factory
│   │
│   ├── optim/
│   │   ├── adamw.py                   # thin wrapper, fused=True
│   │   ├── muon.py                    # gated; only loads if config selects it
│   │   ├── schedules.py               # cosine + WSD + linear warmup
│   │   └── registry.py
│   │
│   ├── train/
│   │   ├── loop.py                    # the actual training loop
│   │   ├── batch_schedule.py          # 1M → 2M → 4M token warmup
│   │   ├── checkpoint.py              # torch.distributed.checkpoint async
│   │   ├── spike_protocol.py          # detect + skip-step + rollback
│   │   └── reduce.py                  # loss / grad norm / metric reductions
│   │
│   ├── eval/
│   │   ├── dclm_core.py               # DCLM CORE/EXTENDED runner
│   │   ├── lm_eval_adapter.py         # bridge to lm-eval-harness
│   │   ├── perplexity.py              # held-out PPL
│   │   └── eval_loop.py               # invoked from train loop on schedule
│   │
│   ├── parallel/
│   │   ├── fsdp.py                    # fully_shard wrapping policy
│   │   ├── meshes.py                  # device mesh (DP-only today, DP×TP shaped)
│   │   └── env.py                     # NCCL env, deterministic seeds
│   │
│   ├── obs/
│   │   ├── wandb_run.py
│   │   ├── metrics.py                 # MFU, throughput, grad/param norms
│   │   ├── attention_stats.py         # per-layer logit max, QK norms
│   │   └── alerts.py                  # spike + NaN + throughput regression
│   │
│   └── util/
│       ├── seed.py
│       ├── timing.py
│       └── git.py                     # capture SHA + diff into checkpoint
│
├── tests/
│   ├── test_model_shapes.py
│   ├── test_indexed_dataset.py
│   ├── test_mix_sampler.py
│   ├── test_schedule_math.py
│   ├── test_checkpoint_roundtrip.py
│   └── e2e/
│       └── test_100m_one_step.py      # the smoke test from M0
│
├── scripts/
│   ├── launch_single_node.sh          # torchrun wrapper
│   ├── prepare_dclm_shards.sh
│   ├── prepare_fineweb_edu_shards.sh
│   ├── prepare_stack_v2_shards.sh
│   └── prepare_proof_pile_shards.sh
│
├── plan/                              # this directory
└── research/                          # the citation-backed recipe
```

### Why this shape

- **Configs separate from code.** Every dial lives in `configs/`; the
  code never has its own defaults beyond unavoidable kernel constants.
- **Functional separation by concern.** A reviewer who wants to know how
  we shard goes to `parallel/`, not to `train/loop.py`. This matters when
  someone new picks the repo up.
- **Registries gate swappability.** Three small registries
  (`model.registry`, `optim.registry`, `data.recipes`) are the only place
  in the code where we resolve a config string to a class. Adding a new
  optimizer is "register + write the class"; nothing else changes.
- **`scripts/` is bash, `cli/` is Python.** Bash files are launchers for
  the cluster; the actual logic is always importable Python.
- **`tests/e2e/test_100m_one_step.py` is the M0 gate.** It builds the
  full pipeline end-to-end on a tiny model. If that test ever breaks, no
  one merges.

## 2. Configuration system — Hydra + Pydantic

**Hydra** for composition (`configs/train/8b_main.yaml` references
`configs/model/llama3_8b_qknorm.yaml`, etc., and supports CLI overrides).
**Pydantic** for validation: every config group has a `dataclass`-style
schema, and Hydra resolves into the schema. Bad configs fail at import,
not at step 4 000.

### The config tree

```yaml
# configs/train/8b_main.yaml
defaults:
  - model: llama3_8b_qknorm
  - data: recipe_v1
  - optim: adamw_default
  - schedule: cosine
  - run: single_node_8xh100
  - _self_

train:
  total_tokens: 150_000_000_000
  seq_len: 4096            # main phase; long-ctx anneal uses 8192
  micro_batch_size: 4
  global_batch_tokens:
    warmup: 1_048_576
    main:   2_097_152
    late:   4_194_304
  warmup_to_main_at_tokens: 4_000_000_000
  main_to_late_at_tokens: 140_000_000_000
  eval_every_tokens: 5_000_000_000
  ckpt_every_tokens: 5_000_000_000

  spike:
    grad_norm_threshold: 5.0
    skip_steps_on_spike: 50
    rollback_on_repeat: true

logging:
  wandb_project: pretrain-8b
  wandb_run_name_template: "8b_main_{git_sha[:8]}_{started_at}"
```

### Why Hydra + Pydantic and not just Hydra

- Hydra alone leaves typos silent (typoed `weight_decay: 0.l` vs `0.1`
  becomes a string and trains your model with no decay). Pydantic's
  schema validation catches that at startup.
- We use Hydra's structured-configs feature so the schema IS the config
  default — same source of truth.

### CLI

A single entrypoint per task:

```bash
torchrun --nproc-per-node=8 -m pretrain.cli.train \
  --config-name 8b_main \
  optim.peak_lr=3e-4 \
  train.total_tokens=300_000_000_000   # stretch
```

Hydra auto-saves the resolved config + git SHA + diff next to the run's
log dir. This becomes part of the checkpoint metadata (§05).

## 3. The registry pattern (the swap mechanism)

```python
# src/pretrain/model/registry.py
ATTENTION = {}
def register_attention(name):
    def deco(cls):
        ATTENTION[name] = cls
        return cls
    return deco

@register_attention("gqa_qknorm")
class GroupedQueryAttentionQKNorm(nn.Module): ...

@register_attention("gqa_plain")
class GroupedQueryAttentionPlain(nn.Module): ...

# configs/model/llama3_8b_qknorm.yaml
attention: gqa_qknorm
```

Same pattern for `ffn` (`swiglu`, `swiglu_glu_variants`), `norm`
(`rmsnorm`, `layernorm`), `optimizer` (`adamw`, `muon_hybrid`), and
`schedule` (`cosine`, `wsd`).

**Constraint that prevents debt:** every registered class MUST conform to
the same call signature as the others in its registry. The signature is
type-asserted at registration time. This forces alternatives to be
genuinely interchangeable rather than "almost but with these special
flags".

## 4. Documentation layout (where things are written down)

- **`research/`** — the *what and why* (recipe + citations). Frozen
  once it lands; do not retro-edit when implementation choices change.
- **`plan/`** — the *how, at architecture level* (this set of docs).
  Reviewed before implementation; updated only via PR with reviewer.
- **Code-level docstrings** — short. We do not duplicate `plan/` in
  module headers.
- **A runbook** — operational, per `08_ops.md`. Updated as we
  learn from running.
- **`plan/09_decisions.md`** — ADR log. Append-only; one entry per
  non-trivial choice.

## 5. What this layout intentionally avoids

- **No `utils.py` graveyard.** Helpers live next to their callers in the
  relevant module.
- **No deeply nested `__init__.py` re-exports.** Imports are explicit:
  `from pretrain.model.modules.attention import GroupedQueryAttention`,
  not `from pretrain.model import GroupedQueryAttention`.
- **No "shared" config defaults inherited from a base config you can't
  see.** Every config file is fully resolved by Hydra's defaults list,
  printable with one command.
- **No mixin-heavy class hierarchies.** Composition via the registry,
  not inheritance.
- **No `train_*.py` scripts proliferating.** One CLI, one config tree.
