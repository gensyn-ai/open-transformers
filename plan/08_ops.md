# 08 — Reproducibility, Run Book, On-Call

*The operational layer. What it takes to run, recover, and reproduce.*

---

## 1. Reproducibility contract

A run is reproducible if, given:

- the NGC container digest,
- the tokenizer hash,
- the data manifest hash,
- the resolved Hydra config + git SHA,
- the seed,

an engineer on a different 8 × H100 box can reproduce the loss curve to
within ε. The run records all five into every checkpoint.

The `runs/<run_id>/` directory contains:

```
runs/<run_id>/
├── config.resolved.yaml           # full resolved Hydra config
├── git.sha
├── git.diff                       # uncommitted diff at run start (small or empty)
├── env.txt                        # output of `pip freeze` + uname + nvidia-smi
├── container.digest               # `docker inspect` of the running image
├── tokenizer.hash
├── data_manifest.hash
├── nccl.env                       # the env file we sourced
├── checkpoints/
│   └── step_<N>/
│       ├── model/                 # DCP shard files
│       ├── optimizer/
│       ├── scheduler.pt
│       ├── sampler.pt
│       └── meta.json              # SHAs, hashes, container digest, restart count
├── evals/
│   └── step_<N>.json
├── nsys/
│   └── M2_proxy_step1500.nsys-rep
└── logs/
    ├── train.log
    └── wandb-run-<id>/
```

We never delete a `runs/` directory while the corresponding W&B run is
not archived.

## 2. Container & environment

### The Dockerfile

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.xx-py3@sha256:<pinned-digest>

WORKDIR /work
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv pip sync --system --no-deps uv.lock

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/work/src
```

Pinned digest, not a floating tag. Bumping the base image is a PR with
re-validation at the smoke-test level.

### The NCCL env file (`scripts/env/nccl.env`)

Sourced by every launcher script. See `06_perf.md` §5 for the values.
Lives in version control; identical across single-node and multi-node.

## 3. Launching a run

### Single-node 8 × H100

```bash
# scripts/launch_single_node.sh
set -euo pipefail
source scripts/env/nccl.env
export RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD)}"

torchrun \
  --standalone \
  --nproc-per-node=8 \
  -m pretrain.cli.train \
  --config-name "${1:-8b_main}" \
  run.run_id="${RUN_ID}" \
  "${@:2}"
```

Run it with:

```bash
./scripts/launch_single_node.sh 8b_main
# or
./scripts/launch_single_node.sh 1b_proxy
```

Hydra overrides pass through directly:

```bash
./scripts/launch_single_node.sh 8b_main optim.peak_lr=2.5e-4
```

### Multi-node (when we get there)

Same script with extra args; the `multi_node_template.yaml` config plus
the host file. We have the scaffolding but do not exercise it in this
project.

## 4. The runbook

A runbook (separate, written as we hit each scenario the first
time). Skeleton:

### 4.1 First-time setup

- Build the container.
- Run `tests/e2e/test_100m_one_step.py` on a single GPU.
- Run unit tests.
- Run data-prep for a 1 GB synthetic shard end-to-end.

### 4.2 Starting a real run

- Confirm tokenizer hash matches the manifest.
- Confirm NVMe has > 2 TB free.
- Confirm W&B credentials.
- `nvidia-smi` to confirm 8 GPUs visible, no other processes.
- Launch.

### 4.3 During the run

- Daily: glance at W&B dashboards (loss, grad norm, MFU, throughput).
- After each eval: confirm DCLM-CORE macro-avg improved.
- Weekly: check disk usage; archive old checkpoints if > 80 %.

### 4.4 Fault recovery scenarios

Document each scenario with the steps the first time it happens. Initial
seed list:

| Scenario | Likely first response |
|---|---|
| GPU drops off the bus / Xid error | `nvidia-smi` + `dmesg`; if persistent, restart the host, resume from last checkpoint |
| NCCL hang | Kill, increase `TORCH_NCCL_TIMEOUT`, capture `py-spy dump` of the stuck rank, file ticket with infra |
| Loss NaN | Check `grad_norm_pre_clip` history; verify spike protocol; if NaN persists, halt and escalate to research before any rollback decision |
| MFU regression | Check NSight on next 100 steps; compare against M2 baseline; common causes: NVMe contention, runaway data prep on host, CUDA driver update |
| Data prep crashed mid-shard | Idempotent re-run of `prepare_data` skips finished shards (idempotency tested) |
| Resume produces different loss curve | First investigation: tokenizer / data manifest hash mismatch. Verify before any other hypothesis. |

### 4.5 Stopping cleanly

- Send SIGTERM to torchrun.
- Loop catches the signal, finishes the in-flight micro-batch,
  flushes a final checkpoint, exits.
- W&B run goes to "finished" state.

## 5. Observability (what we watch)

W&B dashboard layout (one per training run):

- **Top row — health**: loss, grad-norm-pre-clip, NaN counter, spike
  counter, alarms.
- **Second row — progress**: tokens consumed, tokens/sec,
  tokens/sec/GPU, MFU, ETA.
- **Third row — eval**: DCLM-CORE macro-avg, MMLU 5-shot, held-out PPL.
- **Fourth row — internals**: per-block attn_logit_max, QK-norm
  Q-mean / K-mean, per-parameter-group param-norm and update-norm.
- **Diagnostic panels (toggled on demand)**: per-block grad-norm,
  per-block param-norm, optimizer step time, dataloader wait time.

Alerts wired to a Slack channel:

- `NaN/Inf` in loss → page.
- `grad_norm_pre_clip > 5 σ` of trailing 200 steps → notify.
- `mfu` below baseline by > 10 % for > 30 min → notify.
- `tokens_per_sec` below baseline by > 10 % for > 30 min → notify.
- `eval/dclm_core_macro` regression vs trailing best → notify.

## 6. On-call expectations

The 8 B run is 3+ weeks. We do not need 24/7 watch but we do need:

- One named owner for the run lifetime.
- Slack alerts go to a channel monitored by the team during business
  hours; explicit hand-offs over weekends.
- Spike protocol auto-recovery means we should not be paged for normal
  spikes; only NaN, repeated spikes, or hard hangs page.

## 7. Reproducibility risks (and what we did)

| Risk | Mitigation |
|---|---|
| Container update during the run silently changes CUDA / NCCL / TE | Pinned digest; updates are a separate validated PR |
| Tokenizer drift (someone retrains and overwrites) | tokenizer.json stored in artifact storage with content-hash filename; hash recorded in checkpoint |
| Data manifest drift (prep pipeline changed mid-run) | Manifest hash recorded in checkpoint; loader refuses to read shards whose hash doesn't match the running manifest |
| Engineer pulls latest, training resumes with API change | Runs are pinned to a git SHA; resume via `git checkout <sha>` then launch |
| Different `world_size` on resume changes data sharding | DCP supports cross-shape resume; sampler state is per-source token counts, not per-rank shard offsets — sampling resumes correctly. Tested. |
| W&B outage | All metrics also written to local JSONL; offline-replay tool lifts them into W&B post-hoc |
| Lost the eval prompts (someone re-randomised) | Eval prompts are baked into `evals/dclm_core.py` as static; not regenerated |

## 8. End-of-project artefacts

When the run completes, we publish:

- The final checkpoint (DCP format) and a HF-compatible re-shard.
- The tokenizer.
- The data manifest (token counts per source, shard layout, mix
  weights).
- The exact resolved configs for the 1 B proxy and 8 B main runs.
- The run log + final W&B export.
- A short README pointing at the relevant ADRs.

Anyone with our container and these artefacts can either resume training
or reproduce eval results bit-for-bit (modulo cuBLAS noise).
