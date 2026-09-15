# 05 — Training Loop

*Implements `research/04_batch_size.md` and `research/05_optimizer.md`.
The hot loop, the optimizer, the LR / batch schedules, checkpointing,
and the loss-spike protocol.*

---

## 1. Optimizer

**Decision: AdamW (`torch.optim.AdamW`, `fused=True`), exact research
defaults.**

```yaml
# configs/optim/adamw_default.yaml
name: adamw
betas: [0.9, 0.95]
eps: 1.0e-8
weight_decay: 0.1
fused: true
no_decay_param_names: [".bias", "norm.weight", ".rope_"]   # bias / norms / RoPE buffers excluded from decay
```

### Implementation notes

- We use `torch.optim.AdamW` with `fused=True`. PyTorch 2.6's fused
  AdamW matches Apex on H100; one fewer dependency.
- Two parameter groups: decayed (matrices) and non-decayed (biases,
  norm gains). The matcher pattern is in config so it's auditable.
- Optimizer state is fp32 master ("mixed precision: bf16 grads & params,
  fp32 master"). FSDP2's `MixedPrecisionPolicy` handles the cast policy.

### Muon — not on the main path

`configs/optim/muon_hidden_adamw_aux.yaml` is implemented but **disabled
by default**. It is a bake-off candidate to be evaluated on the 1 B
proxy only, and only adopted for the main run if the 1 B comparison
shows ≥ 1.15× wall-clock gain at matched downstream eval (research
recalibrated expectation: 1.0–1.1× at 8 B). See ADR-006.

When enabled, Muon is applied **only to 2-D hidden weight matrices**
(`attn.wo`, `attn.wqkv`, `ffn.w_*` excluding gating norms); embeddings,
lm_head, RMSNorm gains use AdamW. Includes Moonlight's WD + update-RMS
rescaling. This split lives in `src/pretrain/optim/muon.py`.

## 2. LR schedule

**Decision: cosine with linear warmup, exact research defaults.**

```yaml
# configs/schedule/cosine.yaml
name: cosine
warmup_steps: 2000
peak_lr: 3.0e-4
min_lr_frac: 0.10            # decay to 10% of peak
cycle_length: total_tokens   # set by train.total_tokens
```

The cosine cycle length matches the token budget — Chinchilla §5.

### WSD — available, not default

`configs/schedule/wsd.yaml` is implemented (warmup → constant → 10 %
decay tail) and is the right choice if we intend to do mid-run data-mix
experiments or a separate "midtraining anneal". For the 150 B target we
do not need it. Adopting WSD is a config switch, not a code change.

### Why we commit upfront

Switching schedule mid-run is the most painful kind of debt: you
either re-do the whole run or you take a lopsided final loss curve.
Pick one upfront based on whether the project will branch. We currently
will not, so cosine.

## 3. Batch schedule

Three phases match the research recipe (`research/04_batch_size.md`):

| Phase | Token range | Global batch (tokens) | Sequences per step | Grad-accum at micro_bs=4, seq=4096, DP=8 |
|---|---|---|---|---|
| Warmup | 0 → 4 B | 1 048 576 | 256 | 8 |
| Main | 4 B → 140 B | **2 097 152** | 512 | 16 |
| Late | 140 B → 150 B | 4 194 304 | 1024 | 32 |
| Long-ctx anneal (optional) | +20 B at seq=8192 | 4 194 304 | 512 | 32 |

Implementation:
- `src/pretrain/train/batch_schedule.py` returns `grad_accum_steps`
  given the current consumed-token count.
- We do **not** change `micro_batch_size` or `seq_len` to grow the
  batch; we change `grad_accum_steps`. This keeps shapes stable for
  `torch.compile` and FSDP2 prefetch.
- LR is **not re-tuned** at each batch step. The research is explicit
  that this is the primary risk and we accept it; the warmup→main step
  matters most because it is at low loss and we are leaving the "perfect
  scaling" regime.
- We log a synthetic "effective LR per token" as a diagnostic.

## 4. Gradient handling

- **Global L2 grad clip = 1.0**. Implemented via FSDP2's
  `clip_grad_norm_`, which does the cross-shard reduction correctly.
- We log `grad_norm_pre_clip` every step. This is the primary spike
  signal.

## 5. The training loop (skeleton)

```python
for step in count():
    state.consumed_tokens += global_batch_tokens
    accum = batch_schedule.grad_accum_steps(state.consumed_tokens)

    micro_loss_total = 0.0
    for k in range(accum):
        batch = next(loader_iter)
        with autocast_bf16():
            logits, aux = model(batch.input_ids)
            ce_loss = cross_entropy(logits, batch.labels)
            zloss = zloss_coeff * aux.zloss if cfg.model.z_loss.enabled else 0.0
            loss = (ce_loss + zloss) / accum
        loss.backward()
        micro_loss_total += loss.detach()

    grad_norm = clip_grad_norm_(model.parameters(), 1.0)
    if spike_protocol.should_skip(grad_norm, micro_loss_total):
        optimizer.zero_grad(set_to_none=True)
        spike_protocol.record_skip(step)
        continue

    optimizer.step()
    scheduler.step(state.consumed_tokens)
    optimizer.zero_grad(set_to_none=True)

    metrics.log(step, state, micro_loss_total, grad_norm)
    if step % cfg.train.eval_every_steps == 0:
        eval_loop.run_async(state)
    if step % cfg.train.ckpt_every_steps == 0:
        checkpoint.save_async(state, model, optimizer, scheduler, sampler)
```

### Notes

- `loss.backward()` runs reduce-scatter via FSDP2. We do not call
  `model.no_sync()` even within accumulation, because FSDP2 handles
  this with `set_requires_gradient_sync(False)` on accumulation
  micro-steps; the helper in `train/loop.py` toggles this so the model
  module stays clean.
- `optimizer.zero_grad(set_to_none=True)` — reduces memory traffic.
- The eval and checkpoint paths are **async**. They never block the
  next step's `forward` (see §6, §7).

## 6. Checkpointing

**Decision: `torch.distributed.checkpoint` (DCP) with async save.**

### What we save

- Sharded model state (one file per rank shard).
- Sharded optimizer state.
- Scheduler state.
- Data sampler state: `consumed_tokens_per_source`, RNG state, epoch
  counter.
- Run metadata: git SHA, git diff, full resolved config, tokenizer
  hash, NGC container digest, Python `sys.version`, `torch.__version__`.

### Cadence

- Every 5 B tokens (≈ every 30–45 minutes at main-phase batch).
- Keep last 4 + every 10th forever, garbage-collect the rest.
- Async path: save kicks off a background process; main loop continues
  forward immediately. The next checkpoint save blocks if the previous
  is not finished (cheap fence).

### Resume

- Loader fast-forwards using `consumed_tokens_per_source` + sampler RNG
  state.
- LR scheduler is re-derived from `consumed_tokens` (idempotent on
  resume).
- A unit test does a full save/restore cycle and asserts identical loss
  on the next 100 steps. This test is in CI and **blocks merging**.

### Why DCP and not `torch.save`

- DCP is sharded → restore on a different world size works. We will not
  do this in this project, but DGX Spark's GPU count varies; the
  insurance costs nothing.
- DCP is the path Meta and Nvidia both recommend post-PyTorch 2.4.
- `torch.save` of a sharded FSDP2 model requires gathering full state
  on rank 0 → memory pressure at 8 B.

## 7. Logging & observability

`src/pretrain/obs/`:

- **W&B** is the live dashboard. One run per training run; resumed
  runs append.
- Metrics every step (cheap to log):
  - `loss/ce`, `loss/zloss`, `loss/total`
  - `grad_norm_pre_clip`
  - `lr`, `effective_batch_tokens`, `accum_steps`
  - `tokens_per_sec`, `tokens_per_sec_per_gpu`
  - `mfu`, `hfu`
  - `step_time_ms`, `forward_ms`, `backward_ms`, `optimizer_ms`
- Metrics every N steps (more expensive; N=100):
  - per-block `attn_logit_max`, `qk_q_norm_mean`, `qk_k_norm_mean`
  - per-parameter-group `param_norm`, `update_norm`
- Eval results after every eval pass (every 5 B tokens).
- Health alerts (Slack/email):
  - Loss is `NaN`/`Inf`.
  - `grad_norm_pre_clip > 5σ` of trailing 200 steps.
  - `tokens_per_sec` regresses > 10 % from rolling baseline.

## 8. Loss-spike protocol

Pretraining at 8 B routinely sees grad spikes; we have a documented
playbook so on-call doesn't improvise.

`src/pretrain/train/spike_protocol.py`:

1. **Detect** — `grad_norm_pre_clip > spike_threshold` (default 5.0,
   configurable; threshold tuned during 1 B proxy).
2. **Skip** — zero grads, do **not** step optimizer; log a `skipped_step`
   event with full per-block grad-norm breakdown.
3. **Repeat detection** — if more than `K=5` skips occur in a 50-step
   window, halt training and page on-call.
4. **Rollback** — on operator approval, restart from the last
   pre-spike checkpoint with: data sampler advanced past the suspect
   batches, LR temporarily damped to `0.5×` for 200 steps. Both knobs
   are config flags so the rollback procedure is "edit config, restart"
   and not a one-off branch.

This protocol is **rehearsed during the 1 B proxy** with an artificial
spike injection, not waited-for-in-anger during the 8 B run.

## 9. Determinism & seeds

- `torch.manual_seed`, `numpy`, Python `random`, all seeded from
  `cfg.run.seed`.
- `torch.use_deterministic_algorithms(False)` — full determinism is
  too costly at this scale; we accept run-to-run noise on the order of
  loss ± 1e-4 at the same step.
- The dataloader is fully deterministic (§03 §5). Optimizer is
  deterministic given input. The non-determinism is in
  CUDA/cuBLAS/cuDNN reductions, which is acceptable.

## 10. What this loop intentionally does NOT do

- **No automatic LR retuning at batch transitions.** The research notes
  this risk; we monitor it instead of auto-correcting.
- **No gradient noise scale (`B_simple`) tracking in production.** Our
  batch schedule is research-derived, not telemetry-driven; adding
  `B_simple` is a 1 B-proxy experiment, not a main-run feature.
- **No EMA / SWA over weights.** Not in scope; not used by Llama 3 /
  Qwen3.
- **No automatic resume on hardware failure.** That's a runbook
  procedure (`08_ops.md`), not a loop feature. Auto-resume can hide
  silent corruption.
- **No mid-run hyperparameter changes via config-reload.** Restart the
  run cleanly. Live mutation of an in-flight 8 B run is the kind of
  feature that breaks reproducibility.
