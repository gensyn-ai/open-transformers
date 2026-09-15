# 06 — Performance & Nvidia Playbook

*Where we cash in on H100 hardware: parallelism, kernels, compile, NCCL,
profiling. The MFU target is 50 %+; this document is what gets us there
without taking on debt.*

---

## 1. Parallelism plan

**Single-node 8 × H100 today:** **FSDP2 fully-sharded (Zero-3 equivalent),
DP=8, TP=1, PP=1.** The model fits cleanly with activation checkpointing
at `seq=4096, micro_bs=4, n_layers=32, d_model=4096`.

### Why FSDP2 alone (not TP, not PP)

- 8 B fp32 master + bf16 params + bf16 grads + Adam state ≈ 90 GB.
  Fully-sharded across 8 GPUs ≈ 11 GB / GPU. Plus activations at
  `seq=4096, mb=4` with selective AC ≈ 25 GB. Plenty of headroom on 80 GB
  H100.
- TP introduces all-reduces inside the forward pass that are wasted
  bandwidth at single-node scale — NVLink can already saturate with
  FSDP2's reduce-scatter / all-gather pattern.
- PP needs sequence-level micro-batching and is overkill below ~30 B
  parameters.

### What changes when we go multi-node

- `world_size` increases; FSDP2 just shards across more ranks.
- We add a **DP × TP** mesh in `src/pretrain/parallel/meshes.py`. The
  mesh dimension is set in config; today both dims = 1 except DP. This
  scaffolding exists from day 1 so the multi-node move is config, not
  rewrite.
- NCCL env tuning may need re-tuning (cross-node bandwidth differs
  from NVLink).

### FSDP2 wrap policy

- `fully_shard` per `TransformerBlock`.
- `fully_shard` separately on `embedding` and `lm_head`.
- `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32,
  output_dtype=bf16)`.
- `use_orig_params=True` (we need param-name introspection for
  optimizer parameter groups; FSDP2 default).

## 2. Activation checkpointing

- Per-block selective activation checkpointing: re-materialise the
  attention output and FFN output during backward; keep the residual
  stream in memory.
- Configured via `torch.utils.checkpoint.checkpoint` wrapped at the
  block level by a helper in `src/pretrain/parallel/fsdp.py`.
- Tuning knob: which fraction of blocks to checkpoint. Default: every
  block. We measure during the 1 B proxy whether full vs every-other
  is the right trade-off; if MFU at full AC is ≥ 50 % we don't bother
  with the every-other variant.

## 3. Kernels

### Attention

- `torch.nn.functional.scaled_dot_product_attention`, with FA backend
  selected automatically.
- Startup assertion: `flash_sdp_enabled() == True` (we error out if not,
  rather than silently fall back to a slow path).
- For long-context anneal: same SDPA path; FA handles 8 k+ fine.

### Linears

- TransformerEngine `Linear` for `wqkv`, `wo`, `w_gate`, `w_up`,
  `w_down`. TE's `Linear` is a thin bf16/fp8-ready wrapper that fuses
  the cast with the matmul.
- Embedding and lm_head: plain `nn.Embedding` and `nn.Linear`. TE's
  embedding wrapper has stricter shape requirements that buy us nothing
  here.

### Norms

- TransformerEngine `RMSNorm` if available in the container; else plain
  PyTorch RMSNorm with fp32 reduction.

### What we do NOT use

- **Triton custom kernels.** No measured win at 8 B over TE + FA.
- **fp8.** Deferred (`04_model.md` §5).
- **`torch.compile(mode="max-autotune")`.** Compilation time
  explodes at 8 B; the gains over `mode="default"` do not justify it.
  Defaults stay at `mode="default"` (or `"reduce-overhead"` if
  measurement shows it helps without breaking FSDP2 prefetch).

## 4. `torch.compile`

**Decision: `torch.compile(model, mode="default")` after FSDP2
wrapping.**

- Compiled per-block. We let `torch.compile` decide its own boundaries
  inside FSDP2-wrapped modules.
- **Validated on the 1 B proxy** before enabling at 8 B. A `compile=False`
  config flag is the escape valve; we will use it without ceremony if
  we hit a compile-related bug at 8 B that costs us > 4 hours to chase.

## 5. NCCL & environment tuning

These go in `08_ops.md`'s env file but are listed here because they're
performance-critical.

```bash
# Recommended for 8xH100 single-node
NCCL_AVOID_RECORD_STREAMS=1
NCCL_NVLS_ENABLE=1
NCCL_IB_DISABLE=1           # no IB on single-node
NCCL_P2P_DISABLE=0
TORCH_NCCL_AVOID_RECORD_STREAMS=1
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
TORCH_DISTRIBUTED_DEBUG=OFF       # turn ON only when debugging
CUDA_DEVICE_MAX_CONNECTIONS=1     # required for proper TP overlap (set even though TP=1, harmless)
OMP_NUM_THREADS=8
```

These are the Nvidia-recommended defaults for H100; we keep them
identical across single- and multi-node so there's no env drift.

## 6. MFU targets and how we measure

PaLM-style MFU formula:

```
MFU = (6 · N · tokens_per_sec) / (n_gpus · peak_bf16_flops_per_gpu)
```

where `N = active_parameters ≈ 8e9` and `peak_bf16_flops_per_gpu ≈ 1e15`
for H100.

Targets:
- **1 B proxy**: ≥ 45 % MFU. (Smaller models have lower MFU ceiling.)
- **8 B main run**: **≥ 50 % MFU**. This is the gate to start the 150 B
  run; if we are at 35 % we are leaving > 1 week on the table and
  should fix it first.
- **8 B late phase (4 M batch)**: ≥ 55 % expected (larger batches push
  MFU up).

MFU is a W&B panel from step 0. A regression alert fires if MFU drops
> 10 % from the rolling baseline (signals e.g. NVMe bottleneck, NCCL
issue, or a torch.compile recompilation cycle).

## 7. Profiling cadence (Nvidia playbook)

NSight Systems is in the NGC container. We profile at three points:

1. **End of M0** — 100-step `nsys` trace on the 100 M smoke test.
   Sanity check: no CPU bubbles, NCCL overlap clean, no unexpected
   D2H copies.
2. **End of M2** — full 1 B-proxy `nsys` trace, 200 steps mid-run.
   Optimise activation checkpointing fraction if needed.
3. **Day 1 of M4** — 8 B main run, 100-step `nsys` trace.
   This is the **last chance** to find perf bugs before we commit weeks
   of compute.

Profiles are archived in `runs/<run_id>/nsys/` and reviewed by at least
one engineer before we move past each milestone.

We do **not** profile continuously during the main run. The overhead is
non-trivial and we do not need it after M4 day 1 unless something breaks.

## 8. Throughput math (sanity check)

At 50 % MFU, 8 × H100, 1 PFLOP/s peak/GPU:

- Sustained: 4 PFLOP/s.
- Tokens/sec: `4e15 / (6 * 8e9) ≈ 83 000 tokens/sec`.
- 150 B tokens at 83 k tps: `≈ 21 days` of pure forward/backward time.
- Adding overhead (eval, checkpoint, recompiles): **3–4 weeks** —
  matches research §3.3 estimate.

If our measured throughput on day 1 of M4 implies > 5 weeks, we treat
that as a perf bug and stop, not as a "we'll just wait longer" answer.

## 9. Performance debt risks

| Risk | Mitigation |
|---|---|
| `torch.compile` recompiles silently mid-run when shapes change | Fixed `seq_len` and `micro_batch_size`; only `grad_accum_steps` varies. CI test asserts no recompilation in 100 steps after warmup. |
| FSDP2 prefetch + AC interaction misses overlap | Tuned during 1 B proxy with NSight; default config baked in. |
| TE / FA / cuDNN version drift from container update | Pinned NGC tag for the run lifetime; no mid-run upgrades. |
| NVMe bandwidth bottleneck for indexed-binary reads | Shards on local NVMe, mmap reads, `num_workers=4` per rank — measured during proxy. |
| Inefficient dataloader CPU work spikes step time | Dataloader is rank-mmap-reads; tokenization happens at prep time, not in the loop. |
| MFU regression after a perf "improvement" | All perf changes go through a 200-step benchmark on the 1 B proxy with MFU before/after recorded. |
