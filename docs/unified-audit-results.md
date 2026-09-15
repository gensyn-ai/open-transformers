# Unified Cross-Device BFR Training Audit — Results

Bitwise-reproducibility (BFR) audit of the 100M smoke training step: byte-identical
(atol=0) init + per-step loss / grad-norm / state-hash across CPU (ARM + x86), Apple
Metal (MPS), and CUDA (H100). All runs drive the **real** training components —
`build_model`, repop `init_weights_seeded`, the optimizer **registry**
(`adamw_repop` → `FSDPAwareRepopAdamW`), `deterministic_total_norm` + clip,
`compute_state_hash`, `cross_entropy_loss` — for N steps.

Config: `100m_smoke_repop` (d_model 64, 4 heads / 2 kv, head_dim 16, ffn 64, 2 layers,
vocab 4096, seq 64). `REPOP_EXECUTION_MODE=cross_device_reproducible`.

## ★ FINAL: full cross-arch + cross-device BFR (after rope cos/sin → repop)

| backend | machine / toolchain | init | step5 state-hash | verdict |
|---|---|---|---|---|
| ARM-CPU | Mac / clang, torch 2.11 | `5ba1a4ee` | `7ae0ac7a` | ✅ ref |
| MPS | Mac / clang, torch 2.11 | `5ba1a4ee` | `7ae0ac7a` | ✅ == |
| x86-CPU | H100 / gcc+nvcc, torch 2.10 | `5ba1a4ee` | `7ae0ac7a` | ✅ == |
| CUDA | H100 / gcc+nvcc, torch 2.10 | `5ba1a4ee` | `7ae0ac7a` | ✅ == |

**ALL FOUR BACKENDS / TWO MACHINES / TWO COMPILERS / TWO TORCH VERSIONS ARE BYTE-IDENTICAL** across init + 100 training steps (loss, grad-norm, every state-hash). Full cross-arch + cross-device bitwise reproducibility.

### The last leak: RoPE cos/sin (torch libm)
Triaged by diffing Mac-CPU vs H100-x86-CPU at step 1: loss/logits/total-norm matched, but the **attention Q/K/V backward** diverged (`blocks.*.attn.{wq,wk,wv,q_norm,k_norm}.grad_out`, the `q_norm`/`k_norm` weight grads, and 8 optimizer moments). Isolated kernels (flash-bwd, rms-bwd, chunked-matmul, rope-bwd) were all byte-exact cross-machine on random data → the cause was **data-dependent**: torch's `freqs.cos()/sin()` in the RoPE cache (`rope.py` `_build_tables` + YaRN `__init__`) round sub-ULP differently across arch/compiler. The int8-PV forward quantizes that delta away (logits matched); the fp32 rope backward preserves it. Fix = `repop.ops.cos/sin` (correct-rounded, cross-platform).

## (superseded) 4-backend matrix — torch-RNG data, before the cos/sin fix

| backend | machine / toolchain | init | step1 loss | step5 state-hash |
|---|---|---|---|---|
| ARM-CPU | Mac / clang, torch 2.11 | `5ba1a4ee` | 8.333977699279785 | `f0b2f113` |
| MPS | Mac / clang, torch 2.11 | `5ba1a4ee` | 8.333977699279785 | `f0b2f113` |
| x86-CPU | H100 / gcc+nvcc, torch 2.10 | `5ba1a4ee` | 8.333977699279785 | `6aeccbcd` |
| CUDA | H100 / gcc+nvcc, torch 2.10 | `5ba1a4ee` | 8.333977699279785 | `6aeccbcd` |

**Verdicts:**
- ✅ **Within-machine cross-device BFR is PERFECT** — Mac CPU==MPS and H100 CPU==CUDA are byte-identical across init + all 5 steps (loss, grad-norm, state-hash). *This is the guarantee the production audit targets.*
- ✅ **init + step-1 forward are byte-identical even cross-machine** (cross-compiler): same init hash `5ba1a4ee` and same step-1 loss on all four. The forward kernels (matmul, rms_norm, flash int8-PV, the `correct_rounded_*` transcendentals) are cross-compiler-stable.
- ⚠️ **Cross-MACHINE (Mac clang/torch2.11 ↔ H100 gcc+nvcc/torch2.10): a sub-ULP difference enters in step-1's backward/optimizer and compounds** (step5 hash `f0b2f113` vs `6aeccbcd`). It is NOT randomness (data is repop-generated, byte-identical) and NOT the forward (step-1 loss matches). It is a cross-**toolchain** kernel effect — a candidate op is a non-correctly-rounded libm call (e.g. `std::pow(beta, step)` in the AdamW bias-correction) or a default FMA-contraction difference between clang and gcc/nvcc in a backward/optimizer kernel. This is the cross-build frontier, distinct from — and stricter than — the cross-device guarantee.

## Result matrix (initial run, torch-RNG data — superseded by the above)

| Comparison | Machine | Backends | Steps | Verdict |
|---|---|---|---|---|
| ARM-CPU ↔ MPS | MacBook (Apple Silicon) | cpu, mps | 5 | ✅ **ALL BYTE-EXACT** (init + every step: loss, grad-norm, state-hash) |
| CUDA ↔ CPU (static) | — | cuda vs cpu/metal | — | ✅ **COMPLIANT** — all 7 BFR ops, incl. `mm_thread_tma_hfma2` |
| x86-CPU ↔ CUDA (ops) | H100 node (x86_64) | cpu, cuda | — | ✅ **BYTE-EXACT** — chunked_hfma_matmul, sum_dim, rms_norm |
| x86-CPU ↔ CUDA (full step) | H100 node (x86_64) | cpu, cuda | 5 | not run at time of writing |
| x86-CPU ↔ ARM-CPU (cross-arch) | H100 node vs Mac | cpu | 5 | not run at time of writing |

## Mac run (ARM-CPU vs MPS) — verbatim

```
backends: ['cpu', 'mps']  steps: 5
  [cpu] optimizer=FSDPAwareRepopAdamW  init=5ba1a4eea82107eb…
  [mps] optimizer=FSDPAwareRepopAdamW  init=5ba1a4eea82107eb…
  mps_vs_cpu: ALL BYTE-EXACT (init_match=True)
    step1..5: hash=  loss=  grad_norm=   (all =)
```

## What this validates

- **The forward, loss, grad-norm, backward, and optimizer step are all cross-device
  byte-exact** (atol=0) on this stack. Init weights are byte-identical too (repop
  Philox via `stable_trunc_normal`, sampled on CPU and byte-copied to device).
- The **optimizer is the repop BFR AdamW** (`ops.adamw_kernel_step`), not
  `torch.optim.AdamW`. Stock torch AdamW's moment math (sqrt/div) is NOT cross-device
  reproducible — verified: with byte-identical weights+grads, a torch-AdamW step left
  25/35 params and 35/35 optimizer-state tensors diverging (~1e-9). The registry
  selects `adamw_repop`; `build_adamw` was also hardened to route through repop.
- **CUDA**: static algorithmic audit confirms all 7 BFR-critical ops implement the
  identical reduction order / rounding as the CPU reference (the shared contract);
  `mm_thread_tma_hfma2`'s K-reduction IS the strict sequential chunked fold
  (chunk_k=32, fp32 promote, HFMA2 packs along N not K, no split-K).

## Provenance

- repop: CPU/CUDA kernels unchanged from base; the changes under test were
  Metal-only + the fp32 flash kernels.
- transformer-pretraining: grad-norm `torch.dot`→`sum_dim` fix +
  `build_adamw`→repop hardening.

## Depth extension: 5 → 20 → 100 steps (and the bias-correction-pow bug)

Extending the audit surfaced a *patient* cross-machine bug the 5-step run missed:

- **5 steps:** all 4 backends byte-identical.
- **20 steps:** within-machine still perfect, but **Mac ↔ H100 forked at step 15.** Localized: step-15 grads + optimizer moments byte-equal cross-machine, only the *param update* differed → the AdamW **bias correction** `1 - beta^step`, computed with `std::pow(beta, (float)step)` = libm `powf` (fp32, not correctly-rounded, clang≠gcc). Sub-ULP drift took 14 steps to flip a bit.
- **Fix:** route all three backends' bias corrections through `correct_rounded_pow` (fp64 pow → fp32, byte-exact across arch/compiler).
- **100 steps:** all 4 backends byte-identical, `step100 = 7ae0ac7a92ccbc`, zero diverging steps.

Lesson: a reproducibility audit's step count selects which bugs you can see — sub-ULP errors accumulate silently until one flips an output bit.
