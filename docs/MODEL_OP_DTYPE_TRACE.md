# Model op → repop kernel & dtype trace

Per-operation trace of the canonical training step (`100m_smoke_repop`:
untied embedding, GQA + QK-norm, RoPE, SwiGLU, **LSQ int8 QAT** on the
attention/FFN projections, **int8-PV sliding-window flash attention**, fused
CE + z-loss, repop AdamW), from the model module down to the repop kernel, with
the dtype of the forward value and of **both** backward gradients (data/activation
grad and weight grad).

Traced from `src/pretrain/model/**` and the repop source tree
(`repop/**` in the companion repository). Kernel names are the
CUDA backend; the CPU/Metal twins carry identical names/signatures and are
byte-for-byte reproducible (`REPOP_EXECUTION_MODE=cross_device_reproducible`).

**This trace reflects the production training env**, which sets the
QAT-BFR backward stack (and must match the single-GPU BFR reference env, or the
two stop being bitwise-comparable):

- `REPOP_LSQ_STE_BWD_HADAMARD=1` — **changes dtypes/kernels**. The LSQ STE
  backward GEMMs run on the Hadamard-rotated **int8** kernel, not the bf16
  `mmacc_hfma2` path (see op #4b). This also flips on the fused fp32
  `lsq_fused_bwd` kernel.
- `REPOP_LSQ_SAVE_X_BF16=1` — **no-op here**. It downcasts the saved-for-backward
  activation `x` to bf16, but under bf16 mixed precision `x` is *already* bf16
  (autocast/FSDP), so no bytes change. It is set for explicitness / fp32-param
  variants only; it is NOT a regime change on this run.
- `REPOP_LSQ_DEQUANT_BF16` is **not** set → the LSQ forward dequant stays on the
  fp32 path (op #4), output still bf16 via `out.to(x_float.dtype)`.
- `REPOP_INT8PV_FUSED` defaults to 1 (fused int8-PV forward). The int8-PV
  *backward* HMMA gate is intentionally NOT set — that kernel asserts an fp32
  `dout`, incompatible with the bf16 attention-output grad, so the backward uses
  the bf16 flash STE float-shadow (op #6b).

---

## 0. The precision regime (read this first)

Every per-op dtype below is an instance of one global policy
(`pretrain.model.precision.PrecisionPolicy` + FSDP2 `MixedPrecisionPolicy`,
wired in `parallelize_llama3_repop.py`):

| Role | dtype | Set by |
|---|---|---|
| **Master weight** (sharded, what the optimizer owns) | **fp32** | model built in fp32; FSDP2 keeps the shard in fp32 |
| **Compute weight** (all-gathered for fwd/bwd) | **bf16** | FSDP2 `param_dtype=bf16` |
| **Activations** (forward) | **bf16** | flow from bf16 params; CUDA also under `torch.autocast(bf16)` |
| **Activation grad** ("data gradient", ∂L/∂x) | **bf16** | flows through the bf16 compute graph |
| **Weight grad, local** (per-rank, as the kernel emits it) | **bf16** | produced against bf16 compute weights |
| **Weight grad, reduced** (`param.grad` the optimizer reads) | **fp32** | FSDP2 `reduce_dtype=fp32` reduce-scatter (+ HSDP all-reduce) |
| **Optimizer moments + update** | **fp32** | repop AdamW reads param/grad as fp32, moments always fp32 |
| **Norm / loss / softmax reductions** | **fp32** | repop kernels upcast internally regardless of I/O dtype |
| **int8 QAT operands** | **int8**, int32 accumulate | LSQ forward quantizes x and w per step |

So the single most important dtype fact: **activation grads stay bf16; weight
grads are born bf16 and become fp32 only at the FSDP2 reduce boundary**, and the
whole optimizer/master side is fp32. `autocast` is enabled only on CUDA; on
CPU/MPS (audit) it is disabled and the bf16-ness comes purely from the FSDP2
param cast (or its ws=1 emulation).

Two things that are easy to get wrong:

- **The lm_head is NOT quantized.** `UntiedEmbeddingRepop.output` is a plain
  `repop.nn.linear.Linear` (bf16), not `LSQQuantizedLinear`. Only the
  attention (wq/wk/wv/wo) and FFN (w_gate_up/w_down) projections are int8-LSQ.
- **RoPE cos/sin buffers are fp32**, not bf16 — buffers are not cast by the
  FSDP2 `param_dtype`. `apply_rope` therefore runs bf16·fp32 → fp32 → stored
  back to a bf16 output tensor.

---

## 1. Forward pass

| # | Model op | repop entry / kernel | Fwd compute dtype | Fwd output dtype |
|---|---|---|---|---|
| 1 | **Token embedding** `embedding.encode` | `repop.nn.embedding.Embedding` → `_DeterministicEmbeddingFn` (plain `weight[idx]` gather) | index gather (no arithmetic) | **bf16** (weight is bf16 compute param) |
| 2 | **emb_norm** (optional) / **norm1 / norm2 / norm_out** RMSNorm | `repop.nn.rmsnorm.rms_norm` → `rms_norm.cu` (`sum2d_dim1` reduce, `correct_rounded_rsqrt`) | **fp32** (all arithmetic fp32; `rstd` fp32) | **bf16** (`y = x·rstd·w` cast to input dtype) |
| 3 | **q_norm / k_norm** RMSNorm (head_dim, 2-D flattened) | same as #2, `rms_norm.cu` | **fp32** | **bf16** |
| 4 | **Q/K/V/O projections** (LSQ int8) `wq,wk,wv,wo` | `repop.qat.lsq.LSQQuantizedLinear` → `LSQQuantizedMatmulFunction.fwd` | x→**int8** (per-tensor amax, `quantize_per_tensor`), w→**int8** (per-channel scale, `lsq_weight_quant_div`), GEMM **int32** (`int8_mm_t_cublas` / `_int8_mm_int32_t`), dequant fp32 (`lsq_int32_to_fp32_dequant`) | **bf16** (`out.to(x_float.dtype)`) |
| 5 | **RoPE apply** `apply_rope` | torch elementwise (device-native); tables built once via `repop.ops.cos/sin` | bf16·**fp32** → **fp32** (type promotion vs fp32 cos/sin) | **bf16** (written into `empty_like(x)`) |
| 6 | **Attention** int8-PV causal flash (sliding window) | `repop.nn.flash_attention.int8pv_causal_flash_attention` → `bfr_flash_attention.cu` (`causal_int8pv_flash_attention_fwd_bf16_32_256`) | scores **fp32**, online-softmax stats fp32; P,V→**int8** per block; PV **int32** accumulate; dequant fp32; `o_acc` fp32 | **bf16** |
| 7 | **SwiGLU** `w_gate_up` (LSQ) → SiLU·up → `w_down` (LSQ) | gate/up: LSQ (#4); `repop.nn.activations.silu` → `activations_autograd.cu` (`correct_rounded_exp`); elementwise `*`; down: LSQ (#4) | LSQ int8 as #4; SiLU computed on input dtype (bf16) | **bf16** |
| 8 | **lm_head** `embedding.project` (plain, **not** LSQ) | `repop.nn.linear.Linear` → `linear.cu` (`mmacc_hfma2_trans`, bf16 + hfma2) | bf16 GEMM, **fp32 accumulate** (hfma2) | **bf16** logits |
| 9 | **Fused CE + z-loss** `fused_ce_z_loss` | `_FusedCEZLoss` over `repop.ops.{exp,sum_dim,log,mean,pow}` (`cross_entropy.cu`, `softmax.cu`) | logits upcast per-chunk to **fp32**; logsumexp/CE/z-loss all **fp32** | **fp32** scalars `(ce, z_loss)` |

Loss assembly: `loss = (ce + zloss) * (1.0/accum)` — fp32 scalar. The `*1/accum`
(not `/accum`) is deliberate for cross-device bit-equality.

---

## 2. Backward pass

Columns: **data grad** = ∂L/∂input (activation gradient handed upstream);
**weight grad (local)** = what the op's backward emits for its Parameter, before
FSDP reduces it. All local weight grads become **fp32** in `param.grad` after the
FSDP2 `reduce_dtype=fp32` reduce-scatter (op #16).

| # | Model op backward | repop entry / kernel | Data grad (∂L/∂x) dtype | Weight grad (local) dtype |
|---|---|---|---|---|
| 9b | **CE + z-loss backward** | `_FusedCEZLoss.backward` (recompute softmax in row-chunks) | grad computed **fp32**, returned `to(logits.dtype)` = **bf16** | — (no weights) |
| 8b | **lm_head backward** (plain Linear) | `linear.cu` backward: `grad_input = mmacc(go, w)`, `grad_weight = mmacc(goᵀ, x)` (both hfma2, fp32 accumulate) | **bf16** | **bf16** → fp32 after reduce |
| 7b | **SwiGLU backward** | SiLU′ via `activations_autograd.cu`; down/gate/up projections via LSQ backward (#4b) | **bf16** | LSQ weights: **bf16** → fp32 after reduce |
| 6b | **Attention backward** (int8-PV flash) | **STE float-shadow**: reuses the **fp32 BFR** bwd `causal_flash_attention_bwd_bf16_32_256` (`bfr_flash_attention.cu`); produces dq,dk,dv | **bf16** (dq/dk/dv, same dtype as q/k/v) | — (no weights; attention has none) |
| 5b | **RoPE backward** | torch elementwise autograd (device-native) | **bf16** | — (no weights; tables are non-persistent buffers, `requires_grad=False`) |
| 4b | **Q/K/V/O & FFN LSQ backward** | `LSQQuantizedMatmulFunction.backward`: STE. `grad_x=_ste_matmul(go,w)`, `grad_w_pre=_ste_matmul(go,xᵀ)` via **`_ste_matmul_hadamard` → `bfr_hadamard_int8_gemm.cu`** (Hadamard on). Then fused **`lsq_fused_bwd.cu`** (fp32) for scale/clip | operands read bf16, **widened to fp32 on load**, per-row/col **int8** quant, **int32** GEMM, **fp32** dequant (folds 1/K). `grad_x` returned `to(ctx.x_dtype)` = **bf16** | STE `grad_w_pre` is **fp32** (Hadamard int8 GEMM out) → `lsq_fused_bwd` → `grad_w.to(w.dtype)` = **bf16** → fp32 after reduce. **weight_scale grad**: fp32 (`sum2d_dim1`) |
| 3b/2b | **RMSNorm backward** (q_norm/k_norm, norm1/2/out, emb_norm) | `rms_norm.cu` backward (`rms_norm_bwd`): recompute from `normalized` + `rstd` | **fp32** internal arithmetic; `grad_in` returned **bf16** | **grad_weight fp32** (accumulated fp32 in-kernel) → fp32 after reduce |
| 1b | **Embedding backward** | `_DeterministicEmbeddingFn.backward`: stable-sort + fixed-order segment scatter-add | — (input is int ids) | **fp32** grad accumulate (`grad_w` built fp32), returned `to(grad_out.dtype)` = **bf16** → fp32 after reduce |

STE note: both int8 paths (LSQ #4b, int8-PV attention #6b) use
straight-through estimators — the gradient *math* is straight-through (no
derivative of the quantizer itself). But on this run the two paths differ in
the matmul kernel:

- **LSQ #4b**: with `REPOP_LSQ_STE_BWD_HADAMARD=1` the two STE-backward GEMMs
  **do run on int8 tensor cores** (Hadamard-rotated `bfr_hadamard_int8_gemm`,
  fp32 output). The rotation clears the int8 gradient-outlier wall so int8 stays
  BFR. With the gate off (default), these fall back to bf16 `mmacc_hfma2`.
- **int8-PV attention #6b**: pure **fp32/bf16 float-shadow** — reuses the fp32
  BFR flash bwd, never an int8 GEMM (the int8-PV backward kernel is deliberately
  disabled under bf16; see the header note).

---

## 3. Reduction / fold / optimizer (master update)

| # | Op | repop entry / kernel | dtype |
|---|---|---|---|
| 16 | **Grad reduce** (FSDP2 reduce-scatter + HSDP cross-replica all-reduce) | FSDP2 `MixedPrecisionPolicy(reduce_dtype=fp32)`; bucketed HSDP all-reduce (`_replicate_reduce_hook.flush`) | local bf16 grad → **fp32** reduced grad in `param.grad` |
| 17 | **Global grad-norm + clip** (`grad_clip=1.0`) | `_global_grad_norm` → `get_total_norm` (`sq_norms` cast `.to(fp32)**2`), `clip_grads_with_norm_` | **fp32** |
| 18 | **LSQ scale refresh** (every N steps) | `refresh_weight_scale`: `sum_dim` over bf16-cast \|w\| → `×const` | reduce in **bf16** (BFR), scale stored fp32 |
| 19 | **Optimizer step — repop AdamW** | `FSDPAwareRepopAdamW` → `ops.adamw_kernel_step` → `optimizer_kernels.cu` | reads param & grad **as fp32**; `exp_avg`/`exp_avg_sq` **always fp32**; decoupled WD `p*=(1-lr·wd)`; writes master back in its dtype (**fp32**) |
| 20 | **Weight init** | `repop.rand.stable_trunc_normal` → CPU `trunc_normal` | **fp32** (device-independent; routes to CPU) |
| 21 | **State hash** | `compute_state_hash` (blake2b over master weights + fp32 moments + grads) | byte-level over **fp32** master/moment/grad |

---

## 4. Quick dtype cheat-sheet by tensor

- **input_ids**: int64
- **all activations h, attention out, logits**: bf16
- **all ∂L/∂activation (data grads)**: bf16
- **CE/z-loss scalars & their internal LSE/softmax**: fp32
- **RoPE cos/sin tables**: fp32 (buffers, no grad)
- **int8-QAT**: x int8 (per-tensor), w int8 (per-channel), GEMM int32, dequant→bf16
- **compute weights (fwd/bwd)**: bf16 (FSDP all-gather)
- **local weight grads**: bf16 (except RMSNorm gain & embedding & LSQ-scale grads, which are fp32 in-kernel)
- **reduced weight grads (`param.grad`)**: fp32
- **master weights, AdamW moments, grad-norm, clip, update**: fp32

---

## 5. Caveats / things to double-check before quoting

- **bf16 vs fp32 weight grad**: the "local bf16 → reduced fp32" story is the
  FSDP2 (world_size>1, CUDA) path. On the **single-device MPS/CPU audit** at
  ws=1 there is no real reduce-scatter; the emulation casts to bf16 and the
  grad-accumulation dtype across microbatches is a known MPS-audit gap.
  Numeric equality of that emulation to real ws=1 FSDP2 is validated
  end-to-end for the 1.6B step (flash-bwd dV fix), but it is an *emulation*, so
  state the ws when quoting the weight-grad dtype.
- **autocast is CUDA-only.** On CPU/MPS the model is not under autocast; bf16
  comes only from the FSDP param cast / emulated bf16 copy. A few torch-native
  elementwise ops (RoPE, residual adds) therefore promote to fp32 mid-expression
  and round back to bf16 on store — identical on both, but worth knowing when
  reading intermediate dumps.
- **hfma2 accumulate**: the bf16 GEMMs (plain Linear, STE matmuls) accumulate in
  fp32 inside `mmacc_hfma2*` even though inputs/outputs are bf16. "bf16 GEMM"
  here means bf16 operands, fp32 accumulator, bf16 result.
- **Gate reality in the production env** (see the header list): `REPOP_LSQ_STE_BWD_HADAMARD=1`
  is the only set gate that changes kernels/dtypes (LSQ backward → int8 Hadamard,
  op #4b). `REPOP_LSQ_SAVE_X_BF16=1` is set but a no-op under bf16 mixed precision.
  `REPOP_LSQ_DEQUANT_BF16` is **not** set, so the LSQ forward dequant stays fp32→bf16
  (op #4). If you ever run a fp32-param variant, `SAVE_X_BF16` becomes a real
  regime change (backward sees bf16-rounded `x`) and `DEQUANT_BF16` would push the
  forward dequant + downstream saved tensors to bf16 — neither applies here.
