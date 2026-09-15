# 04 — Model Architecture

*Implementation plan for `research/03_architecture.md`. Module-by-module:
which class, which file, which hyperparameters from where, where the swap
points are, how it gets initialised.*

---

## 1. Source of truth — config

All architectural numbers live in
`configs/model/llama3_8b_qknorm.yaml`:

```yaml
name: llama3_8b_qknorm
n_layers: 32
d_model: 4096
n_heads: 32
n_kv_heads: 8
head_dim: 128
ffn_intermediate: 14336
vocab_size: 128256
max_seq_len_pretrain: 8192
rope_theta: 500_000.0
rms_norm_eps: 1.0e-5
qk_norm: true
tie_embeddings: false
init:
  std: 0.02                  # flat OLMo 2 init (2501.00656 §3.2); NO depth scaling
z_loss:
  enabled: true
  coeff: 1.0e-5              # Chameleon, with QK-Norm in place
modules:
  attention: gqa_qknorm
  ffn: swiglu
  norm: rmsnorm
  rope: rope_default
  embedding: untied
```

The 1 B proxy config is the same shape with `n_layers=24, d_model=2048,
n_heads=16, n_kv_heads=4, ffn_intermediate=5632`.

## 2. The model file

**Decision: fork `torchtitan/models/llama3/model.py` as
`src/pretrain/model/llama3.py` and modify it.**

The torchtitan implementation:
- ~ 600 LOC, FSDP2-native, no Megatron dependency
- Used by Meta's PyTorch team for benchmark runs at 8 B–405 B
- Already supports RoPE θ tuning and GQA with our config shape

We change three things:
1. Replace `torchtitan`'s attention with one that supports QK-Norm
   (registered under `gqa_qknorm`).
2. Insert hooks for the registry pattern so the model is composed from
   registered modules rather than hardcoded.
3. Add the optional z-loss pathway (returns auxiliary loss term from
   `forward()` when enabled; the train loop sums it in).

We do **not** rewrite the model from first principles. The reference is
audited; we do not have a reason to do better.

## 3. Module-by-module spec

### 3.1 Embedding (`modules/embedding.py`)

- `nn.Embedding(vocab_size, d_model)`, untied from output.
- Init: `trunc_normal_(std=0.02)`, no scaling.
- Output projection (`lm_head`) is a separate `nn.Linear(d_model,
  vocab_size, bias=False)`, also `trunc_normal_(std=0.02)`.
- Padding is to multiple of 128 (we pick `vocab_size=128256`); embedding
  rows past the trained tokens get zero gradient via mask.

Why untied: research recommendation; ~ 6 % parameter overhead at this
scale, modest quality gain.

### 3.2 RMSNorm (`modules/norm.py`)

- Pre-LN on residual stream (one before attention, one before FFN).
- `eps=1e-5`.
- bf16 compute is fine; we do the variance reduction in fp32 for
  numerical safety (TE's RMSNorm does this; if we use plain PyTorch we
  match that policy explicitly).

Registered as `rmsnorm`. `layernorm` is registered as a bake-off
alternative; never used by default.

### 3.3 RoPE (`modules/rope.py`)

- `theta=500_000.0` (matches Llama 3 8B / 3.1).
- Pre-computed `cos`/`sin` tables for `max_seq_len_pretrain=8192`.
- For the long-context anneal (Phase M6), we recompute tables for 8 192
  with the same θ; YaRN extension to 32 k+ is a separate config that
  re-bases via the standard YaRN scaling formulas — implemented but not
  default.

Registered as `rope_default`. The YaRN variant is `rope_yarn`.

### 3.4 Attention (`modules/attention.py`)

The core swap point. `GroupedQueryAttention` class with:
- 32 Q heads, 8 KV heads, head_dim 128.
- Q/K/V projections as a **single fused `Linear`** for memory efficiency
  (Llama 3 reference style; TE's `Linear` does this fine).
- O projection: separate `Linear`, flat trunc-normal init (std 0.02, no depth scaling).
- RoPE applied to Q and K *after* Q/K-norm (when QK-Norm is enabled).
- Attention computed via
  `torch.nn.functional.scaled_dot_product_attention(...,
  is_causal=True)`. We let SDPA pick the FlashAttention backend; we
  assert at startup that the FA backend is available
  (`torch.backends.cuda.flash_sdp_enabled()`).

**QK-Norm:** per-head `RMSNorm(head_dim, eps=1e-5)` on Q and K, applied
after the projection and before RoPE. Two tiny norm modules per attention
layer; negligible parameter cost.

```python
@register_attention("gqa_qknorm")
class GQAQKNorm(nn.Module):
    def forward(self, x, rope):
        qkv = self.wqkv(x)
        q, k, v = split_qkv(qkv, n_heads=32, n_kv=8, head_dim=128)
        q = self.q_norm(q)              # QK-Norm
        k = self.k_norm(k)              # QK-Norm
        q = apply_rope(q, rope)
        k = apply_rope(k, rope)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out)
```

Registered alternatives:
- `gqa_plain` — no QK-Norm, for ablation only.
- `mha_plain` — full MHA, for ablation at smaller scales.

### 3.5 FFN (`modules/ffn.py`)

SwiGLU, `intermediate_size=14336`:

```python
@register_ffn("swiglu")
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

- `w_gate`, `w_up`, `w_down`: TE `Linear` (or PyTorch `Linear`,
  configurable; default TE in bf16).
- `w_down`: flat trunc-normal init (std 0.02, no depth scaling).

### 3.6 Block (in `model/llama3.py`)

```python
class TransformerBlock(nn.Module):
    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.ffn(self.norm2(x))
        return x
```

This is the load-bearing structure. It does not change between configs.
What changes is the registered submodule chosen for `attn`, `ffn`,
`norm1`, `norm2`.

### 3.7 LM head + loss

- `lm_head` is the untied `Linear` from §3.1.
- Cross-entropy on shifted-by-one targets.
- Optional z-loss term: `coeff · log²(logsumexp(logits))`, summed across
  positions, returned as a separate scalar so we log it independently.
- Logit soft-cap: **off** (research recommendation; conflicts with FA).

## 4. Initialisation

`src/pretrain/model/init.py`:

```python
def init_weights(model, n_layers, std=0.02):
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            nn.init.trunc_normal_(p, std=std, a=-2*std, b=2*std)
        else:
            nn.init.zeros_(p)
    # Scaled init on residual outputs
    scale = (2 * n_layers) ** -0.5
    for block in model.blocks:
        block.attn.wo.weight.data.mul_(scale)
        block.ffn.w_down.weight.data.mul_(scale)
    # RMSNorm gains: ones (default torch init for our class)
    # Embeddings: trunc-normal already from the loop above
```

Tested: same seed → same init. The init is applied **before** FSDP2
wrapping, on the materialised module on CPU/meta-device, then sharded.

## 5. Mixed-precision policy

`src/pretrain/model/precision.py`:

- Activations / forward / backward: **bf16**.
- RMSNorm variance reduction: **fp32** (TE default).
- Loss reduction (cross-entropy): **fp32**.
- Master weights, optimizer state: **fp32** (managed by FSDP2's
  `MixedPrecisionPolicy`).
- Z-loss: computed in **fp32** (it's small and precision matters here).

The single helper that produces both the `MixedPrecisionPolicy` for
FSDP2 and the per-module precision overrides is the only place this
policy is set. Other code asks the helper.

### fp8 — explicitly deferred

TE supports fp8 attention/Linear. We do **not** turn it on for the
primary 150 B-token run. Reasons:
- fp8 needs delayed-scaling calibration; debugging fp8 numeric drift
  during a 3-week run is not where we want to spend cycles.
- The MFU win on 8 × H100 at 8 B is real but modest at our batch size.
- **Revisit trigger**: if/when we extend to a 300 B+ stretch run after
  the 150 B target lands cleanly, evaluate fp8 on the 1 B proxy first.

Documented in ADR-005.

## 6. FSDP2 wrap policy (preview; details in `06_perf.md`)

- `fully_shard` per `TransformerBlock`.
- Embedding and lm_head sharded separately.
- Activation checkpointing: every block (selective AC turned on for the
  `attn` submodule of every other block initially; tuned in M2).

The model file does not contain FSDP wrapping calls; that lives in
`src/pretrain/parallel/fsdp.py` and is applied to the constructed model
in the train loop. The model module is parallelism-agnostic.

## 7. Why this composition prevents debt

- **Three small registries are the entire swap surface.** A new
  attention variant is a single file, a single decorator, a single
  config string.
- **Init / precision / parallelism are NOT in the model file.** Model
  describes the math; train loop / fsdp / precision modules describe
  the execution. This separation is the difference between "I can drop
  this model into a new training stack" and "this model is welded to its
  loop".
- **No conditionals on `is_distributed` inside the model.** Distribution
  is layered on top.
- **Tests at three scales.** `test_model_shapes.py` instantiates 100 M,
  1 B, 8 B configs (8 B with `meta` device only) and asserts
  parameter-count math. These tests catch config drift cheaply.

## 8. Things we are intentionally NOT doing

- **No custom CUDA kernels.** TE + FA + SDPA cover us.
- **No Triton kernels for RMSNorm or RoPE.** Diminishing returns at 8 B
  with TE's fused paths.
- **No Apex.** Stale; PyTorch native covers the same surface now.
- **No dynamic block depth / mixture-of-depths / Mamba blocks.** Outside
  scope.
- **No attention sinks / sliding window.** Not in the recipe; would
  diverge from Llama 3 reference.
- **No bias parameters** anywhere except RMSNorm gains (matches Llama 3
  / Qwen3 / Mistral; biases are on by accident in some HF configs).
