# Architecture for a 7B Dense Decoder-Only LLM Pretraining Run

This document specifies the architectural choices for a fully reproducible
~7B-parameter dense decoder-only LLM, anchored to the recipes used by the
current generation of frontier open-weight models — Llama 3.x, Qwen3, Mistral,
DeepSeek-V3, and Gemma 2/3. The design philosophy is "boring but SOTA":
every ingredient below has converged across at least two independent labs.

---

## 1. Tokenizer

The two dominant choices today are:

* **SentencePiece BPE** (Llama 1/2, Mistral, Gemma) — typically 32k–256k vocab.
* **Byte-level BPE in the tiktoken/GPT-style** (Llama 3, Qwen2/3, GPT-4) —
  100k–150k vocab.

Llama 3 switched to a tiktoken-derived BPE with **128,256** tokens (≈128k),
delivering "up to 15% fewer tokens compared to Llama 2"
([Introducing Meta Llama 3 — Meta AI, 2024](https://ai.meta.com/blog/meta-llama-3/)).
Qwen3 uses a **byte-level BPE (BBPE)** of **151,936** tokens, inherited from
Qwen2's tokenizer, supporting 119 languages
([Qwen3 Technical Report — Alibaba/Qwen, 2025](https://arxiv.org/abs/2505.09388)).
Gemma 2 uses a **256k SentencePiece** vocab inherited from Gemini
([Gemma 2 Technical Report — Google DeepMind, 2024](https://arxiv.org/abs/2408.00118)).

**Trade-offs of larger vocabulary:**

* Pro: shorter sequences (more bytes per token) → cheaper training/inference,
  lower-perplexity fits to multilingual + code data.
* Con: the embedding/unembedding matrices grow as `2 · vocab · d_model`,
  which dominates parameter count at small scales (a 128k × 4096 embedding
  alone is 0.5B parameters), and produces a longer tail of rare tokens that
  see fewer gradient updates.

**Recommendation:** train a fresh **byte-level BPE of 128,000 tokens** on a
sample of the pretraining mix (English + code + ≥10 other languages), with
explicit digit splitting and byte fallback so any input is encodable. This
matches the Llama 3 design point ([The Llama 3 Herd of Models — Meta AI,
2024](https://arxiv.org/abs/2407.21783)) and avoids the parameter-budget
penalty of the 256k Gemma vocabulary at 7B scale.

---

## 2. Positional Embeddings

All recent dense decoder-only models use **Rotary Position Embeddings (RoPE)**
([RoFormer: Enhanced Transformer with Rotary Position Embedding — Su et al.,
2021](https://arxiv.org/abs/2104.09864)). RoPE encodes absolute position via a
rotation matrix applied to Q and K, while the attention dot product depends
only on relative position.

**Base / theta:**
The original RoPE paper used base θ = 10,000 (still the default in Mistral 7B v0.1
— [config.json](https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json)).
Modern long-context training has migrated to much larger bases:

* **Llama 3 / 3.1 8B:** θ = **500,000** at pretrain
  ([Llama-3.1-8B config.json](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/raw/main/config.json)).
* **Qwen3-8B:** θ = **1,000,000** at pretrain
  ([Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)).

This is a form of **RoPE ABF (Adjusted Base Frequency)** — increasing θ
spreads the rotation angles across longer sequences before they alias
([Effective Long-Context Scaling of Foundation Models — Xiong et al., Meta,
2023](https://arxiv.org/abs/2309.16039)).

**Long-context extension** is then performed via continued pretraining on
longer sequences using either:

* **YaRN** ([YaRN: Efficient Context Window Extension of Large Language
  Models — Peng et al., 2023](https://arxiv.org/abs/2309.00071)), an
  NTK-aware piecewise interpolation that needs ~10× fewer tokens than naive
  positional interpolation. Used by Qwen3 to extend 32k → 128k
  ([Qwen3 Technical Report — Alibaba/Qwen, 2025](https://arxiv.org/abs/2505.09388)).
* **Llama 3 RoPE scaling**, an explicit piecewise interpolation specified in
  the HF config (`low_freq_factor=1.0`, `high_freq_factor=4.0`, factor=8.0)
  ([Llama-3.1-8B config.json](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/raw/main/config.json)).

**Recommendation:** RoPE with **θ = 500,000**, native pretrain context
**8,192**, with YaRN-based extension to ≥32k as a separate continued-pretrain
phase. Do not pretrain from scratch at 128k — the FLOPs cost is dominated by
attention.

---

## 3. Attention

The progression for inference-friendly attention:

1. **Multi-Head Attention (MHA)** — original Transformer.
2. **Multi-Query Attention (MQA)** — one shared K and V head across all Q heads
   ([Fast Transformer Decoding: One Write-Head is All You Need — Shazeer,
   2019](https://arxiv.org/abs/1911.02150)). Slashes KV-cache memory but
   degrades quality.
3. **Grouped-Query Attention (GQA)** — middle ground: several Q heads share
   each K/V head ([GQA: Training Generalized Multi-Query Transformer Models
   from Multi-Head Checkpoints — Ainslie et al., EMNLP
   2023](https://arxiv.org/abs/2305.13245)). "Achieves quality close to
   multi-head attention with comparable speed to MQA."

**GQA is the standard.** Q:KV ratio at 7–8B scale has converged on **4:1**:

* Llama 3 8B: 32 Q / 8 KV → ratio 4
  ([NousResearch/Meta-Llama-3-8B config.json](https://huggingface.co/NousResearch/Meta-Llama-3-8B/raw/main/config.json)).
* Mistral 7B v0.1: 32 Q / 8 KV → ratio 4
  ([Mistral-7B-v0.1 config.json](https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json)).
* Qwen3-8B: 32 Q / 8 KV → ratio 4
  ([Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)).

Head dimension is **128** in all three.

**Recommendation:** GQA with **32 Q heads, 8 KV heads, head_dim 128**.

---

## 4. Normalization

**Pre-LN vs Post-LN.** Pre-LN (norm inside the residual block) yields
well-behaved gradients at initialization and removes the need for warmup; the
gradients of Post-LN at init are ill-scaled ([On Layer Normalization in the
Transformer Architecture — Xiong et al., ICML
2020](https://arxiv.org/abs/2002.04745)). Every modern open-weight LLM uses
Pre-LN.

**RMSNorm vs LayerNorm.** ([Root Mean Square Layer Normalization — Zhang &
Sennrich, NeurIPS 2019](https://arxiv.org/abs/1910.07467)) drops the
mean-centering term in LayerNorm; it gives equivalent quality at 7–64% lower
runtime. Used by Llama 1/2/3, Mistral, Gemma 2, Qwen3
([Qwen3 Technical Report — Alibaba/Qwen, 2025](https://arxiv.org/abs/2505.09388)).
`rms_norm_eps` is **1e-5** in Llama 3 and Mistral, **1e-6** in Qwen3.

**QK-Norm.** Applying RMSNorm to Q and K projections before the dot product
fixes attention-logit blow-ups in very large models. Introduced for vision
transformers in [Scaling Vision Transformers to 22 Billion Parameters —
Dehghani et al. /
Google, 2023](https://research.google/blog/scaling-vision-transformers-to-22-billion-parameters/),
and **adopted by Qwen3** as a "newly introduced" ingredient
([Qwen3 Technical Report — Alibaba/Qwen, 2025](https://arxiv.org/abs/2505.09388)).

**Gemma 2 dual-norm.** Gemma 2 uses **both pre-norm and post-norm** RMSNorms
around each sub-layer for additional training stability
([Gemma 2 Technical Report — Google DeepMind, 2024](https://arxiv.org/abs/2408.00118)).

**Recommendation:** Pre-LN **RMSNorm** (eps `1e-5`) on the residual stream,
plus **QK-Norm** (per-head RMSNorm on Q and K). QK-Norm is cheap and provably
helps stability at scales where unnormalized logits begin to drift. Skip
double-norm (Gemma-style) unless instability is observed.

---

## 5. Activation / FFN

The standard FFN is **SwiGLU** ([GLU Variants Improve Transformer — Shazeer,
2020](https://arxiv.org/abs/2002.05202)):
`FFN(x) = (Swish(xW_gate) ⊙ xW_up) W_down`. This requires three weight
matrices instead of two, so to preserve parameter count Llama 1+ keeps the
total FFN parameter budget constant by applying the **2/3 correction**:

```
hidden_dim = int(2 * (4 * dim) / 3)             # SwiGLU correction
hidden_dim = int(ffn_dim_multiplier * hidden_dim) # optional scaling
hidden_dim = round_up_to_multiple(hidden_dim, multiple_of=256)
```

(verbatim from the [Llama 3 reference model.py](https://github.com/meta-llama/llama3/blob/main/llama/model.py)).

For dim = 4096, this gives baseline FFN ≈ 2.67·dim ≈ 10,922 → rounded.
At 8B scale, Llama 3 / Mistral land on **intermediate_size = 14,336**
(≈ 3.5×d_model). Qwen3-8B uses **12,288** (= 3.0×d_model) — the trade-off
is FFN width vs depth/heads.

**Recommendation:** SwiGLU with `intermediate_size = 14,336` (the
Llama/Mistral 7–8B value), `hidden_act = "silu"` (which combined with the
gate is SwiGLU as implemented in HF Transformers).

---

## 6. Architecture Hyperparameters at 7–8B Scale

The closest reference points (all from public `config.json` files):

| Field                | Llama 3 8B | Llama 3.1 8B | Mistral 7B v0.1 | Qwen3-8B |
|----------------------|-----------:|-------------:|----------------:|---------:|
| `num_hidden_layers`  |         32 |           32 |              32 |       36 |
| `hidden_size`        |       4096 |         4096 |            4096 |     4096 |
| `num_attention_heads`|         32 |           32 |              32 |       32 |
| `num_key_value_heads`|          8 |            8 |               8 |        8 |
| `head_dim`           |        128 |          128 |             128 |      128 |
| `intermediate_size`  |     14,336 |       14,336 |          14,336 |   12,288 |
| `vocab_size`         |    128,256 |      128,256 |          32,000 |  151,936 |
| `max_position_embeddings` | 8,192 |     131,072 |          32,768 |   40,960 |
| `rope_theta`         |    500,000 |      500,000 |          10,000 | 1,000,000|
| `tie_word_embeddings`|      false |        false |           false |    false |
| `rms_norm_eps`       |       1e-5 |         1e-5 |            1e-5 |     1e-6 |
| `hidden_act`         |       silu |         silu |            silu |     silu |

Sources: [Llama-3-8B config.json (NousResearch
mirror)](https://huggingface.co/NousResearch/Meta-Llama-3-8B/raw/main/config.json),
[Llama-3.1-8B config.json (NousResearch mirror)](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/raw/main/config.json),
[Mistral-7B-v0.1 config.json](https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json),
[Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json).

The four-way agreement on `d_model=4096`, `n_heads=32`, `n_kv_heads=8`,
`head_dim=128` makes these load-bearing constants for the regime; the only
real degrees of freedom are depth (32 vs 36) and FFN width (12,288 vs 14,336).

---

## 7. Initialization

* **Llama 3 family** uses HF default `initializer_range = 0.02` (truncated
  Normal at ±0.04 in HF), per the Llama 3.1 8B
  [config.json](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/raw/main/config.json).
  Output projection re-scaling ("scaled init", `1/sqrt(2·n_layers)` on
  residual-out projections) is the GPT-NeoX / GPT-2 inheritance and is the de
  facto default in popular pretraining stacks.
* **Qwen3** does not disclose its init scheme in the technical report
  ([Qwen3 Technical Report — Alibaba/Qwen, 2025](https://arxiv.org/abs/2505.09388)),
  but the released weights match the Llama-style σ ≈ 0.02 envelope.
* **μP / μTransfer** ([Tensor Programs V: Tuning Large Neural Networks via
  Zero-Shot Hyperparameter Transfer — Yang et al.,
  NeurIPS 2021](https://arxiv.org/abs/2203.03466)) is the principled
  alternative: parametrize so that the optimal LR and init scales are stable
  as you widen the model. Reported wins: matched GPT-3 6.7B by tuning HPs on
  a 40M proxy, at ~7% of full pretraining cost. This is highly attractive for
  a reproducible reference run because it lets you do an HP sweep at
  ~100M-scale and transfer.

**Recommendation:** Either (a) standard truncated-Normal init with
`std=0.02`, scaled init `1/sqrt(2·n_layers)` on `o_proj` and `down_proj`, OR
(b) μP with a small-model HP sweep. (a) is the path of least resistance and
exactly matches Llama 3.

---

## 8. Tied vs Untied Embeddings

* **Untied:** Llama 1/2/3, Mistral 7B, Qwen3-8B — confirmed by
  `tie_word_embeddings: false` in every config.json above.
* **Tied:** Smaller Qwen3 variants (1.7B, 4B) tie embeddings
  ([Qwen3 Technical Report — Alibaba/Qwen, 2025](https://arxiv.org/abs/2505.09388)),
  Gemma 2 ties for all sizes
  ([Gemma 2 Technical Report — Google DeepMind, 2024](https://arxiv.org/abs/2408.00118)).

The argument for tying is purely parameter-budget: at <2B parameters, an
untied 128k×4096 unembedding matrix is a non-trivial fraction of total params.
At 7–8B, untied wins on quality with only ≈6% parameter overhead.

**Recommendation:** **untied embeddings.**

---

## 9. Context Length & RoPE Base — Pretrain vs Extend

Best practice as evidenced by Llama 3 and Qwen3:

1. **Pretrain at a moderate context** (Llama 3: 8k, Qwen3: 32k) with a
   **large RoPE base** (5e5 or 1e6).
2. **Continued-pretrain on longer sequences** (32k → 128k) using YaRN /
   Llama-3 RoPE scaling. Qwen3 raised θ from 10k → 1M via ABF and applied
   YaRN + Dual Chunk Attention to reach 128k
   ([Qwen3 Technical Report — Alibaba/Qwen, 2025](https://arxiv.org/abs/2505.09388)).

Pretraining at 128k from scratch is FLOP-inefficient (attention is O(L²) and
short-context tokens are far cheaper per gradient step).

**Recommendation:** pretrain at **8,192** with **θ = 500,000**, then extend
to ≥32k with YaRN in a short continued-pretrain phase.

---

## 10. Optional Extras

* **Logit soft-capping (Gemma 2)**:
  `logits ← soft_cap · tanh(logits / soft_cap)`, with 50.0 on attention logits
  and 30.0 on the final LM head
  ([Gemma 2 Technical Report — Google DeepMind, 2024](https://arxiv.org/abs/2408.00118)).
  Mitigates rare large logits but interacts awkwardly with FlashAttention
  kernels — Llama 3 / Qwen3 explicitly do not use it. **Skip** unless you
  observe instability that QK-Norm doesn't fix.

* **z-loss** — auxiliary `1e-4 · log²(Z)` term on the softmax partition
  function, originally used in PaLM and Chameleon to keep the unnormalized
  logits' scale from drifting. Cheap insurance against late-training
  divergence. **Optional**, recommend including with coefficient `1e-4`.

* **MoE vs dense.** DeepSeek-V3 (671B total / 37B active) demonstrates strong
  MoE results at the frontier ([DeepSeek-V3 Technical Report — DeepSeek-AI,
  2024](https://arxiv.org/abs/2412.19437)), but MoE adds substantial
  complexity: load-balancing losses (or auxiliary-loss-free balancing),
  expert sharding, all-to-all communication. For a single-machine, fully
  reproducible reference run we **strongly recommend dense**: deterministic
  outputs, no expert-routing variance, simpler distributed training, far
  smaller engineering surface area. Qwen3 also ships dense 0.6B–8B variants
  precisely to serve this constituency
  ([Qwen3 Technical Report — Alibaba/Qwen, 2025](https://arxiv.org/abs/2505.09388)).

---

## Concrete Reference Architecture

Putting the above together, the recommended 7B configuration:

| Component            | Value                                              | Rationale / Source |
|----------------------|----------------------------------------------------|---------------------|
| Architecture family  | Dense decoder-only Transformer                     | Reproducibility (vs. MoE) |
| n_layers             | **32**                                             | Llama 3 / Mistral 7B |
| d_model              | **4096**                                           | Universal at this scale |
| n_heads (Q)          | **32**                                             | head_dim = 128 |
| n_kv_heads           | **8** (GQA, Q:KV = 4:1)                           | [GQA — Ainslie et al., 2023](https://arxiv.org/abs/2305.13245); Llama 3 / Mistral / Qwen3 |
| head_dim             | **128**                                            | All references |
| FFN type             | **SwiGLU**                                         | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) |
| intermediate_size    | **14,336** (~3.5·d_model w/ 2/3 correction)       | Llama 3 8B / Mistral 7B |
| Normalization        | **Pre-LN RMSNorm**, eps=1e-5                       | [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467); [Xiong et al., 2020](https://arxiv.org/abs/2002.04745) |
| QK-Norm              | **Yes** (per-head RMSNorm on Q,K)                  | [Qwen3, 2025](https://arxiv.org/abs/2505.09388); [ViT-22B / Google Research, 2023](https://research.google/blog/scaling-vision-transformers-to-22-billion-parameters/) |
| Positional encoding  | **RoPE**, θ = 500,000                              | [RoFormer — Su et al., 2021](https://arxiv.org/abs/2104.09864); Llama 3.1 |
| Tokenizer            | **Byte-level BPE, 128,000 vocab**                  | [Llama 3 — Meta AI, 2024](https://ai.meta.com/blog/meta-llama-3/) |
| vocab_size (padded)  | **128,256** (multiple of 128)                      | Llama 3 8B config |
| Tied embeddings      | **No**                                             | Llama 3 / Mistral / Qwen3-8B |
| max_position_embeddings (pretrain) | **8,192**                            | Llama 3 pretrain context |
| max_position_embeddings (extended) | **32,768+ via YaRN**                 | [YaRN — Peng et al., 2023](https://arxiv.org/abs/2309.00071) |
| Initialization       | Truncated Normal **σ=0.02**, scaled init `1/√(2·n_layers)` on `o_proj`/`down_proj` | Llama 3.1 8B config |
| z-loss (optional)    | coeff `1e-4` on `log²(Z)`                          | PaLM-style training stability |
| Logit soft-cap       | **No** (interferes with FlashAttention; unneeded with QK-Norm) | [Gemma 2 — Google DeepMind, 2024](https://arxiv.org/abs/2408.00118) describes the technique |
| Total parameters     | ≈ 8.0B (incl. embeddings)                          | Matches Llama 3 8B / Qwen3-8B class |

This is essentially **the Llama 3 8B architecture with Qwen3's QK-Norm
addition** — the intersection of two independently developed SOTA recipes.
Every choice has at least one production-trained model behind it.

---

### Summary of citations used

* [RoFormer: Enhanced Transformer with Rotary Position Embedding — Su et al., 2021](https://arxiv.org/abs/2104.09864)
* [Fast Transformer Decoding: One Write-Head is All You Need — Shazeer, 2019](https://arxiv.org/abs/1911.02150)
* [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — Ainslie et al., EMNLP 2023](https://arxiv.org/abs/2305.13245)
* [Root Mean Square Layer Normalization — Zhang & Sennrich, NeurIPS 2019](https://arxiv.org/abs/1910.07467)
* [On Layer Normalization in the Transformer Architecture — Xiong et al., ICML 2020](https://arxiv.org/abs/2002.04745)
* [GLU Variants Improve Transformer — Shazeer, 2020](https://arxiv.org/abs/2002.05202)
* [YaRN: Efficient Context Window Extension of Large Language Models — Peng et al., 2023](https://arxiv.org/abs/2309.00071)
* [Effective Long-Context Scaling of Foundation Models — Xiong et al. / Meta, 2023](https://arxiv.org/abs/2309.16039)
* [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer — Yang et al., NeurIPS 2021](https://arxiv.org/abs/2203.03466)
* [Scaling Vision Transformers to 22 Billion Parameters — Dehghani et al. / Google, 2023](https://research.google/blog/scaling-vision-transformers-to-22-billion-parameters/)
* [The Llama 3 Herd of Models — Meta AI, 2024](https://arxiv.org/abs/2407.21783)
* [Introducing Meta Llama 3 — Meta AI, 2024](https://ai.meta.com/blog/meta-llama-3/)
* [Llama 3 reference implementation (model.py) — Meta AI, 2024](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
* [Qwen3 Technical Report — Alibaba/Qwen, 2025](https://arxiv.org/abs/2505.09388)
* [Gemma 2 Technical Report — Google DeepMind, 2024](https://arxiv.org/abs/2408.00118)
* [DeepSeek-V3 Technical Report — DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437)
* [Mistral 7B — Jiang et al., 2023](https://arxiv.org/abs/2310.06825)
* HF model configs:
  [Llama-3-8B](https://huggingface.co/NousResearch/Meta-Llama-3-8B/raw/main/config.json),
  [Llama-3.1-8B](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/raw/main/config.json),
  [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json),
  [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json).

---

## 11. PaLM z-loss verification (follow-up)

The §10 reference to z-loss carried the coefficient `1e-4` from general
knowledge. With direct rendering of the PaLM HTML this is now confirmed
verbatim, and the picture across PaLM / Chameleon / Wortsman et al. /
Gemma 2 is more nuanced than "everyone uses 1e-4."

**PaLM (Chowdhery et al. 2022) — Section 5, "Training Setup".** Quoting
the [ar5iv HTML rendering of arXiv:2204.02311](https://ar5iv.labs.arxiv.org/html/2204.02311):

> "We additionally use an auxiliary loss of `z_loss = 10⁻⁴ · log² Z` to
> encourage the softmax normalizer `log(Z)` to be close to 0, which we
> found increases the stability of training."

So the authoritative PaLM coefficient is **1e-4**, applied to `log²(Z)`
where `Z = Σᵢ exp(xᵢ)` is the partition function of the LM-head softmax,
with the explicit motivation of suppressing softmax-normalizer drift.

**Chameleon (Meta 2024) — Section 2.3 "Stability".** From the
[ar5iv HTML rendering of arXiv:2405.09818](https://ar5iv.labs.arxiv.org/html/2405.09818):

> "Following Chowdhery et al. (2022); Wortsman et al. (2023), we apply
> z-loss regularization. Specifically, we regularize the partition
> function `Z` of the softmax function `σ(x)ᵢ = eˣⁱ / Z` by adding
> `10⁻⁵ · log² Z` to our loss function."

Chameleon-7B reports needing **both** dropout **and** z-loss for
stability; Chameleon-34B requires only z-loss plus the QK-Norm /
norm-reordering changes. Note the coefficient is **10⁻⁵**, an order of
magnitude smaller than PaLM's. Chameleon trained with bf16 and used
QK-Norm; this is consistent with z-loss only needing to be a light touch
once QK-Norm has handled attention-logit growth.

**Wortsman et al. 2023 ("Small-scale proxies for large-scale Transformer
training instabilities", [arXiv:2309.14322](https://arxiv.org/abs/2309.14322)).**
This paper is the canonical small-scale ablation for the two
large-model instabilities — attention-logit growth (Dehghani et al.,
ViT-22B) and output-logit divergence (Chowdhery et al., PaLM). It uses
PaLM's coefficient (`1e-4` on `log²Z`) and shows the resulting
auxiliary loss "resolves this instability" of output-logit divergence.

**Gemma 2 (Google DeepMind 2024).** A direct read of the Gemma 2 tech
report ([arXiv:2408.00118](https://arxiv.org/abs/2408.00118)) does not
mention z-loss; Gemma 2's analogue is the **logit soft-cap**
(`logits ← cap · tanh(logits / cap)`, with caps 50.0 on attention
logits and 30.0 on the LM head). Soft-capping is a hard bound on logit
magnitude rather than a soft penalty on `log Z`; the two techniques
target the same failure mode but are not the same construction.

**Public configs as of 2026.** z-loss is a *training-time* auxiliary
loss; it does not appear in the final HF `config.json` of trained
models, so direct config inspection is not informative. From training
reports: PaLM (1e-4), Chameleon-7B/34B (1e-5), and Wortsman et al.'s
small-scale repros (1e-4) explicitly use it. Llama 3, Mistral, Qwen3,
and Gemma 2 technical reports do **not** mention it; Gemma 2 instead
uses logit soft-cap (Section "Architecture" of
[arXiv:2408.00118](https://arxiv.org/abs/2408.00118)).

**Implication for the §10 recommendation.** Keep z-loss as optional
insurance, but the safer default coefficient is the lower **1e-5**
Chameleon value once QK-Norm is enabled (which the recipe in §4
already specifies). PaLM's `1e-4` predates QK-Norm; with QK-Norm
controlling attention logits, the only remaining job for z-loss is the
output-head divergence, which the lighter Chameleon coefficient is
sufficient for.

Sources:

- [PaLM: Scaling Language Modeling with Pathways — Chowdhery et al. 2022 (ar5iv HTML)](https://ar5iv.labs.arxiv.org/html/2204.02311) — §5 "Training Setup": `z_loss = 10⁻⁴ · log² Z`.
- [Chameleon: Mixed-Modal Early-Fusion Foundation Models — Chameleon Team / Meta 2024 (ar5iv HTML)](https://ar5iv.labs.arxiv.org/html/2405.09818) — §2.3 "Stability": `10⁻⁵ · log² Z`.
- [Small-scale proxies for large-scale Transformer training instabilities — Wortsman et al., ICLR 2024](https://arxiv.org/abs/2309.14322) — replicates PaLM's `1e-4 · log²Z` and confirms "z-loss resolves this instability" of output-logit divergence.
- [Gemma 2 Technical Report — Google DeepMind, 2024](https://arxiv.org/abs/2408.00118) — uses **logit soft-cap** (no z-loss) for the same failure mode.
