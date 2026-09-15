# Optimal Batch Size (in tokens) for Pretraining a 7B LLM

Research note for a fully reproducible 7B pretraining run. Every empirical
claim is cited inline. Topic 4 of the broader research plan in
`research.md` (batch size).

---

## 1. Critical batch size theory: McCandlish et al. (2018)

The foundational reference is McCandlish, Kaplan, Amodei et al. (OpenAI),
"An Empirical Model of Large-Batch Training," arXiv:1812.06162 [^mccandlish].
The paper introduces two coupled quantities:

* **Simple gradient noise scale** `B_simple = tr(Σ) / |G|^2`, where `G` is
  the true gradient and `Σ` its per-example covariance. It is "the ratio of
  gradient variance to its squared mean" and can be estimated cheaply during
  training [^mccandlish].
* **Critical batch size (CBS)** `B_crit = E_min / S_min`, defined via the
  trade-off between minimum optimizer steps `S_min` and minimum total
  examples `E_min` to reach a target loss. The empirical scaling law

  > `(S/S_min − 1) · (E/E_min − 1) = 1`

  follows directly, so at `B = B_crit` both `S` and `E` are exactly twice
  their respective minima ("equal time and compute cost") [^mccandlish].

Operationally:

* For `B ≪ B_crit`, doubling the batch ~halves the number of optimizer
  steps needed (near-linear speedup, "perfect scaling" regime).
* For `B ≫ B_crit`, batches give diminishing wall-clock returns and waste
  compute; loss can also degrade if optimizer hyperparameters are not
  re-tuned.
* `B_simple` increases as the loss decreases over training; in particular
  for a Billion-Word LSTM benchmark it grows from ~`10^3` at the start to
  ~`1.5 × 10^5` tokens by mid-training, and `B_crit` follows the same
  qualitative trend (700 → 100,000 tokens reported) [^mccandlish].
* Critical batch size depends on model size **only via loss**: bigger
  models reach lower loss faster and so unlock larger CBS earlier
  [^mccandlish].

This is exactly the framework GPT-3 used to justify its batch-size schedule:
"We measure the gradient noise scale during training and use it to guide our
choice of batch size [85]" — citing McCandlish et al. directly
[^gpt3-ar5iv].

## 2. Updated estimates

**Shallue et al. 2018** ("Measuring the Effects of Data Parallelism on Neural
Network Training," arXiv:1811.03600) confirmed the three-regime picture
(perfect scaling → diminishing returns → saturation) across many tasks,
with the maximum useful batch varying from `2^9` to `2^16` depending on
workload, and emphasised that LR (and momentum) **must be re-tuned for every
batch size** [^shallue].

**Hoffmann et al. 2022 (Chinchilla)** used conservative batches:
Gopher 280B was trained at `3M → 6M` tokens and Chinchilla 70B at
`1.5M → 3M` tokens (each doubled mid-run). They explicitly note their
batches are likely smaller than McCandlish/Shallue/Zhang would allow
[^chinchilla].

**Zhang et al. ICLR 2025** ("How Does Critical Batch Size Scale in
Pre-training?", arXiv:2410.21676) is the most directly relevant recent
result. With careful HP tuning across 85M–1.2B param transformers on C4,
they find CBS scales primarily with **data size `D`**, not with model size,
and fit:

> `B* ≈ 22.91 · D^0.47`     (D in billions of tokens, B* in sequences)

[^zhang]. Plugging in:

| D (training tokens) | B* (sequences, 4k context) | B* (tokens) |
|---------------------|----------------------------|-------------|
| 100 B               | ~5,000                     | ~20 M       |
| 300 B (Chinchilla 7B) | ~8,400                   | ~34 M       |
| 1 T                 | ~26,300                    | ~108 M      |

These numbers are upper bounds at which scaling stops being efficient; they
are far above what most production 7B runs actually use (see §3), which is
consistent with McCandlish's observation that *operating below* CBS only
costs you wall-clock, never final loss.

## 3. What real frontier models use

| Model | Params | Batch (tokens) | Seq len | Total tokens | Source |
|-------|-------:|---------------:|--------:|-------------:|--------|
| GPT-3 125M | 125 M | 0.5 M | 2048 | 300 B | Table 2.1 [^gpt3-ar5iv] |
| GPT-3 1.3B | 1.3 B | 1 M  | 2048 | 300 B | Table 2.1 [^gpt3-ar5iv] |
| GPT-3 6.7B | 6.7 B | 2 M  | 2048 | 300 B | Table 2.1 [^gpt3-ar5iv] |
| GPT-3 175B | 175 B | **3.2 M** | 2048 | 300 B | Table 2.1 [^gpt3-ar5iv] |
| OPT-6.7B | 6.7 B | **2 M** | 2048 | 180 B | Table 1, Zhang et al. 2022 [^opt] |
| Pythia 6.9B | 6.9 B | **~2.1 M** (1024 × 2048) | 2048 | 300 B | Biderman et al. 2023 [^pythia] |
| MPT-7B | 6.7 B | **~2.1 M** (1024 × 2048) | 2048 | 1 T | llm-foundry yaml [^mpt] |
| LLaMA 1 7B | 6.7 B | **4 M** | 2048 (implicit) | 1 T | Table 2 [^llama1] |
| Llama 2 7B | 6.7 B | **4 M** | 4096 | 2 T | Table 1 [^llama2] |
| PaLM 540B | 540 B | 1M → 2M → **4M** | 2048 | 780 B | §5.3 of paper [^palm] |
| Llama 3 (all sizes) | 8B / 70B / 405B | **4M → 8M → 16M** | 4096 → 8192 | 15 T | §3.4.1 [^llama3] |
| DeepSeek-V3 | 671 B (37 B active) | ~12.6 M → ~62.9 M (3072 → 15360 seq × 4 k) | 4096 | 14.8 T | Tech report [^deepseekv3] |
| Qwen3 (dense) | up to 32 B | not disclosed; "scaling laws for batch size" | 4096 (S1/S2) → 32 k (S3) | ~36 T | Tech report [^qwen3] |
| Chinchilla 70B | 70 B | 1.5M → 3M | 2048 | 1.4 T | Table 4 [^chinchilla] |
| Gopher 280B | 280 B | 3M → 6M | 2048 | 300 B | Table 4 [^chinchilla] |

Two clear patterns:

1. The **modal batch size for ~7B-class models is 2M–4M tokens**.
   GPT-3/OPT/Pythia/MPT cluster at 2M, while Meta's Llama line settled at 4M
   from Llama 1 onward.
2. **Frontier 2024+ runs (Llama 3, DeepSeek-V3, PaLM)** ramp the batch
   *up* over training rather than down, ending at 4M–60M+ tokens. This is
   precisely what McCandlish predicts: as loss falls, `B_crit` rises, so
   you can spend less wall-clock per step without losing efficiency.

Note: Llama 3's "16M tokens" batch is for 405B; the paper says "we use
similar recipes to pre-train the 8B and 70B models" so the 8B almost
certainly uses the same `4M → 8M → 16M` schedule, though Meta does not
break out per-size numbers [^llama3].

## 4. Sequence length × micro-batch interplay

What matters for the optimizer (and for §1's CBS theory) is **total tokens
per gradient step**:

```
batch_tokens = micro_batch × seq_len × grad_accum_steps × DP_world_size
```

All four knobs are interchangeable for the purposes of CBS; they differ
only in **memory** (micro_batch and seq_len cost activations) and
**throughput** (grad_accum costs latency, DP costs interconnect).

Modern 7B runs train at 4k–8k context (Llama 1/2 at 2k/4k, Llama 3 at
4k→8k, Qwen3 stage-1 at 4k, DeepSeek-V3 at 4k) [^llama1][^llama2][^llama3][^qwen3][^deepseekv3].
8k is now the de-facto pretraining context: longer windows (32k–128k) are
typically reserved for a final long-context phase. We adopt **4k** for the
main run and a brief 8k extension (Llama 3 style).

So a 4M-token batch at 4k context is **1024 sequences per step**, which is
exactly the MPT-7B and Pythia-6.9B configurations [^mpt][^pythia].

## 5. Single-machine constraints (8×H100, 80 GB each → 640 GB HBM)

For a 7B model with Adam, ZeRO-3 / FSDP shards parameters (14 GB bf16),
gradients (14 GB bf16) and optimizer state (~84 GB fp32 m/v + ~28 GB fp32
master = ~112 GB) across 8 ranks, leaving roughly
`(14 + 14 + 112) / 8 ≈ 17.5 GB` per GPU for fixed model state, plus
activation memory which grows linearly in
`micro_batch × seq_len × n_layers × d_model`.

Empirically:

* **MPT-7B** trains with `device_train_microbatch_size: 8`, `max_seq_len:
  2048`, FSDP `FULL_SHARD`, `activation_checkpointing: true`,
  `precision: amp_bf16` — fitting comfortably on A100-40GB and
  trivially on H100-80GB [^mpt].
* The HuggingFace Transformers parallelism guide recommends **ZeRO-3 / FSDP
  for any 7B-class single-node training** and refers to the Nanotron
  Ultrascale Playbook for tuning [^hf-parallel].
* DeepSeek-V3 (671 B, MoE) and Megatron-LM all rely on activation
  recomputation; for dense 7B it is essentially free at 4 k context and
  reduces activation memory by ~5–10× [^hf-parallel].

A realistic single-node configuration for a 7B at 4k context:

* `micro_batch_per_gpu = 4` (bf16, FSDP-3, FlashAttention-2, activation
  checkpointing) → 32 sequences per fwd/bwd, ~131k tokens per step
  before grad accumulation.
* To reach a global batch of `4 M tokens` we need
  `4 × 10^6 / 131,072 ≈ 31` grad-accum steps — round to **32**.
* Memory headroom: ~30–40 GB free for activations, leaving room to push
  `micro_batch_per_gpu = 8` if FlashAttention-3 is available, halving
  grad-accum to 16 (this matches MPT-7B's `device_train_microbatch_size:
  8`) [^mpt].

## 6. Batch-size warm-up

Both the McCandlish theory and frontier practice agree that **starting
with a smaller batch and growing it** is preferred:

* **GPT-3** explicitly states "We measure the gradient noise scale during
  training and use it to guide our choice of batch size" [^gpt3-ar5iv]; the
  community-known per-model schedules in Table 2.1 are end-of-training
  values and earlier checkpoints used smaller batches. (The detailed
  per-step schedule lives in the paper's Appendix B and is not
  reproduced verbatim in the ar5iv HTML; we cite the well-documented
  successors PaLM and Llama 3 instead, which inherit the same idea.)
* **PaLM 540B**: "we use batch size 512 (1 M tokens) until step 50k, then
  double it to 1024 (2 M) until step 115k, and finally double it again to
  2048 (4 M) until training is complete at step 255k" [^palm].
* **Llama 3 405B**: "We use a lower batch size early in training to
  improve training stability, and increase it subsequently to improve
  efficiency. Specifically, we use an initial batch size of 4M tokens and
  sequences of length 4,096, and double … to 8M … after 252M tokens. We
  double the batch size again to 16M after pre-training on 2.87T tokens"
  [^llama3].
* **DeepSeek-V3**: "the batch size is gradually increased from 3072 to
  15360 in the training of the first 469B tokens" (sequences at 4k → tokens
  ~12.6M → ~62.9M) [^deepseekv3].
* **Chinchilla / Gopher** also doubled batch size midway through training
  [^chinchilla].

The justification follows directly from McCandlish: `B_crit` *rises* as
loss falls, so a fixed-batch schedule is either wasteful early (`B >
B_crit`, throws away compute) or slow late (`B < B_crit`, leaves
parallelism on the floor) [^mccandlish]. Zhang et al. confirm CBS scales
with data consumed, supporting the same conclusion [^zhang].

## 7. Recommendation for our 7B / single-node run

Working assumptions: ~140B–200B tokens of training (Chinchilla-optimal for
7B is 140B [^chinchilla]; we plan for some over-training as in Llama 2/3),
on 8×H100-80GB with FSDP-3, FlashAttention-2, bf16, activation
checkpointing, and AdamW.

**Schedule (all numbers in tokens):**

| Phase                         | Tokens         | Global batch | seq_len | Sequences/step | micro_bs/GPU × DP × grad_accum |
|-------------------------------|----------------|--------------|---------|----------------|-------------------------------|
| 1. Stability ramp (warm-up)   | 0 → 4 B        | **1 M**      | 4096    | 256            | 4 × 8 × 8                     |
| 2. Main pretraining           | 4 B → 140 B    | **2 M**      | 4096    | 512            | 4 × 8 × 16                    |
| 3. Late efficiency phase      | 140 B → 180 B  | **4 M**      | 4096    | 1024           | 4 × 8 × 32                    |
| 4. (Optional) long-context    | +20–40 B       | 4 M          | 8192    | 512            | 2 × 8 × 32                    |

Justifications:

* The **2 M-token core batch** matches GPT-3 6.7B, OPT-6.7B, Pythia-6.9B,
  and MPT-7B exactly [^gpt3-ar5iv][^opt][^pythia][^mpt]. It is well below
  the Zhang et al. CBS estimate of ~20 M at 100 B tokens [^zhang], so we
  forfeit ~no final-loss quality.
* The **1 M-token warm-up** mirrors PaLM's first phase and Chinchilla's
  early phase [^palm][^chinchilla], improving early-training stability
  per Llama 3's stated rationale [^llama3].
* The **4 M late-phase batch** matches Llama 1/2/3's setting [^llama1][^llama2][^llama3].
* All values are integer multiples of 1024 sequences at 4k, so they map
  cleanly to `micro_bs=4 × DP=8` with grad-accum 8 / 16 / 32.
* `micro_bs=4` at 4k context with FlashAttention-2 + activation
  checkpointing is well within H100-80GB memory headroom; MPT-7B fit
  `micro_bs=8` at 2k on A100-40GB [^mpt], and 4k×4 has the same activation
  footprint.

**FLOP / throughput sanity check.** Per Kaplan-style accounting,
`FLOPs_per_token ≈ 6 · N_params` for transformer pretraining, so 7 B params
× 6 ≈ 4.2 × 10^10 FLOP/token. A 4 M-token step is therefore ~1.7 × 10^17
FLOP. An 8×H100 node at ~50 % bf16 MFU delivers
`8 · 0.5 · 989 TFLOP/s ≈ 4.0 × 10^15 FLOP/s`, so a 4 M step takes ~42 s. A
2 M step takes ~21 s. At 150 B tokens that is ~21 days wall-clock for the
core pretraining phase, consistent with MPT-7B's reported 9.5 days for 1 T
tokens on **440** A100-40GB [^mpt-blog] (scale: 440/8 ≈ 55× more GPUs over
~6× more tokens at similar MFU, giving a single-node estimate in the same
ballpark).

---

## References

[^mccandlish]: McCandlish, Kaplan, Amodei, OpenAI Dota team. *An Empirical
Model of Large-Batch Training*. arXiv:1812.06162, 2018.
https://arxiv.org/abs/1812.06162 ; HTML:
https://ar5iv.labs.arxiv.org/html/1812.06162

[^shallue]: Shallue, Lee, Antognini, Sohl-Dickstein, Frostig, Dahl.
*Measuring the Effects of Data Parallelism on Neural Network Training*.
arXiv:1811.03600, 2018. https://arxiv.org/abs/1811.03600

[^zhang]: Zhang, Bach, Soltanolkotabi, Yu et al. *How Does Critical Batch
Size Scale in Pre-training?*. ICLR 2025. arXiv:2410.21676.
https://arxiv.org/abs/2410.21676 ; HTML:
https://ar5iv.labs.arxiv.org/html/2410.21676 — fits `B* = 22.91 · D^0.47`
with D in billions of tokens, B* in sequences.

[^gpt3-ar5iv]: Brown et al. *Language Models are Few-Shot Learners* (GPT-3).
arXiv:2005.14165, 2020. Table 2.1 (per-model batch sizes from 0.5 M to
3.2 M tokens, n_ctx = 2048) and §2.3 ("We measure the gradient noise scale
… and use it to guide our choice of batch size"). HTML:
https://ar5iv.labs.arxiv.org/html/2005.14165

[^llama1]: Touvron et al. *LLaMA: Open and Efficient Foundation Language
Models*. arXiv:2302.13971, 2023. Table 2: 7B uses **4 M-token batch**, 1.0 T
total tokens. https://ar5iv.labs.arxiv.org/html/2302.13971

[^llama2]: Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat
Models*. arXiv:2307.09288, 2023. Table 1: "All models are trained with a
global batch-size of 4M tokens"; context length 4096.
https://ar5iv.labs.arxiv.org/html/2307.09288

[^llama3]: Grattafiori et al. *The Llama 3 Herd of Models*. arXiv:2407.21783,
2024. §3.4.1 Initial Pre-Training: "initial batch size of 4M tokens and
sequences of length 4,096 … double to 8M … after 252M tokens. Double again
to 16M after pre-training on 2.87T tokens" (405B); §3.4 "We use similar
recipes to pre-train the 8B and 70B models."
https://ar5iv.labs.arxiv.org/html/2407.21783

[^chinchilla]: Hoffmann et al. *Training Compute-Optimal Large Language
Models*. arXiv:2203.15556, 2022. Table 4: Gopher 280B 3 M → 6 M;
Chinchilla 70B 1.5 M → 3 M. https://ar5iv.labs.arxiv.org/html/2203.15556

[^opt]: Zhang et al. (Meta). *OPT: Open Pre-trained Transformer Language
Models*. arXiv:2205.01068, 2022. Table 1: OPT-6.7B uses **2 M-token batch**,
seq_len 2048. https://ar5iv.labs.arxiv.org/html/2205.01068

[^pythia]: Biderman et al. *Pythia: A Suite for Analyzing Large Language
Models Across Training and Scaling*. arXiv:2304.01373, 2023. "1024 samples
with a sequence length of 2048 (2,097,152 tokens) for all models …
consistency across all Pythia model training runs."
https://ar5iv.labs.arxiv.org/html/2304.01373

[^mpt]: MosaicML llm-foundry, `scripts/train/yamls/pretrain/mpt-7b.yaml`:
`global_train_batch_size: 1024`, `max_seq_len: 2048`,
`device_train_microbatch_size: 8`, FSDP `FULL_SHARD`,
`activation_checkpointing: true`. Yields **2,097,152 tokens / batch**.
https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/yamls/pretrain/mpt-7b.yaml

[^mpt-blog]: MosaicML / Databricks. *Introducing MPT-7B: A New Standard for
Open-Source, Commercially Usable LLMs*. May 2023. "~9.5 days to train on
440×A100-40GB GPUs … 1 trillion tokens."
https://www.databricks.com/blog/mpt-7b

[^palm]: Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways*.
arXiv:2204.02311, 2022. §5.3: "we use batch size 512 (1M tokens) until step
50k, then double … to 1024 (2M) until step 115k, and finally … to 2048
(4M) until training is complete at step 255k."
https://ar5iv.labs.arxiv.org/html/2204.02311

[^deepseekv3]: DeepSeek-AI. *DeepSeek-V3 Technical Report*.
arXiv:2412.19437, 2024. Pre-training §: "the batch size is gradually
increased from 3072 to 15360 in the training of the first 469B tokens"
(sequences × 4k context = ~12.6 M → ~62.9 M tokens).
https://ar5iv.labs.arxiv.org/html/2412.19437

[^qwen3]: Yang et al. *Qwen3 Technical Report*. arXiv:2505.09388, 2025.
Three-stage pretraining (S1 4k context / 30T tokens, S2 reasoning, S3 32k
long-context); paper notes "we develop scaling laws for optimal
hyper-parameters (e.g., learning rate scheduler, and batch size)
predictions" but does not publish per-stage batch numbers for the dense
8B/14B/32B models. https://ar5iv.labs.arxiv.org/html/2505.09388 ;
companion blog https://qwenlm.github.io/blog/qwen3/

[^hf-parallel]: HuggingFace Transformers documentation, *Parallelism
methods*, `perf_train_gpu_many` — recommends FSDP / ZeRO-3 for single-node
7B-class training and links to the Nanotron Ultrascale Playbook.
https://huggingface.co/docs/transformers/perf_train_gpu_many

---

## 8. Batch-schedule verification (follow-up)

Targeted re-research (April 2026) to close three open gaps from §3 and §6,
using arXiv full-text, technical reports, and third-party citations.

### 8.1 GPT-3 batch-size warmup (Brown et al. 2020, arXiv:2005.14165)

The PDF of Brown et al. would not render through any of the WebFetch
mirrors attempted (arxiv.org/pdf, NeurIPS proceedings PDF, ar5iv HTML —
arXiv only renders the abstract page; ar5iv truncates before the
appendices). However, the canonical GPT-3 batch-warmup recipe is quoted
verbatim in two well-cited follow-up papers, both attributing it to
Brown et al. 2020:

* Adaptive Batch Size Schedules paper (arXiv:2412.21124v1/v2, §2 Related
  Work):

  > "GPT-3 (Brown et al., 2020) was pretrained by gradually increasing
  > the batch size linearly from a small value (32k tokens) to the full
  > value (3.2M tokens) over the first 4–12 billion tokens of training."

  [^batch-warmup-citation].

* Lambda Labs' GPT-3 technical overview corroborates the per-model
  end-state: "GPT-3 125M use batch size 0.5M and learning rate of
  6.0×10⁻⁴, where GPT-3 175B uses batch size 3.2M and learning rate of
  0.6×10⁻⁴" [^lambda-gpt3].

* The DeepSpeed curriculum-learning blog refers explicitly to "the batch
  size warmup technique introduced by Open AI GPT-3" [^deepspeed-cl],
  confirming the technique is canonically associated with Brown et al.

**Per-model warmup targets:** the "3.2M tokens" target in the standard
quote applies specifically to the **GPT-3 175B** run. The smaller models
ramp linearly to *their* row of Table 2.1 (we already cite this table in
§3): 0.5M for 125M, 0.5M for 350M, 1M for 760M, 1M for 1.3B, 2M for 2.7B,
**2M for 6.7B**, 2M for 13B, 3.2M for 175B [^gpt3-ar5iv]. The 32k starting
value and 4–12 B-token warmup window are reported as a single shared
schedule, not per-model — i.e. all GPT-3 sizes ramp from ~32k tokens to
their model-specific Table-2.1 target over the first 4–12 B tokens.

The exact verbatim text of GPT-3 Appendix B could not be retrieved
(PDF/HTML rendering blocked), so we explicitly mark the per-model 6.7B
ramp endpoint of **2 M tokens** as inferred from Table 2.1 + the shared
linear-warmup recipe quoted by Adaptive Batch Size Schedules
(arXiv:2412.21124).

### 8.2 Llama 3 8B / 70B batch sizes (Grattafiori et al. 2024, arXiv:2407.21783)

The paper is explicit only about the **405B** schedule:

> "We use an initial batch size of 4M tokens and sequences of length
> 4,096, and double these values to a batch size of 8M sequences of
> 8,192 tokens after pre-training 252M tokens. We double the batch size
> again to 16M after pre-training on 2.87T tokens." (§3.4.1)
> [^llama3].

For the 8B and 70B models the paper says only:

> "We use similar recipes to pre-train the 8B and 70B models." (§3.4)
> [^llama3].

No per-size batch table appears in the body, model card, or Meta blog.
The official Hugging Face MODEL_CARD.md for Meta-Llama-3-8B confirms
sequence length (8,192) and total tokens (15T+) but **does not disclose
batch size** [^llama3-modelcard]. The Meta announcement blog likewise
reports only architecture and dataset details [^llama3-meta-blog].

A third-party scaling tutorial ("How To Scale Your Model", JAX-ML)
reports: "LLaMA 3-70B was pretrained with a batch size of about 4M
tokens" (≈1024 sequences × 4k context) [^jax-scaling]. This matches the
Llama 1/2 7B and Llama 3 405B starting batch, but is **not from a
primary Meta source** — treat as a community estimate, not a confirmed
Meta value.

**Conclusion:** the Llama 3 8B max global batch is **not publicly
disclosed by Meta** in either the paper, model card, or blog. The most
defensible reading remains: 8B and 70B "follow similar recipes" to 405B,
which uses 4M → 8M → 16M, but Meta has chosen not to break out per-size
numbers. The community-quoted "4M" for 70B should be cited as
third-party (jax-ml/scaling-book), not Meta-confirmed.

### 8.3 Qwen3 batch schedule (Yang et al. 2025, arXiv:2505.09388)

Direct WebFetch of the arXiv HTML and the Qwen3-8B-Base Hugging Face
README confirms the original assessment in §3:

> "we develop scaling laws for optimal hyper-parameters (e.g., learning
> rate scheduler, and batch size) predictions … we set the predicted
> optimal learning rate and batch size strategy for each dense or MoE
> model" [^qwen3].

The technical report **does not publish concrete per-stage batch sizes
for any of the Qwen3 dense or MoE models**, only the scaling-law
methodology. The Qwen team's blog [^qwen3-blog] and the NVIDIA NeMo
Qwen3 recipe documentation [^nemo-qwen3] likewise list only model-size
options and architecture — **no batch-size value**.

A search hit referencing "global batch size of 4M with 1,000 warmup
steps" and a "dynamic 2M → 4M → 5M → 6M every 125B tokens" sequence
appears to come from arXiv:2601.05034 (a 2026 batch-size paper, "How to
Set the Batch Size for Large-Scale Pre-training?") whose subject is
**generic recommendations**, not a measurement of Qwen3 itself. We
explicitly do **not** attribute these numbers to the Qwen3 team.

**Conclusion:** Qwen3 batch sizes remain **undisclosed** in all primary
and reputable secondary sources we could access. The §3 table entry
("not disclosed; 'scaling laws for batch size'") stands.

### 8.4 DeepSeek-V3 dense baselines (DeepSeek-AI 2024, arXiv:2412.19437)

Re-checking the technical report directly:

* Main DeepSeek-V3 (671B MoE): batch size confirmed verbatim as
  > "the batch size is gradually increased from 3072 to 15360 in the
  > training of the first 469B tokens, and then keeps 15360" [^deepseekv3].

* Ablation models in Table 4 — small-scale baseline (15.7B total params,
  1.33T training tokens) and large-scale baseline (228.7B total params,
  540B training tokens) — are described as **MoE baselines**, not dense
  baselines. The report does not train standalone dense baselines for
  ablation; comparisons against dense models (e.g. LLaMA-3.1 405B) are
  evaluation-only, using the published checkpoints.

* The report **does not specify batch size for the 15.7B / 228.7B
  ablation MoE baselines**.

**Conclusion:** there is no public dense-baseline batch-size figure to
report for DeepSeek-V3; the 3072 → 15360 sequence (≈12.6 M → 62.9 M
tokens at 4 k context) is the only confirmed schedule and refers to the
full 671 B MoE run.

### 8.5 Net edits to §3

* GPT-3 6.7B and 175B end-of-warmup batch values (2 M and 3.2 M tokens
  respectively) are confirmed by Table 2.1; the **shared 32 k → target
  ramp over 4–12 B tokens** is now sourced explicitly via
  arXiv:2412.21124's verbatim quote of Brown et al. 2020.
* Llama 3 8B "almost certainly uses the same 4M → 8M → 16M schedule" —
  this is editorial inference, not a Meta-disclosed number; the only
  third-party numeric anchor is jax-ml/scaling-book's "≈4M tokens" for
  the **70B**, which is itself not from Meta.
* Qwen3 line stays as "not disclosed" — no public number exists.
* DeepSeek-V3 line stays as is; no separate dense baseline batch was
  ever published.

### Additional references (§8)

[^batch-warmup-citation]: Liu, Wong, Tang, et al. *Adaptive Batch Size
Schedules for Distributed Training of Language Models with Data and
Model Parallelism*. arXiv:2412.21124, 2024 (v1) / 2025 (v2). §2 Related
Work, opening sentence:
"GPT-3 [Brown et al., 2020] was pretrained by gradually increasing the
batch size linearly from a small value (32k tokens) to the full value
(3.2M tokens) over the first 4–12 billion tokens of training."
https://arxiv.org/html/2412.21124v1 ; https://arxiv.org/html/2412.21124v2

[^lambda-gpt3]: Lambda Labs blog, *Demystifying GPT-3*: "GPT-3 125M use
batch size 0.5M and learning rate of 6.0×10⁻⁴, where GPT-3 175B uses
batch size 3.2M and learning rate of 0.6×10⁻⁴."
https://lambda.ai/blog/demystifying-gpt-3

[^deepspeed-cl]: Microsoft DeepSpeed, *Curriculum Learning: A
Regularization Method for Efficient and Stable Billion-Scale GPT Model
Pre-Training*: "curriculum learning (seqlen-based) provides much better
training stability than the batch size warmup technique introduced by
Open AI GPT-3."
https://www.deepspeed.ai/tutorials/curriculum-learning/

[^llama3-modelcard]: meta-llama/llama3 MODEL_CARD.md (GitHub) — lists
sequence length (8,192) and 15T+ training tokens; does not disclose
batch size for 8B or 70B.
https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md

[^llama3-meta-blog]: Meta AI, *Introducing Meta Llama 3: The most
capable openly available LLM to date* (April 2024). Architectural and
dataset description; no batch-size disclosure.
https://ai.meta.com/blog/meta-llama-3/

[^jax-scaling]: jax-ml/scaling-book, *Training LLaMA 3 on TPUs* /
*How To Scale Your Model* — community write-up: "LLaMA 3-70B was
pretrained with a batch size of about 4M tokens" (≈1024 × 4096). Not a
Meta-primary source.
https://jax-ml.github.io/scaling-book/applied-training/

[^qwen3-blog]: Qwen Team blog, *Qwen3: Think Deeper, Act Faster* — does
not publish batch-size values.
https://qwenlm.github.io/blog/qwen3/

[^nemo-qwen3]: NVIDIA NeMo Framework User Guide, *Qwen3* recipes — lists
only model-size variants (0.6B – 235B-A30B); no hyperparameter values.
https://docs.nvidia.com/nemo-framework/user-guide/25.07/llms/qwen3.html
