# Scaling Laws and the Optimal Token Budget for a 7B-Parameter LLM

*Research notes for a fully reproducible 7B pretraining run. Every empirical claim is cited.*

---

## 1. Chinchilla (Hoffmann et al., 2022): the compute-optimal scaling law

The reference work for compute-optimal training of dense decoder-only transformers is *Training Compute-Optimal Large Language Models* by Hoffmann et al. at DeepMind ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556)). The authors trained "over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens" and concluded that "for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled" ([Hoffmann et al., 2022, abstract](https://arxiv.org/abs/2203.15556)).

They present three estimation methods. **Approach 3** (parametric loss fit) is the one that gives concrete coefficients usable for budgeting. They model the final pre-training loss as:

> L̂(N, D) = E + A / N^α + B / D^β   *(Eq. 2)*

with the following empirical fit on their training corpus ([Hoffmann et al., 2022, §D.2](https://arxiv.org/abs/2203.15556)):

> E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28.

Treating training compute as **C ≈ 6 N D** ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361); also adopted by Chinchilla, §3.3) and minimizing L̂ subject to fixed C yields the closed-form efficient frontier:

> N_opt(C) = G · (C/6)^a, D_opt(C) = G⁻¹ · (C/6)^b,
> where G = (αA / βB)^(1/(α+β)), a = β/(α+β), b = α/(α+β).   *(Eq. 4)*

The three approaches each yield a/b exponents close to ½: Approach 1 gives a = 0.50, b = 0.50; Approach 2 gives a = 0.49, b = 0.51; Approach 3 gives a = 0.46, b = 0.54 (Hoffmann et al., 2022, Table 2). The flagship empirical test was *Chinchilla*: a 70B-parameter model trained on **1.4T tokens** under the same FLOP budget as the 280B *Gopher* model, which it outperformed broadly (e.g., MMLU 67.6% vs. 60.0%).

**The "20 tokens per parameter" rule.** Chinchilla 70B at 1.4T tokens corresponds to 20 tokens/parameter, and Hoffmann et al.'s Table 3 projects a similar 20:1 ratio across model sizes. This translates directly:

| Model size | Compute-optimal tokens (Chinchilla 20:1) |
|------------|------------------------------------------|
| 7B         | **≈ 140B tokens** |
| 13B        | ≈ 260B tokens |
| 70B        | ≈ 1.4T tokens (the actual Chinchilla run) |

This 140B-token figure is the canonical "Chinchilla-optimal" target for a 7B model.

---

## 2. Kaplan et al. (2020) and why Chinchilla superseded it

The earlier scaling-law paper of record is Kaplan et al., *Scaling Laws for Neural Language Models* ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361)). They likewise fit power laws of loss vs. N, D, and C, but their headline allocation rule was **very different**: increasing the compute budget should mostly go into model size, not data. Specifically, they reported

> N_opt(C) ∝ C^0.73, D_opt(C) ∝ C^0.27

(see [Wikipedia: Neural scaling law](https://en.wikipedia.org/wiki/Neural_scaling_law) summarizing Kaplan et al., 2020). Hoffmann et al. quote this directly: "given a 10× increase computational budget, they suggest that the size of the model should increase 5.5× while the number of training tokens should only increase 1.8×" ([Hoffmann et al., 2022, §1](https://arxiv.org/abs/2203.15556)).

Kaplan's rule produced GPT-3-style models — a 175B-parameter model trained on only ~300B tokens (≈ 1.7 tokens/parameter). Chinchilla's analysis showed those models were **massively under-trained**.

**Why Kaplan was wrong.** Hoffmann et al. (§2) attribute the discrepancy to two methodological choices in Kaplan's work:

1. They "use a fixed number of training tokens and learning rate schedule for all models." Chinchilla showed that the LR cosine schedule should match the training horizon; using a fixed long schedule overestimates the loss of small/short runs and biases the optimum toward bigger models.
2. The majority of Kaplan's models are below 100M parameters, where the FLOP-loss frontier shows curvature.

A third factor, well documented in [Wikipedia: Neural scaling law](https://en.wikipedia.org/wiki/Neural_scaling_law), is that Kaplan et al. did not count token-embedding parameters in N. At small N, embeddings are a large fraction of the total, biasing α. Once these issues are corrected, Kaplan's data is consistent with Chinchilla's a ≈ b ≈ 0.5.

The contemporary consensus is that the **Chinchilla allocation (a ≈ b ≈ 0.5, ≈20 tokens/param)** is the correct compute-optimal rule for vanilla dense transformer pretraining.

---

## 3. The post-Chinchilla "over-training" trend

Chinchilla minimizes *training* compute for a fixed loss target. It deliberately ignores **inference** cost, which often dominates total lifetime FLOPs for a deployed model. The Llama series leans hard into this trade-off:

| Model | Params | Training tokens | Tokens/param | Source |
|-------|--------|-----------------|--------------|--------|
| LLaMA 1 7B | 7B  | 1.0T  | ~143:1 | [Touvron et al. 2023a](https://arxiv.org/abs/2302.13971) |
| Llama 2 7B | 7B  | 2.0T  | ~286:1 | [Touvron et al. 2023b](https://arxiv.org/abs/2307.09288); [Llama-2-7b model card](https://huggingface.co/meta-llama/Llama-2-7b) |
| Llama 3 8B | 8B  | ~15T  | **~1875:1** | [Grattafiori et al. 2024](https://arxiv.org/abs/2407.21783); [Sardana et al. 2024 §1](https://arxiv.org/abs/2401.00448) |
| Chinchilla 70B | 70B | 1.4T | 20:1 | [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556) |

Llama 1's authors explicitly cite this trade-off as motivation: "Touvron et al. (2023a) cites the lower inference cost of smaller models as inspiration for the LLaMA series" ([Sardana et al., 2024, §1](https://arxiv.org/abs/2401.00448)).

**Sardana et al., *Beyond Chinchilla-Optimal*** ([arXiv:2401.00448](https://arxiv.org/abs/2401.00448)) formalizes the trade-off. They modify the Chinchilla scaling law to account for inference cost and conclude:

> "LLM researchers expecting reasonably large inference demand (~10⁹ inference requests) should train models smaller and longer than Chinchilla-optimal" (abstract).

They train 47 models from 150M to 6B parameters at token-to-parameter ratios from 10 to 10,000 and report:

> "Loss continues to decrease as we increase tokens per parameter, even to extreme ratios… we see no evidence of loss flat-lining" (§2.4).

Concretely, their Figure 2 shows that for a "7B-Chinchilla-quality model with an inference demand of 10¹¹ tokens, our formula suggests the compute-optimal method is to train a 6B parameter model on 1.18× the original (Chinchilla-prescribed) amount of data" — i.e., mild over-training when inference demand is significant ([Sardana et al., 2024, §2](https://arxiv.org/abs/2401.00448)).

The trade-off is straightforward: more training tokens → lower loss / better downstream evals, but training FLOPs grow linearly with D while inference FLOPs are unchanged. For research runs where inference demand is small or zero, Chinchilla-optimal is fine. For production deployment (Meta's case), heavy over-training pays back.

---

## 4. Replication and corrections: Besiroglu et al. (2024)

In 2024 the community caught a subtle issue with Hoffmann et al.'s **Approach 3** specifically. Besiroglu, Erdil, Barnett & You, *Chinchilla Scaling: A Replication Attempt* ([arXiv:2404.10102](https://arxiv.org/abs/2404.10102), summarized at [Epoch AI blog](https://epoch.ai/blog/chinchilla-scaling-a-replication-attempt)) showed that the original Approach 3 fit:

- was "inconsistent with their first two estimation methods,"
- "fail[ed] at fitting the extracted data," and
- reported "implausibly narrow" confidence intervals — "intervals this narrow would require over 600,000 experiments, while they likely only ran fewer than 500" (Besiroglu et al., 2024, abstract).

Re-fitting Approach 3 using the data scraped from the Chinchilla paper figures, they obtain the **corrected parametric law**:

> L(N, D) = 1.8172 + 482.01 / N^0.3478 + 2085.43 / D^0.3658

i.e., E = 1.8172, A = 482.01, B = 2085.43, α = 0.3478, β = 0.3658 ([Epoch AI: Chinchilla Scaling: A Replication Attempt](https://epoch.ai/blog/chinchilla-scaling-a-replication-attempt)).

**Does the 20:1 rule survive?** Yes. The corrected exponents give a = β/(α+β) ≈ 0.513 and b = α/(α+β) ≈ 0.487 — i.e., near-equal scaling of N and D with C, exactly the conclusion of Approaches 1 and 2. The Epoch AI write-up notes: "their estimates fit the data better and align with Hoffmann's other approaches… consistent with the scaling policy used for Chinchilla," which used a 20:1 ratio.

So the headline practical recommendation — **~20 tokens per parameter for compute-optimal training** — is robust to the replication corrections; only the noise quantification and the specific Approach-3 coefficients changed materially.

---

## 5. Practical token-budget recommendation for a 7B research run

We can group the candidate budgets:

| Regime | Tokens (7B) | Tokens/param | Justification |
|--------|------------|--------------|---------------|
| Strict Chinchilla-optimal | ~140B | 20:1 | [Hoffmann et al. 2022, Table 3](https://arxiv.org/abs/2203.15556) |
| Modest over-training (LLaMA 1-style) | ~1T | 143:1 | [Touvron et al. 2023a](https://arxiv.org/abs/2302.13971) |
| Llama 2-style | ~2T | 286:1 | [Touvron et al. 2023b](https://arxiv.org/abs/2307.09288) |
| Heavy over-training (Llama 3-style) | ~15T | 1875:1 | [Grattafiori et al. 2024](https://arxiv.org/abs/2407.21783) |

**Recommendation for a single-machine 7B research run: target ~150B tokens (≈ Chinchilla-optimal 20:1), with an option to extend to 300–500B if budget allows.**

Reasoning:

- **Compute economy.** At 20:1 we are sitting on the compute-efficient frontier (Hoffmann et al. 2022, Approach 1: a = 0.50, b = 0.50). Loss drops rapidly with tokens here; beyond ~150–300 tokens/param, the marginal benefit is small per Chinchilla and per Sardana et al.'s extreme-ratio sweep (which still shows monotonic improvement but with diminishing returns).
- **Single-machine feasibility.** A Llama-3-style 1T+ run at 7B is infeasible on a single 8-GPU node in any reasonable wall-clock time (see §6). Sardana et al. explicitly note that the inference-cost over-training argument only justifies aggressive over-training when you expect ≥10⁹ inference requests; a research artifact does not meet that bar.
- **Reproducibility and known good recipe.** 1T tokens / 7B was the LLaMA 1 recipe (Touvron et al. 2023a) and is well within the validated regime of every replication. If the project later wants to extend to LLaMA-1 parity (1T tokens, 143:1), the same recipe can simply be continued; the LR schedule should be planned for the longer horizon as Hoffmann et al. (Appendix B) note that "setting the learning rate schedule to approximately match the number of training tokens results in the best final loss."

A specific concrete plan:

- **Primary target:** 7B params × 150B tokens (≈ 6.3 × 10²² FLOPs at 6ND).
- **Stretch target if time permits:** 7B × 300B tokens (≈ 1.26 × 10²³ FLOPs), still well below LLaMA-1's 1T.

---

## 6. Compute estimates: FLOPs and GPU-hours for a 7B × 140B-token run

Using the **C ≈ 6ND** approximation introduced in Kaplan et al., 2020 and adopted in Hoffmann et al., 2022 (§3.3) — also confirmed by the Wikipedia summary that "C₀ = 6, meaning that it costs 6 FLOPs per parameter to train on one token" ([Wikipedia: Neural scaling law](https://en.wikipedia.org/wiki/Neural_scaling_law)):

> FLOPs(7B, 140B tokens) ≈ 6 × 7 × 10⁹ × 1.4 × 10¹¹ ≈ **5.88 × 10²¹ FLOPs ≈ 5.9 zettaFLOPs**.

For comparison, [Wikipedia: Llama (language model)](https://en.wikipedia.org/wiki/Llama_(language_model)) reports the largest LLaMA-1 (65B × 1.4T tokens) at **6,300 petaFLOP-days ≈ 5.4 × 10²³ FLOPs**, so a 7B × 140B run is ~100× cheaper, as expected.

**Translating to GPU-hours.** Two well-documented anchors:

1. **Llama 2 7B (Meta, 2023):** 2T tokens, 184,320 A100-80GB GPU-hours ([Llama-2-7b model card](https://huggingface.co/meta-llama/Llama-2-7b)). Linearly scaling to 140B tokens: 184,320 × (140 / 2000) ≈ **12,900 A100 GPU-hours**.
2. **MosaicML MPT-7B (2023):** 7B × 1T tokens in "~9.5 days on 440× A100-40GB" ([Databricks/MosaicML, *MPT-7B blog*](https://www.databricks.com/blog/mpt-7b)). That is ~100,320 A100 GPU-hours for 1T; scaled to 140B: **~14,000 A100 GPU-hours.**

Both estimates agree on **~13–15k A100-hours** for a 7B × 140B run. On an 8× A100-80GB node that is ~70 days; on an 8× H100 node it is meaningfully less.

**H100 conversion.** H100s in well-tuned BF16 LLM training routinely deliver 2–3× the throughput of A100s. SemiAnalysis reports "AI labs are achieving FP8 Model FLOPs Utilization (MFU) as high as 35% and FP16 MFU of 40% on trillion parameter training runs" on H100s ([SemiAnalysis: 100,000 H100 Clusters](https://newsletter.semianalysis.com/p/100000-h100-clusters-power-network)). A reasonable rule of thumb is **5,000–7,000 H100-hours for 7B × 140B** (i.e., ~30 days on an 8× H100 node, or ~3 days on an 8-node cluster of 64 H100s).

For the stretch target (7B × 300B tokens), simply double the above: ~12 × 10²¹ FLOPs, ~28k A100-hours, or ~10–14k H100-hours.

---

## Summary table

| Quantity | Value | Source |
|----------|-------|--------|
| Compute-optimal tokens/param | ~20 | [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556) |
| Chinchilla-optimal tokens for 7B | ~140B | Hoffmann et al. 2022, Table 3 |
| Approach-3 fit (corrected) | E=1.8172, A=482.01, B=2085.43, α=0.348, β=0.366 | [Besiroglu et al. 2024](https://arxiv.org/abs/2404.10102) |
| FLOP approximation | C ≈ 6ND | [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) |
| 7B × 140B FLOPs | ~5.9 × 10²¹ | Derived from 6ND |
| 7B × 140B GPU-hours (A100-80) | ~13–15k | Linear-scale from [Llama-2-7b card](https://huggingface.co/meta-llama/Llama-2-7b), [MPT-7B blog](https://www.databricks.com/blog/mpt-7b) |
| 7B × 140B GPU-hours (H100) | ~5–7k | [SemiAnalysis](https://newsletter.semianalysis.com/p/100000-h100-clusters-power-network) MFU figures |
| Recommended budget for 7B research run | ~150B tokens (Chinchilla-optimal), stretch to 300–500B | This document |

---

## References

- [Hoffmann, J. et al. *Training Compute-Optimal Large Language Models* — DeepMind, 2022 (arXiv:2203.15556)](https://arxiv.org/abs/2203.15556)
- [Kaplan, J. et al. *Scaling Laws for Neural Language Models* — OpenAI, 2020 (arXiv:2001.08361)](https://arxiv.org/abs/2001.08361)
- [Touvron, H. et al. *LLaMA: Open and Efficient Foundation Language Models* — Meta, 2023 (arXiv:2302.13971)](https://arxiv.org/abs/2302.13971)
- [Touvron, H. et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models* — Meta, 2023 (arXiv:2307.09288)](https://arxiv.org/abs/2307.09288)
- [Grattafiori, A. et al. *The Llama 3 Herd of Models* — Meta, 2024 (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783)
- [Sardana, N. et al. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws* — MosaicML/Databricks, 2024 (arXiv:2401.00448)](https://arxiv.org/abs/2401.00448)
- [Besiroglu, T., Erdil, E., Barnett, M., You, J. *Chinchilla Scaling: A Replication Attempt* — Epoch AI, 2024 (arXiv:2404.10102)](https://arxiv.org/abs/2404.10102)
- [Epoch AI blog summary of Besiroglu et al.](https://epoch.ai/blog/chinchilla-scaling-a-replication-attempt)
- [Wikipedia: Neural scaling law](https://en.wikipedia.org/wiki/Neural_scaling_law)
- [Wikipedia: Llama (language model)](https://en.wikipedia.org/wiki/Llama_(language_model))
- [Llama-2-7b model card — Meta / Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b)
- [MosaicML/Databricks: *Introducing MPT-7B*](https://www.databricks.com/blog/mpt-7b)
- [SemiAnalysis: 100,000 H100 Clusters: Power, Network Topology, Ethernet vs InfiniBand, Reliability, Failures, Checkpointing](https://newsletter.semianalysis.com/p/100000-h100-clusters-power-network)
