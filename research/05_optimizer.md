# Optimizer & Optimization Hyperparameters for a Single-Machine 7B Pretraining Run

This note surveys the optimizer landscape for autoregressive LLM pretraining as of April 2026, then commits to a single configuration that is reproducible on one node. Empirical claims are cited inline.

---

## 1. AdamW (Loshchilov & Hutter, ICLR 2019) — the dominant baseline

AdamW decouples weight decay from the gradient-based update of Adam, restoring the property that "L2 regularization and weight decay regularization are equivalent for standard SGD" (a property Adam violates). Loshchilov & Hutter showed that decoupling "substantially improves Adam's generalization performance, allowing it to compete with SGD with momentum on image classification datasets" ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101)). Every modern open-weights LLM pretraining recipe of which I am aware uses AdamW (or a close variant) as either the only optimizer or the optimizer for at least the embedding/output/LayerNorm parameters.

The convention for LLM pretraining departs from Adam's textbook defaults in two important ways:

- **β2 = 0.95, not 0.999**. This was set by GPT-3 ("Adam with β1 = 0.9, β2 = 0.95, ε = 1e-8", Brown et al. 2020, Appendix B of [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)) and inherited by essentially every follow-on. The lower β2 makes the second-moment estimate more responsive to recent gradients, which empirically stabilizes large-batch language model training.
- **Weight decay = 0.1**, applied via decoupled AdamW. This is again the GPT-3 setting and is the de-facto standard.

Concrete configurations from public technical reports:

| Model | β1 | β2 | ε | Weight decay | Grad clip | Peak LR | Schedule | Source |
|------|----|----|---|---|---|---|---|---|
| GPT-3 175B | 0.9 | 0.95 | 1e-8 | 0.1 | 1.0 | 0.6e-4 | cosine, 375M-token warmup, decay to 10% | [Brown et al. 2020](https://arxiv.org/abs/2005.14165) |
| Chinchilla 70B | (Adam→AdamW) | — | — | — | — | 1.0e-4 | cosine cycle length matched to total tokens, decays 10× | [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556) |
| LLaMA-1 7B | 0.9 | 0.95 | — | 0.1 | 1.0 | 3.0e-4 | cosine, 2 000-step warmup, decays to 10% of peak | [Touvron et al. 2023](https://arxiv.org/abs/2302.13971) |
| Llama 3 8B | (AdamW; same conventions) | — | — | (0.1×LR for scaling-law runs) | — | 3e-4 | linear warmup → cosine decay, min LR 8e-7 (405B); 8B uses 3e-4 peak | [Llama 3 paper](https://arxiv.org/abs/2407.21783) |
| DeepSeek-V2 | 0.9 | 0.95 | — | 0.1 | 1.0 | 2.4e-4 | warmup 2 000 steps + multi-step decay (×0.316 at 60% and 90%) | [DeepSeek-V2 report](https://arxiv.org/abs/2405.04434) |
| DeepSeek-V3 | 0.9 | 0.95 | — | 0.1 | 1.0 | 2.2e-4 | warmup 2 000 steps → constant → cosine decay → constant tail | [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) |

The convergence of these recipes is striking: AdamW with (β1, β2) = (0.9, 0.95), weight decay 0.1, and gradient clipping 1.0 has effectively become the default since GPT-3.

## 2. Lion (Chen et al., Google 2023)

Chen et al. used symbolic program search to discover Lion ("EvoLved Sign momentum") in [arXiv:2302.06675](https://arxiv.org/abs/2302.06675). Lion stores only first-moment momentum (no second moment), saving ~½ of optimizer memory vs Adam. The update is the sign of an interpolated momentum term, so the per-step update has roughly unit norm; the authors recommend "a smaller learning rate than Adam due to the larger norm of the update" — typically 3–10× smaller — and 3–10× larger weight decay to keep regularization strength constant ([arXiv:2302.06675](https://arxiv.org/abs/2302.06675)). They report a +1.96% ImageNet ViT-B/16 improvement, 88.3% zero-shot on BASIC-L (+2%), and ~2× faster diffusion convergence.

For LLM pretraining specifically, the story is more cautious. The paper notes that on a 1.6T-token text corpus they observed "no perplexity difference throughout training" between Lion and AdamW, with only modest gains in in-context learning. The authors themselves acknowledge "scenarios where its improvements are small or not statistically significant." Independent users have reported sensitivity to the LR/WD coupling. So Lion is a viable memory-saving option but does not beat AdamW on language modeling loss in the regime that matters for a 7B run.

## 3. Sophia (Liu et al., Stanford 2023)

Sophia ([arXiv:2305.14342](https://arxiv.org/abs/2305.14342)) is "a scalable stochastic second-order optimizer" that maintains a light diagonal Hessian estimate (refreshed every ~10 steps) and applies per-coordinate update clipping. Liu et al. report a "2× speedup compared to Adam in the number of steps, total compute, and wall-clock time" on GPT-2 sized models (125M–770M) and GPT-NeoX (1.5B and 6.6B) and claim the gap widens with scale. Default hyperparameters from the paper and [reference repo](https://github.com/Liuhong99/Sophia): β1 = 0.96, β2 = 0.99, ε = 1e-12, ρ (clipping scale) = 0.05 for Sophia-G, weight decay roughly 2× the AdamW value, and "lr slightly smaller than AdamW".

The 2× claim has not been broadly replicated at the 7B+ scale by independent labs; subsequent benchmark studies (see AlgoPerf below) did not find Sophia to be the winner, and large frontier-lab reports continue to use AdamW or Muon rather than Sophia. For a one-shot reproducible run, the risk of regressing under-tuned vs an AdamW baseline outweighs the expected upside.

## 4. Shampoo / Distributed Shampoo (Anil et al.; Shi et al. 2023)

Shampoo is a Kronecker-factored full-matrix preconditioner ("AdaGrad family ... block-diagonal preconditioner with Kronecker product approximation to full-matrix AdaGrad", [Shi et al. 2023, arXiv:2309.06497](https://arxiv.org/abs/2309.06497)). The PyTorch Distributed Shampoo implementation showed that the heavier per-step cost stays "at most a 10% performance reduction in per-step wall-clock time compared against standard diagonal-scaling-based adaptive gradient methods." That makes it competitive on wall-clock — not just on step count.

The most credible head-to-head evidence comes from the MLCommons AlgoPerf benchmark (see §6): Distributed Shampoo won the External-Tuning track of the inaugural 2024 competition. For a single-machine run the question is whether the implementation complexity (block-diagonal preconditioner statistics, eigendecomposition cadence, distribution of preconditioner state across devices) is worth the speedup. Without a mature single-node Shampoo recipe that matches the benchmark conditions, AdamW remains lower-risk.

## 5. Muon (Jordan et al., 2024) and MuonClip (Kimi K2)

Muon is the most interesting recent addition. Keller Jordan's [writeup](https://kellerjordan.github.io/posts/muon/) (Dec 2024) and the [reference implementation](https://github.com/KellerJordan/Muon) describe the algorithm as: take an SGD-Nesterov-momentum step, then orthogonalize the resulting matrix update via a 5-step quintic Newton–Schulz iteration with coefficients (3.4445, −4.7750, 2.0315), tuned to be stable in bf16. This is applied **only to 2-D hidden weight matrices**; embeddings, the final classifier head, biases, RMSNorm gains, and Q/K/V projections (when not separated) should still be optimized with AdamW.

Reported results:

- **CIFAR-10 to 94%**: 3.3 → 2.6 A100-seconds (Jordan).
- **NanoGPT speedrunning**: Muon beat the prior AdamW record by ~1.35× and has held the top of the leaderboard across "12 subsequent benchmarks by multiple researchers since October 2024" ([Jordan 2024](https://kellerjordan.github.io/posts/muon/)).
- **GPT-2 XL (1.5B)**: matched in 10 vs 13.3 8×H100 hours (~25% wall-clock improvement) over AdamW.

**Scaling Muon up.** The Moonshot team's "Moonlight" report ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982)) identified two changes that make Muon work out-of-the-box at scale: (1) add AdamW-style decoupled weight decay (Muon as published has none), and (2) rescale per-parameter updates by √max(A,B) so update RMS matches AdamW's typical 0.2–0.4 range. With these changes, training a 16B-total / 3B-active MoE on 5.7T tokens, they report "Muon achieves ~2× computational efficiency compared to AdamW with compute optimal training," and Moonlight "advances the Pareto frontier" relative to comparable open models. Recommended Muon hyperparameters in that work: lr 9.5e-4 → 8.3e-4 across model sizes, momentum 0.95, weight decay 0.1, 5 NS iterations. Kimi K2 (1T-total MoE, 32B-active) further introduces **MuonClip**, "improves upon Muon with a novel QK-clip technique to address training instability while enjoying the advanced token efficiency of Muon," and reports zero loss spikes over 15.5T tokens ([Kimi K2 report](https://arxiv.org/abs/2507.20534)).

For dense models, public-evidence is currently strongest in the ≤1.5B regime; the Moonlight and Kimi K2 results are MoE. Muon at dense 7B is still under-validated outside individual lab reports, although community speedrun records continue to fall.

## 6. AlgoPerf benchmark (MLCommons, 2024–2025)

AlgoPerf is the MLCommons benchmark for "neural network training algorithms" — it scores submissions on time-to-target across 8 fixed workloads on 8×V100 hardware. The inaugural 2024 competition had 18 submissions from 10 teams. Per the [ICLR 2025 results paper](https://openreview.net/forum?id=CtM5xjRSfm):

- **External-Tuning ruleset winner: Distributed Shampoo**, "demonstrates the effectiveness of non-diagonal preconditioning over popular methods like Adam."
- **Self-Tuning ruleset winner: Schedule-Free AdamW**, "demonstrates a new level of effectiveness for completely hyperparameter-free training algorithms."

The current public [v0.6 leaderboard](https://github.com/mlcommons/submissions_algorithms) (Mar 24, 2025) confirms: Distributed Shampoo at 0.6244 and the NadamW baseline at 0.4590 — a substantial gap. The earlier AlgoPerf benchmarking paper ([arXiv:2306.07179](https://arxiv.org/abs/2306.07179)) also found that "baselines using adaptive methods (AdamW and NadamW) score more highly than baselines using non-adaptive methods" with NadamW the strongest among the simple baselines.

Caveats: AlgoPerf workloads are diverse (vision, speech, MT, LM), not just LLM pretraining; the "winner" reflects average performance across all eight tasks. Still, the headline is clear: in 2024–2025 controlled comparisons, non-diagonal preconditioners (Shampoo) and modern variants (Schedule-Free, NAdamW) have the edge over vanilla AdamW. AlgoPerf has so far featured Shampoo and NAdamW more prominently than Muon.

## 7. Schedule

**Linear warmup → cosine decay to 10% of peak** is the textbook recipe and was used by GPT-3, Chinchilla, LLaMA-1, and Llama 3. LLaMA-1 used 2 000 warmup steps and decayed to 10% of peak ([Touvron et al. 2023](https://arxiv.org/abs/2302.13971)); Chinchilla showed that "setting the cosine cycle length to approximately match the number of training tokens results in the best final loss regardless of model size" ([Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)) — i.e., do not decay to 10% earlier or later than your token budget.

**Warmup-Stable-Decay (WSD)** is the alternative gaining traction. Introduced by MiniCPM ([arXiv:2404.06395](https://arxiv.org/abs/2404.06395)), it has three phases: linear warmup to η, a long constant-LR "stable" phase, and a short final "decay" phase (often the last ~10% of tokens). Formally: WSD(T;s) = {s/W·η, s<W; η, W≤s<T; f(s−T)·η, s≥T}. The MiniCPM authors showed that "10% of total tokens" in the decay phase suffices for full convergence. WSD has two practical advantages over cosine: (a) you can branch from a stable-phase checkpoint and run multiple decay tails (different total token budgets, different data mixes), turning data-mix experiments into cheap fine-tunes rather than separate pretraining runs; (b) it supports continued training — appending more tokens does not require re-doing the schedule. DeepSeek-V2 uses a "warmup-and-step-decay" variant (warmup 2 000 → constant 2.4e-4 → ×0.316 at ~60% → ×0.316 at ~90%; [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)); DeepSeek-V3 uses a hybrid (warmup → constant → cosine decay → short constant tail; [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)).

For a one-shot reproducible 7B run with a fixed token budget, cosine remains lower-risk because every public 7B comparator was trained with cosine. WSD is the right choice if mid-training data-mix changes or extending the run later are likely.

## 8. Peak learning rate at 7–8B

Two complementary anchors:

- **μP / muTransfer** ([Yang et al. 2022, arXiv:2203.03466](https://arxiv.org/abs/2203.03466)) shows that under the Maximal Update Parametrization "many optimal HPs remain stable even as model size changes," allowing zero-shot transfer of LR (and others) from a small proxy (e.g., 13M for BERT-large; 40M for GPT-3 6.7B). The principled recipe: parametrize in μP, sweep LR on a small model, transfer.
- **Empirical LR at this scale**, from public reports: LLaMA-1 7B used **3.0e-4** ([Touvron et al. 2023](https://arxiv.org/abs/2302.13971)); Llama 3's 8B used **3e-4** (Llama 3 paper Table 3, [arXiv:2407.21783](https://arxiv.org/abs/2407.21783); the 70B uses 1.5e-4 and the 405B uses 8e-5); DeepSeek-V2 used 2.4e-4 ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)) and DeepSeek-V3 used 2.2e-4 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)). At 7–8B with AdamW, **3e-4 is the consensus peak**.

If using Muon for hidden 2-D weights, the appropriate range from public reports is much higher — Jordan suggests 0.02 with AdamW at 3e-4 for the auxiliary parameters, and Moonlight tuned to ~0.95e-3 to 8.3e-4 for hidden matrices at 3B-active MoE scale ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982)). For a dense 7B Muon run, an LR sweep on a smaller proxy (μP-style or just empirical) is mandatory.

## 9. Recommendation for a single-machine reproducible 7B run

**Use AdamW.** It is the only optimizer with strong, multi-lab, multi-billion-parameter evidence on dense LLM pretraining, the hyperparameters are converged, and under-performing reproductions are extremely rare with this recipe.

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimizer | AdamW (decoupled weight decay) | [Loshchilov & Hutter 2019](https://arxiv.org/abs/1711.05101); ubiquitous for LLM pretraining |
| β1 | 0.9 | GPT-3, LLaMA-1/3, DeepSeek-V2/V3 |
| β2 | 0.95 | GPT-3 ([Brown et al. 2020](https://arxiv.org/abs/2005.14165)); standard since |
| ε | 1e-8 | GPT-3 |
| Weight decay | 0.1 (decoupled) | GPT-3, LLaMA, DeepSeek |
| Gradient clip | 1.0 (global L2 norm) | LLaMA, DeepSeek |
| Peak LR | 3e-4 | LLaMA-1 7B and Llama 3 8B |
| Min LR | 3e-5 (10% of peak) | LLaMA-1; cosine to 10% of peak |
| Warmup | 2 000 steps, linear from 0 | LLaMA-1 |
| Schedule | cosine decay to 10% of peak, cycle length = total training tokens | [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556) (Chinchilla) |
| Precision | bf16 mixed; fp32 master optimizer state | standard practice |

**If single-machine memory is the binding constraint** (a 7B AdamW state is ~84 GB in fp32 master copy + bf16 params + grads), consider Lion as a memory-saving fallback at 3–10× smaller LR and 3–10× larger WD ([arXiv:2302.06675](https://arxiv.org/abs/2302.06675)) — with the caveat that Lion has not shown gains over AdamW on language modeling at this scale.

**If you want to bet on speed and accept calibration risk**, use Muon for hidden 2-D weights with AdamW for embeddings, output head, and LayerNorm gains. Plan a small μP-style or empirical LR sweep on a 100M-1B proxy, add Moonlight's two fixes (decoupled WD, update-RMS rescaling by √max(A,B)·0.2), and budget for a fallback to AdamW. Expected upside is ~1.5–2× wall-clock at the cost of a more complex configuration and weaker public reproductions at dense 7B as of April 2026.

**Risk vs reward summary.** AdamW with the table above is the safe, reproducible choice that every public 7B comparator uses. Muon (and MuonClip) is genuinely promising — it has held NanoGPT speedrun records since October 2024 and produced the strongest open MoEs (Moonlight, Kimi K2) — but for a one-shot dense 7B reproduction the marginal expected speedup does not justify the calibration cost. Sophia and Lion underperform AdamW on language-modeling loss at this scale in published evidence. Distributed Shampoo wins AlgoPerf but lacks a turnkey single-node 7B-scale recipe. Recommend AdamW with the configuration above; revisit Muon once dense 7B–13B Muon reproductions are published independently.

---

## Sources

- Loshchilov & Hutter, "Decoupled Weight Decay Regularization," ICLR 2019 — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3) — [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla) — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" — [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- Llama 3 — [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- DeepSeek-V2 — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-V3 — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- Chen et al., "Symbolic Discovery of Optimization Algorithms" (Lion) — [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)
- Liu et al., "Sophia: A Scalable Stochastic Second-order Optimizer" — [arXiv:2305.14342](https://arxiv.org/abs/2305.14342) and [reference repo](https://github.com/Liuhong99/Sophia)
- Shi et al., "A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer" — [arXiv:2309.06497](https://arxiv.org/abs/2309.06497)
- Dahl et al., "Benchmarking Neural Network Training Algorithms" (AlgoPerf) — [arXiv:2306.07179](https://arxiv.org/abs/2306.07179)
- AlgoPerf 2024 ICLR results paper — [OpenReview CtM5xjRSfm](https://openreview.net/forum?id=CtM5xjRSfm); [submissions leaderboard](https://github.com/mlcommons/submissions_algorithms); [MLCommons benchmarks](https://mlcommons.org/benchmarks/algorithms/)
- Jordan, "Muon: An optimizer for hidden layers in neural networks," Dec 2024 — [blog](https://kellerjordan.github.io/posts/muon/); [implementation](https://github.com/KellerJordan/Muon)
- Liu et al., "Moonlight" / Muon at scale — [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
- Kimi K2 / MuonClip — [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
- Hu et al., "MiniCPM" (WSD schedule) — [arXiv:2404.06395](https://arxiv.org/abs/2404.06395)
- Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (μP / muTransfer) — [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)

---

## 10. 2025–2026 optimizer landscape (follow-up)

This section updates §§3–6 with public results published after Moonlight
(Feb 2025), with deliberate focus on whether anything has changed the
"AdamW for a one-shot dense 7B run" recommendation.

### 10.1 Muon at 7B+ dense scale: still no public single-lab dense reproduction

The pattern after the Feb 2025 Moonlight report ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982))
is that **every public Muon-trained model in the >10B class is MoE**, not
dense. Specifically:

- **Kimi K2** (Moonshot, July 2025, [arXiv:2507.20534](https://arxiv.org/abs/2507.20534))
  — 1T-total / 32B-active MoE, MuonClip variant.
- **GLM-4.5 / GLM-4.5-Air** (Z.ai / Zhipu AI, July 2025) — 355B-total /
  32B-active and 106B-total / 12B-active **MoE**. Public DeepLearning.AI
  writeup confirms "they used the more efficient Muon optimizer"
  ([Zhipu AI Releases GLM-4.5 — DeepLearning.AI The Batch, 2025](https://www.deeplearning.ai/the-batch/zhipu-ai-z-ai-releases-open-weights-glm-4-5-models-that-perform-comparably-to-the-latest-from-claude-and-deepseek/)).
- **INTELLECT-3** (Prime Intellect, Dec 2025, [arXiv:2512.16144](https://arxiv.org/pdf/2512.16144);
  [PrimeIntellect/INTELLECT-3 — HuggingFace](https://huggingface.co/PrimeIntellect/INTELLECT-3))
  — 106B-total / 12B-active **MoE**, derived from GLM-4.5-Air-Base; the
  Muon usage is in the post-training stage and inherited from a Muon-
  pretrained base. Not a from-scratch dense Muon run.

Keller Jordan's own blog (Dec 2024 update, still authoritative as of
April 2026) lists the largest dense Muon result as a **1.5B GPT-2 XL**
trained "to GPT-2 XL level performance on HellaSwag in 10 8×H100-hours,"
and explicitly leaves "Will Muon scale to larger trainings? (e.g., 20B+
parameters for 1T+ tokens)" as an **open question**
([Muon: An optimizer for hidden layers in neural networks — Jordan, 2024](https://kellerjordan.github.io/posts/muon/)).

The strongest controlled comparison published since is the Stanford /
Marin paper **"Fantastic Pretraining Optimizers and Where to Find Them"**
(Wen et al., Sep 2025, [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)).
Quoting the abstract: "the actual speedup of many proposed optimizers
over well-tuned baselines is lower than claimed and decreases with
model size to only **1.1× for 1.2B parameter models**." Their study
covers ten optimizers (incl. Muon, SOAP, Schedule-Free AdamW, Sophia,
Lion, AdEMAMix, ADOPT, Signum, Prodigy) at 0.1B–1.2B over 1–8×
Chinchilla data, with equal-budget hyperparameter tuning per optimizer.
The matrix-preconditioner family (Muon, SOAP) is fastest, but the
margin shrinks from ~1.4× at 0.1B to ~1.1× at 1.2B — i.e., even at the
largest scale they study, **the AdamW gap is closing, not widening, as
model size increases**. They do not run 7B.

A more bullish counter-data-point is Essential AI's **"Practical
Efficiency of Muon for Pretraining"** (Apr 2025, [arXiv:2505.02222](https://arxiv.org/pdf/2505.02222));
the PDF rendered only partially in our fetch, but searches across
multiple summaries indicate it argues for Muon over AdamW on
**data-efficiency at large batch sizes** rather than wall-clock speedup,
and again does not include a from-scratch dense 7B+ run.

A separate **"Benchmarking Optimizers for Large Language Model
Pretraining"** (Sep 2025, [arXiv:2509.01440](https://arxiv.org/html/2509.01440v1))
covers ADOPT, Signum, Prodigy, SF-AdamW (Schedule-Free AdamW), Muon,
Sophia, AdEMAMix, and SOAP, and reports SOAP as the strongest, with
ADOPT and AdEMAMix also outperforming AdamW. Again not at 7B dense
scale.

**Bottom line.** As of April 2026 there is **still no public dense 7B
reproduction of Muon outside the original Moonshot/MoE line**. The
controlled small-scale benchmarks that do exist (Wen et al. 2509.02046;
Pretraining Optimizers Benchmark 2509.01440) suggest Muon's headline
~2× advantage shrinks toward ~1.1–1.2× at the largest scales they
test. The §9 recommendation — AdamW for the reference run, Muon as a
deliberate-bet alternative — does not need to change.

### 10.2 Schedule-Free AdamW: AlgoPerf 2024 winner, still no public 7B+ run

Schedule-Free AdamW (Defazio et al., "The Road Less Scheduled", May
2024, [arXiv:2405.15682](https://arxiv.org/abs/2405.15682);
[facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free))
**won the self-tuning track of AlgoPerf 2024** with the only submission
to beat the prize-qualification baseline by ~8% ("Announcing the
results of the inaugural AlgoPerf benchmark competition", MLCommons,
Aug 2024 — [link](https://mlcommons.org/2024/08/mlc-algoperf-benchmark-competition/)).
The original paper's largest LM experiment is **a 150M decoder-only
transformer trained on 15B tokens at batch size 32k** with results
matching cosine-LR; subsequent follow-ups including "Through the
River" (Song et al., July 2025, [arXiv:2507.09846](https://arxiv.org/abs/2507.09846))
study Schedule-Free's behaviour relative to the loss-landscape "river"
phenomenon but again do not present a from-scratch dense 7B+ run.

**No public 7B-class Schedule-Free AdamW pretraining run has been
reported as of April 2026.** It is a credible drop-in replacement for
the cosine schedule under §7 if one wants to remove the LR schedule
choice from the reproduction surface, but it has the same "no public
7B comparator" risk as Muon.

### 10.3 Distributed Shampoo: SOAP, SPlus, and the Shampoo family in 2025

The most credible 2025–2026 development in the Shampoo family is **SOAP**
("Improving and Stabilizing Shampoo using Adam", Vyas et al., ICLR 2025,
[arXiv:2409.11321](https://arxiv.org/abs/2409.11321)). SOAP runs
Adafactor in the eigenbasis of Shampoo's preconditioner; the paper
reports **"approximately 40% reduction in iterations and 35% reduction
in wall-clock time vs AdamW, ~20% reduction vs Shampoo"** in their
LM-pretraining sweep. The largest model in the SOAP paper is
**360M** parameters; no 7B reproduction has been published.

A subsequent algorithm, **SPlus** (2025), tightens the Shampoo recipe
with bounded sign-based normalization, shape-aware LR scaling, and
EMA iterate-averaging. Per [Shampoo Algorithms summary —
EmergentMind, 2025](https://www.emergentmind.com/topics/shampoo-family-of-algorithms),
"SPlus achieves superior stability with infrequent eigenbasis updates
and enables practical deployment on large Transformer training
benchmarks, consistently reaching Adam-level performance in fewer
steps and less wall-clock time." Again: no 7B-scale public run as of
April 2026.

The Meta-maintained reference implementation
([facebookresearch/optimizers / distributed_shampoo](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md))
remains the production-grade single-node-distributable codebase. It is
the same implementation as the AlgoPerf 2024 winning entry. **No
turnkey single-node 7B Shampoo recipe has been published**; the
benchmark workloads are smaller (≤350M LM, ResNet, etc.).

### 10.4 AlgoPerf v0.6 leaderboard (as of March 24, 2025) — no movement since

The MLCommons rolling leaderboard
([github.com/mlcommons/submissions_algorithms](https://github.com/mlcommons/submissions_algorithms))
has not been updated since the **2025-03-24 v0.6 snapshot** referenced
in §6. Current state confirmed by direct fetch (April 2026):

| Track | Leader | Score | Affiliation |
|---|---|---|---|
| External-Tuning | **Distributed Shampoo** (PyTorch) | **0.6244** | Meta Platforms (Shi et al.) |
| External-Tuning | NadamW baseline (JAX) | 0.4590 | baseline |
| Self-Tuning | (no completed entries scored at v0.6) | — | — |

**The 2024 winners — Distributed Shampoo (external) and Schedule-Free
AdamW (self) — remain the leaders.** Neither Muon nor SOAP/SPlus
appears on the current leaderboard as of April 2026. The competition
has been re-organised as a rolling leaderboard rather than a yearly
edition, and no new prize round has been announced for 2025 or 2026.

### 10.5 Net effect on the §9 recommendation

None of the 2025–2026 evidence overturns the §9 conclusion:

- **No public dense 7B Muon run** outside Moonshot's MoE line. The
  largest dense Muon result remains GPT-2 XL 1.5B (Jordan 2024).
  Wen et al.'s 1.2B benchmark suggests the speedup shrinks with scale.
- **No public 7B Schedule-Free AdamW run.** Algorithm is robust
  (AlgoPerf self-tuning winner), but its scaling story stops at
  ≤150M LM in the published literature.
- **No turnkey 7B Distributed Shampoo / SOAP / SPlus recipe.** Meta's
  reference implementation is solid but tuned for the AlgoPerf
  workloads; LM coverage caps at ~350M.
- **AlgoPerf leaderboard has not moved since March 2025.** Same
  winners; no new submissions evaluated.

The recipe in §9 (AdamW, β2=0.95, WD=0.1, peak LR 3e-4, cosine to 10%,
2 000-step warmup) remains the lowest-risk choice for a one-shot
single-machine reproducible 7B run. Muon-for-hidden-2D / AdamW-for-the-
rest with Moonlight's two scaling fixes is the only credible
alternative, and it now has the additional cover of Z.ai / Zhipu's
GLM-4.5 line and Prime Intellect's INTELLECT-3 (both MoE), but **the
dense 7B reproduction gap has not closed**.

Sources (new as of this follow-up):

- [Muon is Scalable for LLM Training — Liu et al. (Moonshot), Feb 2025](https://arxiv.org/abs/2502.16982)
- [Kimi K2 — Moonshot AI, July 2025](https://arxiv.org/abs/2507.20534)
- [Zhipu AI Releases GLM-4.5 — DeepLearning.AI The Batch, 2025](https://www.deeplearning.ai/the-batch/zhipu-ai-z-ai-releases-open-weights-glm-4-5-models-that-perform-comparably-to-the-latest-from-claude-and-deepseek/)
- [INTELLECT-3 Technical Report — Prime Intellect, Dec 2025](https://arxiv.org/pdf/2512.16144); [PrimeIntellect/INTELLECT-3 on HuggingFace](https://huggingface.co/PrimeIntellect/INTELLECT-3)
- [Fantastic Pretraining Optimizers and Where to Find Them — Wen et al., Sep 2025](https://arxiv.org/abs/2509.02046)
- [Practical Efficiency of Muon for Pretraining — Essential AI, Apr 2025](https://arxiv.org/pdf/2505.02222)
- [Benchmarking Optimizers for LLM Pretraining — Sep 2025](https://arxiv.org/html/2509.01440v1)
- [The Road Less Scheduled — Defazio et al., May 2024](https://arxiv.org/abs/2405.15682); [facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free)
- [Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training — Song et al., July 2025](https://arxiv.org/abs/2507.09846)
- [SOAP: Improving and Stabilizing Shampoo using Adam — Vyas et al., ICLR 2025](https://arxiv.org/abs/2409.11321)
- [Distributed Shampoo reference implementation — Meta, facebookresearch/optimizers](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md)
- [AlgoPerf v0.6 leaderboard — mlcommons/submissions_algorithms (snapshot 2025-03-24)](https://github.com/mlcommons/submissions_algorithms)
- [Muon: An optimizer for hidden layers in neural networks — Keller Jordan blog, Dec 2024](https://kellerjordan.github.io/posts/muon/)
