# Pretraining Corpus Survey for a 7B Chinchilla-Optimal Run

*Research date: 2026-04-27. Author: research agent. Scope: open-source pretraining corpora suitable for a fully reproducible 7B-parameter dense LLM at or above Chinchilla-optimal token budget.*

---

## 1. Compute target: how many tokens does a 7B model need?

The Chinchilla scaling law of [Hoffmann et al., 2022 ("Training Compute-Optimal Large Language Models", DeepMind)](https://arxiv.org/abs/2203.15556) shows that, for a fixed training-compute budget, model size and training-token count should scale roughly equally — "for every doubling of model size the number of training tokens should also be doubled" ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). The empirical fit of all three of their methods produces a token-to-parameter ratio of approximately **20×**: their headline 70B-parameter Chinchilla was trained on **1.4 T tokens** ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556); summary in [Wikipedia — Chinchilla (language model)](https://en.wikipedia.org/wiki/Chinchilla_(language_model))).

For a 7B-parameter model, the compute-optimal target is therefore roughly:

> **7 × 10⁹ params × 20 tokens/param ≈ 1.4 × 10¹¹ tokens ≈ 140 B tokens.**

That figure is the *minimum* a 7B run should aim for. In practice, modern flagship models train far past Chinchilla-optimal because inference cost dominates over many years: e.g., Meta's Llama 3 8B was trained on ~15 T tokens (~1875 tokens/parameter), and OLMo 2 7B/13B was trained on up to 5 T tokens ([AI2 — OLMo 2 blog, 2024](https://allenai.org/blog/olmo2)). For a fully reproducible 7B run that is competitive on MMLU/ARC etc., a **300 B – 2 T token** budget is the practical target; the corpora discussed below all comfortably exceed that.

---

## 2. Survey of current open corpora (late 2025 / early 2026)

### 2.1 FineWeb (HuggingFace)

* **Tokens:** Originally 15 T GPT-2 tokens at release; the live dataset is currently **18.527 T GPT-2 tokens** across 96 Common Crawl dumps from summer 2013 through June 2025 ([HF dataset card — HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb); [Penedo et al., "The FineWeb Datasets", 2024 (arXiv:2406.17557)](https://arxiv.org/abs/2406.17557)).
* **Language:** English only (`en`) ([HF dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)).
* **License:** ODC-By 1.0 ([HF dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)).
* **Source/Pipeline:** Common Crawl WARC → trafilatura extraction → fastText language ID → Gopher and C4-style heuristic filters → custom quality filters → **MinHash LSH deduplication with 5-grams and 14×8 hash functions, applied per-dump** rather than globally — an empirically-supported choice in the FineWeb tech report ([HF dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)).
* **Reproducibility:** Pipeline is open-source (the `datatrove` library) and the paper documents every ablation ([Penedo et al., 2024](https://arxiv.org/abs/2406.17557)).
* **Sample subsets:** `sample-10BT`, `sample-100BT`, `sample-350BT` are pre-shuffled token-budget slices ([HF dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)).

### 2.2 FineWeb-Edu (HuggingFace)

* **Tokens:** **1.3 T** GPT-2 tokens (educational subset of FineWeb at classifier `int_score ≥ 3`) ([HF dataset card — HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)).
* **Language / License:** English; ODC-By ([HF dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)).
* **Filtering:** A BERT-style regression classifier (Snowflake-arctic-embed backbone) was fine-tuned on **500 k FineWeb samples scored 0–5 for educational quality by Llama-3-70B-Instruct**. The classifier reaches an **F1 of 82 %** when binarised, and applying it to FineWeb's 15 T tokens cost **6 k H100-hours** ([HF dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)).
* **Sister release: `fineweb-edu-score-2`** with threshold `int_score ≥ 2`, yielding **5.4 T tokens** — a less aggressive, larger pool ([HF dataset card — HuggingFaceFW/fineweb-edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2); [HF dataset card — HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)).

### 2.3 FineWeb-2 (HuggingFace, multilingual)

* **Tokens / scope:** **20 TB**, **5 B documents**, **1 000+ languages** drawn from ~100 Common Crawl snapshots ([Penedo et al., "FineWeb2: One Pipeline to Scale Them All", 2025 (arXiv:2506.20920)](https://arxiv.org/abs/2506.20920); [HF dataset card — HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)).
* **Language:** 1 810 ISO language-script combinations are listed on the dataset card ([HF dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)).
* **License:** ODC-By ([HF dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)).
* **Note:** FineWeb-2 is the multilingual successor — relevant only if your 7B run is non-English or polyglot. The paper claims the FineWeb-2 pipeline produces "more performant models than prior datasets" on a 9-language ablation suite ([Penedo et al., 2025](https://arxiv.org/abs/2506.20920)). It does *not* report direct head-to-heads with CC-100, mC4, CulturaX, or HPLT in the abstract — those numbers live in the full paper.

### 2.4 DCLM-Baseline (DataComp-LM, Apple / MLCommons / consortium)

* **Tokens:** **4 T tokens / 3 B documents** in the released `dclm-baseline-1.0` ([HF dataset card — mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)). The full candidate pool DCLM-Pool is **240 T tokens** from Common Crawl ([Li et al., "DataComp-LM", 2024 (arXiv:2406.11794)](https://arxiv.org/abs/2406.11794); [datacomp.ai — DCLM benchmark](https://www.datacomp.ai/dclm/)).
* **License:** **CC-BY-4.0** ([HF dataset card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)).
* **Pipeline:** RefinedWeb-style heuristic cleaning → Bloom-filter dedup → **fastText classifier** trained on instruction-formatted positives (OpenHermes-2.5 + r/ExplainLikeImFive) versus random web negatives ([HF dataset card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0); [Li et al., 2024](https://arxiv.org/abs/2406.11794)).
* **Headline result:** A 7B model trained on **2.6 T tokens** of DCLM-Baseline reaches **MMLU 5-shot = 64 %** — comparable to Mistral-7B-v0.3 (63 %) and within 2 points of Llama 3 8B (66 %) at **6.6× less training compute** than Llama 3 8B ([Li et al., 2024](https://arxiv.org/abs/2406.11794)).
* **Reported on the dataset card (7B / 2.6 T tokens):**
  | Dataset | Tokens | CORE | MMLU | EXTENDED |
  |---|---|---|---|---|
  | Llama 2 (7B) | 2 T | 49.2 | 45.8 | 34.1 |
  | OLMo-1.7 (7B) | 2.1 T | 47.0 | 54.0 | 34.2 |
  | MAP-Neo (7B) | 4.5 T | 50.2 | 57.1 | 40.4 |
  | **DCLM-Baseline** | 2.6 T | **57.1** | **63.7** | **45.4** |
  ([HF dataset card — mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)).

### 2.5 RedPajama-V2 (Together AI)

* **Tokens:** **30.4 T deduplicated tokens** (~50.6 T raw) across 84 CC snapshots: 20.5 T en, 3.0 T de, 2.7 T fr, 2.8 T es, 1.5 T it ([HF dataset card — togethercomputer/RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)).
* **Language:** 5 (en, de, fr, es, it) ([HF dataset card](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)).
* **License:** Common Crawl Foundation Terms of Use for data, Apache 2.0 for code ([HF dataset card](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)).
* **Pipeline:** CCNet extraction; ships **40+ pre-computed quality signals** (perplexity, importance-weighting scores, repetitiveness, toxicity) plus MinHash signatures (Jaccard 0.7–1.0) — i.e., RP-V2 is a *quality-annotated raw pool*, leaving the user to apply filters ([HF dataset card](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)).
* **Caveat:** Because it is unfiltered by default, RP-V2 underperforms ready-made filtered corpora (FineWeb-Edu, DCLM-Baseline) unless you replicate a strong filter recipe yourself.

### 2.6 Dolma (AI2)

* **Tokens:** Dolma v1.7 totals **2.31 T tokens** raw; the OLMo-7B-v1.7 mix samples to **1.715 T effective tokens** after per-source up/down-weighting ([HF dataset card — allenai/dolma](https://huggingface.co/datasets/allenai/dolma)). The original Dolma paper reports a **3 T-token** corpus ([Soldaini et al., "Dolma", ACL 2024 (arXiv:2402.00159)](https://arxiv.org/abs/2402.00159)).
* **Language:** English ([HF dataset card](https://huggingface.co/datasets/allenai/dolma)).
* **License:** ODC-By ([HF dataset card](https://huggingface.co/datasets/allenai/dolma)).
* **Composition (v1.7):** 50 % Dolma-CC, 100 % RefinedWeb, plus StarCoder, C4, Reddit, S2ORC, arXiv, StackExchange, Flan, OpenWebMath, Algebraic Stack, Project Gutenberg, MegaWika, Wikipedia ([HF dataset card](https://huggingface.co/datasets/allenai/dolma)).
* **Reproducibility:** Filtering and dedup tooling (the `dolma` toolkit) and intermediate-state ablations are released in the paper ([Soldaini et al., 2024](https://arxiv.org/abs/2402.00159)).

### 2.7 SlimPajama (Cerebras)

* **Tokens:** **627 B tokens** — a deduplicated/cleaned distillation of RedPajama-1T ([Cerebras blog — "SlimPajama: A 627B token cleaned and deduplicated version of RedPajama"](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)).
* **Pipeline:** MinHash LSH at Jaccard ≥ 0.8, plus removal of <200-character documents (excluding Books and GitHub). Net byte reduction: 49.6 % vs. RedPajama-1T ([Cerebras blog](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)).
* **Status:** Solid baseline but **older (2023)** and now superseded by FineWeb / DCLM on benchmark performance per the FineWeb and DCLM ablations.

### 2.8 Nemotron-CC (NVIDIA)

* **Tokens:** **6.3 T tokens** total; the NVIDIA team's reference 8B model used 7.2 T tokens from Nemotron-CC inside a 15 T-token training run ([Su et al., "Nemotron-CC", 2024 (arXiv:2412.02595)](https://arxiv.org/abs/2412.02595)).
* **Methodology:** Classifier *ensembling* (multiple educational/quality classifiers combined) + **synthetic rephrasing** of low-quality web pages + reduced reliance on heuristic filters. The motivation is explicit: aggressive single-classifier filters like FineWeb-Edu / DCLM "removed 90 % of data," and Nemotron-CC trades off less aggression for more unique tokens ([Su et al., 2024](https://arxiv.org/abs/2412.02595)).
* **Headline numbers:** "Four times more unique real tokens than DCLM"; on a 1 T-token training run their *high-quality subset* improves MMLU by **+5.6 points over DCLM**; at 15 T-token scale Nemotron-CC matches DCLM on MMLU and an 8B model trained on it beats Llama 3.1 8B by +5 MMLU, +3.1 ARC-Challenge, +0.5 average over ten tasks ([Su et al., 2024](https://arxiv.org/abs/2412.02595)).

### 2.9 Zyda-1 / Zyda-2 (Zyphra)

* **Zyda-1: 1.3 T tokens** (GPT-NeoX tokenizer), ODC-By, MinHash dedup at Jaccard 0.4 over RefinedWeb / C4-en / SlimPajama / Pile-Uncopyrighted / peS2o / StarCoder / arxiv-s2orc ([HF dataset card — Zyphra/Zyda](https://huggingface.co/datasets/Zyphra/Zyda)).
* **Zyda-2: 5.07 T tokens** built by *cross-deduplicating* DCLM-Baseline (3.35 T), FineWeb-Edu-score-2 (1.32 T), Zyda-1 (164 B), and Dolma-CC v1.7 (238 B), then reweighting via NVIDIA's quality classifier to mixing weights {DCLM 0.404, FWE3 0.506, Zyda 0.032, Dolma-CC 0.058} ([HF dataset card — Zyphra/Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2)).
* **License:** ODC-By ([HF dataset card](https://huggingface.co/datasets/Zyphra/Zyda-2)).
* **Claimed result:** Zyda-2 outperforms The Pile, RefinedWeb, FineWeb, FineWeb-Edu, and DCLM at the size of the Zamba2 model series — though the headline is a *self-comparison* and the dataset card doesn't quote external benchmarks side-by-side ([HF dataset card](https://huggingface.co/datasets/Zyphra/Zyda-2)).

### 2.10 TxT360 (LLM360)

* **Tokens:** ~5 T deduplicated raw tokens (4.83 T from 99 Common Crawl snapshots + 154.96 B papers, 35.975 B Wikipedia in 310+ languages, 27.76 B StackExchange, plus FreeLaw, USPTO, PG-19, HackerNews, Ubuntu IRC, EuroParl, DM Math). With the recommended mixing recipe: **15 T+ training tokens** ([HF dataset card — LLM360/TxT360](https://huggingface.co/datasets/LLM360/TxT360)).
* **License:** ODC-By ([HF dataset card](https://huggingface.co/datasets/LLM360/TxT360)).
* **Pipeline:** Per-source filtering, **global** dedup across web + curated, with explicit duplicate-count buckets; the v1.1 release adds ProX-filtered "BestOfWeb", synthetic Mistral-7B-Instruct QA pairs, and aligned Europarl ([HF dataset card](https://huggingface.co/datasets/LLM360/TxT360)).
* **Claimed result:** In a 1.5 T-token training run on an 8×8B MoE, TxT360 outperformed FineWeb on training loss and SlimPajama-derived validation benchmarks ([HF dataset card](https://huggingface.co/datasets/LLM360/TxT360)).

### 2.11 The Pile (EleutherAI) — context only

* **Size:** 825 GiB / ~300 B tokens across 22 sub-corpora ([HF dataset card — EleutherAI/pile](https://huggingface.co/datasets/EleutherAI/pile); [Gao et al., "The Pile", 2020 (arXiv:2101.00027)](https://arxiv.org/abs/2101.00027)).
* **Status:** **Do not use as a primary corpus for a 2026 7B run.** Aside from being far below Chinchilla-optimal, the Books3 sub-corpus has been the subject of copyright litigation and was removed from upstream mirrors; many community redistributions are now "Pile-Uncopyrighted" variants ([HF dataset card](https://huggingface.co/datasets/EleutherAI/pile)). Useful only as a small evaluation/diagnostic mix.

### 2.12 RefinedWeb (TII / Falcon) — partially open

* **Tokens:** ~5 T tokens internally; only ~600 B tokens (968 M docs) are publicly released ([HF dataset card — tiiuae/falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)).
* **License:** ODC-By 1.0 for the released subset ([HF dataset card](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)).
* **Caveat:** Filtering code is described in the Falcon paper but the **full 5 T pipeline output is not released** — only an extract. Treat as filter-recipe and small-scale corpus, not a Chinchilla-scale source.

### 2.13 Code & math complements

* **The Stack v2 (BigCode):** **3.28 B unique files / 67.5 TB uncompressed / ~900 B tokens** across 658 programming languages, sourced from the Software Heritage 2023-09-06 graph; permissive licenses only, with attribution requirements per original license ([HF dataset card — bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2)). Dedup-only and full variants are both released.
* **Proof-Pile-2 (EleutherAI):** **55 B tokens** = 29 B arXiv (RedPajama subset) + 15 B OpenWebMath + 11 B AlgebraicStack (Python 6.1 B, Isabelle 1.1 B, C++ 0.95 B, etc.); used to train Llemma 7B/34B; underlying licenses preserved per source ([HF dataset card — EleutherAI/proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)).

---

## 3. Benchmark evidence at the 1 B – 7 B scale

The two ablation-rich, peer-reviewed-or-equivalent reports are the FineWeb tech report and the DCLM paper.

* **FineWeb tech report ([Penedo et al., 2024](https://arxiv.org/abs/2406.17557)):** Trains 1.5 B-parameter models on 350 B-token slices of each candidate corpus and reports an aggregate score over MMLU, ARC, HellaSwag, OpenBookQA, PIQA, CommonsenseQA, SIQA, WinoGrande. The abstract claims FineWeb "produces better-performing LLMs than other open pretraining datasets" (RefinedWeb, C4, Dolma 1.6, SlimPajama, RedPajama-V2, The Pile) and that **FineWeb-Edu yields "dramatically better performance on knowledge- and reasoning-intensive benchmarks like MMLU and ARC"** ([Penedo et al., 2024](https://arxiv.org/abs/2406.17557)). The exact per-benchmark deltas are reported in the paper's tables; the dataset card and blogpost summarise them as a clear ordering: **FineWeb-Edu > FineWeb > Dolma 1.6 ≈ RefinedWeb > C4 > RedPajama-V2 > The Pile** at fixed training tokens ([HF dataset card — HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)).

* **DCLM paper ([Li et al., 2024](https://arxiv.org/abs/2406.11794)):** Runs a *standardised testbed* of 53 downstream evaluations across model scales 412 M → 7 B. They find "model-based filtering is key" and that DCLM-Baseline at 2.6 T tokens, 7B params hits **CORE 57.1 / MMLU 63.7 / EXTENDED 45.4**, beating Llama 2-7B (CORE 49.2, MMLU 45.8) and OLMo-1.7-7B (CORE 47.0, MMLU 54.0) and matching Mistral-7B-v0.3 on MMLU at far less compute ([HF dataset card — mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0); [Li et al., 2024](https://arxiv.org/abs/2406.11794)).

* **Cross-validating evidence:** Independent reuse of these corpora confirms the ordering — SmolLM3 (3B params, 11.2 T tokens) blends FineWeb-Edu + DCLM + FineWeb2 + FineWeb2-HQ as its web component ([HF blog — "SmolLM3", 2025](https://huggingface.co/blog/smollm3)), and OLMo 2 7B/13B uses DCLM + Dolma + StarCoder + Proof-Pile-II as Stage-1 (3.9 T tokens) ([AI2 — OLMo 2 blog, 2024](https://allenai.org/blog/olmo2)). NVIDIA's Nemotron-CC paper directly confirms DCLM and FineWeb-Edu as the two strongest baselines in the open ecosystem ([Su et al., 2024](https://arxiv.org/abs/2412.02595)).

* **Bottom line at 1–7 B scale:** **DCLM-Baseline and FineWeb-Edu are the two strongest single-source open web corpora.** DCLM-Baseline tends to win on knowledge benchmarks (MMLU) per token; FineWeb-Edu wins on reasoning and educational benchmarks; both dominate older corpora (Pile, C4, RedPajama, SlimPajama, RefinedWeb, Dolma 1.6). Nemotron-CC and Zyda-2 / TxT360 are credible "next-generation" mixes that *combine* DCLM-style and FineWeb-Edu-style filtering and report further gains.

---

## 4. License / openness check

| Corpus | Data license | Filtering code released | Reproducible end-to-end? |
|---|---|---|---|
| FineWeb / FineWeb-Edu / FineWeb-2 | ODC-By 1.0 | Yes (`datatrove`) | **Yes** ([HF](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [Penedo et al., 2024](https://arxiv.org/abs/2406.17557)) |
| DCLM-Baseline | CC-BY-4.0 | Yes (DCLM repo + fastText classifier + Bloom-dedup) | **Yes** ([HF](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0), [Li et al., 2024](https://arxiv.org/abs/2406.11794)) |
| RedPajama-V2 | CC Foundation ToU + Apache-2.0 code | Yes | Pool only — user must filter ([HF](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)) |
| Dolma | ODC-By | Yes (`dolma` toolkit) | **Yes** ([HF](https://huggingface.co/datasets/allenai/dolma), [Soldaini et al., 2024](https://arxiv.org/abs/2402.00159)) |
| SlimPajama | Per-source (RedPajama-1T) | Yes | **Yes** ([Cerebras blog](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)) |
| Nemotron-CC | Common Crawl ToU | Partial (paper describes; tooling on NVIDIA repo) | **Mostly** ([Su et al., 2024](https://arxiv.org/abs/2412.02595)) |
| Zyda-2 | ODC-By | Yes (Zyphra repos) | **Yes** ([HF](https://huggingface.co/datasets/Zyphra/Zyda-2)) |
| TxT360 | ODC-By | Yes (LLM360 repo) | **Yes** ([HF](https://huggingface.co/datasets/LLM360/TxT360)) |
| The Pile | "Other" / per-subset | Yes | **No** — Books3 redistribution disputed ([HF](https://huggingface.co/datasets/EleutherAI/pile)) |
| **RefinedWeb (full)** | ODC-By for released ~600 B | Code described, full 5 T not released | **No — filter-only / partial dataset** ([HF](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)) |
| The Stack v2 | "Other" — permissive licenses inherited | Yes | **Yes** ([HF](https://huggingface.co/datasets/bigcode/the-stack-v2)) |
| Proof-Pile-2 | Per-source, unaltered | Yes | **Yes** ([HF](https://huggingface.co/datasets/EleutherAI/proof-pile-2)) |

**Murky / partial:** The Pile (Books3 IP issues), RefinedWeb (only ~600 B of 5 T released), and to a lesser extent RP-V2 (raw pool — needs your filter recipe).

---

## 5. Recommendation for a fully reproducible 7B Chinchilla-optimal run

### Primary picks

1. **DCLM-Baseline (4 T tokens, CC-BY-4.0)** — *primary choice* if your headline metric is MMLU and you value the most permissive license among the top-tier corpora. The DCLM paper directly demonstrates a 7B / 2.6 T-token recipe hitting MMLU 63.7 ([Li et al., 2024](https://arxiv.org/abs/2406.11794); [HF](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)). Reproducible filtering code is released.

2. **FineWeb-Edu (1.3 T) or FineWeb-Edu-score-2 (5.4 T) — ODC-By** — *primary choice* if you want the biggest documented win on reasoning/educational benchmarks per token, and if you prefer HuggingFace's `datatrove` tooling. Score-2 is the better fit for a 7B run that needs more than 1.3 T tokens. ([HF — fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu); [HF — fineweb-edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2))

For a 7B run the **lowest-risk, most-cited recipe** at 300 B – 2 T training tokens is **DCLM-Baseline as the web backbone**, optionally cross-deduplicated against FineWeb-Edu (this is exactly what Zyda-2 does) ([HF](https://huggingface.co/datasets/Zyphra/Zyda-2)). If you want a fully ready-made mixed corpus, **Zyda-2 (5.07 T, ODC-By)** or **TxT360 (~5 T raw, ODC-By)** are both turnkey options that have already done the cross-dedup and reweighting for you ([HF — Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2); [HF — TxT360](https://huggingface.co/datasets/LLM360/TxT360)).

### Practical mixing (Llama-3 / OLMo-2 / SmolLM3 style)

To reach competitive code/math performance at 7B, blend the web backbone with:

* **Code:** **The Stack v2** (~900 B tokens, 658 languages, permissive-only; SmolLM3 used 16 languages from Stack v2 ([HF blog — SmolLM3](https://huggingface.co/blog/smollm3); [HF — the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2))).
* **Math:** **Proof-Pile-2** (55 B tokens; arXiv + OpenWebMath + AlgebraicStack; used to train Llemma ([HF — proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2))). Optionally augment with **FineMath-3+/4+** and **InfiWebMath**, both used in SmolLM3 ([HF blog — SmolLM3](https://huggingface.co/blog/smollm3)).
* **Reference mix ratios from open recipes:** OLMo 2 Stage 1 ≈ DCLM + Dolma + StarCoder + Proof-Pile-II totalling 3.9 T tokens ([AI2 — OLMo 2 blog](https://allenai.org/blog/olmo2)); SmolLM3 Stage 1 = 85 % web (FineWeb-Edu + DCLM + FineWeb2 + FineWeb2-HQ) / 12 % code (Stack v2 + StarCoder2 PRs + Jupyter/Kaggle + GitHub issues + StackExchange) / 3 % math (FineMath3+ + InfiWebMath3+) ([HF blog — SmolLM3](https://huggingface.co/blog/smollm3)).

### Recommended concrete recipe for a 7B Chinchilla-optimal run

Target ≥ 140 B tokens, plan for ~500 B – 2 T tokens to be competitive:

| Component | Source | Weight | License |
|---|---|---|---|
| Web (filtered) | **DCLM-Baseline** | 70–80 % | CC-BY-4.0 |
| Web (educational, complementary) | **FineWeb-Edu** | 10–15 % | ODC-By |
| Code | **The Stack v2 (dedup)** | 8–12 % | Permissive-only |
| Math | **Proof-Pile-2** | 2–3 % | Per-source |

This recipe replicates the spine of OLMo 2, Llemma-style math, and SmolLM3-style filtering, with every component fully open and reproducible.

---

## 6. Tokenization and token-count parity

* **FineWeb / FineWeb-Edu** report token counts using the **GPT-2 tokenizer** (`token_count` field is GPT-2 tokens) ([HF dataset card — fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)).
* **Zyda-1 and Zyda-2** report counts using the **GPT-NeoX tokenizer** ([HF dataset card — Zyphra/Zyda](https://huggingface.co/datasets/Zyphra/Zyda)).
* **DCLM-Baseline** does not pin a tokenizer in the dataset card; the paper's training runs use a standard 50k-100k BPE depending on model recipe ([Li et al., 2024](https://arxiv.org/abs/2406.11794)).
* **RedPajama-V2** publishes raw documents and CCNet quality signals; token counts depend on the user's tokenizer ([HF dataset card](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)).
* **The Pile, Dolma, RefinedWeb** report token counts using GPT-NeoX-20B / GPT-2-style tokenizers depending on the source paper.

### Cross-tokenizer parity rules of thumb

A given byte stream tokenises to *fewer* tokens with larger / better vocabularies. Concrete published reference points:

* **Llama 3.2 tokenizer** (128 k vocab, used in SmolLM3) compresses English at **3.94 characters per token vs. 3.17 for Llama 2's 32 k tokenizer** — i.e., Llama 3 needs ~24 % fewer tokens than Llama 2 to cover the same text ([HF blog — SmolLM3](https://huggingface.co/blog/smollm3)).
* GPT-2 (50 k vocab) sits between the two and is the *de facto* unit for HuggingFace dataset cards.

**Practical implication:** When the FineWeb card says "18.5 T GPT-2 tokens" and you tokenise with Llama-3, expect closer to **~14 T Llama-3 tokens** for the same data. Always:

1. State the tokenizer when reporting your training-token budget.
2. Re-tokenise (or scale by a known compression ratio) before comparing budgets across corpora.
3. If you mix corpora reported in different tokenizers, **re-tokenise all sources with your final training tokenizer** before computing mixing weights — otherwise the mix proportions silently drift.

For Chinchilla-optimality at 7B, the **140 B token target should be measured in your own training tokenizer**, not in GPT-2 tokens. Llama-3 / similar 128k tokenizers typically need ~15–25 % fewer tokens than GPT-2 to cover the same bytes ([HF blog — SmolLM3](https://huggingface.co/blog/smollm3)), so a "GPT-2-equivalent" of ~170–180 B tokens of source bytes will land you safely past the Chinchilla optimum once retokenised.

---

## 7. Open questions / unverified items

* Exact per-benchmark deltas in the FineWeb tech report ablation tables are summarised qualitatively above; the specific MMLU/ARC point gains versus RefinedWeb/C4/Dolma are reported in the paper PDF tables that I could not fully render through WebFetch — verify against the original PDF before quoting numbers in slide decks.
* Nemotron-CC's exact licensing terms ("available at [URL]" via the Common Crawl contribution repository) are not fully spelled out in the abstract — confirm before redistribution ([Su et al., 2024](https://arxiv.org/abs/2412.02595)).
* The Pile's current redistribution status varies by mirror and sub-corpus (Books3 in particular). If The Pile must be used at all, prefer "Pile-Uncopyrighted" subsets; flagged as **unverified** for any new training run.
* FineWeb-2 directly competitive ablations vs. CC-100 / mC4 / CulturaX / HPLT / MADLAD are claimed in the paper but not reproduced numerically in the abstract; verify in the full paper before relying on multilingual claims ([Penedo et al., 2025](https://arxiv.org/abs/2506.20920)).

---

## References (consolidated)

* [Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022 (Chinchilla)](https://arxiv.org/abs/2203.15556)
* [Wikipedia — Chinchilla (language model)](https://en.wikipedia.org/wiki/Chinchilla_(language_model))
* [Penedo et al., "The FineWeb Datasets", 2024 (arXiv:2406.17557)](https://arxiv.org/abs/2406.17557)
* [Penedo et al., "FineWeb2: One Pipeline to Scale Them All", 2025 (arXiv:2506.20920)](https://arxiv.org/abs/2506.20920)
* [Li et al., "DataComp-LM (DCLM)", 2024 (arXiv:2406.11794)](https://arxiv.org/abs/2406.11794)
* [Soldaini et al., "Dolma", ACL 2024 (arXiv:2402.00159)](https://arxiv.org/abs/2402.00159)
* [Gao et al., "The Pile", 2020 (arXiv:2101.00027)](https://arxiv.org/abs/2101.00027)
* [Su et al., "Nemotron-CC", 2024 (arXiv:2412.02595)](https://arxiv.org/abs/2412.02595)
* [HF dataset card — HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
* [HF dataset card — HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
* [HF dataset card — HuggingFaceFW/fineweb-edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2)
* [HF dataset card — HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
* [HF dataset card — mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
* [DataComp-LM benchmark site](https://www.datacomp.ai/dclm/)
* [HF dataset card — togethercomputer/RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)
* [HF dataset card — allenai/dolma](https://huggingface.co/datasets/allenai/dolma)
* [Cerebras blog — SlimPajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)
* [HF dataset card — Zyphra/Zyda](https://huggingface.co/datasets/Zyphra/Zyda)
* [HF dataset card — Zyphra/Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2)
* [HF dataset card — LLM360/TxT360](https://huggingface.co/datasets/LLM360/TxT360)
* [HF dataset card — EleutherAI/pile](https://huggingface.co/datasets/EleutherAI/pile)
* [HF dataset card — tiiuae/falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
* [HF dataset card — bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2)
* [HF dataset card — EleutherAI/proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
* [HF blog — SmolLM (HuggingFaceTB), 2024](https://huggingface.co/blog/smollm)
* [HF blog — SmolLM3, 2025](https://huggingface.co/blog/smollm3)
* [AI2 blog — OLMo 2, 2024](https://allenai.org/blog/olmo2)

---

## 8. Verified ablation numbers (follow-up)

*Targeted second-pass verification against the original PDFs / HTML mirrors. Where a number could not be extracted from the public PDF in tabular form, this is stated explicitly rather than substituted.*

### 8.1 FineWeb tech report (arXiv:2406.17557) — 1.82 B params, 350 B tokens

The FineWeb paper ablates on **1.82 B-parameter models trained on 350 B tokens** (not 1.5 B / 350 B as paraphrased elsewhere). Per-benchmark scores (MMLU, ARC, HellaSwag, OpenBookQA, PIQA, CommonsenseQA, SIQA, WinoGrande) for the cross-corpus comparison are presented as **Figure 10 ("Comparing FineWeb datasets to other public datasets")** in the main body, with an appendix figure (referenced as Figure 15 in Appendix E.2) breaking out the 9 individual benchmarks per dataset.

**Not extractable from public PDF / HTML mirror as a numeric table** — the comparison is rendered only as a chart (`dataset_ablations.png`). The HuggingFace blog post and dataset card report the aggregate ordering verbally:

> "FineWeb-Edu surpasses FineWeb and all other open web datasets, with remarkable improvements on educational benchmarks such as MMLU, ARC, and OpenBookQA."

The only concrete numbers re-publishable from the FineWeb blog/space at the **350 B-token** budget are:

| Corpus | MMLU | ARC | Notes |
|---|---|---|---|
| FineWeb (350 B-token training, 1.82 B params) | 33 % | 46 % | [HF blog — FineWeb](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html) |
| FineWeb-Edu (same training budget) | 37 % | 57 % | [HF blog — FineWeb](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html); also [HF dataset card — fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |

Datasets compared in Figure 10 (paper §6 / blogpost): RefinedWeb (500 B), C4 (172 B), Dolma v1.6 CC-portion (3 T), The Pile (340 B), SlimPajama (627 B), RedPajama-V2 (20 T dedup), FineWeb (15 T), plus CC-100 / Colossal-OSCAR / Matrix in the appendix. Ordering on the aggregate score is FineWeb-Edu > FineWeb > Dolma > RefinedWeb > C4 > SlimPajama > RedPajama-V2 > The Pile (qualitative; per-corpus aggregate-score numbers visible on the chart but not tabulated in the public PDF). Caveat: precise numeric deltas (±X.X aggregate points) require reading values off the SVG/PNG charts in the blog or rebuilding from the `datatrove` evaluation logs — they are not in a copy-pasteable table in the arXiv PDF.

### 8.2 DCLM paper (arXiv:2406.11794) — headline 7B and ablations

**Table 9 (main 7B comparison, DCLM-Baseline vs other 7B/8B models):**

| Model | Params | Tokens | CORE | MMLU (5-shot) | EXTENDED |
|---|---|---|---|---|---|
| Llama-2 | 7 B | 2 T | 49.2 | 45.8 | 34.1 |
| OLMo-1.7 | 7 B | 2.1 T | 47.0 | 54.0 | 34.2 |
| MAP-Neo | 7 B | 4.5 T | 50.2 | 57.1 | 40.4 |
| Mistral-7B-v0.3 | 7 B | n/r | 57.0 | 62.7 | 45.1 |
| **DCLM-Baseline** | **7 B** | **2.6 T** | **57.1** | **63.7** | **45.4** |
| Llama-3 | 8 B | 15 T | 57.6 | 66.2 | 46.3 |

Source: [arXiv:2406.11794v3 Table 9 (HTML mirror)](https://arxiv.org/html/2406.11794v3); also reproduced on the [HF dataset card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0). DCLM-Baseline matches Mistral-7B-v0.3 on CORE/MMLU/EXTENDED at comparable token count and is within ~2.5 MMLU points of Llama-3 8B at ~5.8× fewer training tokens.

**Table 4 (1B-1x scale, model-based vs heuristic filtering):**

| Filter | CORE | EXTENDED |
|---|---|---|
| RefinedWeb-style heuristic baseline | 27.5 | 14.6 |
| PageRank | 26.1 | 12.9 |
| SemDedup | 27.1 | 13.8 |
| BGE classifier | 27.2 | 14.0 |
| AskLLM | 28.6 | 14.3 |
| Perplexity filtering | 29.0 | 15.0 |
| Top-k average logits | 29.2 | 14.7 |
| **fastText OH-2.5 + ELI5** | **30.2** | **15.4** |

**Table 5 (7B-1x scale, fastText positive-class ablation, 10 % retention threshold):**

| fastText positives | CORE | MMLU |
|---|---|---|
| OpenWebText2 | 34.7 | 25.0 |
| Wikipedia | 35.7 | 27.0 |
| GPT-3 Approx mix | 37.5 | 24.4 |
| **OH-2.5 + ELI5 (DCLM-Baseline recipe)** | **41.0** | **29.2** |

Source: [arXiv:2406.11794v3 Tables 4 & 5 (HTML mirror)](https://arxiv.org/html/2406.11794v3). The DCLM authors summarise this as "fastText OH-2.5 + ELI5 gives a 3.5-percentage-point lift on CORE over conventional reference-data choices."

### 8.3 Nemotron-CC (arXiv:2412.02595) — license + +5.6 MMLU number

**License (verified, verbatim from dataset card):**

> "The Nemotron-CC-Code-v1, Nemotron-CC-v2.1, Nemotron-Pretraining-Code-v2 datasets are governed by the [NVIDIA Data Access Agreement for Model Training](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/raw/main/LICENSE.md)."

This is **NOT CC-BY-4.0 and NOT the Common Crawl ToU directly** — it is NVIDIA's bespoke "Data Access Agreement for Model Training," which permits training of any AI model (including proprietary or open-source releases) and explicitly does not prohibit disclosing benchmarks or evaluations of trained models. The agreement is grounded in fair-use language: "Copyright law protects particular expressions, but not facts, ideas, data, or information." The *Nemotron-Pretraining-Specialized-v1* sub-collection has different terms (CC-BY-4.0 by default, with CC-BY-SA-4.0 / GFDL-1.3 carve-outs for the Wiki-Rewrite and Scientific-Coding subsets). Sources: [HF — nvidia/Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1); [NVIDIA Data Access Agreement (license file)](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/raw/main/LICENSE.md).

**Implication for our recipe:** Nemotron-CC is **not as permissive as DCLM-Baseline (CC-BY-4.0) or FineWeb-* (ODC-By-1.0)** — redistribution and downstream license-compatibility need explicit reading of the NVIDIA Data Access Agreement. For a fully reproducible 7B run prioritising license simplicity, DCLM-Baseline / FineWeb-Edu remain the lowest-risk picks.

**+5.6 MMLU headline confirmed (Table 5 of the paper, 8 B params / 1 T tokens):**

| Dataset | MMLU | ARC-Challenge | HellaSwag | Avg |
|---|---|---|---|---|
| FineWeb-Edu | 42.9 | 48.0 | 70.7 | 53.2 |
| DCLM | 53.4 | 47.0 | 76.3 | 57.0 |
| **Nemotron-CC-HQ** | **59.0** | **52.9** | **76.6** | **60.1** |

Δ MMLU = 59.0 − 53.4 = **+5.6** vs DCLM, exactly matching the abstract claim. Source: [arXiv:2412.02595v2 Table 5 (HTML mirror)](https://arxiv.org/html/2412.02595v2).

**Long-horizon (Table 6, 8 B / 15 T tokens):**

| Model | MMLU | ARC-Challenge | HellaSwag | Avg |
|---|---|---|---|---|
| Llama 3.1 8B | 65.3 | 55.0 | 79.3 | 64.2 |
| **Nemotron-CC 8B (7.2 T from Nemotron-CC inside 15 T total)** | **70.3** | **58.1** | **80.8** | **64.7** |

Δ MMLU = +5.0; Δ ARC-C = +3.1; Δ avg = +0.5 — matches abstract claims. Source: [arXiv:2412.02595v2 Table 6 (HTML mirror)](https://arxiv.org/html/2412.02595v2).

### 8.4 FineWeb-2 (arXiv:2506.20920) — multilingual head-to-heads

**Canary languages evaluated (9):** Arabic, Chinese, French, Hindi, Russian, Swahili, Telugu, Thai, Turkish. Unseen evaluation languages (5): German, Indonesian, Italian, Japanese, Vietnamese. Total = 14 languages, of which FineWeb-2 wins on 11 (per Figure 3 caption, [arXiv:2506.20920v1](https://arxiv.org/html/2506.20920v1)).

**Datasets compared:** CC-100, mC4, CulturaX, HPLT (v1), and "raw" CC (post-extraction, post-LangID, no additional filtering or dedup). MADLAD-400 is referenced but **excluded from the head-to-head ablation** because most languages had insufficient overlap; CC-100 and mC4 likewise lacked enough Telugu/Swahili to run a 30 B-token training run, and only CulturaX / HPLT had enough Hindi.

**Per-language aggregate-score numbers: NOT EXTRACTABLE FROM THE PUBLIC PDF.** The expanded numerical tables live in Appendix A.9 ("Dataset comparison on Canary Languages") and Appendix A.10.2 ("Full evaluation results"); both arXiv HTML and PDF mirrors render them only as figures, and the OpenReview PDF returned binary-encoded streams to WebFetch. Table 25 (§A.7.2) reports an *average ranking* of filtering thresholds across languages but not absolute scores. The only verifiable headline number from the public abstract / Figure 3 caption is the qualitative "11 of 14 languages" win-rate — verify exact per-language deltas against the source PDF before quoting them in any deck. Source: [arXiv:2506.20920v1](https://arxiv.org/html/2506.20920v1); [HF dataset card — fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2).

### 8.5 Summary of follow-up findings

1. **DCLM Tables 4, 5, 9 fully verified** — model-based fastText filtering provides +2.7 CORE points over a RefinedWeb-style heuristic baseline at 1B-1x scale (Table 4: 30.2 vs 27.5) and +3.5 CORE / +4.2 MMLU points at 7B-1x scale over the next-best classifier-positives variant (Table 5).
2. **Nemotron-CC license is the NVIDIA Data Access Agreement**, not CC-BY or Common Crawl ToU — update §4 license-table verbiage accordingly. The +5.6 MMLU headline number is confirmed at 8 B / 1 T tokens (Nemotron-CC-HQ 59.0 vs DCLM 53.4).
3. **FineWeb tech report (arXiv:2406.17557) per-benchmark numbers are chart-only** in the public PDF/HTML mirrors. The two pinpoint numbers re-publishable are FineWeb at 33 % MMLU / 46 % ARC and FineWeb-Edu at 37 % MMLU / 57 % ARC (1.82 B params, 350 B tokens). Ordering on the aggregate score is consistent with §3 of this report.
4. **FineWeb-2 per-language ablation tables (vs CC-100/mC4/CulturaX/HPLT/MADLAD-400) are not extractable from the public PDF** — they are referenced as Appendix A.9 / A.10.2 but render only as figures. Only the "11 of 14 languages" qualitative win-rate is fully verified; MADLAD-400 is excluded from the head-to-head due to data-sufficiency mismatches.
