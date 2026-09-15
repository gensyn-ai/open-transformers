# OPEN

**[Audit app](https://open1b.gensyn.ai/)** · **[Models on Hugging Face](https://huggingface.co/collections/Gensyn/open-1b)** · [Verify a step yourself](#verify-a-training-step-yourself) · [Audit runbook](scripts/audit_volunteer/RUNBOOK.md)

OPEN is a collection of dense transformer language models from Gensyn whose
training is **bitwise auditable on consumer hardware**. Every optimizer step of
every published run has a recorded state hash, and anyone can re-execute a step
on a MacBook, a single NVIDIA GPU, or a plain CPU and get back the exact same
bytes. Not "close." Not `atol=1e-6`. The same bits.

This repository is the training harness and the audit-replay harness — the code
that produced the runs and the code that lets a stranger check them.

**OPEN-1B** is the first model in the collection: 1.61 B parameters, 400 B
tokens, 80,957 optimizer steps, every one of them hashed and replayable.

---

## Why this exists

"Open weights" tells you what a model ended up as. It tells you nothing about
how it got there — what data it saw at step 41,000, whether the recipe in the
paper is the recipe that ran, whether the checkpoint you downloaded is the
checkpoint that training produced.

Verifying that normally means trusting the lab, because reproducing a training
step bit-for-bit is not something floating-point hardware does for free.
Floating-point addition is not associative, so reduction order, transcendental
implementations, FMA contraction, RNG streams and optimizer internals all change
the output bits. Two "identical" runs on two machines diverge in the last bit
within a handful of steps, and that divergence compounds.

OPEN pins all of it. The result is a training run where:

- a published hash at step *N* is a **commitment**, and
- checking it costs one command on hardware you already own.

An audit on an Apple laptop is worth exactly as much as an audit on an H100.
That is the point, and it is why the replay path refuses to gate on GPU
architecture.

## How bitwise reproducibility is achieved

Five things have to be nailed down at once. Miss one and the step diverges.

| Layer | Mechanism | Where |
|---|---|---|
| **Kernels** | Every byte-sensitive op — matmul, RMSNorm, SwiGLU, RoPE tables, flash attention, the embedding backward, loss reductions, AdamW — routes through [`repop`](#repop-the-reproducible-operator-library), a reproducible operator library with fixed-order reductions and matching CPU / CUDA / Metal implementations. No stock framework primitive touches a tensor whose bits matter. | `src/pretrain/model/`, `src/pretrain/optim/adamw_repop.py` |
| **Initialization** | Counter-based Philox stream sampled on CPU and byte-copied to device, so init is device-independent and unchained — auditable with no checkpoint and no data. | `src/pretrain/model/init.py` |
| **Data** | A canonical `GlobalStream` that is a pure function of (seed, shard manifests, `seq_len`). A single-device replay rebuilds every virtual rank's slice and feeds identical windows in identical order. Resume is bit-exact. | `src/pretrain/data/global_stream.py`, `mix_sampler.py` |
| **Reduction** | Topology-invariant gradient reduction (`reduction_mode: deterministic_allgather`): ascending-rank reduce-scatter inside the shard group, a balanced binary-blocks tree across replicas. The result does not depend on how many GPUs ran it. | `src/pretrain/parallel/deterministic_reduce.py` |
| **Clipping** | Stateless deterministic global-norm clip, folded in the same ascending-shard order. No clipper state to carry, nothing to grind into fp32 subnormals. | `src/pretrain/train/global_clip.py` |

On top of that sits the **chained state hash**: a blake2b digest over weights,
optimizer moments and param groups, the step's post-clip gradients, and the
running per-rank batch digest — each step's hash folding in the previous one.
OPEN-1B hashed **every** step (`state_hash.every_n_steps: 1`), so any interval,
however short, has a target to compare against.

For the full engineering story, including every divergence found the hard way,
see [`docs/unified-audit-results.md`](docs/unified-audit-results.md).

## Verify a training step yourself

You need no Gensyn account and no credentials. An **audit kit** is a
commit-keyed, world-readable set of artifacts: this harness as a wheel, the
matching `repop` wheels for Linux and macOS, a `trajectory.json` of canonical
hashes, and a `kit.json` manifest binding them by SHA-256. The hashes only
reproduce against the exact code pair that minted them, which is what makes the
manifest load-bearing rather than decorative.

```bash
KIT=https://storage.googleapis.com/gensyn-audit-public/audit-kit/pt-<sha12>_rp-<sha12>
curl -fsSL -O "$KIT/kit.json"      # then verify each file's sha256 before installing

pip install ./repop-*.whl ./pretrain-*.whl

pretrain-audit-replay --from-init --until-step 0 \
    --config-name 1b_repop_v2 --device mps      # or cuda, or cpu
```

It prints `MATCH` or `MISMATCH`. Step-interval replays additionally take a
published checkpoint and the shards that interval consumed:

```bash
pretrain-audit-replay \
    --checkpoint  step_000050300 \
    --until-step  50400 \
    --gcs-root    gs://gensyn-open-1b/data/shards \
    --expect-hash <digest from the published hash log>
```

Memory: ~8 GB covers the 1B init unit on MPS; ~24 GB covers every published
unit including step replays, with the offload flags the harness selects for you.
CPU is the reference device and needs no GPU at all.

The full volunteer path — fetching a kit, checking digests, picking the right
wheel for your machine — is [`scripts/audit_volunteer/RUNBOOK.md`](scripts/audit_volunteer/RUNBOOK.md).
Flags, gotchas and the segment-boundary rules are in
[`docs/audit-replay-usage.md`](docs/audit-replay-usage.md); the Apple-silicon
specifics are in [`docs/mps-audit-runbook.md`](docs/mps-audit-runbook.md).

### What a match proves, and what it does not

A match establishes that the selected replay, with the pinned artifacts and
inputs, produced the expected state hash on the tested backend. An
initialization-only unit does not verify a training interval. Running compiled
kernels does not independently reveal or prove their implementation, and a
matching hash alone does not establish how an earlier training run was executed.
The v3 hash covers weights, optimizer state, gradients and the batch digest — it
does not cover RNG, the data-stream cursor, spike state, or the descriptor keys
in `meta.json`.

**Checkpoint directories are code, not data.** `torch.distributed.checkpoint`
unpickles its metadata index before any of our code runs. Only replay
checkpoints whose provenance you trust, and sandbox anything else. See the trust
model at the top of [`docs/audit-replay-usage.md`](docs/audit-replay-usage.md).

## The models

### OPEN-1B

Decoder-only transformer descended from Llama 3, modified for stability under
low-precision training and for bitwise-reproducible execution.

| | OPEN-1B |
|---|---|
| Parameters | 1.61 B total / 1.08 B non-embedding |
| Layers | 24 |
| Hidden size | 2048 |
| Attention heads (Q / KV) | 16 / 4 (GQA), head dim 128 |
| FFN hidden | 5632 (SwiGLU, fused gate+up GEMM) |
| Vocabulary | 128,256 (byte-level BPE, trained from scratch) |
| Sequence length | 4096 |
| Attention | Hybrid sliding-window: 512-token window, full causal every 5th layer and the last |
| Normalization | RMSNorm pre-norm, plus RMSNorm on the embedding output |
| QK-norm | RMSNorm, **gain-free** (no learnable temperature) |
| RoPE θ | 5 · 10⁵ |
| Z-loss | 10⁻⁴, fused with cross-entropy in one chunked pass |
| Embeddings | Untied; no weight decay on the input embedding |
| Linear precision | int8 W8A8 via LSQ quantization-aware training; int8 P·V inside flash attention |
| Peak LR / warmup | 4.5 · 10⁻⁴ / 667 steps |
| Schedule | Cosine over 400 B tokens to 10 % of peak |
| Training run | 80,957 steps, 400,004,481,024 tokens, 48 × H100 |

Each departure from Llama 3 is motivated by a demonstrated failure rather than
taste: the QK-norm gain was removed after the product of learned query and key
gains acted as an unbounded attention temperature and drove an entropy collapse,
and the embedding norm bounds the residual stream entering the network, which
turned out to be necessary for stable activation quantization. The configs carry
the reasoning inline — see [`configs/train/1b_repop_v2.yaml`](configs/train/1b_repop_v2.yaml)
and [`configs/model/llama3_1b_proxy_repop.yaml`](configs/model/llama3_1b_proxy_repop.yaml).

Larger dense members of the collection are in progress.

### Training corpus

Four fully-open sources, mixed proportionally to what was pulled
(`configs/data/recipe_v1_proportional.yaml`):

| Source | Tokens on disk | Share |
|---|---|---|
| DCLM-Baseline 1.0 | 300.5 B | 66.7 % |
| FineWeb-Edu (`int_score ≥ 3`) | 59.8 B | 13.3 % |
| The Stack v2 dedup (permissive subset) | 54.2 B | 12.0 % |
| Proof-Pile-2 | 36.0 B | 8.0 % |

Every shard manifest records the tokenizer hash and per-shard blake2b digests,
so the published corpus is self-verifying and a re-pull that drifts is
detectable rather than silent.

## Published artifacts

| Artifact | Where |
|---|---|
| Model weights | [Hugging Face — the `Gensyn/open-1b` collection](https://huggingface.co/collections/Gensyn/open-1b) |
| Audit app | <https://open1b.gensyn.ai/> — browse the run and the data behind any step |
| Checkpoints (810, all 7 run segments) | `gs://gensyn-open-1b/ckpt` |
| Corpus shards + tokenizer | `gs://gensyn-open-1b/data` |
| Audit kits | `gs://gensyn-audit-public/audit-kit/` |

The corpus is published as the mirrored shard tree `audit_replay --gcs-root`
expects, so an auditor fetches only the shards an interval actually consumed.
The run's rank-0 loss log is deliberately withheld from the release so that
submissions can be gated against it.

## Repo layout

```
src/pretrain/
├── cli/          train, audit_replay, verify_handoff, prepare_data, eval,
│                 eval_olmes, fetch_audit_data, dcp_safetensors, dump_documents
├── config/       pydantic schema + hydra loader (configs ship inside the wheel)
├── data/         tokenizer, indexed binary shards, canonical global stream,
│                 mix sampler, manifests, interval fetch, doc map
├── model/        Llama-3-derived model, modules, seeded init, fused CE+z-loss,
│                 streaming vocabulary head, precision policy
├── optim/        repop AdamW, schedules, Muon (experimental)
├── parallel/     FSDP2 wrap, mesh dims, deterministic reduction, repop env
├── train/        loop, batch schedule, global clip, spike protocol, state hash,
│                 checkpointing, midtraining
├── eval/         perplexity, lm-eval adapter, OLMES runner, DCLM-CORE
└── obs/ util/    metrics + W&B, registries, git/timing helpers

configs/          hydra tree: model / data / optim / schedule / run / train
docs/             architecture, audit runbooks, kit inventories, BFR write-ups
scripts/
├── audit_kit/    build, gate and publish an audit kit
├── audit_volunteer/  the verifier-facing runbook
├── explorer/     training-data explorer load jobs
└── build_corpus.py, launch_*.sh
tests/            59 test modules; CPU-runnable, repop required for some
plan/ research/   original design docs and recipe citations (historical)
```

## Training

Requires PyTorch and `repop`. Production runs use the NGC PyTorch container
(see [`Dockerfile`](Dockerfile)), which pins CUDA / cuDNN / NCCL / FlashAttention
by image digest.

```bash
# Build the corpus (streams from HF, filters, tokenizes, shards).
python scripts/build_corpus.py --preset 1b_proxy

# Launch. Hydra overrides pass straight through.
./scripts/launch_single_node.sh 1b_repop_v2 optim.peak_lr=3e-4

# Resume (bit-exact: sampler position, RNG, moments, chained hash).
RESUME_FROM=runs/<RUN_ID>/checkpoints/step_000050300 \
    ./scripts/launch_single_node.sh 1b_repop_v2
```

Parallelism composes two axes over a `("dp_replicate", "fsdp")` device mesh
(`pretrain.parallel.parallel_dims.ParallelDims`). Tensor, pipeline, context and
expert parallelism are out of scope — TP was removed and the harness runs
`tp=1`, which is also the only shape `audit_replay` accepts.

```yaml
run:
  dp_replicate_size: 8     # outer replicas — one grad all-reduce per optimizer step
  dp_shard_size: 4         # FSDP2 sharding within a replica
  reduction_mode: deterministic_allgather   # "nccl" opts out of auditability
```

Product must equal `world_size`; `dp_shard_size: -1` auto-fills. Under HSDP the
loop defers FSDP2's gradient sync to the final micro-batch
(`set_requires_gradient_sync`), so *N* accumulation steps cost one cross-replica
all-reduce rather than *N*.

`reduction_mode: nccl` trades auditability for throughput — a run recorded that
way cannot be replayed, and `audit_replay` refuses it rather than
mis-reproducing it.

### Evaluation

```bash
pretrain-eval-olmes \
    --config-name 1b_repop_v2 \
    --checkpoint  runs/<RUN_ID>/checkpoints/step_<N> \
    --tokenizer   data/tokenizer.json \
    --out-dir     runs/<RUN_ID>/evals/olmes
```

OLMES runs against a pinned `ai2-olmes` commit with `lm-eval==0.4.3`; the
DCLM-CORE path wants `0.4.4`. Install the `eval` extra on eval boxes only.

## repop, the reproducible operator library

`repop` is a separate package with its own licence (the published audit wheels
declare MIT) and its own release cadence. It provides the fixed-order CPU, CUDA
and Metal kernels this harness depends on for its guarantee. It is not on PyPI:
install the wheel from an audit kit (the `repop` source tree is not public). Kernel-selecting environment (`REPOP_EXECUTION_MODE`, the LSQ/Hadamard
flags) is recorded per checkpoint in `meta.json` and re-applied by
`audit_replay` — never set it by hand.

## Testing

```bash
uv sync --extra dev
uv run pytest
```

The suite runs on CPU (macOS arm64 and Linux x86-64) and covers indexed-binary
round-trips, sampler determinism and bit-exact resume, schedule math, seeded
init, model shapes, the fused loss and z-loss backward, global clip and
deterministic grad norm, state hashing (full, sharded, and the native wire
format), checkpoint round-trip and untrusted-load behaviour, the spike protocol,
the audit memory plan, spill integrity, the gradient sidecar and hand-off
verifier, and audit-kit provenance and disclosure gates. Tests that exercise the
real model or a real checkpoint need `repop` installed; `uv sync --extra dev`
alone does not supply it, and those tests skip without it.

## License

Code is [Apache 2.0](LICENSE.md). `repop`, checkpoints and corpus are
distributed separately under their own terms.

This harness is published for verification. It is not a general-purpose training
framework, it carries no stability guarantee across kit versions, and issues are
triaged at Gensyn's discretion.
