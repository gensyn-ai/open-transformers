# pretrain

The training harness behind Gensyn's auditable-training claims, packaged so
that anyone can re-run a published training step and check the result against
a published hash.

This wheel is distributed as part of an *audit kit*: a commit-keyed set of
artifacts holding this package, the matching `repop` wheels for Linux and
macOS, a canonical hash trajectory, and a manifest binding them together. The
kit is what makes a verification meaningful, because the hashes only reproduce
against the exact code pair that minted them.

## Verifying a claim

You need no Gensyn account and no credentials. Install the two wheels named by
a kit's `kit.json`, then replay a unit:

```bash
pretrain-audit-replay --from-init --until-step 0 \
    --config-name <unit.config_name> --device <cpu|cuda|mps>
```

The harness prints `MATCH` or `MISMATCH` against the trajectory shipped in the
kit. Apple silicon, NVIDIA and plain CPU are all first-class: `repop`'s kernels
are bitwise reproducible across devices, so the same published hash has to come
back from Metal, CUDA and CPU alike. A verification on a Mac is worth exactly
as much as one on an NVIDIA box.

For CPU replay, `--cpu-threads auto` is the default. It preserves existing
`OMP_NUM_THREADS` or `MKL_NUM_THREADS` settings; otherwise Apple Silicon uses
its performance-core count and other hosts keep the Torch default. Use
`--cpu-threads 8` to select a count explicitly. The effective Torch count and
selection source appear in the log and result JSON. This is a hardware-based
default, not affinity pinning or performance autotuning. The expected state
hash remains the acceptance check at every worker count.

For FP32 MPS replay with an untied repop head, `PRETRAIN_AUDIT_STREAM_HEAD=1`
reduces vocabulary-head memory by streaming row panels. It requires a matching
repop build with `mm_accumulate`; `PRETRAIN_AUDIT_HEAD_CHUNK_ROWS` defaults to
256. Unsupported panel shapes fall back to the ordinary head and loss.
The option is off by default, and the expected hash remains mandatory for
accepting a replay. Separately, `PRETRAIN_AUDIT_MPS_OFFLOAD_GRADIENTS=1` stores
accumulated FP32 gradients on CPU while adding them on MPS; transfers can make
this slower. It is also off by default and was tested separately from the
streamed-head timing pair.

Full instructions, including how to fetch a kit and check its digests before
installing anything, are in the audit-kit volunteer runbook that accompanies
the kit URL you were given.

A match establishes that the selected replay, with the pinned artifacts and
inputs, produced the expected state hash on the tested backend. An
initialization-only unit does not verify a training interval. Running compiled
kernels does not independently reveal or prove their implementation, and a
matching hash alone does not establish how an earlier training run was executed.

## What is in here

A PyTorch-native Llama 3 training stack: FSDP2 with optional tensor and
replicate parallelism, reproducible AdamW and norm/clip paths built on
`repop`, deterministic data sampling with bit-exact resume, and the replay
harness that reconstructs a step from a published checkpoint interval.

The reproducibility guarantee comes from `repop`, which is a separate package
with its own licence.

## Support

This package is published for verification. It is not a general-purpose
training framework, it carries no stability guarantee across kit versions, and
issues are triaged at Gensyn's discretion.
