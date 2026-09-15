# Implementation-time decisions

Judgment calls made while implementing the spec in `plan/`. These are
deliberately separate from `plan/09_decisions.md` (which holds
architectural ADRs decided *before* implementation): the entries here
are smaller, often mechanical, and only emerged once code hit the road.

## D-001 — Pydantic-validated Hydra (no structured-configs binding)

**Decision.** The plan calls for Hydra + Pydantic. We use Hydra purely
for *composition* (config groups, defaults, CLI overrides) and feed the
resolved `OmegaConf` dict into a Pydantic `RootConfig` for validation.
We do **not** use Hydra's structured-configs feature
(`ConfigStore.store(node=Schema)`).

**Why.** Maintaining a parallel `@dataclass` hierarchy that mirrors the
Pydantic schemas doubled the schema surface for zero additional safety
(Pydantic already type-checks). The single `resolve_to_typed()` helper
gives the same effect with one source of truth.

## D-002 — `# @package _global_` on every train YAML

**Decision.** Each `configs/train/*.yaml` declares `# @package _global_`
at the top so its contents populate the root, not nested under `train.`.

**Why.** Without it, Hydra would namespace fields under the directory
name (`cfg.train.train.total_tokens`). The plan-level config tree assumed
flat root composition.

## D-003 — Smoke config is 234 M, not 100 M, due to the 128 k vocab

**Decision.** The "100 M smoke" config (`configs/model/llama3_100m_smoke.yaml`)
in fact materialises a ~234 M-parameter model. The transformer backbone
is ~10 M; the rest is the input + output embedding (each
`128_256 × 768`).

**Why.** The plan locks vocab at 128 256 across all configs (see
`plan/04_model.md` §3.1) so smoke / proxy / main share a tokenizer and
a head shape. Shrinking vocab in the smoke config would invalidate the
test as a true end-to-end exercise of the pipeline. The size remains
small enough to run a few steps in seconds on CPU; the e2e test shrinks
the *vocab* further (4 k) to keep CI fast — but only inside the test, not
in the published config.

**Revisit trigger.** If the smoke test outgrows CI memory, swap the
e2e test to a separate `configs/model/llama3_smoke_tiny_vocab.yaml`
with vocab=4 k and keep the published 100 M smoke as it is.

## D-004 — 1 B proxy lands at 1.6 B due to the same vocab

**Decision.** The 1 B proxy config produces 1.61 B parameters. We accept
this and refer to it as "1 B proxy" everywhere, matching the plan.

**Why.** Same reason as D-003 — the tokenizer is shared with the 8 B
main. A 1.6 B "proxy" still has the property that *every* knob other
than depth/width matches the main run, which is the whole point of the
M2 gate (`plan/07` §1.3).

## D-005 — RoPE `apply_rope` broadcasts cos/sin across heads explicitly

**Decision.** `pretrain/model/modules/rope.py::apply_rope` unsqueezes
cos/sin to shape `[..., T, 1, head_dim/2]` so they broadcast across
the head dim of `[B, T, H, head_dim/2]` Q/K tensors.

**Why.** A naive `unsqueeze(0)` left cos/sin at `[T, head_dim/2]` and
broadcast incorrectly when Q has the head dim *between* T and head_dim.
Caught by an early forward smoke test. The fix is two extra
`unsqueeze` calls — no perf cost.

## D-006 — `torch.optim.AdamW(fused=True)` falls back silently on Mac dev

**Decision.** `build_adamw` requests `fused=True` only when CUDA is
available; on Mac CPU it falls back to the un-fused path so dev tests
can run.

**Why.** PyTorch raises if `fused=True` is requested without a CUDA
device. We don't want every CPU test to special-case this.

## D-007 — DCP fallback path uses `torch.save` with `weights_only=False`

**Decision.** `pretrain/train/checkpoint.py::Checkpointer` tries DCP first;
on `Exception` (typical on single-process CPU dev) falls back to
`torch.save({"model": ..., "optim": ...})`. Load uses
`torch.load(..., weights_only=False)`.

**Why.** DCP requires `torch.distributed` to be initialised. We want
checkpoint round-trip tests to run in pytest without spinning up a
distributed group. The fallback's serialisation format is intentionally
*not* the production path — production runs through DCP — so we keep it
simple.

**Risk.** The fallback's `pickle` deserialisation runs arbitrary code
on load. Acceptable because checkpoints are produced by the same
codebase that loads them; do not load unknown checkpoints.

## D-008 — Mix sampler resume saves RNG + carry-over buffer

**Decision.** `MixSamplerState` holds three things: per-source consumed
counts (+ epoch), the categorical-pick RNG state
(`bit_generator.state`), and the carry-over token buffer. All three
are required for bit-exact resume; the test
`test_mix_sampler.py::test_resume_from_state` asserts equality of the
next 3 chunks after save/restart.

**Why.** The plan calls for "bit-exact resume" but doesn't enumerate
what state that requires. The carry-over buffer holds tokens from a
partially-consumed document; the RNG drives the source-pick at every
document boundary. Without saving both, "same seed + counters" does
not reproduce subsequent chunks.

## D-009 — `_global_grad_norm` prefers FSDP2's helper, falls back to `nn.utils`

**Decision.** The training loop calls `_global_grad_norm(model, max)`,
which delegates to `model.clip_grad_norm_(max)` if available (FSDP2
exposes this) and otherwise to `torch.nn.utils.clip_grad_norm_(...)`.

**Why.** FSDP2's helper does the cross-shard reduction; the standard
helper does the right thing on a non-sharded model. One code path
covers both production and dev/CPU.

## D-010 — Activation checkpoint replaces `block.forward`, not via wrapper class

**Decision.** `parallel/fsdp.py::_wrap_block_with_ac` replaces a
`TransformerBlock`'s `forward` attribute with a closure that calls
`torch.utils.checkpoint.checkpoint(original, ...)`.

**Why.** Wrapping each block in an outer `nn.Module` would change the
module graph that FSDP2 has just sharded. Replacing the bound method
keeps the FSDP2 wrap intact and avoids re-wrapping. Caveat: this is a
load-bearing implementation detail — if a future refactor moves AC
*before* FSDP2, the call order needs to flip.

## D-011 — `compute_perplexity` is single-rank for now

**Decision.** The eval helper streams the held-out set on a single rank.
For 250 M held-out tokens at the documented seq_len of 4 096, this is
~ 1 minute of GPU time on H100 — fine to run from rank 0 every 1 B
training tokens.

**Why.** Sharding eval across ranks adds infrastructure that the plan's
M-series milestones don't need; the bottleneck is the DCLM eval, not
PPL. The plan's "watch list" already flags single-rank eval as something
to revisit if it dominates wall-clock late in the run; no action needed
yet.

## D-012 — `data.tokenizer.train_tokenizer` lives where it's used, not in a script

**Decision.** Tokenizer training is a Python function in
`pretrain/data/tokenizer.py`, exposed via the
`pretrain.cli.prepare_data train-tokenizer` subcommand. It is *not* a
separate top-level script under `scripts/`.

**Why.** `scripts/` is reserved for cluster launchers (bash). Anything
that returns a value or reads typed config goes through the CLI module.
Cleanest division per `plan/02_repo_layout.md` §1: "bash files are
launchers; the actual logic is always importable Python".

## D-013 — DCLM-CORE runner dispatches via lm-eval where possible

**Decision.** `eval/dclm_core.py` does not vendor the full DCLM task
definitions in this skeleton — it dispatches the canonical task names
(hellaswag, mmlu, etc.) through `lm-eval` and structures the result by
the DCLM category breakdown. Tasks not present in `lm-eval` show up as
`error` in the result JSON.

**Why.** Vendoring 53 task scripts at a fixed SHA is a separate piece
of work (tracked in `plan/09_decisions.md`'s watch list under "Eval
drift"). Doing it now bloats this skeleton with task definitions whose
exact form is owned by another team. The shape of `run_dclm_core(...)`
matches the production interface so swapping the dispatcher later is a
one-file change.

## D-015 — Dockerfile installs into the NGC system Python (PEP 668 override)

**Decision.** The Dockerfile installs our Python deps into the NGC base
image's *system* Python via `uv pip install --system
--break-system-packages` (and exports `UV_BREAK_SYSTEM_PACKAGES=1` /
`PIP_BREAK_SYSTEM_PACKAGES=1` for child commands). We do not create a
fresh venv inside the container.

**Why.** NGC ships its tested torch / TransformerEngine / FlashAttention
/ cuDNN combo in the system Python's site-packages. A fresh venv would
either reinstall those (and lose the tested wheels) or require copying
them in by hand. Newer NGC images (Ubuntu 24.04 / Python 3.12) carry
the PEP 668 EXTERNALLY-MANAGED marker, which uv refuses to write past
without explicit consent — `--break-system-packages` is the explicit
consent.

**Why not `rm /usr/lib/python3.12/EXTERNALLY-MANAGED`.** That works too
but silently disables the guard for any future `pip install` step,
including ones added by a hurried PR. The flag form scopes the override
to the lines that asked for it.

**Risk.** A future NGC version could reorganise its torch install (e.g.
via a venv layered on `/usr`). Mitigation: the `RUN python -c "import
torch, transformer_engine; ..."` sanity check at the end of the
install layer fails the build immediately if our deps shadowed or broke
the tested wheels.

## D-016 — uv pinned at 0.5.11 in the container

**Decision.** `pip install uv==0.5.11` in the Dockerfile.

**Why.** 0.4.27 (the original pin) predates PEP 668's
`--break-system-packages` support landing cleanly in uv. 0.5.x is the
oldest stable line that handles the externally-managed marker via the
flag/env. We do not track uv-latest; the container is rebuilt only on
deliberate base-image bumps, so a current-but-not-latest uv minimises
churn-at-build-time.

**Revisit trigger.** When we next bump the NGC base, re-evaluate uv
against whatever Python the new image ships.

## D-017 — `wrap_model` is a no-op when `world_size == 1`

**Decision.** `pretrain/parallel/fsdp.py::wrap_model` skips
`fully_shard` entirely when the distributed world size is 1, returning
the model with only activation-checkpointing applied. The full FSDP2
wrap is still applied for `world_size >= 2`.

**Why.** On single-GPU runs (smoke test on a Spark, single-GPU CI),
FSDP2's `fully_shard` still converts parameters to `DTensor` even when
the mesh has one rank. `nn.Embedding(input_ids)` then fails with
`aten.embedding.default: got mixed torch.Tensor and DTensor` because
the input ids are a plain tensor and there is no input-conversion path
through the per-module `fully_shard` wrap. With nothing to shard, the
DTensor envelope is overhead with no upside.

**Why not "add an input-conversion hook so FSDP also handles ws=1".**
That would mean carrying a code path whose only purpose is to make a
no-op work — and the no-op case has zero performance interest because
there is no parallelism to extract. Skipping is simpler and matches
torchtitan's pattern (its single-GPU benchmarks bypass FSDP wrapping
the same way).

**Revisit trigger.** If we ever want `world_size==1` runs to *still*
exercise FSDP code paths (e.g. for catching FSDP-specific bugs in
CI), invert the policy with a config flag. Today CI's smoke test
intentionally skips the FSDP path; the 1B / 8B production runs always
have `ws >= 2`.

## D-014 — `tests/conftest.py` adds `src/` to `sys.path`

**Decision.** Pytest discovers `tests/conftest.py` first; it inserts
`<repo>/src` onto `sys.path`. We do not require `pip install -e .` to
run tests.

**Why.** Lower friction for contributors, identical behaviour to a
proper editable install.
