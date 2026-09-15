# `audit_replay` usage

Single-device, bitwise reproduction of a cluster checkpoint. It rebuilds the
canonical data stream and the cluster's reduction order on one device, then
either verifies the regenerated init or replays an interval and compares the
chained `state_hash` to a recorded target.

Entry point: `src/pretrain/cli/audit_replay.py`
(`uv run python src/pretrain/cli/audit_replay.py …`).

Scope: `tp_size == 1` only. Replays require an *auditable* run
(`reduction_mode == deterministic_allgather`). HSDP (`dp_replicate > 1`) is
supported.

---

## Trust model: a checkpoint directory is code, not data

`--checkpoint` is designed to take a directory somebody else produced —
`--save-checkpoint-dir` exists so one auditor can hand the next interval to
another, and `--gcs-root` pulls from a mirror. Loading a checkpoint is
therefore not a read-only operation. **Only replay a checkpoint whose
provenance you trust** (your own run, or a dir whose bytes you have verified
against a digest you got out of band).

What we control: `Checkpointer.load` reads `rng.rank_*.pt` and `fallback.pt`
with `weights_only=True`, so a doctored file in either position fails to load
instead of running its author's code
(`tests/test_checkpoint_untrusted_load.py`).

What we do **not** control: `torch.distributed.checkpoint` reads the
`dcp/.metadata` index with a plain `pickle.load` (checked against torch
2.11.0), so a hostile `.metadata` executes arbitrary code inside `dcp.load`
before any of our code runs. There is no flag to turn that off — it is why
provenance, not file-format hardening, is the actual control here. (The
`.distcp` shards themselves are read with `weights_only=True` upstream; it is
the index that is unguarded.)

Practical rule: treat "audit this checkpoint" like "run this script". If the
dir came from outside your trust boundary, replay it in a sandbox (throwaway
container/VM, no credentials mounted) rather than on a workstation with
cluster access.

---

## The three things every check needs

1. **A run descriptor** — seed, topology (`dp_world_size/replicate/shard`),
   reduction order, `repop_env`. Lives in a checkpoint's `meta.json`. Required
   for any **replay**; optional for a pure init check (use `--config-name`).
2. **Data** — the shards the interval consumes (`--data-root` or `--gcs-root`).
3. **A comparison target** — a recorded digest. Sources, in order:
   - `--expect-hash <hex|path>` (explicit), else
   - the checkpoint's own `state_hash.txt`, **only** when `--until-step` equals
     that checkpoint's step (or `state_hash_init.txt` next to it, for init).
   - There is **no** automatic lookup in `logs/state_hashes.jsonl`.

---

## Common commands

### 1. Pure init verification (no replay, no checkpoint)

```bash
uv run python src/pretrain/cli/audit_replay.py \
    --from-init --config-name <name> --until-step 0 \
    [--expect-hash <init-digest|path/to/state_hash_init.txt>]
```

Regenerates init from the seed and compares to the recorded init hash. No
checkpoint, data, or topology needed (init is device-independent and unchained).

### 2. Init verification against a run (checkpoint present)

```bash
uv run python src/pretrain/cli/audit_replay.py \
    --checkpoint <ckpt_dir> --from-init --until-step 0
```

Auto-targets `state_hash_init.txt` sibling of the checkpoints. If that file
isn't present it does **not** fail — it just reports the regenerated digest
(unless the checkpoint is step 0, whose `state_hash.txt` *is* the init hash).

### 3. Replay an existing interval (checkpoint → checkpoint)

```bash
uv run python src/pretrain/cli/audit_replay.py \
    --checkpoint <step_N_dir> --until-step <M> \
    --gcs-root gs://…/data/shards \
    --expect-hash <path/to/step_M/state_hash.txt>
```

Loads `step_N`'s weights/optimizer and replays N→M. Omit `--expect-hash` to
replay exactly to the checkpoint's own next step and auto-compare its
`state_hash.txt`.

### 4. Init → step N replay

```bash
uv run python src/pretrain/cli/audit_replay.py \
    --checkpoint <any_ckpt_from_the_run> --from-init --until-step <N> \
    --data-root <tree with the first N steps' shards> \
    --expect-hash <step-N hash from logs/state_hashes.jsonl>
```

`--from-init` ignores the checkpoint's weights (regenerates from seed) — the
checkpoint is used **only** as the topology/reduction/env descriptor, so any
checkpoint from the same run works. The target comes from `--expect-hash`.

---

## Gotchas (learned the hard way)

- **`--from-init` + `--gcs-root` does not work.** The fetcher anchors to the
  checkpoint's step + saved stream position, not init/position 0, so it fetches
  the wrong (or zero) shards. For from-init replays, stage data and use
  `--data-root` instead. *(Fix candidate: teach `fetch_audit_interval` a
  from-init mode.)*

- **Auditing to an intermediate step needs `--expect-hash`.** Only the
  checkpoint's own step auto-resolves a target. For any other step, pull the
  digest from the run's `logs/state_hashes.jsonl` and pass it. Without it the
  audit runs but only **prints** the digest (no MATCH, exits 0).

- **A step is only checkable if the run hashed it.** A digest exists at step `k`
  only when `state_hash.every_n_steps` divides `k`. Auditing to a non-hashed
  step yields a non-chained fallback hash that matches nothing. To bisect at
  fine granularity (e.g. step 4), the run must have trained with a small
  `every_n_steps`.

- **Older FINAL / EMERGENCY checkpoints carry an unreproducible
  `state_hash.txt`.** Before the writer-side hash fix, the loop's final
  and emergency (SIGTERM) saves fired *outside* the step: the recompute ran
  after `zero_grad` (grads hashed as `grad_none`) and after the chain had
  advanced to the step's own hash (double-link), so the stamped digest matches
  **no log and no replay** — even when the replay is bit-perfect. These are
  exactly the run's off-cadence checkpoints (step not a multiple of the ckpt
  cadence). **Known published case: the 1B run's final checkpoint
  `step_000080957`** (`20260722-213626-ad3276b-resume-7`): its
  `state_hash.txt` (`95367696…`) is a write-off; the state itself verifies
  against `logs/state_hashes.jsonl` step 80957 / `meta.json`'s `chained_hash`
  (`b6567153…`) / `reference_state_hashes.jsonl`. For any such checkpoint,
  take `--expect-hash` from those references, never from its
  `state_hash.txt`. Post-fix saves always stamp the at-step hash, so this
  caveat is history-only; a FAIL against a file target now also prints this
  hint.

- **The state hash is chained.** The digest at step N folds in every prior
  hashed step, so it depends on `every_n_steps`. Checkpoints made with different
  `every_n_steps` are **not** hash-comparable even when the tensors are
  identical. Let the audit read `every_n_steps` from `meta.json` (omit
  `--config-name`); never force a `--config-name` whose `every_n_steps` differs
  from the run that wrote the checkpoint.

- **Only `clip_algo="global"` checkpoints are auditable from this tree.** A
  checkpoint recording any other clip algorithm predates the stateless
  deterministic global-norm clip; `audit_replay` refuses it loudly rather than
  mis-reproducing — audit such a checkpoint from an older checkout.

- **fp32 subnormals are the cross-device fault line — know the two floors.**
  (1) STATE: any persistent fp32 state that grinds toward zero (a retired
  per-tensor clipper's EMA state reached 7e-44) diverges MPS-vs-CUDA the first time it is multiplied
  (MPS flushes, CUDA keeps — reproduced on M4/M5). Current runs have no such
  state (global clip is stateless; Adam moments are repop-kernel-hardened).
  (2) FOLD: a gradient ELEMENT with 0 < |g| < ~1.08e-19 squares into the
  subnormal range inside the norm fold and FTZ-diverges the norm on MPS.
  Exact zeros are safe on both backends. Healthy grads sit orders of magnitude
  above this; a violation shows up loudly as a state-hash mismatch, not silent
  corruption — but if an MPS audit mismatches with no other cause, scan the
  grads for sub-1e-19 nonzero elements before suspecting kernels.

- **Don't point `--checkpoint` at a non-step-0 checkpoint for an init target.**
  The fallback to `<checkpoint>/state_hash.txt` is valid only for step-0
  checkpoints; for later checkpoints that file is the post-step chained hash and
  comparing init against it always (falsely) fails.

---

## Cold-start runs (the current `1b_repop`)

The canonical 1B run is a **cold start**: `configs/train/1b_repop.yaml`
initialises from `run.seed` at step 0 with a fresh optimizer, so there is a
single chained `state_hash` from init forward. Audit it with the ordinary
commands above — init verification, then init→step / step→step replays. The
chain has no phase boundary *until a mid-run fork introduces one* — an
earlier long run had segment boundaries at step 38,500 (`repop_env` flip) and
step 50,200 (clipper fix); see **Segment boundaries** below.

Loss and clipping are **fixed**, so nothing is selected by config or by a code
checkout:

- **Loss** — always the fused CE + z-loss (one shared softmax; z-loss is
  differentiable). There is no separate or inert z-loss variant; the same path
  runs on CUDA, CPU, and MPS.
- **Clipping** — always the stateless deterministic global-norm clip
  (`train.grad_clip`, `pretrain.train.global_clip`). `audit_replay` reproduces
  it from the checkpoint's `meta.json` (`clip_algo="global"` and the resolved
  config); there is no clipper state to load. No flags to set.

A checkpoint that predates this — a legacy per-tensor-clip / `full_tensor`-norm /
`ascending_allgather` / separate-loss run — is **no longer auditable from this
tree** (those compatibility branches were removed with the cold-start reset).
`audit_replay` fails loudly with a `clip_algo`/`replicate_reduce_algo`
assertion rather than mis-reproducing; audit such a checkpoint from an older
checkout.

### Reset-optimizer resumes (supported, but unused by the cold start)

`audit_replay` still supports auditing a run started with
`train.resume_reset_optimizer=true` — a **model-weights-only** load of the start
checkpoint plus a fresh (cold) optimizer. The current `1b_repop` cold start sets
`resume_reset_optimizer=false`, so this path is dormant. If a future run
re-enables it (e.g. to change the optimizer's param grouping mid-training, where
reusing the checkpoint's positionally-keyed DCP moments would misalign), point
`--checkpoint` at the pre-reset step and force the new config with
`--config-name`; the audit then does the model-weights-only load + fresh
optimizer to reproduce it, while `meta.json` still supplies the chained-hash
anchor and data-stream position. Note that `resume_reset_optimizer` is a config
flag and resets on *every* resume, so keep `RESUME_FROM` pinned at the intended
boundary (a crash-resume onto a later checkpoint would wrongly re-reset a warmed
optimizer).

## Segment boundaries (mid-run forks) and `--descriptor-checkpoint`

A mid-run fork that changes anything **segment-scoped** — `repop_env` (the
Hadamard flip at step 38,500), the clipper, the resolved config, topology —
splits the run into segments. One
replay carries exactly one descriptor (**one `meta.json` = one clipper/env**),
so each situation around a boundary is audited differently. The worked example
below is a fork at step 50,200: old tree `runs/<old-run>`, fixed
continuation `runs/<new-run>` (resumed from the old step-50,200 checkpoint
with a changed segment-scoped descriptor).

### 1. Entirely before the boundary — ordinary replay

```bash
uv run python src/pretrain/cli/audit_replay.py \
    --checkpoint  runs/<old-run>/checkpoints/step_000050100 \
    --until-step  50200 \
    --expect-hash runs/<old-run>/checkpoints/step_000050200/state_hash.txt
```

Keys absent from a pre-fork meta are reconstructed with their legacy
defaults, so the replay **reproduces the old segment's behavior faithfully**
— bugs included. That is
correct: the audit's job is to reproduce what ran, not what should have run.

### 2. The boundary interval — state from the old segment, descriptor from the new

```bash
uv run python src/pretrain/cli/audit_replay.py \
    --checkpoint  runs/<old-run>/checkpoints/step_000050200 \
    --descriptor-checkpoint runs/<new-run>/checkpoints/step_000050300 \
    --until-step  50300 \
    --expect-hash runs/<new-run>/checkpoints/step_000050300/state_hash.txt
```

The starting STATE (weights / moments / RNG / stream position / chained
hash) loads from the old checkpoint, but the interval was *trained* by the new
segment — `--descriptor-checkpoint` overlays the segment-scoped meta keys
(`repop_env`, `config_resolved`, topology, reduction/clip algos)
from the new segment's first checkpoint. Position keys and the chain anchor
always come from `--checkpoint`; seeds must agree (hard error otherwise).
Needed for exactly **one interval per fork**. Without the flag this replay
runs the interval under the old descriptor and mismatches on the first step —
a "divergence" that is really a
descriptor mismatch. Mind the direction: `--expect-hash` must point at the
**new** tree (the old run's own post-50,200 checkpoints will not match).


**Gotcha (found the hard way):** the descriptor overlay must carry EVERY
segment-scoped meta field, including `rewarm_anchor_tokens` — the LR re-warm
anchor is a token count but describes the segment's fork point. Before it was
added to `_SEGMENT_META_KEYS`, an Option-A boundary audit replayed the first
post-fork step at full schedule LR while the live run took it at `lr = 0`
(elapsed-from-anchor = 0), mismatching immediately. If a future feature adds
resume-anchored behavior, its anchor must be (a) recorded in meta by the loop
and (b) listed in `_SEGMENT_META_KEYS`, or boundary audits of that fork break.

### 3. Entirely after the boundary — ordinary replay again

```bash
uv run python src/pretrain/cli/audit_replay.py \
    --checkpoint  runs/<new-run>/checkpoints/step_000050300 \
    --until-step  50400 \
    --expect-hash runs/<new-run>/checkpoints/step_000050400/state_hash.txt
```

New-segment checkpoints record their own descriptor,
so no flag is ever needed again.

### 4. "Through" the boundary at fine granularity (e.g. 50,199 → 50,201)

Not possible in a single pass — the clipper (and potentially kernel-selecting
env vars) would have to hot-swap mid-replay, which is exactly the invariant a
segment boundary defines. But nothing is left unverified: the hashes are
per-step (`state_hash.every_n_steps: 1`) and **chained across the boundary**
(the fix run's 50,201 hash chains from the old run's 50,200 hash), so two
abutting passes share the 50,200 link with zero gap:

```bash
# Pass 1 (old descriptor): enter at the nearest real checkpoint ≤ target.
uv run python src/pretrain/cli/audit_replay.py \
    --checkpoint  runs/<old-run>/checkpoints/step_000050100 \
    --until-step  50200 \
    --expect-hash runs/<old-run>/checkpoints/step_000050200/state_hash.txt

# Pass 2 (stitch, one step): literal digest from the NEW run's
# logs/state_hashes.jsonl at step 50201.
uv run python src/pretrain/cli/audit_replay.py \
    --checkpoint  runs/<old-run>/checkpoints/step_000050200 \
    --descriptor-checkpoint runs/<new-run>/checkpoints/step_000050300 \
    --until-step  50201 \
    --expect-hash <step-50201 hex from the new run's logs/state_hashes.jsonl>
```

To stop pass 1 at an arbitrary hashed step (e.g. mint a resumable state at
50,199), add `--save-checkpoint-dir` and continue the chain from the hand-off
— every step is hash-due in this run. Hand-off checkpoints written during a
pass-2 stitch carry the **new** segment's descriptor (the overlay is applied
to the in-memory meta before it is copied), so the chain continues cleanly.

---

## `--save-checkpoint-dir` hand-off and the gradients sidecar

When an audited interval reaches its target with `--save-checkpoint-dir DIR`, the
audit writes a fresh, fully-loadable checkpoint (weights, optimizer moments, RNG,
stream position, `meta.json`, `state_hash.txt`) so the **next** interval can be
audited starting from it. Alongside that checkpoint it also writes
**`gradients.safetensors`** — the target step's final gradients — so a recipient
can recompute the logged hash without replaying the interval.

- **What is in it.** The target step's reduced, post-clip gradients as the
  canonical hash saw them — captured at the canonical hash point, **before**
  `optimizer.zero_grad()`. One entry per `model.named_parameters()` name, each
  `.detach().contiguous()` — dtype, shape, and bits preserved with **no cast**
  (bf16 gradients round-trip losslessly).
- **None vs. zero.** A parameter whose `grad is None` is **not** written as a
  tensor; its name is listed in the file's metadata key `none_grad_names` (a JSON
  list, encoded as a string). The metadata also records `format =
  pretrain-audit-gradients` and `format_version = 1`. A verifier must assert that
  the tensor keys ∪ `none_grad_names` account for **every** named parameter, so a
  missing/corrupt entry is never mistaken for a legitimate None gradient.
- **When it is written.** Only for a real hand-off of a target whose run hashed
  gradients (`include_grads=true`). Init-only verification and ordinary replays
  without `--save-checkpoint-dir` write nothing. The gradients are captured at
  the canonical hash point (before `zero_grad`), each device gradient released
  as its host copy is taken, avoiding a simultaneous full device/host duplicate.
  Host copies are real allocations, not deferred I/O: they remain resident
  through finalization and hash verification. After the hash gate, they are
  written once into the unpublished checkpoint directory and released BEFORE
  DCP saves the model/optimizer. Overall peak memory still depends on the device,
  allocator and finalization work; this is not a whole-run peak-neutral guarantee.
  On a hash mismatch no sidecar has been written.
  A hand-off from a run that hashed with `include_grads=false` has **no**
  sidecar (and logs a warning) — those gradients were never hashed, so there is
  nothing to export.
  Pass `--expect-hash` to verify against the published commitment: as before,
  omitting it permits an explicitly warned, unverified handoff.
- **Existing output is refused, not overwritten.** Re-running an audit into a
  `DIR` that already contains `step_<N>` hard-fails with `FileExistsError`
  (it used to overwrite silently). Delete or rename the old directory first.
- **Publication.** After verification, a hidden `.handoff-*` directory under
  `DIR` holds the entire output while it is written. The gradients, checkpoint,
  `_COMPLETE` marker and all per-rank batch files become visible together via
  one same-filesystem directory rename to `DIR/step_<N>/`. There is no separate
  gradient staging file, cross-filesystem copy, or replay-loop staging context.
  Save/write/publication exceptions (including KeyboardInterrupt) clean up the
  hidden directory. SIGKILL or power loss can leave hidden staging directories,
  but `Checkpointer.latest()` will not select them as completed handoffs.
- **Downstream use.** `pretrain-audit-verify-handoff` does it — see the next
  section. By hand it is: load the checkpoint with `Checkpointer.load` (weights +
  optimizer), reattach the gradients (`p.grad = tensors[name]`, or `None` for the
  `none_grad_names` list), then recompute `audit_shard_state_digest(…,
  include_grads=True)` + `finalize_state_hash(...)` using the saved topology,
  optimizer groups and per-rank batch chains, plus the previous published hash.
  Compare against an independently published target, not just a hash supplied
  by the checkpoint author. This verifies the hashed state, not unhashed
  RNG/stream/descriptor metadata.
  The committed CPU test (`tests/test_audit_gradient_sidecar.py`) loads the
  actual DCP handoff into fresh objects and checks v3 hash equality, including
  single-rank, replicated/sharded, and uneven-shard layouts. It also covers
  None-versus-zero, bf16, mutation detection, host-gradient release before DCP,
  and failures during sidecar writing, checkpoint saving and final publication.
  The full checkpoint/v3-hash round-trip runs on CPU but requires repop installed
  (the parallel package imports it). Run it in the audit-kit/training environment;
  `uv sync --extra dev` alone does not supply the runtime. The narrower sidecar
  serialization and publication tests can run without repop.

---

## Verifying a hand-off you were given: `pretrain-audit-verify-handoff`

Before replaying the next interval from someone else's hand-off, reconstruct
the hash the run published *for that hand-off's own step* and compare. No
forward, no backward, no data: a checkpoint load and some blake2b.

```bash
uv run pretrain-audit-verify-handoff \
    --checkpoint  handoffs/step_000050300 \
    --expect-hash <step-50300 hex from the run's logs/state_hashes.jsonl> \
    --prev-hash   <step-50299 hex from the same log>
```

Exits 0 and prints a result object on a match; exits 1 with the reason
otherwise. `--json PATH` also writes the object. `--device` defaults to `cpu`
(the digest is device-independent by construction).

- **Both hashes are yours to supply, from the published log.** Only full literal
  hex digests are accepted, not file paths; their provenance remains the caller's
  responsibility. Nothing here reads `state_hash.txt` as the target — an author who can write the tensors can
  write that file too. It is read only to fail early when the author's own claim
  already contradicts the log you passed.
- **`--prev-hash` is the previous HASHED step**, which with a cadence of N is N
  steps back — not step-1, and not the previous available checkpoint. For the
  first periodic hash of a cold-start run, pass 64 zeroes: the separate step-0
  initialization hash is NOT part of the periodic chain.
- **Off-cadence hand-offs are refused, not reinterpreted.** Their
  `state_hash.txt` is a side-link the published chain never folds in, while
  `meta.json` keeps the older running chain; computing something from the pair
  would be a different schema than the one that was published.
- **The gradients sidecar must account for every named parameter**, as a tensor
  or in `none_grad_names`, with matching dtype and shape. A missing entry is an
  error, never a legitimate `grad is None`. The sidecar format/version and
  None-name list are validated. Every rank's `batch_hasher.rank_<r>.bin` is
  likewise required and must be exactly 32 bytes.
- **Gradients are transferred to the chosen device without dtype conversion**
  and released after hashing, including failures after attachment. They are
  ~1x the model size and a continuation must not inherit them.

**Automated regression:** `tests/test_verify_handoff.py` includes a tiny real
Llama/repop checkpoint exported with the production handoff helper. It launches
`pretrain-audit-verify-handoff` in a fresh process, checks the published v3 hash,
then alters a saved gradient and requires a nonzero exit and mismatch result.
It runs on CPU and additionally on MPS when available. Install this checkout
and pytest in an environment with a matching repop/PyTorch pair (for example,
the audit kit's pinned versions). `uv sync --extra dev` alone does not install
repop. Missing repop or an uninstalled console entrypoint causes a skip; native
integration runs must install both and confirm these cases actually run.
No model-builder or verifier subprocess is stubbed in that test.

**What a match proves.** The hashed state — weights, optimizer moments and
param groups, the target step's gradients, and the running batch digest — is
bitwise the state the run committed to at that step. It says nothing about RNG,
the data-stream cursor, spike state, or the descriptor keys in `meta.json`:
none of those are in the v3 hash, and a continuation still trusts them.

**It does not make the directory safe to open.** The check runs inside a process
that has already loaded the checkpoint, and `dcp.load` unpickles its metadata
index first. `Checkpointer.load` gates that read with an allowlisting unpickler,
but an allowlist is not a proof — verify provenance and bytes *before* handing a
directory to this tool, and sandbox an untrusted one (see the trust model at the
top of this file). Safetensors for the gradients alone does not change that.

---

## Useful flags

| Flag | Purpose |
|------|---------|
| `--from-init` | Regenerate start state from the seed instead of loading the checkpoint. |
| `--until-step N` | Replay to absolute optimizer step N. `0` = init-only. |
| `--expect-hash X` | Comparison target: hex digest or path to a `state_hash*.txt`. |
| `--data-root DIR` | Read shards from an already-staged tree. |
| `--gcs-root gs://…` | Fetch only the interval's shards from a GCS mirror (not for `--from-init`). |
| `--config-name NAME` | Use this config instead of the checkpoint's `meta.config_resolved`. |
| `--descriptor-checkpoint DIR` | Overlay the SEGMENT descriptor (repop_env / config / topology / clipper) from another checkpoint's `meta.json` — for the one boundary interval after a mid-run fork (see **Segment boundaries**). |
| `--device cuda\|cpu\|mps` | Backend (arch-agnostic by design; defaults to CUDA, falls back to CPU). |
| `--offload-optimizer` | Stream AdamW moments disk↔GPU per param (large models; repop-AdamW only). |
| `--optimizer-offload-dir DIR` | Where to spill offloaded moments (default: auto temp dir). Cosmetic — controls location only. |
| `--fold-spill-dir DIR` | Spill the cross-replica gradient fold to disk. Auto-engages when `dp_replicate>2`; pass to force/relocate it. |
| `--no-fold-spill` | Keep the fold in RAM (overrides the `dp_replicate>2` auto-spill). |
| `--offload-master` | Spill the fp32 master to disk during fwd/bwd, reloading it for the step/hash; frees ~1× model size. Works wherever a separate grad model owns the fwd/bwd, which is both the MPS emulation and the CUDA `fully_shard` path, and is a no-op on a single-rank fp32 run where the master IS the compute model. Includes an `empty_cache` after the spill (without it the freed memory stays as wired MPS cache). Bitwise-identical. |
| `--master-offload-dir DIR` | Where to spill the offloaded master (default: auto temp dir). |
| `--offload-grads` | Accumulate each micro-batch's gradients on the host as fp32 and clear the device leaf, so a micro-batch does not start with the previous one's gradients resident; frees ~1x model size at peak. Costs one device-to-host copy per micro-batch. Bitwise-identical. |

### The memory plan picks these for you

You do not normally pass the three offload flags. On CUDA the audit measures
free VRAM at two points and turns them on when the card cannot hold the default
plan: `--offload-optimizer` after the model is resident and before the optimizer
exists (that choice changes the checkpoint load path, so it cannot be revisited),
and `--offload-master` plus `--offload-grads` once the grad model is placed. A
card with room takes exactly the path it took before and logs one line saying
so. Passing a flag explicitly still forces it on; the plan only ever escalates,
it never turns off something you asked for.

`PRETRAIN_AUDIT_MEM_RESERVE_GB` (default 3.0) is the headroom left for the
all-gather and the activations, the one term with no closed form. Raise it if a
card still runs out, lower it to keep more work on a card that has room to
spare.

Measured on an RTX 4090 (23.52 GiB usable) at the 1.6B audit shape, all three
are needed: a bare FSDP2 grad model is 6.43 GB and a single micro-batch peaks at
23.12 GB, so the first micro-batch fits and the second does not.

## Environment variables

On the MPS/emul path the audit **always** emulates fully_shard's
`cast_forward_inputs=True` — it bf16-casts each block's float forward inputs
(incl. the RoPE cos/sin tables) so `apply_rope` runs in bf16 like the cluster.
This is required for cross-arch BFR (without it the audit diverges at seq 4096)
and is **not configurable** — there is no env var to set or unset.

Launch-env knobs:

| Var | Default | Purpose |
|-----|---------|---------|
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | unset | Set `0.0` to disable the MPS allocator cap on tight-RAM Macs. NB: this is *not* a memory lever — it leaves the peak unchanged but ~47% faster than the default; keep `0.0`. |

Memory knobs:

| Var | Default | Purpose |
|-----|---------|---------|
| `PRETRAIN_AUDIT_MEMLOG` | `0` | `1` logs host RSS + MPS current/driver allocation at each phase boundary. |
| `PRETRAIN_AUDIT_EMPTY_CACHE_PER_MB` | unset | `1` calls `torch.mps.empty_cache()` after every micro-batch. Bounds the cross-microbatch fragmentation creep (full-audit peak ~84→~43 GB) — but the micro-batch after each trim must re-allocate + zero-fill its working set (~+60%/mb). Off by default; enable only when memory-bound. Bitwise-neutral. |

Applied **automatically from the checkpoint's `meta.json` `repop_env`** (don't set by hand — they must match the trained run): `REPOP_EXECUTION_MODE=cross_device_reproducible`, `REPOP_LSQ_STE_BWD_HADAMARD`, `REPOP_LSQ_SAVE_X_BF16`, `CUBLAS_WORKSPACE_CONFIG`, `TORCH_CUDA_ARCH_LIST`.

Debug / localization knobs (`PRETRAIN_MICROBATCH_HASH`, `PRETRAIN_PERPARAM_GRADS`, `PRETRAIN_DEEP_BWD`, `PRETRAIN_DUMP_BWD`, `REPOP_AUDIT_FORCE_EMUL`, `REPOP_AUDIT_CE_COMPARE`, …) are **not in this tree**; they lived on an internal debugging branch.

See the module docstring in `audit_replay.py` for the full reduction/grad-norm
reproduction details.
