# Apple-Silicon (MPS/Metal) audit runbook

How to run a single-device `audit_replay` of a cluster checkpoint on an
Apple-Silicon Mac, and every flag/env var it needs. This is the cross-device
gate: the MPS replay must reproduce the cluster's canonical `state_hash`
bit-for-bit (repop's cross-device-reproducible kernels + device-independent init).

## Memory requirements

The 1.6B audit's resident working set is ~48 GB (activations + bf16 grad-model +
2-gradient fold + fp32 accum), even with the offload flags below. A Mac with
less unified memory than that swaps hard and slows dramatically (a step can go
from ~13 h to multiple days). Use a machine with enough unified memory (≳ 48 GB)
for full 1.6B steps.

## 1. Build repop's Metal backend (once per machine / repop change)

repop's `setup.py` links `-lomp` with **no** `-L`/rpath for libomp, and the
prebuilt `.so` points at a dead `/opt/llvm-openmp`. So install Homebrew libomp
and bake its path in. CUTLASS is **CUDA-only** — not needed for the Metal build.
The MPP metallib (`backend/metal/int8_gemm_mpp.metallib`) is a committed binary;
no build step.

```bash
brew install libomp                       # /opt/homebrew/opt/libomp/{lib,include}
cd <repop-source-checkout>
rm -f repop/backend/*.cpython-312-darwin.so && rm -rf build   # force fresh
export CPATH=/opt/homebrew/opt/libomp/include
export LIBRARY_PATH=/opt/homebrew/opt/libomp/lib
export LDFLAGS="-Wl,-rpath,/opt/homebrew/opt/libomp/lib"
uv pip install --python <repo>/.venv/bin/python --no-build-isolation --no-deps -e .
# verify: PYTHONPATH=<repop>:src .venv/bin/python -c "from repop.backend import metal; import repop"
#         otool -L repop/backend/metal.cpython-312-darwin.so | grep omp  → /opt/homebrew/opt/libomp/...
```

## 2. Run flags — env vars

| Env var | Value | Why |
|---------|-------|-----|
| `PYTHONPATH` | `<repop>:<repo>/src` | import the intended repop + pretrain |
| `WANDB_MODE` | `disabled` | no wandb in a headless audit |
| `GOOGLE_CLOUD_PROJECT` | your GCP project id | GCS client needs a project; if the machine has ADC creds but no `gcloud`/default project, this env is REQUIRED for `--gcs-root` fetches |
| `PRETRAIN_AUDIT_EMPTY_CACHE_PER_MB` | `1` | trim the MPS allocator cache each micro-batch to curb fragmentation growth. NOTE: the code calls this a "runtime-dominating thrash" (forces a device sync + realloc per mb) — needed to avoid OOM on tight RAM, but it costs speed |
| `REPOP_INT8_MPP` | `1` to enable | Metal-4 tensor-ops int8 GEMM fast path (~2× GEMM, byte-exact + deterministic). Applies only when M,N are multiples of 64. Off by default; needs `backend/metal/int8_gemm_mpp.metallib` present. Byte-exact ⇒ does not change the reproduced hash |
| `PRETRAIN_AUDIT_MEMLOG` | `1` (diagnostic) | log host RSS + MPS current/driver allocation per phase — use to watch the footprint / diagnose swap |
| `REPOP_METAL_SHADER_DIR` | **leave UNSET** | `repop.ops` auto-defaults it to `<repop>/backend/metal` (where the `.metal` + `.metallib` live). Overriding it breaks MPP metallib resolution |
| `ulimit -n` | `65536` | ~1700 shard memmaps blow past the default 256 open-files → "too many open files" |

## 3. Run flags — `audit_replay` CLI

```
--checkpoint <dir>          the pulled cluster checkpoint (trust-gate: its
                            state_hash.txt must equal the run's published hash)
--device mps
--gcs-root  gs://<your-bucket>/pretrain-1b/data/shards   # NOT .../500B/...
--fetch-dest ./data/audit_data
--until-step <T+1>          one step past the checkpoint (or omit for one ckpt interval)
--expect-hash <canonical H_{T+1}>   verify against the run's state_hashes.jsonl
--offload-optimizer         AdamW moments → disk, streamed per-param (frees ~12 GB)
--offload-master            fp32 master → disk during micro-batch/fold (frees ~6 GB)
--offload-grads             per-micro-batch gradients → host fp32 (frees ~6 GB;
                            needed on 24 GB CUDA cards, not on a large Mac)
--save-checkpoint-dir <dir> (optional) write a hand-off checkpoint for a chained audit
```

## 4. Canonical command (MPP on)

```bash
cd <this-repo>
ulimit -n 65536
export PYTHONPATH=<repop-checkout>:src
export WANDB_MODE=disabled GOOGLE_CLOUD_PROJECT=<your-gcp-project>
export PRETRAIN_AUDIT_EMPTY_CACHE_PER_MB=1 PRETRAIN_AUDIT_MEMLOG=1 REPOP_INT8_MPP=1
unset REPOP_METAL_SHADER_DIR
nohup .venv/bin/python -m pretrain.cli.audit_replay \
  --checkpoint <pulled-checkpoint>/step_<NNNNNNNNN> \
  --device mps \
  --gcs-root gs://<your-bucket>/pretrain-1b/data/shards --fetch-dest ./data/audit_data \
  --until-step <T+1> \
  --expect-hash <canonical H_T+1> \
  --offload-optimizer --offload-master \
  --save-checkpoint-dir <handoff-dir> \
  > ~/audit_mps.log 2>&1 < /dev/null &   # detach: MPS steps run hours
```

Long audits MUST run detached (`nohup`, survives the SSH session). Result: the
log ends with a `MATCH=True/False`
line and (if `--save-checkpoint-dir`) a saved hand-off checkpoint.

## Where the expected hashes come from

Take `--expect-hash` from the run's published per-step reference hashes
(`reference_state_hashes.jsonl` in the release artifacts), never from a
checkpoint's own `state_hash.txt` — see docs/audit-replay-usage.md for the
off-cadence-checkpoint caveat.
