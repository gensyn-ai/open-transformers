# Audit-kit volunteer runbook

This is the path for verifying a Gensyn auditable-training claim on your own
machine, with no source checkout: install two wheels, run one command, read
pass/fail. The kit is the third delivery vehicle of the RepOps packaging
strategy — the repop audit wheels, the replay harness, and published
canonical hashes, bound together by a manifest.

Apple silicon and NVIDIA are both first-class here, and that is the claim
rather than a convenience. repop's kernels are bitwise reproducible across
devices, so the same published hash has to come back from Metal, CUDA and CPU
alike. A verification on an NVIDIA box is worth exactly as much as one on a
Mac. The replay refuses to gate on GPU architecture for the same reason.

## What you need

- One of:
  - An Apple-silicon Mac on macOS 14 or newer. This is `--device mps`.
  - A Linux x86-64 machine with an NVIDIA GPU. This is `--device cuda`.

  Either way 8 GB of free memory covers the 1B init unit (measured 5.8 GB
  peak on MPS); 24 GB covers every published unit including step replays with
  the offload flags. Any of the three devices can also run `--device cpu`,
  which is the reference and needs no GPU at all, but wants more RAM: the 1B
  init unit peaked at 21.9 GB there.
- Python 3.11 (`brew install python@3.11` on macOS; your distribution's
  `python3.11` package on Linux).
- The kit URL prefix you were given, e.g.
  `https://storage.googleapis.com/gensyn-audit-public/audit-kit/pt-<sha12>_rp-<sha12>`.
  Everything below comes from that one prefix: `kit.json` (the manifest, which
  lists wheel names, sha256s, and the commit pair), the wheels, and
  `trajectory.json` (the canonical hashes).

The kit carries a repop wheel for both platforms, so one kit id serves both
kinds of volunteer. You install the one for your machine and ignore the other.

You do not need a Gensyn account, `gcloud`, or any credential. The kit is
served over plain https, and the prefix is also recorded inside `kit.json` as
`public_url_base`.

## Install

Set the prefix once:

```bash
KIT=https://storage.googleapis.com/gensyn-audit-public/audit-kit/pt-<sha12>_rp-<sha12>
```

Fetch the manifest, then the files it names, and check each digest before you
install anything:

```bash
curl -fsSL -O "$KIT/kit.json"
python3 - "$KIT" <<'PY'
import hashlib, json, os, sys, urllib.request

base = sys.argv[1]
for entry in json.load(open("kit.json"))["files"]:
    # Download under a temporary name and adopt it only once the digest
    # matches. Rejected bytes must not become an installable wheel.
    part = entry["name"] + ".part"
    urllib.request.urlretrieve(f"{base}/{entry['name']}", part)
    digest = hashlib.sha256(open(part, "rb").read()).hexdigest()
    if digest != entry["sha256"]:
        os.remove(part)
        raise SystemExit(
            f"{entry['name']}: sha256 {digest} does not match the published "
            f"{entry['sha256']}"
        )
    os.replace(part, entry["name"])
    print(f"ok  {entry['name']}")
PY
```

A mismatch means the bytes you hold are not the bytes that were published.
Stop there. Verifying them would tell you about some other artifact, not about
the claim.

Now install the repop and pretrain wheels for your machine. The kit contains one pair per
platform, so this picks the right one and sets the matching `DEVICE`, which
every command below uses. Paste the whole block into bash or zsh. The function
returns on failure without closing your shell or reporting a successful install:

```bash
install_audit_kit() {
  DEVICE=
  local selected_device WHEEL PRETRAIN_WHEEL TORCH_INDEX TORCH_REQUIREMENT
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
      selected_device=mps
      set -- ./repop-*-macosx_14_0_arm64.whl
      TORCH_INDEX=https://pypi.org/simple
      ;;
    Linux-x86_64)
      selected_device=cuda
      set -- ./repop-*-linux_x86_64.whl
      TORCH_INDEX=https://download.pytorch.org/whl/cu129
      ;;
    *) echo "no audit wheel for $(uname -s)-$(uname -m)" >&2; return 1 ;;
  esac
  if [[ $# != 1 || ! -f "$1" ]]; then
    echo "expected exactly one repop wheel for this platform" >&2
    return 1
  fi
  WHEEL="$1"
  case "$selected_device" in
    mps) set -- ./pretrain-*-cp311-cp311-macosx_*_arm64.whl ;;
    cuda) set -- ./pretrain-*-cp311-cp311-linux_x86_64.whl ;;
  esac
  if [[ $# != 1 || ! -f "$1" ]]; then
    echo "expected exactly one pretrain wheel for this platform" >&2
    return 1
  fi
  PRETRAIN_WHEEL="$1"
  TORCH_REQUIREMENT=$(python3 - <<'PY'
import json, re
requirement = json.load(open("kit.json")).get("torch_requirement", "")
if not re.fullmatch(r"torch==[0-9][a-zA-Z0-9.+!-]*", requirement):
    raise SystemExit("kit.json must contain an exact torch_requirement from its repop wheels")
print(requirement)
PY
  ) || return 1
  python3.11 -m venv audit-venv || return 1
  source audit-venv/bin/activate || return 1

  # Keep both resolutions on the same index, using the kit's exact pin.
  pip install --index-url "$TORCH_INDEX" "$TORCH_REQUIREMENT" || return 1
  pip install --index-url "$TORCH_INDEX" "$WHEEL" || return 1
  pip install "$PRETRAIN_WHEEL" || return 1
  DEVICE="$selected_device"
  echo "installed for --device $DEVICE"
}
install_audit_kit
```

The torch version comes from the kit's repop wheel metadata, not from this
runbook: older kits can require 2.10.0 while newer ones require 2.12.1. Kits
without `torch_requirement` must be republished under a new kit id; do not guess
the version. The Linux wheels covered here require the CUDA 12.9 index. Both
torch and repop installs use that index so dependency resolution cannot switch
to a different CUDA build on PyPI. Apple silicon uses PyPI.

Confirm it before going further:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# The version must match kit.json's torch_requirement (possibly with +cu129).
# On Linux torch.version.cuda must be 12.9. Otherwise stop and recreate the venv.
```

On a Linux machine with no NVIDIA GPU, set `DEVICE=cpu` by hand. CPU is the
reference the other two are compared against, so it is a real verification and
not a downgrade; it only wants more RAM.

Then confirm the kernel build is the one the kit claims, and that it can
actually serve the device you chose:

```bash
python -c "import repop, json; print(json.dumps(repop.build_info()))"
# "commit" must equal kit.json's repop_commit.
# "backends" must include "metal" for mps, or "cuda" for cuda.
```

You do not have to check that second line by eye. The replay refuses to start
on a build that cannot serve the requested device, and names the wheel you
should have installed. A CPU-only build would otherwise fall back op by op and
report a pass without ever touching the hardware the verification claims to
cover.

## Verify a published unit

Take a unit from `trajectory.json` — for an `init` unit:

```bash
pretrain-audit-replay --from-init --until-step 0 \
    --config-name <unit.config_name> --device "$DEVICE" \
    --expect-hash <unit.state_hash>
```

- **Exit 0** and `"match": true` in the JSON — you independently re-derived
  the published state hash on your hardware. That is the verification.
- **Exit 1** with `AUDIT FAILED` — your bits differ from the canonical hash.
  Check the two provenance lines first (below); if they match the kit and it
  still fails, that is a reportable reproducibility break, and the result
  JSON is the report.

The result JSON records what actually ran: `"repop"` (kernel-build commit +
compiled backends) and `"device"`. A result whose `repop.commit` differs from
the kit's `repop_commit` verified a different kernel build and proves nothing
about the published trajectory. The harness refuses `--device mps` outright if
the build has no Metal backend, and `--device cuda` if it has no CUDA backend,
so a silent CPU-only "pass" cannot happen on either.

The published hash does not depend on which of the two you ran. That is the
point of the whole exercise: repop's cross-device-reproducible kernels and the
device-independent init are expected to give the same bits on CUDA, Metal and
CPU, so the replay deliberately applies no GPU-architecture gate. If your
hardware disagrees with the published hash, the disagreement is the finding.

For a step-interval unit (`kind: "interval"` in `trajectory.json`), the same
command takes `--checkpoint` + `--gcs-root` per the unit's fields; see
`pretrain-audit-replay --help`. Interval units exist only for runs whose
checkpoints and data shards are published.

## The audit unit, and why it is sized the way it is

An **init unit** verifies that the run's initial state (seeded,
device-independent trunc-normal init + zeroed optimizer moments) reproduces
bit-for-bit. Timed on an M4 Max from a clean wheel install at repop
`244c0791e378`, torch 2.10. The repop pin has moved three times since, most
recently to the `c9ca6e71e673` this kit ships, and each repin re-verified the
units byte-identical rather than re-timing them. So read the table as sizing
guidance for what an init unit costs, not as a measurement of the wheel you
just installed:

| unit | device | wall | peak RSS |
|---|---|---|---|
| `100m_smoke_repop` init | mps | 5.0 s | 1.8 GB |
| `1b_repop_run3` init | mps | 29.7 s | 5.8 GB |
| `1b_repop_run3` init | cpu | 25.0 s | 21.9 GB |

Every digest above was byte-identical between cpu and mps.

A **step-interval unit** is one checkpoint interval (one training step at the
1B run config). The current MPS baseline for a full audited step at the real
run shape is on the order of half a day on an M4-class machine, so interval
units are sized at **one step per unit**: a volunteer verifies one interval,
not a stretch. Memory fits a 24 GB Mac with `--offload-optimizer`
`--offload-master` (the flags exist for exactly this). As kernel perf levers
land, the per-unit step count can grow; the unit size is a property of the
published trajectory, not of this harness.

The unit is sized for the slowest device it has to serve, which today is MPS.
An NVIDIA volunteer runs the same unit and should expect to finish sooner, but
this runbook does not quote a CUDA figure because none has been measured the
way the table above was. Do not read the sizing as a statement about your
hardware. It is a statement about the published trajectory.

## What the wheels disclose

The repop wheel ships compiled kernels and no source. There is no shader
source, no Triton source and no Python implementation in it, and the Metal
entry points carry generated names rather than descriptive ones, so reading an
implementation back out of the wheel takes real work.

It does not take impossible work, and this runbook will not imply otherwise. A
`.metallib` carries Apple's intermediate representation, and stock tools such
as `metal-objdump` decode every module in it. The CUDA wheel is the same story
with PTX and SASS. Shipping machine code means shipping something that can be
disassembled, and no packaging choice available to us removes that.

For a verification kit that is the right trade rather than a defect in it. You
are not being asked to run a sealed box and trust what it prints. The bytes you
install are the bytes whose sha256 `kit.json` publishes, and they are open to
whatever inspection you care to do. What the kit asks you to accept is
arithmetic you can re-derive, not code you cannot look at.
