#!/usr/bin/env bash
# Publish the audit kit: the pretrain wheel, the matching repop audit wheels,
# a canonical-hash trajectory file, and a manifest binding them — one durable,
# commit-keyed GCS prefix a verifier installs from with no source checkout.
#
# Publication is blocked until every wheel payload has a reviewed digest in
# implementation_disclosure_policy.json. The shipped policy approves nothing.
#
# The disclosure scan below also needs the team-name list that is deliberately
# not tracked: set PRETRAIN_DISCLOSURE_NAMES or write
# scripts/audit_kit/disclosure_names.txt (see its .example) before publishing.
# Without it the scan aborts rather than run with that pattern disabled.
#
# Usage (from the repo root, on the commit to publish):
#   scripts/audit_kit/publish_audit_kit.sh \
#       --repop-commit <full-sha-of-the-repop-source-repo> \
#       --trajectory <path/to/trajectory.json> \
#       [--kit-bucket gs://gensyn-audit-artifacts] \
#       [--public-bucket gs://gensyn-audit-public] \
#       [--promote-only] \
#       [--repop-wheel <local .whl>]...
#       [--pretrain-wheel <local native .whl>]...
#
# The kit goes to two places, because durability and distribution are two
# different requirements and one bucket cannot hold both.
#
#   --kit-bucket    keeps the provenance copy. Closed, credentialed, and holding
#                   every commit-addressed artifact CI writes.
#   --public-bucket serves the verifier. World-readable, and holding only the
#                   kits that a maintainer promoted into it.
#
# The verifier is a person with no Gensyn credentials, so the public copy is the
# one the runbook cites, over https rather than gs://, which is also the only
# form pip can install from. Pass --public-bucket "" to publish the provenance
# copy alone; the kit is then unreachable by its audience, so do this only for a
# kit that is not meant to be verified yet.
# Later, use --promote-only with the same commit pair to copy the existing kit
# without rebuilding wheels or requiring --trajectory.
#
# The repop wheels are pulled from the CI publish prefix
# (gs://gensyn-audit-artifacts/repop-wheels/<commit>/, falling back to the old
# gs://gensyn-buildkite-artifacts prefix for commits published before
# the CI publish cutover) unless local wheel files are passed with --repop-wheel;
# either way they are re-published into the kit prefix so the kit is
# self-contained
# and a volunteer needs exactly one URL prefix.
#
# Objects are written with --no-clobber: the first publish of a kit id wins,
# matching the immutability rule the repop wheel publish established — a kit a
# receipt refers to must never change bytes underneath it.
set -euo pipefail

KIT_BUCKET="gs://gensyn-audit-artifacts"
PUBLIC_BUCKET="gs://gensyn-audit-public"
REPOP_COMMIT=""
TRAJECTORY=""
LOCAL_REPOP_WHEELS=()
LOCAL_PRETRAIN_WHEELS=()
PROMOTE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repop-commit) REPOP_COMMIT="$2"; shift 2 ;;
    --trajectory) TRAJECTORY="$2"; shift 2 ;;
    --kit-bucket) KIT_BUCKET="$2"; shift 2 ;;
    --public-bucket) PUBLIC_BUCKET="$2"; shift 2 ;;
    --repop-wheel) LOCAL_REPOP_WHEELS+=("$2"); shift 2 ;;
    --pretrain-wheel) LOCAL_PRETRAIN_WHEELS+=("$2"); shift 2 ;;
    --promote-only) PROMOTE_ONLY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$REPOP_COMMIT" ]] || { echo "--repop-commit is required" >&2; exit 2; }
if [[ "$PROMOTE_ONLY" == 0 ]]; then
  [[ -n "$TRAJECTORY" && -f "$TRAJECTORY" ]] || { echo "--trajectory <file> is required" >&2; exit 2; }
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
CHECKS="$REPO_ROOT/scripts/audit_kit"

check_kit() {
  local directory="$1"
  python3 "$CHECKS/verify_kit_inventory.py" \
    "$directory" "$PUBLIC_URL_BASE" "$PRETRAIN_COMMIT" "$REPOP_COMMIT"
  python3 "$CHECKS/check_native_wheels.py" \
    "$directory" "$PRETRAIN_COMMIT" "$REPOP_COMMIT"
  python3 "$CHECKS/check_wheel_disclosure.py" "$directory"/*.whl
  python3 "$CHECKS/check_implementation_disclosure.py" "$directory"/*.whl
}

# The kit is keyed by the exact code pair that produces the hashes. A dirty
# tree has no commit identity, so it cannot be published as audit evidence.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to publish from a dirty tree: the kit must be traceable to a commit" >&2
  exit 1
fi
PRETRAIN_COMMIT="$(git rev-parse HEAD)"
KIT_ID="pt-${PRETRAIN_COMMIT:0:12}_rp-${REPOP_COMMIT:0:12}"
DEST="${KIT_BUCKET}/audit-kit/${KIT_ID}"

# The same kit id in both buckets, so one id names one set of bytes wherever it
# is read from. The https form is what a verifier uses: a public GCS object is
# served at storage.googleapis.com/<bucket>/<object>, and pip installs from that
# directly, while it cannot resolve a gs:// URL at all.
PUBLIC_DEST=""
URL_BUCKET="${PUBLIC_BUCKET:-gs://gensyn-audit-public}"
PUBLIC_URL_BASE="https://storage.googleapis.com/${URL_BUCKET#gs://}/audit-kit/${KIT_ID}"
if [[ -n "$PUBLIC_BUCKET" ]]; then
  PUBLIC_DEST="${PUBLIC_BUCKET}/audit-kit/${KIT_ID}"
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

if [[ "$PROMOTE_ONLY" == 0 ]]; then
  if [[ ${#LOCAL_PRETRAIN_WHEELS[@]} -gt 0 ]]; then
    cp "${LOCAL_PRETRAIN_WHEELS[@]}" "$STAGE/"
  else
    echo "--- Building the native pretrain wheel at ${PRETRAIN_COMMIT:0:12}"
    # Build outside the staging directory and copy in only the wheel.
    # The staging directory becomes the kit, and verify_kit_inventory
    # requires its file set to equal the manifest exactly. uv writes a
    # .gitignore into --out-dir, which is enough to fail the publish with
    # "kit directory contains unmanifested or missing files". Copying the
    # wheel in keeps that true whatever a build tool leaves beside it.
    BUILD_DIR="$(mktemp -d)"
    PRETRAIN_PYTHON_NATIVE=1 uv build --python 3.11 --wheel \
      --out-dir "$BUILD_DIR" >/dev/null
    cp "$BUILD_DIR"/pretrain-*.whl "$STAGE/"
    rm -rf "$BUILD_DIR"
  fi
  ls "$STAGE"/pretrain-*.whl >/dev/null

  echo "--- Staging repop wheels for ${REPOP_COMMIT:0:12}"
  if [[ ${#LOCAL_REPOP_WHEELS[@]} -gt 0 ]]; then
    cp "${LOCAL_REPOP_WHEELS[@]}" "$STAGE/"
  else
    # Durable bucket first, then the old expiring one. repop CI published
    # wheels to the expiring bucket until the CI publish cutover; that bucket
    # deletes every object at 5 days, so this kit could only ever be built from
    # a repop commit less than 5 days old, while publishing its own output to
    # the non-expiring bucket. Wheels built after the cutover are only in the
    # durable bucket and wheels built before it are only in the old one, so the
    # fallback is what makes the kit buildable across the cutover rather than
    # for one side of it. Drop it once no pre-cutover commit needs a kit.
    gcloud storage cp \
      "gs://gensyn-audit-artifacts/repop-wheels/${REPOP_COMMIT}/*.whl" "$STAGE/" \
      || gcloud storage cp \
        "gs://gensyn-buildkite-artifacts/repop-wheels/${REPOP_COMMIT}/*.whl" "$STAGE/"
  fi
  ls "$STAGE"/repop-*.whl >/dev/null || { echo "no repop wheels staged" >&2; exit 1; }

  # Stage the trajectory with the publishing commit stamped in: the repo copy
  # carries SET-BY-PUBLISHER because the hashes cannot know the commit that
  # will publish them. Refuse a trajectory whose repop_commit disagrees with
  # the wheels being bundled — that pairing is the whole provenance claim.
  python3 - "$TRAJECTORY" "$STAGE/trajectory.json" "$PRETRAIN_COMMIT" "$REPOP_COMMIT" <<'PY'
import json, sys
src, dst, pretrain_commit, repop_commit = sys.argv[1:5]
t = json.load(open(src))
if t.get("repop_commit") != repop_commit:
    raise SystemExit(
        f"trajectory {src} was minted against repop_commit={t.get('repop_commit')!r} "
        f"but this kit bundles {repop_commit!r}; hashes and wheels must pair."
    )
t["pretrain_commit"] = pretrain_commit
json.dump(t, open(dst, "w"), indent=2)
PY

  echo "--- Writing kit manifest"
  python3 - "$STAGE" "$PRETRAIN_COMMIT" "$REPOP_COMMIT" "$PUBLIC_URL_BASE" <<'PY'
import email, hashlib, json, pathlib, re, sys, zipfile
stage, pretrain_commit, repop_commit, public_url_base = sys.argv[1:5]
stage = pathlib.Path(stage)
files = sorted(p for p in stage.iterdir() if p.suffix in (".whl", ".json"))
torch_requirements = set()
for wheel in stage.glob("repop-*.whl"):
    with zipfile.ZipFile(wheel) as archive:
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise SystemExit(f"{wheel.name}: expected one wheel METADATA file")
        requirements = email.message_from_bytes(archive.read(metadata[0])).get_all("Requires-Dist", [])
    torch = [r for r in requirements if re.match(r"^torch\b", r, re.I)]
    pin = (
        re.fullmatch(r"torch\s*\(?\s*==\s*([0-9][a-zA-Z0-9.+!-]*)\s*\)?", torch[0], re.I)
        if len(torch) == 1 else None
    )
    if pin is None:
        raise SystemExit(f"{wheel.name}: expected one unconditional exact torch== pin, got {torch}")
    torch_requirements.add(f"torch=={pin[1]}")
if len(torch_requirements) != 1:
    raise SystemExit(f"repop wheels must agree on their torch pin: {sorted(torch_requirements)}")
manifest = {
    "kit_format": 1,
    "pretrain_commit": pretrain_commit,
    "repop_commit": repop_commit,
    # Additive, so kit_format stays 1: a reader that does not know the field
    # loses nothing it had before. It carries the prefix that every name in
    # "files" resolves under, so a verifier holding only this manifest can fetch
    # the rest without being told a second URL.
    "public_url_base": public_url_base,
    "torch_requirement": torch_requirements.pop(),
    "files": [
        {
            "name": p.name,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in files
    ],
}
(stage / "kit.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

  # Before anything is written anywhere. The provenance bucket is immutable, so
  # a wheel carrying internal text cannot be taken back out of it once copied,
  # only orphaned behind a new kit id.
  echo "--- Checking the staged kit inventory and disclosure approvals"
  check_kit "$STAGE"

  echo "--- Publishing to ${DEST} (immutable: --no-clobber)"
  gcloud storage cp --no-clobber "$STAGE"/* "${DEST}/"
fi

# Read the immutable winner, never the newly built stage: a rerun may have
# skipped existing objects, and partial prior publishes may be inconsistent.
CANONICAL="$STAGE/canonical"
mkdir -p "$CANONICAL"
gcloud storage cp "${DEST}/*" "$CANONICAL/"
check_kit "$CANONICAL"

if [[ -n "$PUBLIC_DEST" ]]; then
  # Upload the validated local snapshot, including on retries and deferred
  # promotion. A fresh GCS wildcard could include new or changed remote objects.
  echo "--- Mirroring to ${PUBLIC_DEST} (world-readable, immutable)"
  gcloud storage cp --no-clobber "$CANONICAL"/* "${PUBLIC_DEST}/"

  # Read it back over plain https with no credentials, which is the path the
  # runbook tells a verifier to use. A publish that lands in the bucket but is
  # not actually readable that way has not delivered the kit, and the failure is
  # silent from the writer's side: the maintainer holds credentials that hide it.
  echo "--- Verifying the kit is fetchable with no credentials"
  curl --fail --silent --show-error --location \
    --output "$STAGE/public-kit.json" "${PUBLIC_URL_BASE}/kit.json"
  cmp "$CANONICAL/kit.json" "$STAGE/public-kit.json"
  # Also detect a conflicting wheel left by an earlier partial mirror. A
  # successful --no-clobber copy alone does not establish byte equality.
  for file in "$CANONICAL"/*; do
    name="${file##*/}"
    [[ "$name" == kit.json ]] && continue
    curl --fail --silent --show-error --location \
      --output "$STAGE/public-file" "${PUBLIC_URL_BASE}/${name}"
    cmp "$file" "$STAGE/public-file"
  done
  echo "  ok: ${PUBLIC_URL_BASE}/kit.json"
fi

cat <<EOF

Kit published.
  provenance copy: ${DEST}/kit.json
EOF

if [[ -n "$PUBLIC_DEST" ]]; then
  cat <<EOF
  volunteer copy:  ${PUBLIC_URL_BASE}/kit.json

Give a verifier that second URL (see scripts/audit_volunteer/RUNBOOK.md).
EOF
else
  cat <<EOF

No public mirror was written, so no verifier can install this kit yet.
EOF
fi
