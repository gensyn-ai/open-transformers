#!/usr/bin/env python3
"""Refuse to publish a wheel that carries internal text.

The audit kit is mirrored into a bucket that grants ``allUsers`` read, so every
byte of every wheel in it is public. The first kits published the internal
README as ``dist-info/METADATA`` (a wheel's readme is embedded verbatim) along
with a config comment naming a machine, because nothing between editing a repo
file and mirroring it into GCS ever looks at the content.

This is that look, and it runs against the built artifact rather than the
source. Source checks are the fast feedback and live in
``tests/test_audit_kit_metadata_disclosure.py``, which imports ``DISCLOSURES``
from here so the two cannot drift; the artifact is what actually ships, and is
the only thing that catches a build embedding something the source did not
obviously say.

Usage:
    check_wheel_disclosure.py <wheel> [<wheel> ...]
    check_wheel_disclosure.py --self-test
"""

from __future__ import annotations

import os
import re
import sys
import zipfile
from pathlib import Path

# The bucket the kit is mirrored into. Any other gs:// URL in published text is
# a closed bucket the audience cannot read, and naming it only says it exists.
PUBLIC_BUCKET = "gensyn-audit-public"

# Infrastructure gets named after whoever provisioned it — "<name>-disk-1t",
# "<name>-hf-token" — so recognising one means knowing the team's names. Those
# names are themselves the identifying text this repo is scrubbed of, and a
# gate that carries its own answer key publishes exactly what it defends
# against. So the list lives outside the tree: PRETRAIN_DISCLOSURE_NAMES
# (comma- or whitespace-separated), else a gitignored ``disclosure_names.txt``
# beside this file. See ``disclosure_names.txt.example``.
#
# An absent list is a gate that silently stops looking, which is worse than the
# leak it prevents, so --self-test and every wheel scan refuse to run without
# one. Importing this module stays cheap and total: the source-side tests read
# DISCLOSURES on machines that have no list, and skip the name pattern there.
_NAMES_FILE = Path(__file__).with_name("disclosure_names.txt")


def load_disclosure_names() -> list[str]:
    """The team names to match resources against; empty when unconfigured."""
    raw = os.environ.get("PRETRAIN_DISCLOSURE_NAMES")
    if raw is None and _NAMES_FILE.is_file():
        raw = _NAMES_FILE.read_text()
    names = set()
    for line in (raw or "").splitlines():
        for token in re.split(r"[,\s]+", line.split("#", 1)[0]):
            if token:
                names.add(token.strip().lower())
    return sorted(names)


DISCLOSURE_NAMES = load_disclosure_names()

# ``(?!)`` never matches, so an unconfigured list yields a pattern that finds
# nothing rather than one that matches everything. Nothing relies on that
# silence: require_disclosure_names() is what keeps it from passing for a pass.
NAME_PATTERN = (
    r"\b(?:%s)-[a-z0-9-]+" % "|".join(re.escape(n) for n in DISCLOSURE_NAMES)
    if DISCLOSURE_NAMES
    else r"(?!)"
)


def require_disclosure_names() -> None:
    """Abort rather than scan with the teammate-name pattern disabled."""
    if DISCLOSURE_NAMES:
        return
    sys.exit(
        "no disclosure name list configured, so this scan cannot recognise a "
        "resource named after a teammate. Set PRETRAIN_DISCLOSURE_NAMES or "
        f"write {_NAMES_FILE} (see {_NAMES_FILE}.example)."
    )

# (pattern, what it would disclose, applies-to-code). Anchored on infrastructure
# nouns rather than prose, so rewording a sentence cannot keep the identifier and
# slip past.
#
# The third field is the difference between prose and code. Published
# documentation (METADATA, markdown) and shipped config data should never name
# our cluster at all, so they get the full list. Source comments legitimately
# explain runtime behaviour in the same words: train/loop.py describes a SIGTERM
# arriving from `kubectl delete`, and fetch_interval.py documents its argument as
# `gs://bucket/prefix`. Those are engineering prose about how the code behaves,
# not a description of our infrastructure, and a gate that fails on them is a
# gate someone switches off. What is never acceptable in code either is a real
# machine, a real secret name or a real bucket.
_ALL, _CODE_TOO = False, True
DISCLOSURES: list[tuple[str, str, bool]] = [
    (NAME_PATTERN, "a resource named after a teammate", _CODE_TOO),
    (r"[a-z0-9-]*-hf-token", "the name of the Hugging Face token secret", _CODE_TOO),
    # `gs://bucket` and `gs://bucket/prefix` are the placeholders our own help
    # text uses; a real bucket name is what matters.
    (
        rf"gs://(?!{re.escape(PUBLIC_BUCKET)}\b|bucket\b)[a-z0-9-]+",
        "a closed bucket",
        _CODE_TOO,
    ),
    (r"\bkubectl\b", "internal cluster operations", _ALL),
    (r"/workspace/[a-z]", "the PVC checkout path a training pod mounts", _ALL),
    (r"\bprep_1t_[a-z]+\.yaml", "internal data-prep job manifests", _ALL),
    # A shipped kernel cited the branch it was copied from, by its author's
    # name. `origin/main` is fine; `origin/<person>/<topic>` names an individual
    # and a ref that only exists in a repo the reader cannot see.
    (
        r"\borigin/(?!main\b|master\b|HEAD\b)[a-z0-9._-]+/[a-z0-9._/-]+",
        "a personal branch in a repo the reader cannot fetch",
        _CODE_TOO,
    ),
    (
        r"github\.com/gensyn-ai/[a-z0-9._-]+",
        "a private repository, as a link that resolves only for us",
        _CODE_TOO,
    ),
    # Our monorepo's directory name. It reached a wheel inside a hard-coded
    # `$HOME/gensyn/node/python/ree-workspace/repop` shader search root, which
    # described exactly one laptop and matched no verifier's machine.
    (r"\bree-workspace\b", "the internal source tree layout", _CODE_TOO),
]

# Suffixes worth reading inside a wheel. Compiled extensions are excluded on
# purpose: their build paths are a separate problem, fixed at compile time with
# -ffile-prefix-map in the node repo, and grepping a .so here would report
# every vendored third-party string as ours.
#
# .metal and .h are here because the repop wheel ships 24 of them: the Metal
# backend compiles its shaders from source at runtime
# (newLibraryWithSource:), so those files are not build leftovers, they are
# the kernel. They are the largest body of human-written text in the kit and
# the first version of this check never opened one.
_TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".cfg", ".toml", ".json"}
_CODE_SUFFIXES = {".py", ".metal", ".h"}


def _is_text_member(name: str) -> bool:
    suffix = Path(name).suffix
    return (
        name.endswith("METADATA")
        or suffix in _TEXT_SUFFIXES
        or suffix in _CODE_SUFFIXES
    )


def scan_bytes(name: str, blob: bytes, *, is_code: bool = False) -> list[str]:
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return []
    findings = []
    for pattern, discloses, code_too in DISCLOSURES:
        if is_code and not code_too:
            continue
        for hit in re.finditer(pattern, text, re.IGNORECASE):
            findings.append(f"{name}: {hit.group(0)!r} discloses {discloses}")
    return findings


def scan_wheel(wheel: Path) -> list[str]:
    findings: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if _is_text_member(member):
                findings.extend(
                    scan_bytes(
                        f"{wheel.name}:{member}",
                        archive.read(member),
                        is_code=Path(member).suffix in _CODE_SUFFIXES,
                    )
                )
    return findings


def _self_test() -> int:
    """A red control: prove each pattern still fires on text it was drawn from.

    Without this the list can rot into a set of patterns that match nothing, and
    a publish sails through because the check had nothing left to find.
    """
    require_disclosure_names()
    samples = {
        # Built from the external list rather than stored here, so the control
        # stays honest without the repo holding a name or a real machine.
        NAME_PATTERN: f"mounted on {DISCLOSURE_NAMES[0]}-scratch-01 today",
        r"[a-z0-9-]*-hf-token": "secret cluster-hf-token (key=token)",
        r"\bkubectl\b": "run kubectl apply -f job.yaml",
        r"/workspace/[a-z]": "cloned at /workspace/transformer-pretraining",
        r"\bprep_1t_[a-z]+\.yaml": "see prep_1t_stack.yaml for the fanout",
        rf"gs://(?!{re.escape(PUBLIC_BUCKET)}\b|bucket\b)[a-z0-9-]+": "gs://gensyn-audit-artifacts/x",
        r"\borigin/(?!main\b|master\b|HEAD\b)[a-z0-9._-]+/[a-z0-9._/-]+": (
            "VERBATIM from origin/someone/metal-randomness."
        ),
        r"github\.com/gensyn-ai/[a-z0-9._-]+": (
            "TODO: see https://github.com/gensyn-ai/node/issues/2151"
        ),
        r"\bree-workspace\b": "$HOME/gensyn/node/python/ree-workspace/repop",
    }
    failures = []
    for pattern, _, _code_too in DISCLOSURES:
        sample = samples.get(pattern)
        if sample is None:
            failures.append(f"no self-test sample for pattern {pattern!r}")
        elif not re.search(pattern, sample, re.IGNORECASE):
            failures.append(f"pattern {pattern!r} no longer matches its sample")
    # And prove the public bucket is not itself reported, or every kit trips it.
    if scan_bytes("x", f"gs://{PUBLIC_BUCKET}/audit-kit/".encode()):
        failures.append("the public bucket is being reported as a disclosure")
    # origin/main is the ordinary way to name our own trunk and says nothing
    # about a person; a pattern that also caught it would be switched off.
    if scan_bytes("x", b"rebased onto origin/main before the run"):
        failures.append("origin/main is being reported as a personal branch")
    for line in failures:
        print(f"SELF-TEST FAIL: {line}", file=sys.stderr)
    if failures:
        return 1
    print(f"self-test ok: {len(DISCLOSURES)} patterns fire, public bucket allowed")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return _self_test()
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    require_disclosure_names()
    findings: list[str] = []
    for arg in argv:
        wheel = Path(arg)
        if not wheel.is_file():
            print(f"no such wheel: {wheel}", file=sys.stderr)
            return 2
        findings.extend(scan_wheel(wheel))
    if findings:
        print(
            "refusing to publish: these wheels carry internal text, and the kit "
            f"is mirrored into {PUBLIC_BUCKET} where anyone can read it.",
            file=sys.stderr,
        )
        for line in findings:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"disclosure check ok: {len(argv)} wheel(s) carry no internal identifiers")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
