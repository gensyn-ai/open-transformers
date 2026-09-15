#!/usr/bin/env python3
"""Validate the complete local upload set against its kit manifest."""

import hashlib
import json
from pathlib import Path
import re
import sys

from check_implementation_disclosure import safe_name
from check_wheel_disclosure import require_disclosure_names, scan_bytes


def verify(root: Path, expected_identity: tuple[str, str, str]) -> None:
    manifest = json.loads((root / "kit.json").read_text())
    identity = tuple(
        manifest.get(key)
        for key in ("public_url_base", "pretrain_commit", "repop_commit")
    )
    if identity != expected_identity:
        raise ValueError("provenance manifest does not match this URL/commit pair")
    if manifest.get("kit_format") != 1:
        raise ValueError("unsupported kit format")
    if not re.fullmatch(
        r"torch==[0-9][a-zA-Z0-9.+!-]*", manifest.get("torch_requirement", "")
    ):
        raise ValueError("kit needs an exact torch_requirement")
    expected = {"kit.json"}
    for entry in manifest["files"]:
        name = entry["name"]
        if (
            not isinstance(name, str)
            or not safe_name(name)
            or "/" in name
            or name in expected
        ):
            raise ValueError(f"invalid or duplicate kit filename: {name!r}")
        if name != "trajectory.json" and not name.endswith(".whl"):
            raise ValueError(f"unexpected kit payload: {name!r}")
        expected.add(name)
        file = root / name
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"kit payload must be a regular file: {name!r}")
        blob = file.read_bytes()
        if (
            len(blob) != entry["bytes"]
            or hashlib.sha256(blob).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"provenance digest/size mismatch: {name!r}")
    if "trajectory.json" not in expected or not any(
        name.endswith(".whl") for name in expected
    ):
        raise ValueError("kit needs a trajectory and wheels")
    if {p.name for p in root.iterdir()} != expected:
        raise ValueError("kit directory contains unmanifested or missing files")
    # scan_bytes silently drops the teammate-name pattern when no list is
    # configured, and this is the last look before promotion.
    require_disclosure_names()
    for name in ("kit.json", "trajectory.json"):
        if (root / name).is_symlink():
            raise ValueError(f"kit payload must not be a symlink: {name!r}")
        findings = scan_bytes(name, (root / name).read_bytes())
        if findings:
            raise ValueError("\n".join(findings))


if __name__ == "__main__":
    try:
        if len(sys.argv) != 5:
            raise ValueError(
                "usage: verify_kit_inventory.py DIR URL PRETRAIN_COMMIT REPOP_COMMIT"
            )
        verify(Path(sys.argv[1]), tuple(sys.argv[2:]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"kit inventory check failed: {error}") from error
