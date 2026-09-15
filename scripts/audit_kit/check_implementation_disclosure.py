#!/usr/bin/env python3
"""Require explicit, content-addressed disclosure approval for wheel payloads.

This is a publication inventory, not an obfuscator or a proof that native code
cannot be reverse engineered. No wheel code is imported or executed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zipfile


POLICY = Path(__file__).with_name("implementation_disclosure_policy.json")
SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".pyc",
    ".pyo",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".m",
    ".mm",
    ".metal",
    ".ptx",
    ".ll",
}
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd", ".a", ".o", ".cubin", ".metallib"}
# Members whose exact bytes another gate already fixes, so a digest approval
# here would add no coverage and cannot be satisfied. This stamp records the
# publishing commit, and this policy is versioned in the same repository, so
# approving its digest would require the policy to contain the hash of a file
# containing the hash of the commit that contains the policy. That fixed point
# does not exist. check_native_wheels.py asserts this member byte for byte
# against the publishing commit, which is the stronger check of the two: it
# pins the content rather than merely recording that someone approved it.
#
# Keyed by distribution, because that assertion only runs for the pretraining
# wheel. The same path inside a repop wheel is read by nothing, so exempting it
# there would leave arbitrary bytes unreviewed behind a trusted name.
COMMIT_STAMPED = {"pretrain": {"pretrain/_native_build.json"}}


def safe_name(name: str) -> bool:
    return (
        bool(name)
        and not name.startswith("/")
        and "\\" not in name
        and all(part not in {"", ".", ".."} for part in name.split("/"))
        and not any(ord(c) < 32 or ord(c) == 127 for c in name)
    )


def load_policy(path: Path = POLICY) -> dict[str, set[str]]:
    data = json.loads(path.read_text())
    if set(data) != {"format", "approved_members"} or data["format"] != 1:
        raise ValueError("unsupported implementation disclosure policy")
    members = data["approved_members"]
    if not isinstance(members, dict):
        raise ValueError("approved_members must be an object")
    approved = {}
    for name, entries in members.items():
        if not safe_name(name) or any(c in name for c in "*?["):
            raise ValueError(f"approval must name an exact wheel member: {name!r}")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"approval needs a nonempty digest list: {name!r}")
        digests = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"sha256", "reason"}:
                raise ValueError(f"approval needs sha256 and reason: {name!r}")
            digest, reason = entry["sha256"], entry["reason"]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid approval digest: {name!r}")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"approval needs a disclosure rationale: {name!r}")
            digests.add(digest)
        approved[name] = digests
    return approved


def payload_kind(name: str, blob: bytes) -> str:
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in SOURCE_SUFFIXES:
        return "source or bytecode"
    if suffix in NATIVE_SUFFIXES or blob.startswith(b"\x7fELF"):
        return "native implementation"
    return "metadata or data"


def inspect_wheel(wheel: Path, approved: dict[str, set[str]]) -> dict:
    raw = wheel.read_bytes()
    result = {
        "wheel": wheel.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "members": [],
        "findings": [],
    }
    findings = result["findings"]
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.comment:
            findings.append("unreviewed ZIP archive comment")
        if not raw.startswith(b"PK\x03\x04") or raw[-22:-18] != b"PK\x05\x06":
            findings.append("unsupported ZIP envelope or data outside the archive")
        files = {}
        seen = set()
        for member in archive.infolist():
            name = member.filename
            normalized = name[:-1] if member.is_dir() else name
            if not safe_name(normalized) or normalized in seen:
                findings.append(f"unsafe or duplicate wheel member: {name!r}")
                continue
            seen.add(normalized)
            mode = member.external_attr >> 16
            local_extra_size = struct.unpack_from("<H", raw, member.header_offset + 28)[
                0
            ]
            if stat.S_ISLNK(mode) or member.comment or member.extra or local_extra_size:
                findings.append(f"symlink or unreviewed ZIP member metadata: {name!r}")
            if not member.is_dir():
                files[name] = archive.read(member)
        records = [name for name in files if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            findings.append("expected exactly one wheel RECORD")
            record = None
        else:
            record = records[0]
            try:
                rows = list(
                    csv.reader(io.StringIO(files[record].decode("utf-8")), strict=True)
                )
                entries = {}
                for row in rows:
                    if len(row) != 3 or row[0] in entries:
                        raise ValueError("malformed or duplicate RECORD row")
                    entries[row[0]] = row[1:]
                if set(entries) != set(files):
                    raise ValueError("RECORD does not cover exactly the wheel payload")
                for name, blob in files.items():
                    expected = (
                        ["", ""]
                        if name == record
                        else [
                            "sha256="
                            + base64.urlsafe_b64encode(hashlib.sha256(blob).digest())
                            .decode()
                            .rstrip("="),
                            str(len(blob)),
                        ]
                    )
                    if entries[name] != expected:
                        raise ValueError(f"RECORD digest/size mismatch: {name!r}")
            except (UnicodeError, csv.Error, ValueError) as error:
                findings.append(str(error))
        for name, blob in files.items():
            if name == record:
                continue
            digest = hashlib.sha256(blob).hexdigest()
            kind = payload_kind(name, blob)
            exempt = name in COMMIT_STAMPED.get(wheel.name.split("-", 1)[0], ())
            accepted = exempt or digest in approved.get(name, set())
            result["members"].append(
                {
                    "name": name,
                    "sha256": digest,
                    "bytes": len(blob),
                    "kind": kind,
                    "approved": accepted,
                    "exempt": exempt,
                }
            )
            if not accepted:
                findings.append(f"unapproved {kind}: {name!r} (sha256={digest})")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", type=Path, nargs="+")
    parser.add_argument(
        "--json", type=Path, help="write the inventory even when rejected"
    )
    args = parser.parse_args()
    try:
        approved = load_policy()
        reports = [inspect_wheel(wheel, approved) for wheel in args.wheels]
    except (
        OSError,
        ValueError,
        TypeError,
        struct.error,
        zipfile.BadZipFile,
        RuntimeError,
    ) as error:
        parser.exit(1, f"implementation disclosure check failed: {error}\n")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        print(f"{report['wheel']}: {len(report['findings'])} disclosure finding(s)")
        for finding in report["findings"]:
            print(f"  {finding}")
    return int(any(report["findings"] for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
