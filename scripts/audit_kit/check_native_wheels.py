"""Require source-free wheels and pretraining provenance before publication."""

import json
from pathlib import Path
import sys
from zipfile import ZipFile

SOURCE_SUFFIXES = {
    ".py",
    ".pyc",
    ".pyo",
    ".pyx",
    ".pxd",
    ".c",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".metal",
    ".mm",
    ".air",
}


def check(directory: Path, commit: str, repop_commit: str) -> None:
    platforms = {"pretrain": set(), "repop": set()}
    for wheel in sorted(directory.glob("*.whl")):
        package = wheel.name.split("-", 1)[0]
        if package not in platforms:
            raise ValueError(f"unexpected native wheel: {wheel.name}")
        tag = wheel.stem.rsplit("-", 3)[-3:]
        if tag[:2] != ["cp311", "cp311"] or tag[2] == "any":
            raise ValueError(f"expected a native CPython 3.11 wheel: {wheel.name}")
        family = (
            "macosx_arm64"
            if tag[2].startswith("macosx_") and tag[2].endswith("_arm64")
            else "linux_x86_64"
            if "linux" in tag[2] and tag[2].endswith("_x86_64")
            else ""
        )
        if not family or family in platforms[package]:
            raise ValueError(f"unsupported or duplicate platform: {wheel.name}")
        platforms[package].add(family)
        with ZipFile(wheel) as archive:
            for name in archive.namelist():
                if Path(name).suffix.lower() in SOURCE_SUFFIXES:
                    raise ValueError(f"source or bytecode in {wheel.name}: {name}")
            if package == "pretrain":
                stamp = json.loads(archive.read("pretrain/_native_build.json"))
                if stamp != {"commit": commit, "python_native": True}:
                    raise ValueError(
                        f"pretraining wheel provenance mismatch: {wheel.name}"
                    )
            else:
                # repop's setup.py stamps "unknown" when it cannot run git, and
                # appends "-dirty" for an unclean tree. Either way the wheel
                # cannot be tied to audited source, and the replay harness
                # rejects it on the auditor's machine rather than ours. Compare
                # the stamp here, where a rebuild is still cheap. The rest of
                # the payload is platform-specific, so only COMMIT is pinned.
                try:
                    stamp = json.loads(archive.read("repop/_build_info.json"))
                except KeyError:
                    raise ValueError(
                        f"repop wheel carries no build provenance stamp: {wheel.name}"
                    ) from None
                if stamp.get("COMMIT") != repop_commit:
                    raise ValueError(
                        f"repop wheel provenance mismatch: {wheel.name} is "
                        f"stamped {stamp.get('COMMIT')!r}, not {repop_commit}"
                    )
    if not platforms["repop"] or platforms["pretrain"] != platforms["repop"]:
        raise ValueError("pretraining and repop native platforms must match")


if __name__ == "__main__":
    check(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
