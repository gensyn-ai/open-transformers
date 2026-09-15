import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

GATE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/audit_kit/check_native_wheels.py"
)
spec = importlib.util.spec_from_file_location("check_native_wheels", GATE_PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
COMMIT = "a" * 40
REPOP_COMMIT = "c" * 40


def wheels(
    root,
    *,
    pretrain_platform="linux_x86_64",
    commit=COMMIT,
    repop_commit=REPOP_COMMIT,
    extra=None,
):
    for package, platform in [
        ("repop", "linux_x86_64"),
        ("pretrain", pretrain_platform),
    ]:
        path = root / f"{package}-0.1.0-cp311-cp311-{platform}.whl"
        with ZipFile(path, "w") as archive:
            archive.writestr(f"{package}/__init__.cpython-311.so", b"native")
            if package == "pretrain":
                archive.writestr(
                    "pretrain/_native_build.json",
                    json.dumps({"commit": commit, "python_native": True}),
                )
                if extra:
                    archive.writestr(extra, b"source")
            else:
                archive.writestr(
                    "repop/_build_info.json",
                    json.dumps({"BACKENDS": ["cpu", "cuda"], "COMMIT": repop_commit}),
                )


def test_native_pair_passes(tmp_path):
    wheels(tmp_path)
    gate.check(tmp_path, COMMIT, REPOP_COMMIT)


@pytest.mark.parametrize(
    "extra",
    [
        "pretrain/model.py",
        "pretrain/model.pyc",
        "pretrain/kernel.cu",
        "pretrain/model.PY",
    ],
)
def test_readable_or_bytecode_payload_is_rejected(tmp_path, extra):
    wheels(tmp_path, extra=extra)
    with pytest.raises(ValueError, match="source or bytecode"):
        gate.check(tmp_path, COMMIT, REPOP_COMMIT)


def test_wrong_commit_is_rejected(tmp_path):
    wheels(tmp_path, commit="b" * 40)
    with pytest.raises(ValueError, match="provenance"):
        gate.check(tmp_path, COMMIT, REPOP_COMMIT)


def test_platform_sets_must_match(tmp_path):
    wheels(tmp_path, pretrain_platform="macosx_14_0_arm64")
    with pytest.raises(ValueError, match="platforms must match"):
        gate.check(tmp_path, COMMIT, REPOP_COMMIT)


def test_empty_kit_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="platforms must match"):
        gate.check(tmp_path, COMMIT, REPOP_COMMIT)


@pytest.mark.parametrize("stamp", ["unknown", REPOP_COMMIT + "-dirty", "c" * 39])
def test_repop_wheel_not_built_at_the_declared_commit_is_rejected(tmp_path, stamp):
    """The kit published on September 14 shipped a repop wheel stamped
    "unknown": setup.py falls back to that string when it cannot run git, and
    the build pod held an exported tree with no .git. No gate compared the
    stamp with the commit the kit declared, so it reached the public bucket and
    the audit CLI rejected it there instead."""
    wheels(tmp_path, repop_commit=stamp)
    with pytest.raises(ValueError, match="repop wheel provenance"):
        gate.check(tmp_path, COMMIT, REPOP_COMMIT)
