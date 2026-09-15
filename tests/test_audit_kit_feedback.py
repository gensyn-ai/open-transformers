"""Execute the documented installer and publisher with local command doubles."""

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / "scripts/audit_kit/publish_audit_kit.sh"
RUNBOOK = ROOT / "scripts/audit_volunteer/RUNBOOK.md"
REPOP_COMMIT = "b" * 40
KIT_ID = f"pt-{'a' * 12}_rp-{'b' * 12}"
PRETRAIN_WHEEL = "pretrain-1.0-cp311-cp311-linux_x86_64.whl"
REPOP_WHEEL = "repop-1.0-cp311-cp311-linux_x86_64.whl"


def executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


def wheel(path, requirement="torch==2.10.0", repop_commit=REPOP_COMMIT):
    members = {
        "repop-1.0.dist-info/METADATA": (
            f"Metadata-Version: 2.1\nName: repop\nVersion: 1.0\n"
            f"Requires-Dist: {requirement}\n"
        ).encode(),
        "repop/_build_info.json": json.dumps(
            {"BACKENDS": ["cpu", "cuda"], "COMMIT": repop_commit}
        ).encode(),
    }
    record = "repop-1.0.dist-info/RECORD"
    rows = io.StringIO()
    writer = csv.writer(rows, lineterminator="\n")
    for name, blob in members.items():
        writer.writerow(
            [
                name,
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(blob).digest())
                .decode()
                .rstrip("="),
                len(blob),
            ]
        )
    writer.writerow([record, "", ""])
    with zipfile.ZipFile(path, "w") as archive:
        for name, blob in members.items():
            archive.writestr(name, blob)
        archive.writestr(record, rows.getvalue())


@pytest.fixture
def commands(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(
        bin_dir / "uname",
        """
import os, sys
print(os.environ['PLATFORM'].split('-')[0 if sys.argv[1] == '-s' else 1])
""",
    )
    executable(
        bin_dir / "pip",
        """
import json, os, sys
from pathlib import Path
p = Path(os.environ['COMMAND_LOG'])
with p.open('a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
sys.exit(int(os.environ.get('PIP_FAILURE', '0')))
""",
    )
    executable(
        bin_dir / "python3.11",
        """
from pathlib import Path
p = Path('audit-venv/bin')
p.mkdir(parents=True, exist_ok=True)
(p / 'activate').write_text(':\\n')
""",
    )
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "COMMAND_LOG": str(tmp_path / "commands.jsonl"),
    }


def install_block():
    blocks = re.findall(r"```bash\n(.*?)```", RUNBOOK.read_text(), re.S)
    return next(block for block in blocks if "python3.11 -m venv" in block)


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize(
    "platform,device,suffix",
    [
        ("Darwin-arm64", "mps", "macosx_14_0_arm64"),
        ("Linux-x86_64", "cuda", "linux_x86_64"),
    ],
)
@pytest.mark.parametrize("pin", ["torch==2.10.0", "torch==2.12.1"])
def test_installer_uses_kit_pin_and_resolved_wheel(
    tmp_path, commands, shell, platform, device, suffix, pin
):
    if not shutil.which(shell):
        pytest.skip(f"{shell} is unavailable")
    for plat in ("macosx_14_0_arm64", "linux_x86_64"):
        wheel(tmp_path / f"repop-1.0-{plat}.whl", pin)
        (tmp_path / f"pretrain-1.0-cp311-cp311-{plat}.whl").touch()
    (tmp_path / "kit.json").write_text(json.dumps({"torch_requirement": pin}))
    result = subprocess.run(
        [shell, "-c", install_block()],
        cwd=tmp_path,
        env={**commands, "PLATFORM": platform},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [
        json.loads(line)
        for line in Path(commands["COMMAND_LOG"]).read_text().splitlines()
    ]
    index = (
        "https://pypi.org/simple"
        if device == "mps"
        else "https://download.pytorch.org/whl/cu129"
    )
    assert calls[0] == ["install", "--index-url", index, pin]
    assert calls[1] == ["install", "--index-url", index, f"./repop-1.0-{suffix}.whl"]
    assert calls[2] == ["install", f"./pretrain-1.0-cp311-cp311-{suffix}.whl"]
    assert "installed for --device " + device in result.stdout


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("platform", ["Linux-aarch64", "Darwin-x86_64"])
def test_unsupported_platform_does_not_install_or_close_shell(
    tmp_path, commands, shell, platform
):
    if not shutil.which(shell):
        pytest.skip(f"{shell} is unavailable")
    (tmp_path / "pretrain-1.0-py3-none-any.whl").touch()
    result = subprocess.run(
        [shell, "-c", "DEVICE=stale\n" + install_block() + "\necho shell-alive"],
        cwd=tmp_path,
        env={**commands, "PLATFORM": platform},
        capture_output=True,
        text=True,
    )
    assert "no audit wheel" in result.stderr
    assert "shell-alive" in result.stdout
    assert "installed for" not in result.stdout
    assert not Path(commands["COMMAND_LOG"]).exists()


@pytest.fixture
def publisher(tmp_path, commands):
    # These doubles copy actual bytes and model --no-clobber, rather than just
    # grepping a command string. Nothing contacts GCS or builds real wheels.
    bin_dir = tmp_path / "bin"
    program = r"""
import base64, csv, glob, hashlib, io, json, os, shutil, sys, zipfile
from pathlib import Path
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
root = Path(os.environ['FAKE_ROOT'])
def resolve(value):
    if value.startswith('gs://'):
        return str(root / 'gcs' / value[5:])
    if value.startswith('https://storage.googleapis.com/'):
        return str(root / 'gcs' / value.removeprefix('https://storage.googleapis.com/'))
    return value
if tool == 'git':
    if args == ['rev-parse', '--show-toplevel']: print(root)
    elif args == ['rev-parse', 'HEAD']: print('a' * 40)
    elif args != ['status', '--porcelain']: sys.exit(2)
elif tool == 'uv':
    if os.environ.get('FORBID_BUILD'): sys.exit('unexpected rebuild')
    out = Path(args[args.index('--out-dir') + 1])
    payload = os.environ.get('BUILD_TAG', 'first').encode()
    name = 'pretrain-1.0.dist-info/METADATA'
    record = 'pretrain-1.0.dist-info/RECORD'
    rows = io.StringIO()
    writer = csv.writer(rows, lineterminator='\n')
    writer.writerow([name, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip('='), len(payload)])
    stamp_name = 'pretrain/_native_build.json'
    stamp = json.dumps({'commit': os.environ.get('PRETRAIN_STAMP', 'a' * 40), 'python_native': True}).encode()
    writer.writerow([stamp_name, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(stamp).digest()).decode().rstrip('='), len(stamp)])
    writer.writerow([record, '', ''])
    platform = os.environ.get('PRETRAIN_PLATFORM', 'linux_x86_64')
    with zipfile.ZipFile(out / f'pretrain-1.0-cp311-cp311-{platform}.whl', 'w') as archive:
        archive.writestr(name, payload)
        archive.writestr(stamp_name, stamp)
        archive.writestr(record, rows.getvalue())
elif tool == 'gcloud':
    if args[:2] == ['storage', 'ls']:
        paths = glob.glob(resolve(args[2]).rstrip('/') + '/*')
        print('\n'.join(paths))
        sys.exit(0 if paths else 1)
    assert args[:2] == ['storage', 'cp'], args
    no_clobber = '--no-clobber' in args
    paths = [a for a in args[2:] if a != '--no-clobber']
    dest = Path(resolve(paths[-1]))
    for pattern in paths[:-1]:
        matches = glob.glob(resolve(pattern))
        if not matches: sys.exit('missing source: ' + pattern)
        for source in matches:
            target = dest / Path(source).name if paths[-1].endswith('/') else dest
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (no_clobber and target.exists()): shutil.copyfile(source, target)
    if os.environ.get('MUTATE_AFTER_DOWNLOAD') and dest.name == 'canonical':
        source = Path(resolve(paths[0])).parent
        (source / 'private.cu').write_text('kernel implementation')
elif tool == 'curl':
    data = Path(resolve(args[-1])).read_bytes()
    if '--output' in args: Path(args[args.index('--output') + 1]).write_bytes(data)
    else: sys.stdout.buffer.write(data)
"""
    for name in ("git", "uv", "gcloud", "curl"):
        executable(bin_dir / name, program)
    local_wheel = tmp_path / REPOP_WHEEL
    wheel(local_wheel)
    checks = tmp_path / "scripts/audit_kit"
    checks.mkdir(parents=True)
    for name in (
        "check_wheel_disclosure.py",
        "check_implementation_disclosure.py",
        "verify_kit_inventory.py",
        "check_native_wheels.py",
    ):
        shutil.copyfile(ROOT / "scripts/audit_kit" / name, checks / name)
    # The real team-name list is untracked on purpose (it is the identifying
    # text the kit is scrubbed of), and check_wheel_disclosure.py aborts rather
    # than scan without one. Synthetic names keep that pattern live here.
    (checks / "disclosure_names.txt").write_text("ada\ngrace\nalan\n")
    with zipfile.ZipFile(local_wheel) as archive:
        metadata = archive.read("repop-1.0.dist-info/METADATA")
        build_info = archive.read("repop/_build_info.json")
    approved = {
        "repop-1.0.dist-info/METADATA": [metadata],
        "repop/_build_info.json": [build_info],
        "pretrain-1.0.dist-info/METADATA": [b"first", b"different rebuild"],
        "pretrain/_native_build.json": [
            json.dumps({"commit": "a" * 40, "python_native": True}).encode()
        ],
    }
    (checks / "implementation_disclosure_policy.json").write_text(
        json.dumps(
            {
                "format": 1,
                "approved_members": {
                    name: [
                        {
                            "sha256": hashlib.sha256(blob).hexdigest(),
                            "reason": "Synthetic test metadata",
                        }
                        for blob in blobs
                    ]
                    for name, blobs in approved.items()
                },
            }
        )
    )
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(json.dumps({"repop_commit": REPOP_COMMIT}))

    def run(*extra, **env):
        inputs = (
            []
            if "--promote-only" in extra
            else ["--trajectory", str(trajectory), "--repop-wheel", str(local_wheel)]
        )
        return subprocess.run(
            [
                "bash",
                str(PUBLISHER),
                "--repop-commit",
                REPOP_COMMIT,
                *inputs,
                *extra,
            ],
            cwd=tmp_path,
            env={**commands, "FAKE_ROOT": str(tmp_path), **env},
            capture_output=True,
            text=True,
        )

    return run


def kit_path(tmp_path, bucket):
    return tmp_path / "gcs" / bucket / "audit-kit" / KIT_ID


def test_deferred_publish_stamps_pin_and_future_url(tmp_path, publisher):
    result = publisher("--public-bucket", "")
    assert result.returncode == 0, result.stderr
    provenance = kit_path(tmp_path, "gensyn-audit-artifacts")
    manifest = json.loads((provenance / "kit.json").read_text())
    assert (
        manifest["public_url_base"]
        == f"https://storage.googleapis.com/gensyn-audit-public/audit-kit/{KIT_ID}"
    )
    assert manifest["torch_requirement"] == "torch==2.10.0"
    assert not kit_path(tmp_path, "gensyn-audit-public").exists()


@pytest.mark.parametrize(
    "env, message",
    [
        ({"PRETRAIN_STAMP": "c" * 40}, "provenance mismatch"),
        ({"PRETRAIN_PLATFORM": "macosx_14_0_arm64"}, "platforms must match"),
    ],
)
def test_publisher_rejects_wrong_native_build_before_upload(
    tmp_path, publisher, env, message
):
    result = publisher(**env)
    assert result.returncode != 0
    assert message in result.stderr
    assert not kit_path(tmp_path, "gensyn-audit-artifacts").exists()
    assert not kit_path(tmp_path, "gensyn-audit-public").exists()


def test_republish_mirrors_provenance_not_fresh_stage(tmp_path, publisher):
    first = publisher("--public-bucket", "")
    assert first.returncode == 0, first.stderr
    second = publisher(BUILD_TAG="different rebuild")
    assert second.returncode == 0, second.stderr
    provenance = kit_path(tmp_path, "gensyn-audit-artifacts")
    public = kit_path(tmp_path, "gensyn-audit-public")
    assert {p.name: p.read_bytes() for p in public.iterdir()} == {
        p.name: p.read_bytes() for p in provenance.iterdir()
    }


def test_promotion_does_not_rebuild(tmp_path, publisher):
    first = publisher("--public-bucket", "")
    assert first.returncode == 0, first.stderr
    result = publisher("--promote-only", FORBID_BUILD="1")
    assert result.returncode == 0, result.stderr
    assert (kit_path(tmp_path, "gensyn-audit-public") / "kit.json").is_file()


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("failure", ["pip", "missing-pin", "multiple-wheels"])
def test_installer_stops_on_failure(tmp_path, commands, shell, failure):
    if not shutil.which(shell):
        pytest.skip(f"{shell} is unavailable")
    wheel(tmp_path / "repop-1.0-macosx_14_0_arm64.whl")
    (tmp_path / "pretrain-1.0-cp311-cp311-macosx_14_0_arm64.whl").touch()
    if failure == "multiple-wheels":
        wheel(tmp_path / "repop-2.0-macosx_14_0_arm64.whl")
    manifest = (
        {} if failure == "missing-pin" else {"torch_requirement": "torch==2.10.0"}
    )
    (tmp_path / "kit.json").write_text(json.dumps(manifest))
    result = subprocess.run(
        [shell, "-c", install_block()],
        cwd=tmp_path,
        env={**commands, "PLATFORM": "Darwin-arm64", "PIP_FAILURE": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "installed for" not in result.stdout
    log = Path(commands["COMMAND_LOG"])
    if failure == "pip":
        assert len(log.read_text().splitlines()) == 1
    else:
        assert not log.exists()


@pytest.mark.parametrize(
    "requirement",
    [
        "numpy==1.0",
        "torch>=2.10",
        "torch==2.10.*",
        "torch==2.10.0; sys_platform == 'linux'",
    ],
)
def test_publisher_rejects_non_exact_torch_pin(tmp_path, publisher, requirement):
    wheel(tmp_path / REPOP_WHEEL, requirement)
    result = publisher("--public-bucket", "")
    assert result.returncode != 0
    assert "exact torch== pin" in result.stderr
    assert not kit_path(tmp_path, "gensyn-audit-artifacts").exists()


def test_publisher_rejects_conflicting_platform_pins(tmp_path, publisher):
    other = tmp_path / "repop-1.0-cp311-cp311-macosx_14_0_arm64.whl"
    wheel(other, "torch==2.12.1")
    result = publisher("--repop-wheel", str(other))
    assert result.returncode != 0
    assert "agree on their torch pin" in result.stderr


@pytest.mark.parametrize("name", ["kit.json", PRETRAIN_WHEEL])
def test_publisher_detects_existing_public_conflict(tmp_path, publisher, name):
    result = publisher("--public-bucket", "")
    assert result.returncode == 0, result.stderr
    public = kit_path(tmp_path, "gensyn-audit-public")
    public.mkdir(parents=True)
    (public / name).write_bytes(b"conflicting bytes")
    result = publisher("--promote-only")
    assert result.returncode != 0
    assert (public / name).read_bytes() == b"conflicting bytes"
    assert "Kit published." not in result.stdout


def test_publisher_detects_partial_provenance_conflict(tmp_path, publisher):
    result = publisher("--public-bucket", "")
    assert result.returncode == 0, result.stderr
    provenance = kit_path(tmp_path, "gensyn-audit-artifacts")
    (provenance / PRETRAIN_WHEEL).write_bytes(b"conflict")
    result = publisher("--promote-only")
    assert result.returncode != 0
    assert "provenance digest/size mismatch" in result.stderr
    assert not kit_path(tmp_path, "gensyn-audit-public").exists()


def test_alternate_public_bucket(tmp_path, publisher):
    result = publisher("--public-bucket", "gs://alternate-public")
    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (kit_path(tmp_path, "alternate-public") / "kit.json").read_text()
    )
    assert (
        manifest["public_url_base"]
        == f"https://storage.googleapis.com/alternate-public/audit-kit/{KIT_ID}"
    )


def test_empty_disclosure_policy_blocks_before_any_upload(tmp_path, publisher):
    policy = tmp_path / "scripts/audit_kit/implementation_disclosure_policy.json"
    policy.write_text(json.dumps({"format": 1, "approved_members": {}}))
    result = publisher()
    assert result.returncode != 0
    assert "unapproved" in result.stdout
    assert not kit_path(tmp_path, "gensyn-audit-artifacts").exists()
    assert not kit_path(tmp_path, "gensyn-audit-public").exists()


def test_promote_only_rechecks_disclosure_policy(tmp_path, publisher):
    assert publisher("--public-bucket", "").returncode == 0
    policy = tmp_path / "scripts/audit_kit/implementation_disclosure_policy.json"
    policy.write_text(json.dumps({"format": 1, "approved_members": {}}))
    result = publisher("--promote-only")
    assert result.returncode != 0
    assert "unapproved" in result.stdout
    assert not kit_path(tmp_path, "gensyn-audit-public").exists()


def test_unmanifested_object_is_not_promoted(tmp_path, publisher):
    assert publisher("--public-bucket", "").returncode == 0
    (kit_path(tmp_path, "gensyn-audit-artifacts") / "private.cu").write_text(
        "kernel implementation"
    )
    result = publisher("--promote-only")
    assert result.returncode != 0
    assert "unmanifested" in result.stderr
    assert not kit_path(tmp_path, "gensyn-audit-public").exists()


def test_remote_changes_after_inspection_are_not_copied(tmp_path, publisher):
    assert publisher("--public-bucket", "").returncode == 0
    result = publisher("--promote-only", MUTATE_AFTER_DOWNLOAD="1")
    assert result.returncode == 0, result.stderr
    assert (kit_path(tmp_path, "gensyn-audit-artifacts") / "private.cu").exists()
    assert not (kit_path(tmp_path, "gensyn-audit-public") / "private.cu").exists()


@pytest.mark.parametrize("mutation", ["duplicate", "size", "trajectory-comment"])
def test_invalid_manifest_or_private_sidecar_cannot_be_promoted(
    tmp_path, publisher, mutation
):
    assert publisher("--public-bucket", "").returncode == 0
    provenance = kit_path(tmp_path, "gensyn-audit-artifacts")
    manifest = json.loads((provenance / "kit.json").read_text())
    if mutation == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif mutation == "size":
        manifest["files"][0]["bytes"] += 1
    else:
        data = json.loads((provenance / "trajectory.json").read_text())
        data["notes"] = "kernel copied from origin/someone/private-kernel"
        blob = json.dumps(data).encode()
        (provenance / "trajectory.json").write_bytes(blob)
        entry = next(e for e in manifest["files"] if e["name"] == "trajectory.json")
        entry.update(sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))
    (provenance / "kit.json").write_text(json.dumps(manifest))
    result = publisher("--promote-only")
    assert result.returncode != 0
    assert not kit_path(tmp_path, "gensyn-audit-public").exists()
