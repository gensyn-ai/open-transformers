"""Adversarial wheel payloads must not pass without exact disclosure approval."""

import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest


CHECKER = (
    Path(__file__).resolve().parents[1]
    / "scripts/audit_kit/check_implementation_disclosure.py"
)
spec = importlib.util.spec_from_file_location("implementation_checker", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def make_wheel(
    tmp_path,
    files,
    *,
    record_transform=lambda text: text,
    comment=b"",
    distribution="repop",
):
    record = "repop-1.0.dist-info/RECORD"
    rows = io.StringIO()
    writer = csv.writer(rows, lineterminator="\n")
    for name, blob in files.items():
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
        )
        writer.writerow([name, "sha256=" + digest, len(blob)])
    writer.writerow([record, "", ""])
    wheel = tmp_path / f"{distribution}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, blob in files.items():
            archive.writestr(name, blob)
        archive.writestr(record, record_transform(rows.getvalue()))
        archive.comment = comment
    return wheel


def approved(files):
    return {name: {hashlib.sha256(blob).hexdigest()} for name, blob in files.items()}


@pytest.mark.parametrize(
    "name,blob",
    [
        ("repop/qat/attention.py", b"@triton.jit\ndef attention(x): return x\n"),
        ("repop/backend/attention.metal", b"kernel void attention() {}"),
        (
            "repop/backend/helpers.h",
            b"inline float reciprocal(float x) { return 1/x; }",
        ),
        ("repop/ops.pyc", b"bytecode"),
        ("repop/backend/cuda.so", b"\x7fELF\x00.nv_fatbin\x00kernel_name"),
        ("repop/backend/shader.metallib", b"MTLB binary"),
        ("repop/data.bin", b"kernel void renamed_implementation() {}"),
        ("repop-1.0.dist-info/METADATA", b"Metadata\nprivate implementation"),
        ("repop/hidden.zip", b"PK\x03\x04nested payload"),
    ],
)
def test_default_policy_rejects_source_native_metadata_and_unknown_payloads(
    tmp_path, name, blob
):
    report = checker.inspect_wheel(
        make_wheel(tmp_path, {name: blob}), checker.load_policy()
    )
    assert any(
        "unapproved" in finding and name in finding for finding in report["findings"]
    )


def test_exact_member_approval_does_not_authorize_changed_or_renamed_source(tmp_path):
    files = {"repop/__init__.py": b"from .backend import cpu\n"}
    assert not checker.inspect_wheel(make_wheel(tmp_path, files), approved(files))[
        "findings"
    ]
    changed = {"repop/__init__.py": b"def private_kernel(): return 42\n"}
    assert checker.inspect_wheel(make_wheel(tmp_path, changed), approved(files))[
        "findings"
    ]
    renamed = {"repop/ops.py": files["repop/__init__.py"]}
    assert checker.inspect_wheel(make_wheel(tmp_path, renamed), approved(files))[
        "findings"
    ]


@pytest.mark.parametrize(
    "transform",
    [
        lambda text: text.replace("sha256=", "sha512="),
        lambda text: text.replace(",1\n", ",2\n"),
        lambda text: text + "private.py,,\n",
        lambda text: text + text.splitlines()[0] + "\n",
        lambda text: "repop-1.0.dist-info/RECORD,,\n",
    ],
)
def test_record_cannot_hide_or_misdescribe_payloads(tmp_path, transform):
    files = {"repop/a.py": b"x"}
    report = checker.inspect_wheel(
        make_wheel(tmp_path, files, record_transform=transform), approved(files)
    )
    assert report["findings"]


@pytest.mark.parametrize(
    "name",
    ["../a.py", "/a.py", "repop//a.py", "repop/./a.py", "repop\\a.py", "repop/a\nb.py"],
)
def test_unsafe_names_are_rejected_even_with_digest_approval(tmp_path, name):
    files = {name: b"x"}
    assert checker.inspect_wheel(make_wheel(tmp_path, files), approved(files))[
        "findings"
    ]


def test_archive_comment_is_not_covered_by_member_approval(tmp_path):
    files = {"repop/a.py": b"x"}
    wheel = make_wheel(tmp_path, files, comment=b"private implementation")
    assert (
        "unreviewed ZIP archive comment"
        in checker.inspect_wheel(wheel, approved(files))["findings"]
    )


def test_duplicate_payload_is_rejected(tmp_path):
    files = {"repop/a.py": b"x"}
    wheel = make_wheel(tmp_path, files)
    with pytest.warns(UserWarning), zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("repop/a.py", b"secret")
    assert any(
        "duplicate" in f
        for f in checker.inspect_wheel(wheel, approved(files))["findings"]
    )


@pytest.mark.parametrize("placement", ["before", "after", "extra"])
def test_unreviewed_zip_payload_is_rejected(tmp_path, placement):
    files = {"repop/a.py": b"x"}
    wheel = make_wheel(tmp_path, files)
    if placement == "before":
        wheel.write_bytes(b"private implementation" + wheel.read_bytes())
    elif placement == "after":
        wheel.write_bytes(wheel.read_bytes() + b"private implementation")
    else:
        with zipfile.ZipFile(wheel) as archive:
            content = [(info, archive.read(info)) for info in archive.infolist()]
        with zipfile.ZipFile(wheel, "w") as archive:
            for info, blob in content:
                info.extra = b"\xff\xff\x06\x00secret"
                archive.writestr(info, blob)
    assert checker.inspect_wheel(wheel, approved(files))["findings"]


@pytest.mark.parametrize(
    "members",
    [
        {"repop/*": [{"sha256": "a" * 64, "reason": "too broad"}]},
        {"repop/a.py": [{"sha256": "a" * 64, "reason": ""}]},
        {"repop/a.py": [{"sha256": "not a digest", "reason": "test"}]},
        {"repop/a.py": []},
    ],
)
def test_invalid_policy_is_rejected(tmp_path, members):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"format": 1, "approved_members": members}))
    with pytest.raises(ValueError):
        checker.load_policy(policy)


def test_policy_requires_exact_digest_and_rationale(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "format": 1,
                "approved_members": {
                    "repop/__init__.py": [
                        {"sha256": "a" * 64, "reason": "Reviewed API import stub"}
                    ],
                },
            }
        )
    )
    assert checker.load_policy(policy) == {"repop/__init__.py": {"a" * 64}}


def test_commit_stamp_is_exempt_but_the_exemption_does_not_widen(tmp_path):
    """The pretrain commit stamp cannot be digest-approved, so it is exempt.

    Its content carries the publishing commit and this policy is versioned in
    the same repository, so an approval would have to contain the hash of a
    file containing the hash of the commit that contains the approval. No such
    fixed point exists. check_native_wheels.py pins these bytes instead. The
    exemption is one exact name: nothing else metadata-shaped rides along with
    it, and a renamed payload does not inherit it.
    """

    def where(slug):
        directory = tmp_path / slug
        directory.mkdir()
        return directory

    stamp = json.dumps({"commit": "a" * 40, "python_native": True}).encode()
    stamped = {"pretrain/_native_build.json": stamp}
    report = checker.inspect_wheel(
        make_wheel(where("pt"), stamped, distribution="pretrain"),
        checker.load_policy(),
    )
    assert not report["findings"]
    assert report["members"][0]["exempt"] is True

    # The same path in a repop wheel is read by no other gate, so it must not
    # inherit the exemption: that would park arbitrary bytes behind the name.
    report = checker.inspect_wheel(
        make_wheel(where("rp"), stamped, distribution="repop"),
        checker.load_policy(),
    )
    assert report["findings"]
    assert report["members"][0]["exempt"] is False

    for impostor in ("repop/_build_info.json", "pretrain/_native_build.json.bak"):
        assert checker.inspect_wheel(
            make_wheel(where(impostor[:4]), {impostor: stamp}, distribution="pretrain"),
            checker.load_policy(),
        )["findings"], impostor
