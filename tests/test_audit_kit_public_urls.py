"""The volunteer runbook has to cite a URL its audience can actually fetch.

The runbook's opening promise is verification "on your own Mac, with no source
checkout". Its audience therefore holds no Gensyn credential and no ``gcloud``.
For most of this kit's life the install step named
``gs://gensyn-audit-artifacts/audit-kit/<kit-id>/``, which fails that audience
twice over: the bucket sets ``public_access_prevention = "enforced"`` and grants
read to one service account, and ``gs://`` is not a scheme pip can resolve even
with credentials.

Neither half announces itself to the person who edits the runbook. A maintainer
can read the file, copy a prefix out of a publish log, and see a plausible
command, because the command is plausible; it is only unusable by someone
standing outside. So the check is static and lives here rather than in a review
checklist.

Pure file reads. No repop, no torch, no network, no credentials.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_RUNBOOK = _REPO_ROOT / "scripts" / "audit_volunteer" / "RUNBOOK.md"
_PUBLISHER = _REPO_ROOT / "scripts" / "audit_kit" / "publish_audit_kit.sh"

# The bucket the runbook must send a verifier to. Named once here so a rename
# has to be made deliberately in three places rather than drifting in one.
_PUBLIC_BUCKET = "gensyn-audit-public"
_PUBLIC_URL_PREFIX = f"https://storage.googleapis.com/{_PUBLIC_BUCKET}/audit-kit/"

# The torch index repop's kernels are built against, named here for the same
# reason as the bucket above: it appears in the runbook and in two CI lanes in
# the repop source repo, and a bump has to be made deliberately in each.
_CU12_INDEX = "https://download.pytorch.org/whl/cu129"


def _section(markdown: str, heading: str) -> str:
    """Return one ``## heading`` section, up to the next heading of any level.

    The maintainer section at the end of the runbook names the closed bucket on
    purpose, and correctly. Checking the whole file for ``gs://`` would flag
    that true sentence, and a guard that reports a correct line as a defect is
    one somebody deletes.

    Headings are recognised outside fenced code blocks only. A shell comment is
    also a line beginning with ``#``, so a plain regex ends the section at the
    first ``# comment`` inside a ```` ```bash ```` block and then checks a
    fragment while claiming to check the section. That is not hypothetical: it
    truncated this file's Install section at the wheel-install comment, which
    left every assertion below reading the two lines above it.
    """
    target = re.compile(rf"^##\s+{re.escape(heading)}\s*$")
    any_heading = re.compile(r"^#{1,6}\s")

    body: list[str] = []
    in_fence = False
    started = False
    for line in markdown.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            if started:
                body.append(line)
            continue
        if not in_fence and target.match(line):
            started = True
            continue
        if started:
            if not in_fence and any_heading.match(line):
                break
            body.append(line)

    assert started, (
        f"RUNBOOK.md has no '## {heading}' section. This guard reads that "
        f"section by name, so a rename silently disables it."
    )
    return "".join(body)


@pytest.fixture(scope="module")
def runbook() -> str:
    return _RUNBOOK.read_text()


def test_install_section_fetches_over_https(runbook: str):
    install = _section(runbook, "Install")
    assert _PUBLIC_URL_PREFIX in install, (
        f"the Install section does not name {_PUBLIC_URL_PREFIX}. A verifier "
        f"with no credentials can only fetch the kit from the public bucket "
        f"over https."
    )


def test_install_section_names_no_gs_url(runbook: str):
    install = _section(runbook, "Install")
    offenders = re.findall(r"gs://\S+", install)
    assert not offenders, (
        f"the Install section tells a verifier to fetch {offenders}. gs:// is "
        f"not resolvable by pip, and the durable bucket is not readable "
        f"without Gensyn credentials, which this runbook's audience does not "
        f"have."
    )


def test_what_you_need_names_no_gs_url(runbook: str):
    # The prefix a verifier is "given" is quoted here as an example, and it was
    # the gs:// one. An unusable example is followed before the prose is.
    section = _section(runbook, "What you need")
    offenders = re.findall(r"gs://\S+", section)
    assert not offenders, (
        f"'What you need' offers {offenders} as the kit prefix. Give the https "
        f"prefix, which is the one that works without credentials."
    )


def test_install_section_covers_both_published_wheels(runbook: str):
    """The kit ships a wheel per platform, so the runbook must install either.

    ``publish_audit_kit.sh`` stages ``repop-wheels/<commit>/*.whl``, an
    unfiltered glob over a prefix holding both the linux_x86_64 wheel (cpu +
    cuda) and the macosx_14_0_arm64 one (cpu + metal). Every kit therefore
    carries both, and ``kit.json`` lists both.

    The runbook named only the macOS wheel for its whole life, which left an
    NVIDIA volunteer holding a wheel nobody told them to install, running a
    harness whose ``--device`` defaults to ``cuda``. Cross-device
    reproducibility is the claim being audited, so a CUDA verification is
    worth exactly as much as a Metal one and the path to it has to be written
    down.
    """
    install = _section(runbook, "Install")
    for wheel in ("repop-*-macosx_14_0_arm64.whl", "repop-*-linux_x86_64.whl"):
        assert wheel in install, (
            f"the Install section never names {wheel}, so the volunteers whose "
            f"machine needs it have no documented path. Both wheels are in "
            f"every kit."
        )
    for device in ("mps", "cuda"):
        assert f"selected_device={device}" in install, (
            f"the Install section does not select device={device}, which the "
            f"replay command later interpolates."
        )


def test_install_section_pins_the_cuda_12_torch_index_on_linux():
    """The Linux wheel is unusable with the torch the default index serves.

    repop's kernels are compiled against the CUDA-12.9 torch build that
    ``repop/pyproject.toml`` pins through ``[tool.uv.sources]``. That pin is
    uv-only, and a wheel's metadata can carry a version but not an index, so
    ``pip install repop-*.whl`` resolves the CUDA-13 build and pairs it with a
    cu129 extension. The result is wrong numbers rather than an error, because
    the failing cuBLAS calls go unchecked over an uninitialised output buffer.

    Handing that to a volunteer is worse than handing it to CI. Their whole
    task is to report whether the published hash reproduces, so a wrong answer
    arrives as a reproducibility break in the trajectory rather than as a
    broken install.
    """
    install = _section(_RUNBOOK.read_text(), "Install")
    assert _CU12_INDEX in install, (
        f"the Install section does not name {_CU12_INDEX}, so a Linux "
        f"volunteer installs the CUDA-13 torch against CUDA-12.9 kernels."
    )
    assert "--index-url" in install, (
        "the Install section names the index but never passes it to pip, so "
        "nothing makes the resolution deterministic."
    )


def test_install_section_installs_torch_before_the_repop_wheel():
    # Order is what makes the index stick. Installing the wheel first lets pip
    # satisfy its `torch==` requirement from the default index, and the
    # explicit install afterwards then finds the requirement already met.
    install = _section(_RUNBOOK.read_text(), "Install")
    torch_at = install.find(
        'pip install --index-url "$TORCH_INDEX" "$TORCH_REQUIREMENT"'
    )
    wheel_at = install.find('pip install --index-url "$TORCH_INDEX" "$WHEEL"')
    assert torch_at != -1 and wheel_at != -1, (
        "the Install section no longer has both an indexed torch install and a "
        "indexed repop install; this ordering check reads them by name."
    )
    assert torch_at < wheel_at, (
        "the repop wheel is installed before torch, so pip resolves torch from "
        "the default index and the explicit index install becomes a no-op."
    )


def test_install_section_tells_the_reader_how_to_check_the_pairing():
    # Every other refusal in this runbook is one the reader can see. A silently
    # wrong CUDA build has no symptom until the hash mismatches, so the check
    # has to be written down next to the step that can get it wrong.
    install = _section(_RUNBOOK.read_text(), "Install")
    assert "torch.version.cuda" in install, (
        "the Install section never has the reader print torch.version.cuda, so "
        "a wrong-index install stays invisible until it looks like a failed "
        "verification."
    )


def test_publisher_default_matches_the_documented_bucket():
    # The runbook is only correct while the publisher writes where it points.
    # These two files are edited by different people at different times.
    publisher = _PUBLISHER.read_text()
    assert f'PUBLIC_BUCKET="gs://{_PUBLIC_BUCKET}"' in publisher, (
        f"publish_audit_kit.sh does not default --public-bucket to "
        f"gs://{_PUBLIC_BUCKET}, which is the bucket RUNBOOK.md tells "
        f"verifiers to fetch from."
    )


def test_publisher_verifies_the_kit_is_readable_without_credentials():
    # A publish that lands in the bucket but is not world-readable looks
    # successful to the maintainer, whose own credentials resolve it. The
    # credential-free read-back is what separates those two outcomes, so it is
    # part of the contract rather than a convenience.
    publisher = _PUBLISHER.read_text()
    assert "storage.googleapis.com" in publisher, (
        "publish_audit_kit.sh never forms the https URL, so it cannot check "
        "that what it published is fetchable the way the runbook says it is."
    )
    assert "curl --fail" in publisher, (
        "publish_audit_kit.sh does not read the kit back over https, so a kit "
        "that is written but unreadable would be reported as published."
    )
