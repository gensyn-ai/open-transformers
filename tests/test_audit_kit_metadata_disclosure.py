"""What a wheel says about itself is published, and this wheel is published wide.

``publish_audit_kit.sh`` mirrors the pretrain wheel into ``gensyn-audit-public``,
a bucket that grants ``allUsers`` read because its whole audience is people with
no relation to Gensyn. A wheel's ``readme`` is not merely a repo file: setuptools
embeds it verbatim into ``dist-info/METADATA``, which ships inside the wheel and
is therefore served anonymously alongside it.

For the first kits that readme was ``README.md``, the file we write for
ourselves. It is a good internal document and a bad public one: it names the PVC
and the Hugging Face token secret a data-prep job mounts, the cluster namespace,
the ``/workspace`` checkout path, and the shape of a run we have not announced.
None of that is reachable by an outsider, so none of it is a credential, but it
is free reconnaissance and it went out without anyone choosing to send it.

The failure is silent in both directions. Nothing in an ordinary README edit
hints that the text is about to be published, and nothing in the publisher
inspects what it is mirroring. So the check is static and lives here, next to
the other kit checks, rather than in a review checklist.

This gate covers infrastructure identifiers, which are unambiguous. Judgment
calls about how much of a roadmap or a benchmark belongs in a public artifact
stay with the person who writes the text.

Pure file reads. No repop, no torch, no network, no credentials.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import tomllib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_CHECKER = _REPO_ROOT / "scripts" / "audit_kit" / "check_wheel_disclosure.py"


def _load_checker():
    """Import the publisher's checker by path; scripts/ is not an importable package.

    The patterns live there rather than here so the two gates cannot drift: the
    publisher's copy is the one that actually blocks a kit, and a pattern added
    in only one place would leave the other quietly weaker.
    """
    spec = importlib.util.spec_from_file_location("check_wheel_disclosure", _CHECKER)
    assert spec and spec.loader, f"cannot load {_CHECKER}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_checker = _load_checker()
_PUBLIC_BUCKET = _checker.PUBLIC_BUCKET
# (pattern, what it would disclose); the third field scopes a pattern to prose
# versus code and only matters when scanning a built wheel, so it is dropped
# here, where every target is prose or config.
_DISCLOSURES: list[tuple[str, str]] = [(p, why) for p, why, _ in _checker.DISCLOSURES]


def _project() -> dict:
    return tomllib.loads(_PYPROJECT.read_text())["project"]


def _published_text() -> tuple[str, str]:
    """The readme that becomes METADATA, with the path it came from."""
    readme = _project()["readme"]
    # A table form (``{file = ...}``) would still resolve to a file; a plain
    # string is what we use and what this asserts, so a switch to the table
    # form fails loudly here instead of silently skipping the scan below.
    assert isinstance(readme, str), f"expected a readme path, got {readme!r}"
    path = _REPO_ROOT / readme
    assert path.is_file(), f"pyproject readme points at a missing file: {readme}"
    return path.read_text(), readme


def test_readme_shipped_in_metadata_is_the_distribution_one() -> None:
    """README.md is ours. The wheel has to carry the one written for a verifier."""
    _, readme = _published_text()
    assert readme != "README.md", (
        "pyproject readme is back to README.md, which is written for us and is "
        "embedded verbatim into the wheel METADATA published in "
        f"{_PUBLIC_BUCKET}. Point it at README-dist.md."
    )


@pytest.mark.parametrize(("pattern", "discloses"), _DISCLOSURES)
def test_published_readme_names_no_internal_infrastructure(
    pattern: str, discloses: str
) -> None:
    text, readme = _published_text()
    hit = re.search(pattern, text, re.IGNORECASE)
    assert hit is None, (
        f"{readme} contains {hit.group(0)!r}, which discloses {discloses}. "
        "This file is embedded into the wheel METADATA and served anonymously "
        f"from {_PUBLIC_BUCKET}."
    )


@pytest.mark.parametrize(("pattern", "discloses"), _DISCLOSURES)
def test_published_description_names_no_internal_infrastructure(
    pattern: str, discloses: str
) -> None:
    """``description`` becomes the METADATA Summary line, published the same way."""
    description = _project()["description"]
    hit = re.search(pattern, description, re.IGNORECASE)
    assert hit is None, (
        f"the project description contains {hit.group(0)!r}, which discloses "
        f"{discloses}, and it is published as the wheel's Summary."
    )


def test_published_wheel_states_a_licence() -> None:
    """A verifier is asked to install and run this. Say what they may do with it.

    For the kits published so far this repo had no LICENSE at all, so the wheel
    carried no grant of any kind while sitting in a world-readable bucket. That
    is the one finding here that is legal rather than operational, and it is
    invisible from inside: the wheel installs and runs perfectly well without a
    licence, so nothing fails until someone asks the question.
    """
    project = _project()
    assert project.get("license"), (
        "no license declared; the wheel published in "
        f"{_PUBLIC_BUCKET} would again grant a verifier nothing"
    )
    declared = project.get("license-files")
    assert declared, "license declared with no license-files, so no text ships"
    for pattern in declared:
        matches = list(_REPO_ROOT.glob(pattern))
        assert matches, f"license-files entry matches nothing: {pattern}"
        for path in matches:
            assert path.read_text().strip(), f"licence file is empty: {pattern}"


def test_published_configs_name_no_internal_hosts() -> None:
    """The Hydra config tree ships inside the wheel, so its comments ship too.

    ``configs/`` is packaged as ``pretrain/_configs`` (so a verifier can
    resolve ``--config-name`` with no checkout). That makes every comment in it
    published text, which is easy to forget while editing what reads like a
    local knob file. One of them recorded which box a corpus re-pull had been
    sized against, by hostname.

    Narrower than the readme scan: a config legitimately names buckets and
    paths, so this looks only for machines and people.
    """
    configs = _REPO_ROOT / "configs"
    assert configs.is_dir(), "configs/ moved; this gate no longer covers the wheel"
    if not _checker.DISCLOSURE_NAMES:
        pytest.skip(
            "no disclosure name list configured; set PRETRAIN_DISCLOSURE_NAMES "
            "or scripts/audit_kit/disclosure_names.txt (the list is untracked "
            "on purpose — see disclosure_names.txt.example)"
        )
    offenders = [
        f"{path.relative_to(_REPO_ROOT)}: {hit.group(0)!r}"
        for path in sorted(configs.rglob("*.yaml"))
        for hit in [
            re.search(_checker.NAME_PATTERN, path.read_text(), re.IGNORECASE)
        ]
        if hit
    ]
    assert not offenders, (
        "config comments name internal machines, and these files ship inside "
        "the wheel published in " + _PUBLIC_BUCKET + ":\n  " + "\n  ".join(offenders)
    )


def test_every_pattern_still_fires_on_the_text_it_was_drawn_from() -> None:
    """The red control for the gate above.

    Without one, the patterns can rot into a list that matches nothing and the
    gate goes green because it has nothing left to find.

    This asked README.md to still be dirty until #56 sanitized it in place,
    which is the outcome the whole effort wanted and which left the control
    asserting something that should now be false. The checker's own
    ``--self-test`` is the better control anyway: it holds every pattern
    against the specific text it was derived from, so one rotted pattern out of
    nine fails here, where "at least one still matches README.md" would not
    have noticed.
    """
    if not _checker.DISCLOSURE_NAMES:
        pytest.skip("no disclosure name list configured; --self-test refuses to run")
    assert _checker._self_test() == 0, (
        "the disclosure patterns no longer match the text they were drawn "
        "from; run scripts/audit_kit/check_wheel_disclosure.py --self-test"
    )
