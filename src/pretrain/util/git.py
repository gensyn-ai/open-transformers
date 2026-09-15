"""Capture git SHA + diff into the checkpoint metadata, and resolve the
commit/dirty provenance of the repos a run executes from (for the startup
system-info log)."""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path


def git_sha() -> str:
    if shutil.which("git") is None:
        return "no-git"
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return r.stdout.strip() or "no-sha"
    except Exception:
        return "no-sha"


def git_diff() -> str:
    if shutil.which("git") is None:
        return ""
    try:
        r = subprocess.run(
            ["git", "diff", "--no-color"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return r.stdout
    except Exception:
        return ""


# --- repo provenance (commit + dirty) for the startup system-info log --------
#
# This reads git *live* from the repo a module lives in. It has to keep working
# inside a training pod, where two things differ from a laptop:
#   * the checkout on the PVC is owned by a different uid than the process, so
#     git refuses with "dubious ownership" unless we pass -c safe.directory=*;
#   * the code may have been copied WITHOUT .git (e.g. the regression tar-pipe
#     strips it), in which case there is simply nothing to read -> available
#     stays False and we log "unknown" rather than crashing the run.


@dataclasses.dataclass
class RepoProvenance:
    """Live git state of one repo, best-effort. ``available`` is False when the
    checkout has no reachable .git (commit/branch/dirty then stay None)."""

    name: str
    available: bool = False
    root: str | None = None
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None

    def describe(self) -> str:
        if not self.available:
            return f"{self.name}=unknown (no .git)"
        flag = "dirty" if self.dirty else "clean"
        return f"{self.name}={self.commit}@{self.branch} ({flag})"


def _git(repo: Path, *args: str, timeout: float = 10.0) -> str | None:
    """Run a read-only git command inside ``repo``; None on any failure.

    ``safe.directory=*`` lets this work against a PVC checkout owned by another
    uid (the pod case) — without it git aborts with "dubious ownership"."""
    if shutil.which("git") is None:
        return None
    try:
        r = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def repo_provenance(name: str, start: Path) -> RepoProvenance:
    """Resolve the commit / branch / dirty flag of the repo enclosing ``start``.

    ``start`` is any path inside the repo (typically a module's __file__); we
    let git walk up to the enclosing toplevel, so this is correct whether the
    code lives at ~/workdir/<repo> locally or /workspace/<repo> on the PVC."""
    prov = RepoProvenance(name=name)
    start = start if start.is_dir() else start.parent
    root = _git(start, "rev-parse", "--show-toplevel")
    if not root:
        return prov  # no .git up the tree (or no git binary) -> unknown
    prov.available = True
    prov.root = root
    rootp = Path(root)
    prov.commit = _git(rootp, "rev-parse", "--short=12", "HEAD")
    prov.branch = _git(rootp, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(rootp, "status", "--porcelain")
    # status is "" for a clean tree; None means the call failed -> leave dirty
    # unknown rather than reporting a clean tree we never actually checked.
    if status is not None:
        prov.dirty = bool(status.strip())
    return prov


def run_provenance() -> list[RepoProvenance]:
    """Provenance of the two repos a training run executes from: this repo
    (transformer-pretraining) and the node repo that ships repop. Each is
    located via an imported module so we find the real on-disk checkout
    regardless of where it was synced."""
    provs = [repo_provenance("transformer-pretraining", Path(__file__))]
    try:
        import repop

        provs.append(repo_provenance("node", Path(repop.__file__)))
    except Exception:
        provs.append(RepoProvenance(name="node"))
    return provs
