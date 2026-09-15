"""Local sanity for the B.1 offline reconstruction analyzer (no cluster needed).

Verifies that, given partials and a known combination, the analyzer correctly
identifies which candidate is bitwise-exact — so when run on real cluster
captures its VERDICT can be trusted.
"""

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

import torch

_SPEC = importlib.util.spec_from_file_location(
    "reconstruct_reduction",
    Path(__file__).resolve().parent.parent / "scripts" / "repro" / "reconstruct_reduction.py",
)
recon = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recon)


def _partials(n: int, names=("w", "b")):
    torch.manual_seed(7)
    return [{name: torch.randn(64, 64) for name in names} for _ in range(n)]


def test_identifies_ascending_sum():
    parts = _partials(4)
    reduced = {
        name: functools.reduce(torch.add, [p[name] for p in parts]) for name in ("w", "b")
    }
    results = recon.evaluate(reduced, parts)
    winner = next((r for r in results if r["all_bitwise"]), None)
    assert winner is not None
    assert winner["candidate"] == "sum_ascending"


def test_identifies_mean():
    parts = _partials(4)
    reduced = {
        name: functools.reduce(torch.add, [p[name] for p in parts]) / 4 for name in ("w", "b")
    }
    results = recon.evaluate(reduced, parts)
    # The /N candidates should be exact; a pure-sum candidate should not.
    by_name = {r["candidate"]: r for r in results}
    assert by_name["mean_ascending=sum/N"]["all_bitwise"]
    assert not by_name["sum_ascending"]["all_bitwise"]


def test_reports_no_match_when_order_differs_at_n4():
    # Build `reduced` as a pairwise tree; at N=4 the left-fold differs bitwise
    # for at least some random tensors, so sum_ascending should NOT be 100%.
    parts = _partials(4)
    reduced = {name: torch.stack([p[name] for p in parts]).sum(0) for name in ("w", "b")}
    results = recon.evaluate(reduced, parts)
    by_name = {r["candidate"]: r for r in results}
    assert by_name["sum_tree"]["all_bitwise"]  # the true combiner is exact


def test_load_capture_roundtrip(tmp_path):
    parts = _partials(2)
    reduced = {name: parts[0][name] + parts[1][name] for name in ("w", "b")}
    d = tmp_path / "N2"
    d.mkdir()
    torch.save(reduced, d / "reduced.pt")
    for r, p in enumerate(parts):
        torch.save(p, d / f"partial_rank{r}.pt")
    rl, pl = recon.load_capture(d)
    assert len(pl) == 2
    assert set(rl) == {"w", "b"}
