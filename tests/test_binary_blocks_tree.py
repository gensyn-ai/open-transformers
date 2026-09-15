"""Canonical combination-tree tests for non-power-of-2 replicate counts.

The deterministic cross-replica all-reduce reduces ``dp_replicate`` per-replicate
gradients in a fixed **binary-blocks** order: split the replicas into consecutive
power-of-2 blocks (largest first, one per set bit of ``dp_replicate``), reduce each
block as a balanced adjacent-pair tree, then combine the block sums by an ascending
left-fold (lowest-rank block on the left). Three implementations must agree on this
exact grouping or the single-device audit stops matching the cluster bitwise:

  * ``deterministic_reduce.tree_reduce_sum``  — the reference spec,
  * ``deterministic_reduce._recursive_doubling_allreduce_avg`` — the cluster path
    (covered by the gloo test in tests/distributed/test_deterministic_reduce.py),
  * ``audit_replay._DiskTreeFold``            — the single-device replay.

These are CPU-only and fast (no torch.distributed). ``6 = 4 + 2`` is the headline
case; ``7 = 4 + 2 + 1`` is the one that distinguishes a left-fold from a right-fold
across blocks (they coincide for <=2 blocks, e.g. 6).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

# Load the reduce module by file path so we don't trigger pretrain.parallel's
# __init__ (which imports the repop-backed model stack, absent in CPU CI) — same
# pattern as tests/distributed/test_deterministic_reduce.py.
_SPEC = importlib.util.spec_from_file_location(
    "det_reduce_bb",
    Path(__file__).resolve().parents[1]
    / "src" / "pretrain" / "parallel" / "deterministic_reduce.py",
)
_det = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_det)
tree_reduce_sum = _det.tree_reduce_sum
_binary_blocks = _det._binary_blocks


# --------------------------------------------------------------------------- #
# Symbolic: pin the exact parenthesisation tree_reduce_sum produces.
# --------------------------------------------------------------------------- #
class _Expr:
    """Records its build order under ``+`` so we can assert the exact grouping."""

    def __init__(self, s: str) -> None:
        self.s = s

    def __add__(self, other: "_Expr") -> "_Expr":
        return _Expr(f"({self.s}+{other.s})")


# Hand-derived expected groupings (lower index always the left operand).
_EXPECTED = {
    2: "(g0+g1)",
    3: "((g0+g1)+g2)",
    4: "((g0+g1)+(g2+g3))",
    5: "(((g0+g1)+(g2+g3))+g4)",
    6: "(((g0+g1)+(g2+g3))+(g4+g5))",
    7: "((((g0+g1)+(g2+g3))+(g4+g5))+g6)",
    8: "(((g0+g1)+(g2+g3))+((g4+g5)+(g6+g7)))",
}


@pytest.mark.parametrize("n", sorted(_EXPECTED))
def test_tree_reduce_sum_grouping(n: int) -> None:
    parts = [_Expr(f"g{i}") for i in range(n)]
    assert tree_reduce_sum(parts).s == _EXPECTED[n]


def test_binary_blocks_decomposition() -> None:
    assert _binary_blocks(6) == [(0, 4), (4, 2)]
    assert _binary_blocks(7) == [(0, 4), (4, 2), (6, 1)]
    assert _binary_blocks(8) == [(0, 8)]
    assert _binary_blocks(1) == [(0, 1)]
    # blocks tile [0, n) exactly and are descending in size.
    for n in range(1, 65):
        blocks = _binary_blocks(n)
        assert sum(sz for _, sz in blocks) == n
        assert [s for s, _ in blocks] == sorted(s for s, _ in blocks)
        assert all((sz & (sz - 1)) == 0 for _, sz in blocks)


# --------------------------------------------------------------------------- #
# Float: tree_reduce_sum bit-matches an independent left-fold reference.
# --------------------------------------------------------------------------- #
def _ref_blocks_leftfold(parts: list[torch.Tensor]) -> torch.Tensor:
    """Independent binary-blocks left-fold: block sizes via the bin() string."""
    n = len(parts)
    bits = bin(n)[2:]
    sizes = [1 << (len(bits) - 1 - i) for i, c in enumerate(bits) if c == "1"]
    acc = None
    idx = 0
    for sz in sizes:
        block = list(parts[idx : idx + sz])
        idx += sz
        while len(block) > 1:
            block = [block[i] + block[i + 1] for i in range(0, len(block), 2)]
        acc = block[0] if acc is None else acc + block[0]
    return acc


def _wide_parts(n: int, k: int = 64) -> list[torch.Tensor]:
    """Per-replicate tensors with wide dynamic range so a wrong grouping would
    reassociate to a different fp32 result (makes bitwise equality meaningful)."""
    g = torch.Generator().manual_seed(20260619 + n)
    return [
        (torch.randn(k, generator=g, dtype=torch.float32) * (10.0 ** (i % 7 - 3)))
        for i in range(n)
    ]


@pytest.mark.parametrize("n", list(range(2, 13)))
def test_tree_reduce_sum_matches_independent_reference(n: int) -> None:
    parts = _wide_parts(n)
    got = tree_reduce_sum([p.clone() for p in parts])
    ref = _ref_blocks_leftfold([p.clone() for p in parts])
    assert torch.equal(got, ref), f"n={n} tree_reduce_sum != independent left-fold"


# --------------------------------------------------------------------------- #
# Audit replay: _DiskTreeFold must reproduce tree_reduce_sum bitwise (any n).
# --------------------------------------------------------------------------- #
def _load_disk_tree_fold():
    try:
        from pretrain.cli.audit_replay import _DiskTreeFold
    except Exception as exc:  # heavy deps (tqdm/pretrain.config) absent
        pytest.skip(f"audit_replay import unavailable: {exc}")
    return _DiskTreeFold


@pytest.mark.parametrize("n", list(range(2, 13)))
@pytest.mark.parametrize("spill", [False, True])
def test_disk_tree_fold_matches_tree_reduce_sum(n: int, spill: bool, tmp_path) -> None:
    _DiskTreeFold = _load_disk_tree_fold()
    parts = _wide_parts(n)
    ref = tree_reduce_sum([p.clone() for p in parts])  # the spec
    spill_dir = (tmp_path if spill else None)
    fold = _DiskTreeFold(spill_dir)
    for p in parts:
        fold.push({"g": p.clone()})  # ascending replicate order
    out = fold.result()["g"]
    assert torch.equal(out, ref), f"n={n} spill={spill} _DiskTreeFold != tree_reduce_sum"
