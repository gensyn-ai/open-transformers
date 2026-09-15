"""B.1 offline reconstruction — which combiner reproduces FSDP's reduce-scatter?

Loads a capture of one reduction (``reduced.pt`` + per-rank
``partial_rank*.pt``) and, for each candidate way of combining the N per-rank
partial gradients, reports how many parameters it reproduces **bitwise**. The
verdict decides the audit's reduction path:

  * A candidate matches 100% of params across N=4 (and N=8) → the audit combines
    partials that exact way; NO cluster change needed (plan B.2).
  * Nothing matches bitwise → FSDP/NCCL's order isn't replicable on one device;
    switch the cluster + audit to a custom deterministic reduction (plan B.3).

Candidates cover SUM vs AVG (FSDP averages by default) and the float-association
that matters at N>=4: left-fold ascending/descending rank order, a balanced
pairwise tree (torch's ``stack().sum(0)``), and divide-then-sum vs sum-then-divide.

Usage:
    python scripts/repro/reconstruct_reduction.py repro_out/N4
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import torch


def load_capture(run_dir: str | Path) -> tuple[dict, list[dict]]:
    """Return ``(reduced, [partial_rank0, partial_rank1, ...])`` for a run dir."""
    run_dir = Path(run_dir)
    reduced = torch.load(run_dir / "reduced.pt", map_location="cpu", weights_only=False)
    partials = []
    r = 0
    while (run_dir / f"partial_rank{r}.pt").exists():
        partials.append(
            torch.load(run_dir / f"partial_rank{r}.pt", map_location="cpu", weights_only=False)
        )
        r += 1
    if not partials:
        raise FileNotFoundError(f"no partial_rank*.pt under {run_dir}")
    return reduced, partials


def candidate_combiners(n: int) -> dict[str, callable]:
    """Map name -> fn(list_of_tensors)->tensor. ``n`` is the world size."""

    def fold(ts):  # left-fold ascending: ((g0+g1)+g2)+...
        return functools.reduce(torch.add, ts)

    def fold_rev(ts):  # left-fold descending rank order
        return functools.reduce(torch.add, list(reversed(ts)))

    def tree(ts):  # balanced pairwise reduction (torch's own)
        return torch.stack(ts, dim=0).sum(dim=0)

    return {
        "sum_ascending": fold,
        "sum_descending": fold_rev,
        "sum_tree": tree,
        "mean_ascending=sum/N": lambda ts: fold(ts) / n,
        "mean_descending=sum/N": lambda ts: fold_rev(ts) / n,
        "mean_tree=tree/N": lambda ts: tree(ts) / n,
        "divide_then_sum_ascending": lambda ts: fold([t / n for t in ts]),
        "mean_builtin=stack.mean": lambda ts: torch.stack(ts, dim=0).mean(dim=0),
    }


def _max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max ULP distance between two finite fp32 tensors (diagnostic for near-misses)."""
    ai = a.contiguous().view(torch.int32).to(torch.int64)
    bi = b.contiguous().view(torch.int32).to(torch.int64)
    # Map to a monotonic ordering so ULP distance across the sign boundary is sane.
    ai = torch.where(ai < 0, torch.tensor(0x80000000, dtype=torch.int64) - ai, ai)
    bi = torch.where(bi < 0, torch.tensor(0x80000000, dtype=torch.int64) - bi, bi)
    return int((ai - bi).abs().max().item())


def evaluate(reduced: dict, partials: list[dict]) -> list[dict]:
    """For each candidate, per-parameter bitwise match stats. Sorted best-first."""
    n = len(partials)
    names = list(reduced.keys())
    results = []
    for cname, fn in candidate_combiners(n).items():
        exact = 0
        worst_abs = 0.0
        worst_ulp = 0
        for name in names:
            recon = fn([p[name].float() for p in partials])
            ref = reduced[name].float()
            if torch.equal(recon, ref):
                exact += 1
            else:
                worst_abs = max(worst_abs, float((recon - ref).abs().max().item()))
                try:
                    worst_ulp = max(worst_ulp, _max_ulp(recon, ref))
                except Exception:
                    pass
        results.append(
            {
                "candidate": cname,
                "params_matched": exact,
                "params_total": len(names),
                "all_bitwise": exact == len(names),
                "max_abs_diff": worst_abs,
                "max_ulp": worst_ulp,
            }
        )
    results.sort(key=lambda r: (-r["params_matched"], r["max_ulp"]))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="capture dir, e.g. repro_out/N4")
    args = ap.parse_args()
    reduced, partials = load_capture(args.run_dir)
    n = len(partials)
    print(f"world_size={n}, params={len(reduced)}")
    results = evaluate(reduced, partials)
    print(f"{'candidate':<28} {'matched':>10} {'bitwise':>8} {'max_abs':>12} {'max_ulp':>9}")
    for r in results:
        print(
            f"{r['candidate']:<28} {r['params_matched']:>4}/{r['params_total']:<5} "
            f"{str(r['all_bitwise']):>8} {r['max_abs_diff']:>12.3e} {r['max_ulp']:>9}"
        )
    winner = next((r for r in results if r["all_bitwise"]), None)
    print()
    if winner:
        print(f"VERDICT: '{winner['candidate']}' reproduces ALL params bitwise → "
              f"audit uses this combiner (plan B.2, no cluster change).")
    else:
        best = results[0]
        print(f"VERDICT: no candidate is bitwise-exact (best '{best['candidate']}' "
              f"max_ulp={best['max_ulp']}). NCCL order not replicable as-is → "
              f"consider the custom deterministic reduction (plan B.3).")


if __name__ == "__main__":
    main()
