#!/usr/bin/env python3
"""Compare an audit's rank-0 loss log against the cluster's metrics.jsonl.

The audit (``pretrain.cli.audit_replay --loss-log out.json``) reproduces
virtual rank 0's per-step mean CE / z-loss with the same fp64 host fold the
training loop used for its rank-local ``loss_ce`` / ``loss_zloss`` metrics
keys, so on an honest same-arch replay every value matches the cluster's
logged fp64 bit-for-bit. Default comparison is therefore EXACT equality;
pass --rtol for a tolerant cross-check (e.g. eyeballing a foreign replay).

Usage:
  python scripts/repro/compare_rank0_losses.py \
      --loss-log audit_losses.json \
      --metrics runs/<run_id>/logs/metrics.jsonl

Resumed runs re-log overlapping steps; the LAST occurrence of a step in
metrics.jsonl wins (the surviving segment, matching the stitched logs).
Exit 0 on full match, 1 on any mismatch or step missing from the metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEYS = ("loss_ce", "loss_zloss")


def load_metrics(path: Path) -> dict[int, dict[str, float]]:
    """{step: {loss_ce, loss_zloss}} from metrics.jsonl, last occurrence wins.

    Rows without the loss keys (e.g. the Halt row, which logs only ``halt``)
    are skipped rather than clobbering an earlier complete row for that step.
    """
    out: dict[int, dict[str, float]] = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: {path}:{lineno}: unparseable line skipped", file=sys.stderr)
                continue
            if not all(k in row for k in KEYS):
                continue
            out[int(row["step"])] = {k: float(row[k]) for k in KEYS}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--loss-log", required=True, type=Path,
                    help="JSON written by audit_replay --loss-log")
    ap.add_argument("--metrics", required=True, type=Path,
                    help="the run's logs/metrics.jsonl (rank-0 metrics stream)")
    ap.add_argument("--rtol", type=float, default=0.0,
                    help="relative tolerance; default 0.0 = exact bit-for-bit "
                    "equality (the contract for a same-arch honest replay)")
    args = ap.parse_args()

    records = json.loads(args.loss_log.read_text(encoding="utf-8"))["records"]
    if not records:
        print("FAIL: loss log contains no records (zero-step replay?)")
        return 1
    metrics = load_metrics(args.metrics)

    mismatches = 0
    missing = 0
    for rec in records:
        step = int(rec["step"])
        got = {k: float(rec[k]) for k in KEYS}
        want = metrics.get(step)
        if want is None:
            print(f"step {step}: MISSING from metrics.jsonl")
            missing += 1
            continue
        for k in KEYS:
            if got[k] == want[k]:
                continue
            denom = max(abs(want[k]), 1e-300)
            rel = abs(got[k] - want[k]) / denom
            if rel <= args.rtol:
                continue
            print(f"step {step}: {k} MISMATCH audit={got[k]!r} cluster={want[k]!r} rel={rel:.3e}")
            mismatches += 1

    n = len(records)
    if mismatches == 0 and missing == 0:
        mode = "exact" if args.rtol == 0.0 else f"rtol={args.rtol:g}"
        print(f"PASS: {n} steps, {n * len(KEYS)} values match ({mode})")
        return 0
    print(f"FAIL: {mismatches} mismatched values, {missing} steps missing, over {n} steps")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
