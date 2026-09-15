#!/usr/bin/env python3
"""Pull a representative multi-source sample for tokenizer training.

Streams documents from each of the four recipe_v1 sources using the same
pullers as ``scripts/build_corpus.py``, then leaves one JSONL per source
under ``--out-dir``. The downstream ``pretrain.cli.prepare_data
train-tokenizer`` step globs ``*.jsonl`` from that directory so all four
sources contribute to the BPE merges.

Doc counts default to ~1.1 M total (~1.6 B tokens at the proxy tok/doc
averages), distributed roughly by recipe_v1 weights so the merge
statistics reflect the production mix rather than FineWeb-Edu alone (the
proxy prep job's tokenizer step was FW-only). Tune
via flags if you want a smaller / bigger sample.

Idempotent: each per-source JSONL is gated on its own existence. If a
prior run wrote dclm.jsonl, a re-run skips DCLM and continues with the
remaining sources.

Usage:
    huggingface-cli login                    # one-time, gated datasets
    python scripts/sample_for_tokenizer.py   # defaults below
    python scripts/sample_for_tokenizer.py --docs-dclm 100000 ...
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# scripts/ isn't a package, so ensure this directory is on sys.path so
# build_corpus.py (next to this file) imports cleanly regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_corpus import pull_dclm, pull_fineweb, pull_proof, pull_stack  # noqa: E402

LOG = logging.getLogger("sample_for_tokenizer")

# Defaults: ~1.1 M docs / ~1.6 B tokens at proxy tok/doc averages, split
# roughly by recipe_v1 weights. Slightly over-weights the smaller sources
# vs strict 75/12/10/3 so Stack + Proof contribute enough unique
# substrings to influence the merges; the absolute proportions of training
# *data* are still set by the per-source mix at train time.
DEFAULTS = {
    "dclm":    750_000,
    "fineweb": 120_000,
    "stack":   200_000,
    "proof":    30_000,
}

PULLERS = {
    "dclm":    ("dclm.jsonl",    pull_dclm),
    "fineweb": ("fineweb.jsonl", pull_fineweb),
    "stack":   ("stack.jsonl",   pull_stack),
    "proof":   ("proof.jsonl",   pull_proof),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--out-dir", default="data/raw/tok_sample",
                   help="directory to write per-source JSONL into")
    p.add_argument("--docs-dclm",    type=int, default=DEFAULTS["dclm"])
    p.add_argument("--docs-fineweb", type=int, default=DEFAULTS["fineweb"])
    p.add_argument("--docs-stack",   type=int, default=DEFAULTS["stack"])
    p.add_argument("--docs-proof",   type=int, default=DEFAULTS["proof"])
    p.add_argument("--stack-concurrency", type=int, default=64,
                   help="parallel SWH S3 fetches for Stack v2 sample")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "dclm":    args.docs_dclm,
        "fineweb": args.docs_fineweb,
        "stack":   args.docs_stack,
        "proof":   args.docs_proof,
    }
    LOG.info("plan: %s -> %s", targets, out_dir)

    for src, n in targets.items():
        if n <= 0:
            LOG.info("[%s] count=0 — skipping", src)
            continue
        fname, puller = PULLERS[src]
        out = out_dir / fname
        if out.is_file():
            LOG.info("[%s] %s already present — skipping", src, out)
            continue

        # .part rename guards against a half-written file being reused
        # by the next run — same pattern as build_corpus.py.
        out_part = out.with_suffix(out.suffix + ".part")
        out_part.unlink(missing_ok=True)
        try:
            if src == "stack":
                puller(n, out_part, concurrency=args.stack_concurrency)
            else:
                puller(n, out_part)
        except BaseException:
            out_part.unlink(missing_ok=True)
            raise
        out_part.rename(out)

    LOG.info("DONE — tokenizer sample under %s", out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
