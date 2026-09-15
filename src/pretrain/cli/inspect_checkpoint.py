"""Print metadata of a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="pretrain.cli.inspect_checkpoint")
    p.add_argument("checkpoint")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    meta = ckpt / "meta.json"
    if meta.exists():
        print(meta.read_text())
    else:
        print(f"no meta.json in {ckpt}")
    # Per-rank sampler state — print rank 0 as a representative sample.
    sampler_rank0 = ckpt / "sampler.rank_0.json"
    if sampler_rank0.exists():
        blob = json.loads(sampler_rank0.read_text())
        print(
            "rank 0 sampler consumed:",
            blob.get("consumed_documents_per_source"),
        )
    else:
        # Legacy: rank-0-only single-file format.
        sampler_consumed = ckpt / "sampler.consumed.json"
        if sampler_consumed.exists():
            print("sampler consumed:", json.loads(sampler_consumed.read_text()))


if __name__ == "__main__":
    main()
