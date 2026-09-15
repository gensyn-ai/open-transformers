"""CLI: dump the step -> document map for a run (training data explorer).

Replays the canonical global stream index-only from step 0 (or from a
checkpoint's stream state) and writes one Parquet row per document fragment
per window: which document, which token range, which step / micro-batch /
slot / DP rank. Reads only ``manifest.yaml`` + ``.idx``; no ``.bin``, no model.

Usage:
    python -m pretrain.cli.dump_doc_map \\
        --checkpoint runs/<id>/checkpoints/step_000080957 \\
        --out-dir /path/to/step_docs [--data-root data/shards] \\
        [--until-step N] [--chunk-steps 1000] [--from-checkpoint-state]

By default the walk starts at step 0 so the map covers the whole run, and when
``--until-step`` equals the checkpoint's step the final stream position is
compared against the checkpoint's ``global_stream.json``; a mismatch exits 2.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("pretrain.dump_doc_map")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="pretrain.cli.dump_doc_map")
    p.add_argument("--checkpoint", required=True, help="checkpoint dir (meta.json [+ global_stream.json])")
    p.add_argument("--out-dir", required=True, help="directory for step_docs-*.parquet + manifest")
    p.add_argument(
        "--data-root",
        default=None,
        help="directory holding one <source>/ subdir per source; default: the paths in the run config, relative to cwd",
    )
    p.add_argument("--config-name", default=None, help="train config name (else from meta.json config_resolved)")
    p.add_argument("--until-step", type=int, default=None, help="walk steps < N (default: the checkpoint's step)")
    p.add_argument("--chunk-steps", type=int, default=1000, help="steps per Parquet file")
    p.add_argument(
        "--from-checkpoint-state",
        action="store_true",
        help=(
            "start at the checkpoint's stream position instead of step 0 and walk forward to "
            "--until-step, which is then required; refused when the checkpoint carries tokens over "
            "into its next window, because those have no document attribution"
        ),
    )
    p.add_argument("--dp-world-size", type=int, default=None, help="override meta.json dp_world_size")
    return p.parse_args(argv)


def resolve_until_step(until_step: int | None, checkpoint_step: int, *, from_checkpoint_state: bool) -> int:
    """The exclusive last step of the walk.

    From step 0 the default is the checkpoint's step, which also arms the
    position gate. From the checkpoint's state the default would equal the
    start step and walk nothing while exiting 0, so the flag must say where to
    stop, and it has to be past the start.
    """
    if from_checkpoint_state:
        if until_step is None:
            raise SystemExit("--from-checkpoint-state needs --until-step (the walk starts at the checkpoint's step)")
        if until_step <= checkpoint_step:
            raise SystemExit(
                f"--until-step {until_step} is not past the checkpoint's step {checkpoint_step}; nothing to walk"
            )
        return until_step
    return checkpoint_step if until_step is None else until_step


def run(args) -> int:
    from pretrain.data.doc_map import compare_position, write_doc_map, write_manifest
    from pretrain.data.fetch_interval import load_run_config, read_checkpoint_descriptor, rebase_sources
    from pretrain.data.global_stream import ShardedWindowView
    from pretrain.data.loader import _resolve_sources

    ckpt = Path(args.checkpoint)
    meta, state = read_checkpoint_descriptor(ckpt)
    cfg = load_run_config(meta, args.config_name)
    data_cfg = rebase_sources(cfg.data, args.data_root) if args.data_root else cfg.data
    manifests, dirs, weights = _resolve_sources(data_cfg)
    seed = int(meta["seed"])
    dp_world = int(args.dp_world_size or meta["dp_world_size"])
    until = resolve_until_step(
        args.until_step, int(meta["step"]), from_checkpoint_state=args.from_checkpoint_state
    )
    mb = cfg.train.micro_batch_size

    if args.from_checkpoint_state:
        view = ShardedWindowView(
            manifests, dirs, weights, cfg.train, seed=seed, rank=0, world_size=1,
            eos_id=data_cfg.document_separator_id, state=state,
            start_consumed_tokens=int(meta["consumed_tokens"]), start_step=int(meta["step"]),
            index_only=True,
        )
    else:
        view = ShardedWindowView(
            manifests, dirs, weights, cfg.train, seed=seed, rank=0, world_size=1,
            eos_id=data_cfg.document_separator_id, index_only=True,
        )
    if view.has_carry_over:
        # Same condition walk_step_spans refuses; checked here so the operator
        # gets a one-line exit rather than a traceback out of the walk.
        raise SystemExit(
            f"checkpoint {ckpt} carries tokens over from the previous step; they have no document "
            "attribution. Drop --from-checkpoint-state and walk from step 0."
        )
    log.info(
        "run=%s seed=%d dp_world=%d mb=%d until_step=%d sources=%s",
        meta.get("git_sha", "?")[:12], seed, dp_world, mb, until, [m.name for m in manifests],
    )
    t0 = time.time()

    def progress(step, rows):
        log.info("step %d / %d, %d rows, %.0fs", step, until, rows, time.time() - t0)

    summary = write_doc_map(
        view, args.out_dir, until_step=until, micro_batch_size=mb, dp_world_size=dp_world,
        chunk_steps=args.chunk_steps, progress=progress,
    )
    extra = {
        "checkpoint": str(ckpt),
        "git_sha": meta.get("git_sha"),
        "seed": seed,
        "dp_world_size": dp_world,
        "micro_batch_size": mb,
        "seq_len": cfg.train.seq_len,
        "sources": [{"name": m.name, "dir": d} for m, d in zip(manifests, dirs)],
        "elapsed_s": round(time.time() - t0, 1),
    }
    rc = 0
    if until == int(meta["step"]) and not args.from_checkpoint_state:
        gs = json.loads((ckpt / "global_stream.json").read_text())
        problems = compare_position(summary["position"], gs)
        extra["validation"] = {"against": str(ckpt / "global_stream.json"), "match": not problems, "problems": problems}
        if problems:
            log.error("stream position does NOT match checkpoint: %s", problems)
            rc = 2
        else:
            log.info("stream position matches checkpoint %s", ckpt.name)
    path = write_manifest(args.out_dir, summary, extra)
    print(json.dumps({k: v for k, v in {**summary, **extra}.items() if k != "files"}, indent=2))
    print(f"manifest: {path}")
    return rc


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    sys.exit(run(_parse_args(argv)))


if __name__ == "__main__":
    main()
