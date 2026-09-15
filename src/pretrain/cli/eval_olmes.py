"""Standalone OLMES eval (Gu et al., 2024) on a checkpoint.

Same loading path as ``pretrain.cli.eval`` (keep the two in sync), then
hands the model to ``pretrain.eval.olmes_runner``. Run single-rank under
``torchrun --standalone --nproc-per-node=1`` so DCP has a process group.

Requires the pinned olmes install (see ``pretrain.eval.olmes_runner``); the default
task set is the full 10-task OLMES standard in both MCF and CF.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from pretrain.config import load_config
from pretrain.eval.olmes_runner import OLMES_DEFAULT_SUITES, OLMES_MAX_LENGTH, run_olmes
from pretrain.model import build_model
from pretrain.parallel import ParallelDims, parallelize_llama3_repop


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(prog="pretrain.cli.eval_olmes")
    p.add_argument("--config-name", required=True)
    p.add_argument("--checkpoint", required=True, help="checkpoint dir to load")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--task",
        nargs="*",
        default=None,
        help=f"olmes task/suite names (default: {' '.join(OLMES_DEFAULT_SUITES)})",
    )
    p.add_argument(
        "--extended",
        action="store_true",
        help="also run the OLMo-2-paper extension tasks (AGIEval, MMLU-Pro, "
        "NQ, DROP, TriviaQA, GSM8K) — several extra hours at 1B; meant for "
        "anchor/final checkpoints",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--max-length",
        type=int,
        default=OLMES_MAX_LENGTH,
        help="input cap in tokens; 2048 is the OLMES standard",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap instances per task — smoke tests only, breaks comparability",
    )
    p.add_argument("--model-label", default=None, help="model name recorded in metrics")
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()

    cfg = load_config(args.config_name, overrides=args.override)
    ckpt_dir = Path(args.checkpoint)
    # Untrusted-input gate up front: reject a doctored dcp/.metadata before
    # paying for the model build (the check is millisecond-scale).
    from pretrain.train.checkpoint import _validate_dcp_metadata, load_dcp_validated

    _validate_dcp_metadata(ckpt_dir / "dcp")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model, device=device)
    # Reproduce the training-time wrap so DCP keys line up (see
    # pretrain.cli.eval for the full rationale).
    pdims = ParallelDims(dp_replicate=1, dp_shard=1, world_size=1)
    model = parallelize_llama3_repop(model, cfg, pdims)
    state = {"model": model.state_dict()}
    load_dcp_validated(state, ckpt_dir / "dcp")
    model.load_state_dict(state["model"])
    model.eval()

    summary = run_olmes(
        model=model,
        tokenizer_path=args.tokenizer,
        out_dir=args.out_dir,
        tasks=args.task,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
        extended=args.extended,
        device=device,
        model_label=args.model_label or f"{args.config_name}:{ckpt_dir.name}",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
