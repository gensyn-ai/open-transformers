"""Standalone eval: PPL on held-out + DCLM CORE on a checkpoint.

Run on a single rank — the eval is fast enough on 8 × H100 that we don't
shard it (plan/07 §2). For deeper post-hoc analysis use the lm-eval CLI
directly with the adapter in ``pretrain.eval.lm_eval_adapter``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from pretrain.config import load_config
from pretrain.data.tokenizer import Tokenizer
from pretrain.eval.dclm_core import run_dclm_core
from pretrain.eval.perplexity import compute_perplexity
from pretrain.model import build_model
from pretrain.parallel import ParallelDims, parallelize_llama3_repop


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(prog="pretrain.cli.eval")
    p.add_argument("--config-name", required=True)
    p.add_argument("--checkpoint", required=True, help="checkpoint dir to load")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--held-out", help="held-out shard prefix for PPL")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-dclm", action="store_true", help="skip the DCLM CORE run")
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
    # Reproduce the training-time wrap so DCP keys line up. With ws=1
    # parallelize_llama3_repop skips FSDP but still applies AC, which
    # is what the saved state_dict keys carry
    # (``_checkpoint_wrapped_module.``). TP=1, dp_replicate=1 here:
    # eval runs single-rank.
    pdims = ParallelDims(dp_replicate=1, dp_shard=1, world_size=1)
    model = parallelize_llama3_repop(model, cfg, pdims)
    state = {"model": model.state_dict()}
    load_dcp_validated(state, ckpt_dir / "dcp")
    model.load_state_dict(state["model"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    if args.held_out:
        ppl = compute_perplexity(
            model,
            held_out_prefix=args.held_out,
            seq_len=cfg.train.seq_len,
            device=device,
            micro_batch_size=cfg.train.micro_batch_size,
        )
        summary["held_out_ppl"] = ppl

    if not args.no_dclm:
        tok = Tokenizer(args.tokenizer)
        result = run_dclm_core(
            model=model,
            tokenizer=tok,
            out_dir=out_dir,
            step=0,
            device=device,
        )
        summary["dclm_core_macro"] = result.get("macro_average")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
