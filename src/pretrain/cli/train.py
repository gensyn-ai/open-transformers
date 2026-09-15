"""Training entrypoint.

Usage (typically wrapped by ``scripts/launch_single_node.sh``):

    torchrun --standalone --nproc-per-node=8 \\
        -m pretrain.cli.train \\
        --config-name 1b_repop_v2 \\
        optim.peak_lr=3e-4

Hydra-style overrides flow through directly via ``sys.argv``. The CLI
resolves the config, prints it once for the run log, and hands off to
``pretrain.train.loop.train``.
"""

from __future__ import annotations

import argparse
import logging
import sys

from pretrain.config import load_config
from pretrain.train.loop import train


def _parse_args() -> tuple[str, list[str], str | None]:
    parser = argparse.ArgumentParser(
        prog="pretrain.cli.train",
        description="Pretraining entrypoint",
    )
    parser.add_argument("--config-name", required=True, help="Hydra train config")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a checkpoint directory to resume from (optional)",
    )
    args, overrides = parser.parse_known_args()
    return args.config_name, overrides, args.resume_from


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    config_name, overrides, resume_from = _parse_args()
    cfg = load_config(config_name, overrides=overrides)
    logging.getLogger("pretrain").info(
        "resolved config:\n%s", cfg.model_dump_json(indent=2)
    )
    train(cfg, resume_from=resume_from)


if __name__ == "__main__":
    main()
