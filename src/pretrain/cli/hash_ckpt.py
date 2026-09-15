"""Print the state hash of a saved checkpoint.

Used to compare two training runs bit-for-bit: run this on each
checkpoint dir and ``diff`` the digests. This emits the *full-tensor,
topology-invariant* hash (:func:`compute_state_hash`) — a mesh-independent
standalone fingerprint of weights+optimizer, NOT chained and NOT
batch-aware. It is the right tool for cross-run / cross-topology diffing.

NOTE: it does NOT reproduce a checkpoint's ``state_hash.txt``. As of the
sharded-hash change, ``state_hash.txt`` is the run's *chained, shard-local
(v3)* canonical hash (weights+optim+grads+batch digest, chained to the
prior step) — reproduced only by ``pretrain.cli.audit_replay``, which
replays the interval. The two are intentionally different fingerprints;
this CLI prints its own digest and only reports equality with
``state_hash.txt`` for the rare legacy case where they coincide.

This CLI is needed for:
  - a mesh-independent fingerprint for diffing two runs/checkpoints,
  - hashing weights-only when the optimizer state can't be reloaded
    (e.g. comparing checkpoints from different optimizers).

Usage:
    python -m pretrain.cli.hash_ckpt CKPT_DIR
    python -m pretrain.cli.hash_ckpt CKPT_DIR --weights-only
    python -m pretrain.cli.hash_ckpt CKPT_DIR --config-name 100m_smoke_repop

By default the original config is recovered from ``meta.json``'s
``config_resolved`` field. ``--config-name`` overrides this for cases
where the schema has evolved.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from pretrain.config import load_config
from pretrain.config.schema import RootConfig
from pretrain.model import build_model
from pretrain.optim.registry import build_optimizer
from pretrain.parallel import ParallelDims, parallelize_llama3_repop
from pretrain.train.state_hash import compute_state_hash


def _cfg_from_meta(ckpt_dir: Path) -> RootConfig:
    from pretrain.config import parse_config_resolved

    meta = json.loads((ckpt_dir / "meta.json").read_text())
    return parse_config_resolved(meta["config_resolved"])


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(prog="pretrain.cli.hash_ckpt")
    p.add_argument("checkpoint", help="checkpoint dir (containing dcp/ + meta.json)")
    p.add_argument(
        "--config-name",
        default=None,
        help="override the embedded config (advanced; usually unset)",
    )
    p.add_argument(
        "--weights-only",
        action="store_true",
        help="skip optimizer state — useful when comparing across optimizer changes",
    )
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="hydra-style overrides; only honored with --config-name",
    )
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint)
    # Untrusted-input gate up front: reject a doctored dcp/.metadata before
    # paying for the model + optimizer build (the check is millisecond-scale).
    from pretrain.train.checkpoint import _validate_dcp_metadata, load_dcp_validated

    _validate_dcp_metadata(ckpt_dir / "dcp")
    if args.config_name is not None:
        cfg = load_config(args.config_name, overrides=args.override)
    else:
        cfg = _cfg_from_meta(ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model, device=device)
    # Reproduce the training-time wrap so DCP keys line up. Single-rank
    # post-hoc → ws=1; parallelize_llama3_repop skips FSDP at ws=1 but
    # still applies AC, matching the saved state_dict's wrapper keys.
    pdims = ParallelDims(dp_replicate=1, dp_shard=1, world_size=1)
    model = parallelize_llama3_repop(model, cfg, pdims)

    state: dict = {"model": model.state_dict()}
    optimizer = None
    if not args.weights_only:
        optimizer = build_optimizer(model, cfg.optim)
        state["optim"] = optimizer.state_dict()
    load_dcp_validated(state, ckpt_dir / "dcp")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        try:
            optimizer.load_state_dict(state["optim"])
        except Exception as e:
            logging.warning(
                "could not reload optimizer state (%s) — falling back to "
                "weights-only hash", e,
            )
            optimizer = None

    digest = compute_state_hash(model, optimizer=optimizer)
    print(digest)

    saved = ckpt_dir / "state_hash.txt"
    if saved.exists():
        saved_digest = saved.read_text().strip()
        if saved_digest == digest:
            logging.info("state_hash.txt: %s (MATCH)", saved_digest)
        else:
            # Expected for any run with periodic hashing: state_hash.txt is the
            # chained shard-local (v3) hash, which this standalone full-tensor
            # fingerprint does not reproduce. Use ``audit_replay`` to verify it.
            logging.info(
                "state_hash.txt: %s (differs — this is the run's chained "
                "shard-local v3 hash; reproduce it with audit_replay, not this "
                "standalone fingerprint)", saved_digest,
            )


if __name__ == "__main__":
    main()
