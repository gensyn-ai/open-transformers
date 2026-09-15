"""CLI: fetch only the dataset shards an audit interval consumes.

Given a (locally present) auditable checkpoint and a GCS mirror of the dataset,
download just the manifests + ``.idx`` for every source and the *whole* ``.bin``
for the shards the canonical stream touches between this checkpoint and the next
— a fraction of a percent of the corpus — into ``./data/audit_data/shards/``.
Then audit against it with ``--data-root``.

Usage:
    python -m pretrain.cli.fetch_audit_data \\
        --checkpoint runs/<id>/checkpoints/step_000000010 \\
        --gcs-root gs://<bucket>/<path-to>/data/shards \\
        [--dest ./data/audit_data] [--config-name 1b_repop_v2] \\
        [--until-step N] [--no-verify]

    python -m pretrain.cli.audit_replay \\
        --checkpoint runs/<id>/checkpoints/step_000000010 \\
        --data-root ./data/audit_data/shards \\
        --expect-hash runs/<id>/checkpoints/step_000000020/state_hash.txt
"""

from __future__ import annotations

import argparse
import json
import logging


def _parse_args():
    p = argparse.ArgumentParser(prog="pretrain.cli.fetch_audit_data")
    p.add_argument("--checkpoint", required=True, help="auditable checkpoint dir to start from")
    p.add_argument(
        "--gcs-root",
        required=True,
        help="gs:// root holding one <source>/ subdir per source (the GCS mirror "
        "of the dataset's data/shards directory)",
    )
    p.add_argument(
        "--dest",
        default="./data/audit_data",
        help="local destination root; shards land under <dest>/shards/<source>/ "
        "(default: ./data/audit_data)",
    )
    p.add_argument(
        "--config-name",
        default=None,
        help="train config name (else resolved from the checkpoint's meta.json)",
    )
    p.add_argument(
        "--until-step",
        type=int,
        default=None,
        help="fetch through this optimizer step (default: one ckpt_every_tokens "
        "interval — the audit's default target)",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="skip blake2b verification of downloaded .idx/.bin against the manifest",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    args = _parse_args()
    from pretrain.data.fetch_interval import fetch_audit_interval

    result = fetch_audit_interval(
        args.checkpoint,
        args.gcs_root,
        dest=args.dest,
        config_name=args.config_name,
        until_step=args.until_step,
        verify=not args.no_verify,
    )
    print(json.dumps(result.__dict__, indent=2))
    print(
        "\nAudit this interval with:\n"
        f"  python -m pretrain.cli.audit_replay \\\n"
        f"      --checkpoint {args.checkpoint} \\\n"
        f"      --data-root {result.data_root} \\\n"
        f"      --expect-hash <next-checkpoint>/state_hash.txt"
    )


if __name__ == "__main__":
    main()
