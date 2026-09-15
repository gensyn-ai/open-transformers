#!/usr/bin/env bash
# Single-node launcher — see plan/08 §3.
#
# Usage:
#   ./scripts/launch_single_node.sh 8b_main
#   ./scripts/launch_single_node.sh 1b_proxy optim.peak_lr=2.5e-4
#
# Env vars (optional):
#   NPROC_PER_NODE    override torchrun --nproc-per-node (default: 8)
#   RUN_ID            run identifier (default: timestamp + git short SHA)
#   RESUME_FROM       checkpoint dir to resume from
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# shellcheck disable=SC1091
source "$HERE/env/nccl.env"

CONFIG_NAME="${1:?missing config name (e.g. 8b_main)}"
shift || true

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)}"
RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    RESUME_ARGS+=(--resume-from "$RESUME_FROM")
fi

cd "$ROOT"
exec torchrun \
    --standalone \
    --nproc-per-node="$NPROC_PER_NODE" \
    -m pretrain.cli.train \
    --config-name "$CONFIG_NAME" \
    "${RESUME_ARGS[@]}" \
    "run.run_id=$RUN_ID" \
    "$@"
