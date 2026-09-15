#!/usr/bin/env bash
# DGX Spark launcher (GB10) — same code path, different default nproc.
#
# Usage:
#   ./scripts/launch_dgx_spark.sh 1b_proxy
#   NPROC_PER_NODE=2 ./scripts/launch_dgx_spark.sh 100m_smoke
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# shellcheck disable=SC1091
source "$HERE/env/nccl.env"

CONFIG_NAME="${1:?missing config name}"
shift || true

# DGX Spark SKUs vary; default to 4 GPUs and let the user override.
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)}"

cd "$ROOT"
exec torchrun \
    --standalone \
    --nproc-per-node="$NPROC_PER_NODE" \
    -m pretrain.cli.train \
    --config-name "$CONFIG_NAME" \
    "run.run_id=$RUN_ID" \
    "run.nproc_per_node=$NPROC_PER_NODE" \
    "$@"
