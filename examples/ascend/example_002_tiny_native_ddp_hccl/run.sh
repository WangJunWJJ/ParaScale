#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

exec "${TORCHRUN:-torchrun}" --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m parascale.cli train --config "$SCRIPT_DIR/config.json" "$@"
