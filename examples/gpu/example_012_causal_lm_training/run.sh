#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

: "${PARASCALE_MODEL:?Set the local model/tokenizer directory}"
: "${PARASCALE_TOKEN_CORPUS:?Set the prepared token corpus directory}"
: "${PARASCALE_RUN_DIR:?Set an experiment output directory outside the repository}"
NPROC="${NPROC:-1}"
case "$NPROC" in
  1) export PARASCALE_BACKEND="${PARASCALE_BACKEND:-native}" ;;
  2|4) export PARASCALE_BACKEND="${PARASCALE_BACKEND:-native_ddp}" ;;
  *) printf '%s\n' 'NPROC must be 1, 2, or 4' >&2; exit 2 ;;
esac
export PARASCALE_PRECISION="${PARASCALE_PRECISION:-bf16}"

if [[ "$NPROC" == 1 ]]; then
  exec "${PYTHON:-python}" -m parascale.cli train --config "$SCRIPT_DIR/config.json" "$@"
fi
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m parascale.cli train --config "$SCRIPT_DIR/config.json" "$@"
