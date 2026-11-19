#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

exec "${PYTHON:-python}" tests/benchmarks/scripts/validate_llama_distillation.py \
  --device "${DEVICE:-cpu}" \
  --output "${OUTPUT_DIR:-runs/examples/llama_distillation}" \
  "$@"
