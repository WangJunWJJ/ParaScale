#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

has_artifact=false
for argument in "$@"; do
  case "$argument" in
    --help|-h)
      exec "${PYTHON:-python}" tests/benchmarks/scripts/benchmark_quantized_inference.py --help
      ;;
    --reference-artifact|--reference-artifact=*) has_artifact=true ;;
  esac
done
if [[ "$has_artifact" != true ]]; then
  printf '%s\n' 'error: --reference-artifact is required for this example' >&2
  exit 2
fi

exec "${PYTHON:-python}" tests/benchmarks/scripts/benchmark_quantized_inference.py \
  --backend gemlite_int8 --static-cache --compile \
  --output "${OUTPUT_FILE:-runs/examples/quantized_artifact/comparison.json}" \
  "$@"
