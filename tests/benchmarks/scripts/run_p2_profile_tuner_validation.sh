#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace}
OUTPUT_DIR=${OUTPUT_DIR:-runs/validation/p2}

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"
cd "${ROOT_DIR}"

python3 tests/benchmarks/scripts/validate_p2_profile_tuner.py \
  --output "${OUTPUT_DIR}/summary.json" \
  --markdown "tests/reports/archive/p2_profile_tuner_validation_report.md"
