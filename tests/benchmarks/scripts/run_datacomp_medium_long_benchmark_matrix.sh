#!/usr/bin/env bash
set -u

ROOT_DIR=${ROOT_DIR:-/workspace}
OUTPUT_DIR=${OUTPUT_DIR:-runs/benchmarks/datacomp_medium_long}
mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"
cd "${ROOT_DIR}" || exit 1

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

run_and_capture() {
  local name="$1"
  shift
  local output="${OUTPUT_DIR}/${name}.json"
  local error_output="${OUTPUT_DIR}/${name}.error.json"
  echo "[ParaScale] running ${name}: $*"
  if "$@" --output "${output}"; then
    rm -f "${error_output}"
  else
    local code=$?
    python3 - "$name" "$code" "$error_output" "$*" <<'PY'
import json
import sys
name, code, path, command = sys.argv[1:5]
payload = {
    "backend": name,
    "status": "error",
    "returncode": int(code),
    "command": command,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
    return "${code}"
  fi
}

run_and_capture native python3 -m parascale.cli benchmark --config tests/benchmarks/configs/benchmark_datacomp_medium_long_native.json

if command -v torchrun >/dev/null 2>&1; then
  run_and_capture fsdp torchrun --standalone --nproc_per_node=2 -m parascale.cli benchmark --config tests/benchmarks/configs/benchmark_datacomp_medium_long_fsdp.json
else
  python3 - <<'PY'
import json
from pathlib import Path
path = Path("runs/benchmarks/datacomp_medium_long/fsdp.error.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"backend": "fsdp", "status": "error", "error": "torchrun is not available"}, indent=2) + "\n")
PY
fi

if command -v deepspeed >/dev/null 2>&1; then
  run_and_capture deepspeed deepspeed --num_gpus=2 --module parascale.cli benchmark --config tests/benchmarks/configs/benchmark_datacomp_medium_long_deepspeed.json
else
  python3 - <<'PY'
import json
from pathlib import Path
path = Path("runs/benchmarks/datacomp_medium_long/deepspeed.error.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"backend": "deepspeed", "status": "error", "error": "deepspeed launcher is not available"}, indent=2) + "\n")
PY
fi

python3 tests/benchmarks/tools/aggregate_benchmark_matrix.py \
  --input "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/comparison.json" \
  --markdown "tests/reports/archive/datacomp_medium_long_backend_benchmark_report.md" \
  --benchmark-id "datacomp_wds_clip_medium_long_backend_matrix" \
  --title "DataComp Medium Long-Run Backend Benchmark Report" \
  --workload-label "DataComp WDS streaming CLIP-B/ViT-B style 1000-step benchmark" \
  --primary-metric "stable_end_to_end_image_text_pairs_per_second"
