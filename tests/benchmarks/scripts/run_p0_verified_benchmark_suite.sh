#!/usr/bin/env bash
set -u

ROOT_DIR=${ROOT_DIR:-/workspace}
CLIP_OUTPUT_DIR=${CLIP_OUTPUT_DIR:-runs/benchmarks/p0_clip_b8_verified}
YOLO_OUTPUT_DIR=${YOLO_OUTPUT_DIR:-runs/benchmarks/yolo_world_objects365_official}

mkdir -p "${ROOT_DIR}/${CLIP_OUTPUT_DIR}" "${ROOT_DIR}/${YOLO_OUTPUT_DIR}"
cd "${ROOT_DIR}" || exit 1

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}

run_and_capture() {
  local name="$1"
  local output_dir="$2"
  shift 2
  local output="${output_dir}/${name}.json"
  local error_output="${output_dir}/${name}.error.json"
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

if command -v torchrun >/dev/null 2>&1; then
  run_and_capture native_ddp_b8_bf16_hook "${CLIP_OUTPUT_DIR}" \
    torchrun --standalone --nproc_per_node=2 -m parascale.cli benchmark \
    --config tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp_b8_bf16_hook.json
  run_and_capture fsdp_b8 "${CLIP_OUTPUT_DIR}" \
    torchrun --standalone --nproc_per_node=2 -m parascale.cli benchmark \
    --config tests/benchmarks/configs/benchmark_datacomp_medium_fsdp_b8.json
else
  python3 - <<'PY'
import json
from pathlib import Path

base = Path("runs/benchmarks/p0_clip_b8_verified")
base.mkdir(parents=True, exist_ok=True)
for name in ("native_ddp_b8_bf16_hook", "fsdp_b8"):
    (base / f"{name}.error.json").write_text(
        json.dumps(
            {"backend": name, "status": "error", "error": "torchrun is not available"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
PY
fi

if command -v deepspeed >/dev/null 2>&1; then
  run_and_capture deepspeed_b8 "${CLIP_OUTPUT_DIR}" \
    deepspeed --num_gpus=2 --module parascale.cli benchmark \
    --config tests/benchmarks/configs/benchmark_datacomp_medium_deepspeed_b8.json
else
  python3 - <<'PY'
import json
from pathlib import Path

base = Path("runs/benchmarks/p0_clip_b8_verified")
base.mkdir(parents=True, exist_ok=True)
(base / "deepspeed_b8.error.json").write_text(
    json.dumps(
        {
            "backend": "deepspeed_b8",
            "status": "error",
            "error": "deepspeed launcher is not available",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
fi

python3 tests/benchmarks/tools/aggregate_benchmark_matrix.py \
  --input "${CLIP_OUTPUT_DIR}" \
  --output "${CLIP_OUTPUT_DIR}/comparison.json" \
  --markdown "tests/reports/archive/p0_clip_b8_verified_benchmark_report.md" \
  --benchmark-id "p0_datacomp_clip_b8_verified_backend_matrix" \
  --title "P0 CLIP B8 Verified Backend Benchmark Report" \
  --workload-label "DataComp WDS CLIP-B style b8 native-DDP bf16-hook vs FSDP/DeepSpeed" \
  --target-backend native_ddp

if [ -x tests/benchmarks/scripts/run_yolo_world_objects365_official_benchmark_matrix.sh ]; then
  OUTPUT_DIR="${YOLO_OUTPUT_DIR}" tests/benchmarks/scripts/run_yolo_world_objects365_official_benchmark_matrix.sh
else
  bash tests/benchmarks/scripts/run_yolo_world_objects365_official_benchmark_matrix.sh
fi
