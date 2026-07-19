#!/usr/bin/env bash
set -u

ROOT_DIR=${ROOT_DIR:-/workspace}
OUTPUT_DIR=${OUTPUT_DIR:-runs/benchmarks/ascend_validation}
REPORT_PATH=${REPORT_PATH:-tests/benchmarks/reports/ascend_validation.md}
SUITE_ID=${SUITE_ID:-ascend_validation}
IMAGE_NAME=${IMAGE_NAME:-quay.io/ascend/llamafactory:latest-npu-a2}
SCENARIOS=${SCENARIOS:-doctor tiny_single tiny_hccl}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}" "${ROOT_DIR}/tests/benchmarks/reports"
cd "${ROOT_DIR}" || exit 1

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}

if [ "${CLEAN_OUTPUT}" = "1" ]; then
  rm -f "${OUTPUT_DIR}"/*.json "${OUTPUT_DIR}"/*.log
fi

has_scenario() {
  local name="$1"
  for scenario in ${SCENARIOS}; do
    if [ "${scenario}" = "${name}" ]; then
      return 0
    fi
  done
  return 1
}

write_error() {
  local name="$1"
  local code="$2"
  local output="${OUTPUT_DIR}/${name}.error.json"
  shift 2
  python3 - "$name" "$code" "$output" "$*" <<'PY'
import json
import sys

name, code, path, command = sys.argv[1:5]
payload = {
    "name": name,
    "status": "error",
    "returncode": int(code),
    "command": command,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
}

run_json() {
  local name="$1"
  local output="${OUTPUT_DIR}/${name}.json"
  local log="${OUTPUT_DIR}/${name}.log"
  shift
  echo "[ParaScale Ascend] running ${name}: $*"
  if "$@" --output "${output}" >"${log}" 2>&1; then
    rm -f "${OUTPUT_DIR}/${name}.error.json"
    return 0
  else
    local code=$?
    tail -n 80 "${log}" || true
    write_error "${name}" "${code}" "$@"
    return 0
  fi
}

run_no_output() {
  local name="$1"
  local log="${OUTPUT_DIR}/${name}.log"
  shift
  echo "[ParaScale Ascend] running ${name}: $*"
  if "$@" >"${log}" 2>&1; then
    rm -f "${OUTPUT_DIR}/${name}.error.json"
    python3 - "$name" "${OUTPUT_DIR}/${name}.json" <<'PY'
import json
import sys

name, path = sys.argv[1:3]
payload = {"name": name, "status": "ok", "ok": True}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
    return 0
  else
    local code=$?
    tail -n 80 "${log}" || true
    write_error "${name}" "${code}" "$@"
    return 0
  fi
}

if has_scenario doctor; then
  run_json doctor python3 -m parascale.cli doctor --strict --require npu
fi

if has_scenario tiny_single; then
  run_json tiny_single python3 -m parascale.cli train \
    --config examples/ascend/example_001_tiny_ascend_native/config.json
fi

if has_scenario tiny_hccl; then
  run_no_output tiny_hccl torchrun --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m parascale.cli train \
    --config examples/ascend/example_002_tiny_native_ddp_hccl/config.json
fi

python3 tests/benchmarks/tools/summarize_ascend_validation.py \
  --input-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/summary.json" \
  --markdown "${REPORT_PATH}" \
  --suite-id "${SUITE_ID}" \
  --hardware "Ascend 910B4" \
  --image "${IMAGE_NAME}"
