#!/usr/bin/env bash
set -u

ROOT_DIR=${ROOT_DIR:-/workspace}
OUTPUT_DIR=${OUTPUT_DIR:-runs/benchmarks/direct_pytorch_clip_comparison}
REPORT_PATH=${REPORT_PATH:-tests/benchmarks/reports/direct_pytorch_clip_comparison.md}
SUITE_ID=${SUITE_ID:-direct_pytorch_clip_comparison}
IMAGE_NAME=${IMAGE_NAME:-parascale-ci:cu121-torch24}
SCENARIOS=${SCENARIOS:-parascale_native_ddp parascale_fsdp torch_ddp torch_fsdp}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}" "${ROOT_DIR}/tests/benchmarks/reports"
cd "${ROOT_DIR}" || exit 1

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

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
    "backend": name,
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
  shift
  local output="${OUTPUT_DIR}/${name}.json"
  local log="${OUTPUT_DIR}/${name}.log"
  echo "[Direct PyTorch comparison] running ${name}: $*"
  if "$@" --output "${output}" >"${log}" 2>&1; then
    rm -f "${OUTPUT_DIR}/${name}.error.json"
    return 0
  fi
  local code=$?
  tail -n 80 "${log}" || true
  write_error "${name}" "${code}" "$@"
  return 0
}

write_parascale_config() {
  local backend="$1"
  local output="$2"
  python3 - "$backend" "$output" "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

backend, output, output_dir = sys.argv[1:4]
base = Path("tests/benchmarks/configs")
config_name = {
    "native_ddp": "benchmark_datacomp_medium_native_ddp_b8_bf16_hook.json",
    "fsdp": "benchmark_datacomp_medium_fsdp_b8.json",
}[backend]
cfg = json.loads((base / config_name).read_text(encoding="utf-8"))
cfg.setdefault("data", {})["type"] = "synthetic_clip"
cfg["data"].pop("data_dir", None)
cfg["data"]["streaming"] = False
cfg["data"]["num_workers"] = 0
cfg["data"]["persistent_workers"] = False
cfg["data"]["num_samples"] = 640
cfg.setdefault("training", {})["benchmark_steps"] = 80
cfg["training"]["max_steps"] = 80
cfg["training"]["warmup_steps"] = 10
cfg["training"]["skip_final_checkpoint"] = True
cfg["training"]["checkpoint_dir"] = str(Path(output_dir) / f"parascale_{backend}_ckpt")
cfg.setdefault("parascale", {})["checkpoint_save_path"] = cfg["training"]["checkpoint_dir"]
cfg["parascale"]["training_backend"] = backend
Path(output).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

if has_scenario parascale_native_ddp; then
  write_parascale_config native_ddp /tmp/parascale_direct_compare_native_ddp.json
  run_json parascale_native_ddp torchrun --standalone --nproc_per_node=2 \
    -m parascale.cli benchmark --config /tmp/parascale_direct_compare_native_ddp.json
fi

if has_scenario parascale_fsdp; then
  write_parascale_config fsdp /tmp/parascale_direct_compare_fsdp.json
  run_json parascale_fsdp torchrun --standalone --nproc_per_node=2 \
    -m parascale.cli benchmark --config /tmp/parascale_direct_compare_fsdp.json
fi

if has_scenario torch_ddp; then
  run_json torch_ddp torchrun --standalone --nproc_per_node=2 \
    tests/benchmarks/scripts/run_direct_pytorch_clip_baseline.py \
    --backend ddp --steps 80 --warmup-steps 10 --batch-size 8
fi

if has_scenario torch_fsdp; then
  run_json torch_fsdp torchrun --standalone --nproc_per_node=2 \
    tests/benchmarks/scripts/run_direct_pytorch_clip_baseline.py \
    --backend fsdp --steps 80 --warmup-steps 10 --batch-size 8
fi

python3 tests/benchmarks/tools/summarize_direct_pytorch_comparison.py \
  --input-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/summary.json" \
  --markdown "${REPORT_PATH}" \
  --suite-id "${SUITE_ID}" \
  --hardware "dual RTX 4090D 24GB" \
  --image "${IMAGE_NAME}"
