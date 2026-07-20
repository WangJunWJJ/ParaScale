#!/usr/bin/env bash
set -u

ROOT_DIR=${ROOT_DIR:-/workspace}
OUTPUT_DIR=${OUTPUT_DIR:-runs/benchmarks/dual_4090_full_validation}
SUITE_ID=${SUITE_ID:-dual_4090_full_validation}
DATA_ROOT=${DATA_ROOT:-/dataset}
MODEL_ROOT=${MODEL_ROOT:-/models}
YOLO_MODEL_DIRS=${PARASCALE_MODEL_DIRS:-/yolo_models:/models}
SUMMARY_PATH=${SUMMARY_PATH:-tests/benchmarks/reports/dual_4090_full_validation/summary.json}
REPORT_PATH=${REPORT_PATH:-${OUTPUT_DIR}/dual_4090_full_validation.md}
UNIFIED_REPORT_PATH=${UNIFIED_REPORT_PATH:-tests/benchmarks/reports/BENCHMARK_REPORT.md}
IMAGE_NAME=${IMAGE_NAME:-parascale-ci:cu121-torch24}
SCENARIOS=${SCENARIOS:-local_tests tiny_smoke clip_native_ddp clip_fsdp clip_deepspeed vlm_native_ddp yolo_native_ddp yolo_native yolo_proxy_native ground_native}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}
WRITE_SUMMARY=${WRITE_SUMMARY:-1}

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}" "${ROOT_DIR}/$(dirname "${SUMMARY_PATH}")"
cd "${ROOT_DIR}" || exit 1

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export YOLO_CONFIG_DIR=${YOLO_CONFIG_DIR:-/tmp/ultralytics}
export PARASCALE_MODEL_DIRS="${YOLO_MODEL_DIRS}"

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
  echo "[ParaScale dual4090] running ${name}: $*"
  if "$@" --output "${output}" >"${log}" 2>&1; then
    rm -f "${OUTPUT_DIR}/${name}.error.json"
    return 0
  fi
  local code=$?
  tail -n 80 "${log}" || true
  write_error "${name}" "${code}" "$@"
  return 0
}

write_config() {
  local scenario="$1"
  local output="$2"
  python3 - "$scenario" "$output" "$OUTPUT_DIR" "$DATA_ROOT" "$MODEL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

scenario, output, output_dir, data_root, model_root = sys.argv[1:6]
out = Path(output)
base = Path("tests/benchmarks/configs")

def load(name):
    return json.loads((base / name).read_text(encoding="utf-8"))

configs = {
    "clip_native_ddp": "benchmark_datacomp_medium_native_ddp_b8_bf16_hook.json",
    "clip_fsdp": "benchmark_datacomp_medium_fsdp_b8.json",
    "clip_deepspeed": "benchmark_datacomp_medium_deepspeed_b8.json",
    "vlm_native_ddp": "benchmark_vlm_lora_datacomp_native_ddp.json",
    "yolo_native_ddp": "benchmark_yolo_world_objects365_official_native_ddp.json",
    "yolo_native": "benchmark_yolo_world_objects365_official_native_ddp.json",
    "yolo_proxy_native": "benchmark_yolo_world_objects365_official_native_ddp.json",
    "ground_native": "benchmark_ground_dino_phrase_official_native.json",
}

cfg = load(configs[scenario])
training = cfg.setdefault("training", {})
data = cfg.setdefault("data", {})
parascale = cfg.setdefault("parascale", {})
training["benchmark_steps"] = int(training.get("benchmark_steps", 8) or 8)
training["max_steps"] = int(training.get("max_steps", training["benchmark_steps"]) or 8)
training["warmup_steps"] = min(2, max(0, int(training.get("warmup_steps", 1) or 1)))
training["skip_final_checkpoint"] = False
training["checkpoint_interval"] = 999999
training["checkpoint_dir"] = str(Path(output_dir) / f"{scenario}_ckpt")
parascale["checkpoint_save_path"] = training["checkpoint_dir"]

if scenario.startswith("clip"):
    data["data_dir"] = f"{data_root}/datacomp_subsets/final/datacomp_10k_wds"
    data["num_samples"] = int(data.get("num_samples", 512) or 512)
if scenario.startswith("vlm"):
    data["type"] = "synthetic_image_text"
    data.pop("data_dir", None)
    data["streaming"] = False
    data["num_workers"] = 0
    data["persistent_workers"] = False
    data["num_samples"] = int(data.get("num_samples", 256) or 256)
if scenario.startswith("yolo"):
    if scenario in {"yolo_native", "yolo_proxy_native"}:
        parascale["training_backend"] = "native"
        parascale["data_parallel_size"] = 1
    if scenario == "yolo_proxy_native":
        cfg.setdefault("task", {})["objective"] = "detection_proxy_training"
        cfg.setdefault("model", {})["loss_type"] = "proxy"
        training["loss_type"] = "proxy"
        data["loss_type"] = "proxy"
        data["type"] = "coco_zip"
        data["zip_path"] = f"{data_root}/102-训练数据集/2-CoCo017/train2017.zip"
    data["num_samples"] = int(data.get("num_samples", 32) or 32)
if scenario.startswith("ground"):
    data["data_dir"] = f"{data_root}/ground_dino_phrase_mini/train"
    data["image_dir"] = f"{data_root}/ground_dino_phrase_mini/train/images"
    data["annotation_dir"] = f"{data_root}/ground_dino_phrase_mini/train/annotations"
    cfg.setdefault("model", {})["pretrained_model_name_or_path"] = (
        f"{model_root}/grounding-dino-tiny"
    )
    data["num_samples"] = int(data.get("num_samples", 16) or 16)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_no_output() {
  local name="$1"
  shift
  local log="${OUTPUT_DIR}/${name}.log"
  echo "[ParaScale dual4090] running ${name}: $*"
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
  fi
  local code=$?
  tail -n 80 "${log}" || true
  write_error "${name}" "${code}" "$@"
  return 0
}

if has_scenario local_tests; then
  run_no_output local_tests python3 tests/run_tests.py
fi

if has_scenario tiny_smoke; then
  run_json tiny_smoke python3 -m parascale.cli smoke \
    --config configs/server_tiny_torch.json
fi

for scenario in ${SCENARIOS}; do
  case "${scenario}" in
    clip_native_ddp|clip_fsdp|clip_deepspeed|vlm_native_ddp|yolo_native_ddp|yolo_native|yolo_proxy_native|ground_native)
      write_config "${scenario}" "/tmp/parascale_${scenario}.json"
      ;;
  esac
done

if command -v torchrun >/dev/null 2>&1; then
  if has_scenario clip_native_ddp; then
  run_json clip_native_ddp torchrun --standalone --nproc_per_node=2 \
    -m parascale.cli benchmark --config /tmp/parascale_clip_native_ddp.json
  fi
  if has_scenario clip_fsdp; then
  run_json clip_fsdp torchrun --standalone --nproc_per_node=2 \
    -m parascale.cli benchmark --config /tmp/parascale_clip_fsdp.json
  fi
  if has_scenario vlm_native_ddp; then
  run_json vlm_native_ddp torchrun --standalone --nproc_per_node=2 \
    -m parascale.cli benchmark --config /tmp/parascale_vlm_native_ddp.json
  fi
  if has_scenario yolo_native_ddp; then
  run_json yolo_native_ddp torchrun --standalone --nproc_per_node=2 \
    -m parascale.cli benchmark --config /tmp/parascale_yolo_native_ddp.json
  fi
else
  has_scenario clip_native_ddp && write_error clip_native_ddp 127 torchrun
  has_scenario clip_fsdp && write_error clip_fsdp 127 torchrun
  has_scenario vlm_native_ddp && write_error vlm_native_ddp 127 torchrun
  has_scenario yolo_native_ddp && write_error yolo_native_ddp 127 torchrun
fi

if has_scenario clip_deepspeed && command -v deepspeed >/dev/null 2>&1; then
  run_json clip_deepspeed deepspeed --num_gpus=2 --module parascale.cli benchmark \
    --config /tmp/parascale_clip_deepspeed.json
elif has_scenario clip_deepspeed; then
  write_error clip_deepspeed 127 deepspeed
fi

if has_scenario ground_native; then
  run_json ground_native python3 -m parascale.cli benchmark \
    --config /tmp/parascale_ground_native.json
fi

if has_scenario yolo_native; then
  run_json yolo_native python3 -m parascale.cli benchmark \
    --config /tmp/parascale_yolo_native.json
fi

if has_scenario yolo_proxy_native; then
  run_json yolo_proxy_native python3 -m parascale.cli benchmark \
    --config /tmp/parascale_yolo_proxy_native.json
fi

if [ "${WRITE_SUMMARY}" = "1" ]; then
  python3 tests/benchmarks/tools/summarize_dual_4090_validation.py \
    --input-dir "${OUTPUT_DIR}" \
    --output "${SUMMARY_PATH}" \
    --markdown "${REPORT_PATH}" \
    --suite-id "${SUITE_ID}" \
    --hardware "dual RTX 4090D 24GB" \
    --image "${IMAGE_NAME}"
  python3 tests/benchmarks/tools/build_benchmark_report.py \
    --report-root tests/benchmarks/reports \
    --output "${UNIFIED_REPORT_PATH}"
fi
