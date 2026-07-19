#!/usr/bin/env bash
set -u

ROOT_DIR=${ROOT_DIR:-$(pwd)}
OUTPUT_DIR=${OUTPUT_DIR:-runs/benchmarks/ascend_parallel_matrix}
REPORT_PATH=${REPORT_PATH:-tests/benchmarks/reports/ascend_parallel_matrix.md}
SUITE_ID=${SUITE_ID:-ascend_parallel_matrix}
IMAGE_NAME=${IMAGE_NAME:-quay.io/ascend/llamafactory:latest-npu-a2}
SCENARIOS=${SCENARIOS:-single_docker_2card two_docker_1card two_docker_2card}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}
STEPS=${STEPS:-80}
WARMUP_STEPS=${WARMUP_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-8}

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}" "${ROOT_DIR}/tests/benchmarks/reports"
cd "${ROOT_DIR}" || exit 1

export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
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
    "run_id": name,
    "status": "error",
    "returncode": int(code),
    "command": command,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
}

write_config() {
  local name="$1"
  local backend="$2"
  local cards="$3"
  local output="$4"
  python3 - "$name" "$backend" "$cards" "$output" "$OUTPUT_DIR" "$STEPS" "$WARMUP_STEPS" "$BATCH_SIZE" <<'PY'
import json
import sys
from pathlib import Path

name, backend, cards, output, output_dir, steps, warmup_steps, batch_size = sys.argv[1:9]
cards = int(cards)
steps = int(steps)
warmup_steps = int(warmup_steps)
batch_size = int(batch_size)
base = Path("tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp_b8_bf16_hook.json")
cfg = json.loads(base.read_text(encoding="utf-8"))

parascale = cfg.setdefault("parascale", {})
data = cfg.setdefault("data", {})
training = cfg.setdefault("training", {})
hardware = cfg.setdefault("hardware_profile", {})

parascale["training_backend"] = backend
parascale["precision"] = "fp32"
parascale["data_parallel_size"] = cards
parascale["ddp_comm_hook"] = "none"
parascale["batch_size"] = batch_size
parascale["dataloader_num_workers"] = 0
parascale["dataloader_prefetch_factor"] = 2
parascale["dataloader_persistent_workers"] = False

data["type"] = "synthetic_clip"
data["streaming"] = False
data.pop("data_dir", None)
data["num_samples"] = max(steps * warmup_steps, steps * batch_size * cards)
data["batch_size"] = batch_size
data["num_workers"] = 0
data["persistent_workers"] = False
data["image_size"] = 224
data["text_length"] = 64

training["max_steps"] = steps
training["benchmark_steps"] = steps
training["warmup_steps"] = warmup_steps
training["checkpoint_interval"] = 999999
training["checkpoint_dir"] = str(Path(output_dir) / f"{name}_ckpt")
training["skip_final_checkpoint"] = True
parascale["checkpoint_save_path"] = training["checkpoint_dir"]

hardware["device_type"] = "npu"
hardware["num_gpus"] = cards
hardware["gpus_per_node"] = cards
hardware["gpu_memory"] = 64000000000
hardware["available_memory"] = 60000000000

Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

device_args() {
  local devices="$1"
  IFS=',' read -ra ids <<<"${devices}"
  for id in "${ids[@]}"; do
    printf '%s\n' "--device=/dev/davinci${id}"
  done
  printf '%s\n' "--device=/dev/davinci_manager"
  printf '%s\n' "--device=/dev/devmm_svm"
  printf '%s\n' "--device=/dev/hisi_hdc"
}

docker_run() {
  local name="$1"
  local devices="$2"
  shift 2
  local -a args=()
  local visible_devices=""
  local visible_index=0
  while IFS= read -r arg; do
    args+=("${arg}")
  done < <(device_args "${devices}")
  IFS=',' read -ra ids <<<"${devices}"
  for _ in "${ids[@]}"; do
    if [ -n "${visible_devices}" ]; then
      visible_devices="${visible_devices},"
    fi
    visible_devices="${visible_devices}${visible_index}"
    visible_index=$((visible_index + 1))
  done
  docker run --rm --ipc=host \
    "${args[@]}" \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    -e ASCEND_RT_VISIBLE_DEVICES="${visible_devices}" \
    -e HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT}" \
    -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    --name "${name}" \
    "${IMAGE_NAME}" "$@"
}

run_container_json() {
  local run_id="$1"
  local devices="$2"
  local cards="$3"
  local backend="$4"
  local nproc="$5"
  local config="${OUTPUT_DIR}/configs/${run_id}.json"
  local output="${OUTPUT_DIR}/${run_id}.json"
  local log="${OUTPUT_DIR}/${run_id}.log"
  write_config "${run_id}" "${backend}" "${cards}" "${config}"
  echo "[ParaScale Ascend matrix] running ${run_id} on devices ${devices}: backend=${backend}, nproc=${nproc}"
  if [ "${nproc}" -gt 1 ]; then
    docker_run "parascale_${run_id}" "${devices}" \
      torchrun --standalone --nproc_per_node="${nproc}" \
      -m parascale.cli benchmark --config "${config}" --output "${output}" >"${log}" 2>&1
  else
    docker_run "parascale_${run_id}" "${devices}" \
      python3 -m parascale.cli benchmark --config "${config}" --output "${output}" >"${log}" 2>&1
  fi
  local code=$?
  if [ "${code}" -ne 0 ]; then
    tail -n 80 "${log}" || true
    write_error "${run_id}" "${code}" "docker ${run_id}"
  else
    rm -f "${OUTPUT_DIR}/${run_id}.error.json"
  fi
  return 0
}

run_pair() {
  local left_run="$1"
  local left_devices="$2"
  local left_cards="$3"
  local left_backend="$4"
  local left_nproc="$5"
  local right_run="$6"
  local right_devices="$7"
  local right_cards="$8"
  local right_backend="$9"
  local right_nproc="${10}"

  run_container_json "${left_run}" "${left_devices}" "${left_cards}" "${left_backend}" "${left_nproc}" &
  local left_pid=$!
  run_container_json "${right_run}" "${right_devices}" "${right_cards}" "${right_backend}" "${right_nproc}" &
  local right_pid=$!
  wait "${left_pid}" || true
  wait "${right_pid}" || true
}

if has_scenario single_docker_2card; then
  run_container_json single_docker_2card 0,1 2 native_ddp 2
fi

if has_scenario two_docker_1card; then
  run_pair \
    two_docker_1card_a 0 1 ascend_native 1 \
    two_docker_1card_b 1 1 ascend_native 1
fi

if has_scenario two_docker_2card; then
  run_pair \
    two_docker_2card_a 0,1 2 native_ddp 2 \
    two_docker_2card_b 2,3 2 native_ddp 2
fi

python3 tests/benchmarks/tools/summarize_ascend_parallel_matrix.py \
  --input-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/summary.json" \
  --markdown "${REPORT_PATH}" \
  --suite-id "${SUITE_ID}" \
  --hardware "Ascend 910B4" \
  --image "${IMAGE_NAME}" \
  --steps "${STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --batch-size "${BATCH_SIZE}"
