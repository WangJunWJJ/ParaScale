#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR=${ROOT_DIR:-/home/wangjun/work/ParaScale}
DATASET_ROOT=${DATASET_ROOT:-/home/wangjun/work/dataset}
YOLO_MODEL_ROOT=${YOLO_MODEL_ROOT:-/home/wangjun/work/yolo_models}
TMP_DIR=${TMP_DIR:-/tmp/parascale_multinode_smoke}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
CLIP_PORT=${CLIP_PORT:-29941}
YOLO_PORT=${YOLO_PORT:-29942}
CLIP_IMAGE=${CLIP_IMAGE:-parascale-ci:cu121-torch24}
YOLO_IMAGE=${YOLO_IMAGE:-parascale-yolo:cu121-torch24-ultralytics83161}

mkdir -p "${TMP_DIR}"
rm -f "${TMP_DIR}"/*.json "${TMP_DIR}"/*.log

write_config() {
  local base_config="$1"
  local output_config="$2"
  local workload_name="$3"
  python3 - "${ROOT_DIR}/${base_config}" "${output_config}" "${workload_name}" "${TMP_DIR}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
workload_name = sys.argv[3]
tmp_dir = Path(sys.argv[4])

config = json.loads(base_path.read_text(encoding="utf-8"))
parascale = config.setdefault("parascale", {})
parascale["target_scale"] = "small_cluster"
parascale["training_backend"] = "native_ddp"
parascale["data_parallel_size"] = 2
parascale["batch_size"] = int(parascale.get("batch_size", 2) or 2)
parascale["checkpoint_save_path"] = str(tmp_dir / "checkpoints" / workload_name)
parascale["checkpoint_save_interval"] = 999999

hardware = config.setdefault("hardware_profile", {})
hardware["num_gpus"] = 2
hardware["world_size"] = 2
hardware["gpus_per_node"] = 1
hardware["num_nodes"] = 2

launch = config.setdefault("launch", {})
launch["nnodes"] = 2
launch["nproc_per_node"] = 1
launch["master_addr"] = "127.0.0.1"

data = config.setdefault("data", {})
data["batch_size"] = int(data.get("batch_size", parascale["batch_size"]) or 2)
data["num_samples"] = min(int(data.get("num_samples", 64) or 64), 64)
data["num_workers"] = min(int(data.get("num_workers", 0) or 0), 2)
if data.get("num_workers", 0) == 0:
    data.pop("prefetch_factor", None)
    data["persistent_workers"] = False

training = config.setdefault("training", {})
training["max_steps"] = 4
training["benchmark_steps"] = 4
training["warmup_steps"] = 1
training["checkpoint_dir"] = str(tmp_dir / "checkpoints" / workload_name)
training["checkpoint_interval"] = 999999
training["skip_final_checkpoint"] = True

output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

docker_common_args() {
  local image="$1"
  local cuda_visible_devices="$2"
  local container_name="$3"
  printf '%s\n' \
    run --rm \
    --name "${container_name}" \
    --network host \
    --gpus all \
    --shm-size=8g \
    -e CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
    -e NCCL_SOCKET_IFNAME=lo \
    -e NCCL_DEBUG=WARN \
    -e OMP_NUM_THREADS=1 \
    -e YOLO_CONFIG_DIR=/tmp/ultralytics \
    -e PARASCALE_MODEL_DIRS=/models:/yolo_models \
    -v "${ROOT_DIR}:/workspace/ParaScale" \
    -v "${DATASET_ROOT}:/dataset" \
    -v "${YOLO_MODEL_ROOT}:/models" \
    -v "${TMP_DIR}:${TMP_DIR}" \
    -w /workspace/ParaScale \
    "${image}"
}

LAST_PID=""

run_node_background() {
  local case_name="$1"
  local image="$2"
  local node_rank="$3"
  local cuda_visible_devices="$4"
  local master_port="$5"
  local config_path="$6"
  local output_path="$7"
  local container_name="parascale-${case_name}-node-${node_rank}"
  local log_path="${TMP_DIR}/${case_name}_node${node_rank}.log"
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
  mapfile -t common_args < <(docker_common_args "${image}" "${cuda_visible_devices}" "${container_name}")
  docker "${common_args[@]}" bash -lc \
    "torchrun --nnodes=2 --node_rank=${node_rank} --master_addr=${MASTER_ADDR} --master_port=${master_port} --nproc_per_node=1 -m parascale.cli benchmark --config ${config_path} --output ${output_path}" \
    >"${log_path}" 2>&1 &
  LAST_PID=$!
}

summarize_case() {
  local case_name="$1"
  local output_path="$2"
  python3 - "${case_name}" "${output_path}" <<'PY'
import json
import sys
from pathlib import Path

case_name = sys.argv[1]
path = Path(sys.argv[2])
if not path.exists():
    print(f"{case_name}: output missing")
    sys.exit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
metrics = payload.get("metrics", {})
train = payload.get("train", {})
print(
    ",".join(
        [
            case_name,
            payload.get("capability_level", ""),
            train.get("backend", payload.get("backend", "")),
            str(train.get("global_step", "")),
            f"{float(metrics.get('stable_end_to_end_images_per_second', metrics.get('end_to_end_images_per_second', 0.0))):.3f}",
            f"{float(metrics.get('stable_end_to_end_image_text_pairs_per_second', metrics.get('end_to_end_image_text_pairs_per_second', 0.0))):.3f}",
            f"{float(metrics.get('stable_peak_memory_bytes', metrics.get('peak_memory_bytes', 0.0))) / 1024 / 1024:.1f}",
            str(payload.get("validation", {}).get("checkpoint", {}).get("ok")),
        ]
    )
)
PY
}

run_case() {
  local case_name="$1"
  local image="$2"
  local base_config="$3"
  local master_port="$4"
  local config_path="${TMP_DIR}/${case_name}.json"
  local output_path="${TMP_DIR}/${case_name}_result.json"
  write_config "${base_config}" "${config_path}" "${case_name}"
  echo "[ParaScale] ${case_name}: launching 2 containers, nnodes=2, nproc_per_node=1"
  run_node_background "${case_name}" "${image}" 0 0 "${master_port}" "${config_path}" "${output_path}"
  pid0="${LAST_PID}"
  sleep 4
  run_node_background "${case_name}" "${image}" 1 1 "${master_port}" "${config_path}" "${output_path}"
  pid1="${LAST_PID}"
  wait "${pid0}"
  rc0=$?
  wait "${pid1}"
  rc1=$?
  if [[ "${rc0}" -ne 0 || "${rc1}" -ne 0 ]]; then
    echo "[ParaScale] ${case_name}: failed rc0=${rc0} rc1=${rc1}"
    tail -80 "${TMP_DIR}/${case_name}_node0.log" || true
    tail -80 "${TMP_DIR}/${case_name}_node1.log" || true
    return 1
  fi
  summarize_case "${case_name}" "${output_path}"
}

echo "case,capability_level,backend,global_step,images_per_second,image_text_pairs_per_second,peak_memory_mb,checkpoint_ok"
status=0
run_case "clip_datacomp_multinode" "${CLIP_IMAGE}" "tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp.json" "${CLIP_PORT}" || status=1
run_case "yolo_world_multinode" "${YOLO_IMAGE}" "tests/benchmarks/configs/benchmark_yolo_world_objects365_official_native_ddp.json" "${YOLO_PORT}" || status=1

rm -rf "${TMP_DIR}/checkpoints"
exit "${status}"
