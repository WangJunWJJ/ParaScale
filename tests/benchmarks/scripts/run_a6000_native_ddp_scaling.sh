#!/usr/bin/env bash
set -u

ROOT_DIR=${ROOT_DIR:-/workspace}
OUTPUT_DIR=${OUTPUT_DIR:-runs/benchmarks/a6000_native_ddp_scaling}
CONFIG_DIR=${CONFIG_DIR:-runs/benchmarks/a6000_native_ddp_scaling_configs}
REPORT_DIR=${REPORT_DIR:-tests/benchmarks/reports/a6000_native_ddp_scaling}
BASE_CONFIG=${BASE_CONFIG:-tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp.json}
DATA_DIR=${DATA_DIR:-/dataset/datacomp_subsets/final/datacomp_10k_wds}
STEPS=${STEPS:-120}
WARMUP_STEPS=${WARMUP_STEPS:-20}
DEFAULT_BATCH_PER_GPU=${DEFAULT_BATCH_PER_GPU:-8}
mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}" "${ROOT_DIR}/${CONFIG_DIR}" "${ROOT_DIR}/${REPORT_DIR}"
cd "${ROOT_DIR}" || exit 1

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

rm -f "${OUTPUT_DIR}"/*.json
nvidia-smi topo -m > "${REPORT_DIR}/nvidia_topology.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "${REPORT_DIR}/nvidia_gpus.csv" 2>/dev/null || true

write_error() {
  local name="$1"
  local code="$2"
  local path="$3"
  local command="$4"
  python3 - "$name" "$code" "$path" "$command" <<'PY'
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
    write_error "${name}" "${code}" "${error_output}" "$*"
    return "${code}"
  fi
}

make_config() {
  local run_id="$1"
  local gpus="$2"
  local precision="$3"
  local hook="$4"
  local batch="$5"
  local workers="$6"
  local prefetch="$7"
  local persistent="$8"
  local bucket_cap="$9"
  local visible_devices="${10}"
  python3 - "$BASE_CONFIG" "$CONFIG_DIR" "$run_id" "$gpus" "$precision" "$hook" "$batch" "$workers" "$prefetch" "$persistent" "$bucket_cap" "$visible_devices" "$DATA_DIR" "$STEPS" "$WARMUP_STEPS" <<'PY'
import json
import sys

(
    base,
    config_dir,
    run_id,
    gpus,
    precision,
    hook,
    batch,
    workers,
    prefetch,
    persistent,
    bucket_cap,
    visible_devices,
    data_dir,
    steps,
    warmup_steps,
) = sys.argv[1:16]
gpus = int(gpus)
batch = int(batch)
workers = int(workers)
prefetch = int(prefetch)
steps = int(steps)
warmup_steps = int(warmup_steps)
bucket_cap_value = int(bucket_cap) if str(bucket_cap).strip() else 0
persistent_bool = persistent.lower() in {"1", "true", "yes"}
with open(base, "r", encoding="utf-8") as handle:
    config = json.load(handle)
backend = "native" if gpus == 1 else "native_ddp"
parascale = config.setdefault("parascale", {})
parascale["training_backend"] = backend
parascale["precision"] = precision
parascale["data_parallel_size"] = gpus
parascale["ddp_gradient_as_bucket_view"] = True
parascale["ddp_static_graph"] = True
parascale["ddp_comm_hook"] = "none" if gpus == 1 else hook
parascale["ddp_bucket_cap_mb"] = bucket_cap_value or None
parascale["batch_size"] = batch
parascale["dataloader_num_workers"] = workers
parascale["dataloader_prefetch_factor"] = prefetch
parascale["dataloader_persistent_workers"] = persistent_bool
parascale["checkpoint_save_path"] = f"{config_dir}/{run_id}_ckpt"
parascale["checkpoint_save_interval"] = 999999
data = config.setdefault("data", {})
data["data_dir"] = data_dir
data["batch_size"] = batch
data["num_workers"] = workers
data["prefetch_factor"] = prefetch
data["persistent_workers"] = persistent_bool
data["num_samples"] = max(8192, batch * max(gpus, 1) * max(steps + warmup_steps, 1) * 2)
hardware = config.setdefault("hardware_profile", {})
hardware["num_gpus"] = gpus
hardware["gpus_per_node"] = gpus
training = config.setdefault("training", {})
training["max_steps"] = steps
training["benchmark_steps"] = steps
training["warmup_steps"] = warmup_steps
training["checkpoint_dir"] = f"{config_dir}/{run_id}_ckpt"
training["checkpoint_interval"] = 999999
training["skip_final_checkpoint"] = True
metadata = config.setdefault("benchmark_metadata", {})
metadata.update(
    {
        "suite": "a6000_native_ddp_scaling",
        "run_id": run_id,
        "gpus": gpus,
        "precision": precision,
        "ddp_comm_hook": parascale["ddp_comm_hook"],
        "ddp_bucket_cap_mb": parascale["ddp_bucket_cap_mb"],
        "visible_devices": visible_devices,
        "batch_per_gpu": batch,
        "num_workers": workers,
        "prefetch_factor": prefetch,
        "persistent_workers": persistent_bool,
        "data_dir": data_dir,
        "steps": steps,
        "warmup_steps": warmup_steps,
    }
)
out_path = f"{config_dir}/{run_id}.json"
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
print(out_path)
PY
}

run_config() {
  local run_id="$1"
  local gpus="$2"
  local precision="$3"
  local hook="$4"
  local batch="$5"
  local workers="$6"
  local prefetch="$7"
  local persistent="$8"
  local bucket_cap="${9:-0}"
  local visible_devices="${10:-}"
  local config
  config=$(make_config "${run_id}" "${gpus}" "${precision}" "${hook}" "${batch}" "${workers}" "${prefetch}" "${persistent}" "${bucket_cap}" "${visible_devices}")
  if [ "${gpus}" -eq 1 ]; then
    if [ -n "${visible_devices}" ]; then
      CUDA_VISIBLE_DEVICES="${visible_devices}" run_and_capture "${run_id}" python3 -m parascale.cli benchmark --config "${config}"
    else
      run_and_capture "${run_id}" python3 -m parascale.cli benchmark --config "${config}"
    fi
  else
    if [ -n "${visible_devices}" ]; then
      CUDA_VISIBLE_DEVICES="${visible_devices}" run_and_capture "${run_id}" torchrun --standalone --nproc_per_node="${gpus}" -m parascale.cli benchmark --config "${config}"
    else
      run_and_capture "${run_id}" torchrun --standalone --nproc_per_node="${gpus}" -m parascale.cli benchmark --config "${config}"
    fi
  fi
}

for precision in bf16 fp16 fp32; do
  for gpus in 1 2 4; do
    run_config "scale_${gpus}gpu_${precision}_none_b${DEFAULT_BATCH_PER_GPU}_w2" "${gpus}" "${precision}" none "${DEFAULT_BATCH_PER_GPU}" 2 2 false 0 "" || true
  done
done

for gpus in 2 4; do
  run_config "hook_${gpus}gpu_bf16_bf16_compress_b${DEFAULT_BATCH_PER_GPU}_w2" "${gpus}" bf16 bf16_compress "${DEFAULT_BATCH_PER_GPU}" 2 2 false 0 "" || true
  run_config "hook_${gpus}gpu_fp16_fp16_compress_b${DEFAULT_BATCH_PER_GPU}_w2" "${gpus}" fp16 fp16_compress "${DEFAULT_BATCH_PER_GPU}" 2 2 false 0 "" || true
done

for bucket_cap in 25 50 100 200; do
  run_config "bucket_4gpu_bf16_bf16_compress_bucket${bucket_cap}_b${DEFAULT_BATCH_PER_GPU}_w2" 4 bf16 bf16_compress "${DEFAULT_BATCH_PER_GPU}" 2 2 false "${bucket_cap}" "" || true
done

for topology in 0123 0134 1234; do
  visible=$(python3 - "${topology}" <<'PY'
import sys
print(",".join(sys.argv[1]))
PY
)
  run_config "topo_4gpu_bf16_bf16_compress_bucket100_cuda${topology}_b${DEFAULT_BATCH_PER_GPU}_w2" 4 bf16 bf16_compress "${DEFAULT_BATCH_PER_GPU}" 2 2 false 100 "${visible}" || true
done

for workers in 0 2 4 8; do
  if [ "${workers}" -eq 0 ]; then
    run_config "data_4gpu_bf16_none_b${DEFAULT_BATCH_PER_GPU}_w0" 4 bf16 none "${DEFAULT_BATCH_PER_GPU}" 0 2 false 0 "" || true
  else
    run_config "data_4gpu_bf16_none_b${DEFAULT_BATCH_PER_GPU}_w${workers}_p2" 4 bf16 none "${DEFAULT_BATCH_PER_GPU}" "${workers}" 2 false 0 "" || true
    run_config "data_4gpu_bf16_none_b${DEFAULT_BATCH_PER_GPU}_w${workers}_p4_persist" 4 bf16 none "${DEFAULT_BATCH_PER_GPU}" "${workers}" 4 true 0 "" || true
  fi
done

python3 tests/benchmarks/tools/summarize_a6000_native_ddp_scaling.py \
  --input "${OUTPUT_DIR}" \
  --output "${REPORT_DIR}/summary.json" \
  --markdown "${REPORT_DIR}/README.md" \
  --hardware "5x RTX A6000, measured with 1/2/4 visible GPUs" \
  --image "${IMAGE:-parascale-ci:a6000-cu126-torch25}" \
  --dataset "${DATA_DIR}" \
  --model "clip_medium" \
  --steps "${STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --batch-per-gpu "${DEFAULT_BATCH_PER_GPU}"
