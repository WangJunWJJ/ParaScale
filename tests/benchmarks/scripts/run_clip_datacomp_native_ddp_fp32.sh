#!/usr/bin/env bash
set -u

ROOT_DIR=${ROOT_DIR:-/workspace}
OUTPUT_DIR=${OUTPUT_DIR:-runs/benchmarks/clip_datacomp_native_ddp_fp32}
RUN_ID=${RUN_ID:-clip_datacomp_native_ddp_fp32}
DATA_DIR=${DATA_DIR:-/dataset/datacomp_subsets/final/datacomp_10k_wds}
ACCELERATOR=${ACCELERATOR:-cuda}
COMMUNICATION_BACKEND=${COMMUNICATION_BACKEND:-nccl}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
STEPS=${STEPS:-80}
WARMUP_STEPS=${WARMUP_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-2}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PRECISION=${PRECISION:-fp32}
DDP_COMM_HOOK=${DDP_COMM_HOOK:-none}

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"
cd "${ROOT_DIR}" || exit 1

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

CONFIG_PATH="${OUTPUT_DIR}/${RUN_ID}.config.json"
OUTPUT_PATH="${OUTPUT_DIR}/${RUN_ID}.json"
LOG_PATH="${OUTPUT_DIR}/${RUN_ID}.log"

python3 - "${CONFIG_PATH}" "${OUTPUT_DIR}" <<'PY'
import json
import os
import sys
from pathlib import Path

config_path, output_dir = sys.argv[1:3]
base = Path("tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp_b8_bf16_hook.json")
cfg = json.loads(base.read_text(encoding="utf-8"))

steps = int(os.environ.get("STEPS", "80"))
warmup_steps = int(os.environ.get("WARMUP_STEPS", "10"))
batch_size = int(os.environ.get("BATCH_SIZE", "8"))
num_workers = int(os.environ.get("NUM_WORKERS", "2"))
prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "2"))
nproc = int(os.environ.get("NPROC_PER_NODE", "2"))
accelerator = os.environ.get("ACCELERATOR", "cuda")
communication_backend = os.environ.get("COMMUNICATION_BACKEND", "nccl")
precision = os.environ.get("PRECISION", "fp32")
ddp_comm_hook = os.environ.get("DDP_COMM_HOOK", "none")
data_dir = os.environ["DATA_DIR"]

parascale = cfg.setdefault("parascale", {})
data = cfg.setdefault("data", {})
training = cfg.setdefault("training", {})
runtime = cfg.setdefault("runtime", {})
hardware = cfg.setdefault("hardware_profile", {})

parascale["training_backend"] = "native_ddp"
parascale["precision"] = precision
parascale["data_parallel_size"] = nproc
parascale["ddp_comm_hook"] = ddp_comm_hook
parascale["batch_size"] = batch_size
parascale["dataloader_num_workers"] = num_workers
parascale["dataloader_prefetch_factor"] = prefetch_factor
parascale["dataloader_persistent_workers"] = False

data["type"] = "datacomp_wds"
data["streaming"] = True
data["data_dir"] = data_dir
data["num_samples"] = max(512, steps * batch_size * nproc)
data["batch_size"] = batch_size
data["num_workers"] = num_workers
data["prefetch_factor"] = prefetch_factor
data["persistent_workers"] = False
data["image_size"] = 224
data["text_length"] = 64

training["max_steps"] = steps
training["benchmark_steps"] = steps
training["warmup_steps"] = warmup_steps
training["checkpoint_interval"] = 999999
training["checkpoint_dir"] = str(Path(output_dir) / f"{Path(config_path).stem}_ckpt")
training["skip_final_checkpoint"] = True
parascale["checkpoint_save_path"] = training["checkpoint_dir"]

runtime["accelerator"] = accelerator
runtime["communication_backend"] = communication_backend
hardware["device_type"] = "npu" if accelerator == "npu" else "cuda"
hardware["num_gpus"] = nproc
hardware["gpus_per_node"] = nproc

Path(config_path).write_text(
    json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

echo "[ParaScale strict CLIP] running ${RUN_ID}: accelerator=${ACCELERATOR}, backend=${COMMUNICATION_BACKEND}, precision=${PRECISION}, data=${DATA_DIR}"
if torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m parascale.cli benchmark --config "${CONFIG_PATH}" --output "${OUTPUT_PATH}" >"${LOG_PATH}" 2>&1; then
  rm -f "${OUTPUT_DIR}/${RUN_ID}.error.json"
  exit 0
fi

code=$?
tail -n 80 "${LOG_PATH}" || true
python3 - "${RUN_ID}" "${code}" "${OUTPUT_DIR}/${RUN_ID}.error.json" <<'PY'
import json
import sys

run_id, code, path = sys.argv[1:4]
payload = {
    "run_id": run_id,
    "status": "error",
    "returncode": int(code),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
exit 0
