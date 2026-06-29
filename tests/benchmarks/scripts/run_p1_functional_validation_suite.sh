#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace}
OUTPUT_DIR=${OUTPUT_DIR:-runs/validation/p1}
DATA_DIR=${DATA_DIR:-/dataset/datacomp_subsets/final/datacomp_10k_wds}
YOLO_MODEL_DIRS=${PARASCALE_MODEL_DIRS:-/yolo_models:/models}
export OUTPUT_DIR
export DATA_DIR

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"
cd "${ROOT_DIR}"

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export YOLO_CONFIG_DIR=${YOLO_CONFIG_DIR:-/tmp/ultralytics}
export PARASCALE_MODEL_DIRS="${YOLO_MODEL_DIRS}"

rm -f "${OUTPUT_DIR}"/*.json

python3 -m parascale.cli smoke \
  --config configs/server_tiny_torch.json \
  --output "${OUTPUT_DIR}/server_tiny_smoke.json"

python3 - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
data_dir = os.environ["DATA_DIR"]
output_dir.mkdir(parents=True, exist_ok=True)

clip = json.loads(
    Path("tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp_b8_bf16_hook.json").read_text(
        encoding="utf-8"
    )
)
clip["training"]["max_steps"] = 4
clip["training"]["benchmark_steps"] = 4
clip["training"]["warmup_steps"] = 1
clip_ckpt = str(output_dir / "clip_native_ddp_ckpt")
clip["training"]["checkpoint_dir"] = clip_ckpt
clip["training"]["checkpoint_interval"] = 999999
clip["training"]["skip_final_checkpoint"] = False
clip["training"]["validate_resume"] = True
clip["training"]["resume_validation_steps"] = 1
clip["data"]["num_samples"] = 128
clip["data"]["data_dir"] = data_dir
clip["parascale"]["checkpoint_save_path"] = clip_ckpt
Path("/tmp/parascale_p1_clip_native_ddp.json").write_text(
    json.dumps(clip, ensure_ascii=False), encoding="utf-8"
)

yolo = json.loads(
    Path("tests/benchmarks/configs/benchmark_yolo_world_objects365_official_native_ddp.json").read_text(
        encoding="utf-8"
    )
)
yolo["training"]["max_steps"] = 3
yolo["training"]["benchmark_steps"] = 3
yolo["training"]["warmup_steps"] = 1
yolo_ckpt = str(output_dir / "yolo_native_ddp_ckpt")
yolo["training"]["checkpoint_dir"] = yolo_ckpt
yolo["training"]["checkpoint_interval"] = 999999
yolo["training"]["skip_final_checkpoint"] = False
yolo["training"]["validate_resume"] = True
yolo["training"]["resume_validation_steps"] = 1
yolo["data"]["num_samples"] = 16
yolo["parascale"]["checkpoint_save_path"] = yolo_ckpt
Path("/tmp/parascale_p1_yolo_native_ddp.json").write_text(
    json.dumps(yolo, ensure_ascii=False), encoding="utf-8"
)
PY

torchrun --standalone --nproc_per_node=2 -m parascale.cli benchmark \
  --config /tmp/parascale_p1_clip_native_ddp.json \
  --output "${OUTPUT_DIR}/clip_native_ddp_resume_benchmark.json"

torchrun --standalone --nproc_per_node=2 -m parascale.cli benchmark \
  --config /tmp/parascale_p1_yolo_native_ddp.json \
  --output "${OUTPUT_DIR}/yolo_native_ddp_resume_benchmark.json"

python3 tests/benchmarks/tools/summarize_p1_validation.py \
  --input-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/summary.json" \
  --markdown "tests/reports/archive/p1_functional_validation_report.md"
