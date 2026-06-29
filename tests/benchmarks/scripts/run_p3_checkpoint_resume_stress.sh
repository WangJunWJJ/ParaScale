#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace}
OUTPUT_DIR=${OUTPUT_DIR:-runs/p3}
DATA_DIR=${DATA_DIR:-/dataset/datacomp_subsets/final/datacomp_10k_wds}
HF_MODEL_DIR=${HF_MODEL_DIR:-/models/openai_clip-vit-base-patch32}
RUN_HF_PRETRAINED=${RUN_HF_PRETRAINED:-1}
export OUTPUT_DIR
export DATA_DIR
export HF_MODEL_DIR
mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"
cd "${ROOT_DIR}"
rm -f "${OUTPUT_DIR}"/*.json

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

python3 - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
data_dir = os.environ["DATA_DIR"]
hf_model_dir = os.environ["HF_MODEL_DIR"]

def write_config(source, target, *, max_steps=None, data_samples=None, checkpoint_dir=None):
    config = json.loads(Path(source).read_text(encoding="utf-8"))
    if max_steps is not None:
        config["training"]["max_steps"] = max_steps
    if data_samples is not None:
        config["data"]["num_samples"] = data_samples
    if checkpoint_dir is not None:
        config["training"]["checkpoint_dir"] = checkpoint_dir
        config["parascale"]["checkpoint_save_path"] = checkpoint_dir
    config["data"]["data_dir"] = data_dir
    Path(target).write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

write_config(
    "tests/validation/configs/p3_datacomp_medium_bf16_native.json",
    "/tmp/parascale_p3_native_bf16.json",
    max_steps=4,
    data_samples=32,
    checkpoint_dir=str(output_dir / "native_bf16_ckpt"),
)
write_config(
    "tests/validation/configs/p3_datacomp_medium_zero3_deepspeed.json",
    "/tmp/parascale_p3_deepspeed_zero3.json",
    max_steps=4,
    data_samples=32,
    checkpoint_dir=str(output_dir / "deepspeed_zero3_ckpt"),
)
write_config(
    "tests/validation/configs/p3_datacomp_medium_zero3_deepspeed_activation_ckpt.json",
    "/tmp/parascale_p3_deepspeed_zero3_activation_ckpt.json",
    max_steps=2,
    data_samples=16,
    checkpoint_dir=str(output_dir / "deepspeed_zero3_activation_ckpt"),
)
hf = json.loads(Path("tests/validation/configs/p3_datacomp_hf_clip_pretrained_offline_smoke.json").read_text(encoding="utf-8"))
hf["data"]["data_dir"] = data_dir
hf["data"]["num_samples"] = 8
hf["training"]["max_steps"] = 1
hf["training"]["checkpoint_dir"] = str(output_dir / "hf_clip_pretrained_offline_ckpt")
hf["parascale"]["checkpoint_save_path"] = str(output_dir / "hf_clip_pretrained_offline_ckpt")
hf["model"]["pretrained_model_name_or_path"] = hf_model_dir
Path("/tmp/parascale_p3_hf_clip_offline.json").write_text(
    json.dumps(hf, ensure_ascii=False), encoding="utf-8"
)
PY

python3 -m parascale.cli train \
  --config /tmp/parascale_p3_native_bf16.json \
  --output "${OUTPUT_DIR}/native_bf16_train.json"

python3 -m parascale.cli train \
  --config /tmp/parascale_p3_native_bf16.json \
  --resume-step 2 \
  --output "${OUTPUT_DIR}/native_bf16_resume.json"

if command -v deepspeed >/dev/null 2>&1; then
  deepspeed --num_gpus=2 --module parascale.cli train \
    --config /tmp/parascale_p3_deepspeed_zero3.json \
    --output "${OUTPUT_DIR}/deepspeed_zero3_train.json"
  deepspeed --num_gpus=2 --module parascale.cli train \
    --config /tmp/parascale_p3_deepspeed_zero3.json \
    --resume-step 2 \
    --output "${OUTPUT_DIR}/deepspeed_zero3_resume.json"
  deepspeed --num_gpus=2 --module parascale.cli train \
    --config /tmp/parascale_p3_deepspeed_zero3_activation_ckpt.json \
    --output "${OUTPUT_DIR}/deepspeed_zero3_activation_ckpt_train.json"
else
  python3 - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["OUTPUT_DIR"]) / "deepspeed_zero3.error.json"
path.write_text(json.dumps({"backend": "deepspeed", "status": "error", "error": "deepspeed launcher is not available"}, indent=2) + "\n")
PY
fi

if [ "${RUN_HF_PRETRAINED}" = "1" ]; then
  if [ -d "${HF_MODEL_DIR}" ] && python3 -c "import transformers" >/dev/null 2>&1; then
    python3 -m parascale.cli train \
      --config /tmp/parascale_p3_hf_clip_offline.json \
      --output "${OUTPUT_DIR}/hf_clip_pretrained_offline_smoke.json"
  else
    python3 - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["OUTPUT_DIR"]) / "hf_clip_pretrained_offline_smoke.json"
path.write_text(json.dumps({"status": "skipped", "reason": "HF model directory or transformers is unavailable"}, indent=2) + "\n")
PY
  fi
fi

python3 tests/benchmarks/tools/summarize_p3_validation.py \
  --input-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/p3_stress_summary.json" \
  --markdown "tests/reports/archive/p3_mixed_precision_zero3_resume_report.md"
