#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-runs/benchmarks/vlm_lora_datacomp}
CONFIG=${CONFIG:-tests/benchmarks/configs/benchmark_vlm_lora_datacomp_native_ddp.json}
OUTPUT=${OUTPUT:-${OUTPUT_DIR}/native_ddp.json}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-29651}
export NCCL_SOCKET_IFNAME=${PARASCALE_NCCL_SOCKET_IFNAME:-lo}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

mkdir -p "${OUTPUT_DIR}"
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" -m parascale.cli benchmark \
  --config "${CONFIG}" --output "${OUTPUT}"
