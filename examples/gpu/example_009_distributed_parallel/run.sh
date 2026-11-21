#!/usr/bin/env bash
set -euo pipefail

mode=${1:-tp2pp2}
output=${2:-/tmp/parascale-parallel-$(date +%Y%m%d-%H%M%S)}
case "$mode" in
  single) dp=1; tp=1; pp=1 ;;
  dp2) dp=2; tp=1; pp=1 ;;
  dp4) dp=4; tp=1; pp=1 ;;
  tp2) dp=1; tp=2; pp=1 ;;
  pp2) dp=1; tp=1; pp=2 ;;
  dp2tp2) dp=2; tp=2; pp=1 ;;
  dp2pp2) dp=2; tp=1; pp=2 ;;
  tp2pp2) dp=1; tp=2; pp=2 ;;
  *) printf 'Unsupported mode: %s\n' "$mode" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python}" -m torch.distributed.run --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" --nproc_per_node="$((dp * tp * pp))" \
  tests/benchmarks/scripts/validate_distributed_parallel.py \
  --dp "$dp" --tp "$tp" --pp "$pp" --model "${MODEL:-residual_mlp}" --output "$output"
