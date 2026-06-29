# YOLO-World Dataloader Optimized Report (2026-06-22)

## Setup

- Model: `/models/yolov8s-worldv2.pt`
- Dataset: `/dataset/cache/objects365_tiny_yolo`
- Dataloader: `num_workers=4`, `persistent_workers=true`, `prefetch_factor=2`, `pin_memory=true`
- Tensor cache: `/tmp/parascale_yolo_tensor_cache`
- Steady-state table skips the first 5 steps to remove worker startup/cache warmup cost.

## Steady-State Results

| Run | Steps used | Loss first | Loss last | Avg images/s | Avg e2e images/s | Avg wait ms | Avg cache hit | Avg decode ms | Avg resize ms | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native_cold | 45 | 36.854 | 43.442 | 113.575 | 72.464 | 10.370 | 0.111 | 2.082 | 3.888 | 680.5 MiB |
| native_warm | 45 | 24.499 | 21.703 | 116.354 | 103.662 | 2.107 | 1.000 | 2.205 | 3.915 | 678.8 MiB |
| native_ddp_warm | 45 | 23.679 | 27.520 | 115.224 | 92.586 | 9.172 | 0.433 | 2.347 | 3.388 | 739.8 MiB |

## Notes

- `native_cold` builds the tensor cache; `native_warm` reuses it.
- `native_ddp_warm` runs two GPU ranks with warmed cache and `ddp_find_unused_parameters=true`.
- The first optimized run was executed in Docker default shared memory. PyTorch reported worker bus errors at shutdown; production runs should use `--shm-size=8g` or reduce workers/pin memory.
- Full per-step metrics are in the remote file `/home/wangjun/work/ParaScale/tests/benchmarks/reports/yolo_remote_20260622_optimized/yoloworld_s_objects365_optimized_50step_report.json`.
