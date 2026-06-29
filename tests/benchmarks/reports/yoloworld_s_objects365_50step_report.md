# YOLO-World Remote Training Report (2026-06-22)

## Setup

- Model: `/models/yolov8s-worldv2.pt`
- Dataset: `/dataset/cache/objects365_tiny_yolo`
- Container: `parascale-yolo:cu121-torch24-ultralytics83161`
- GPUs: NVIDIA GeForce RTX 4090 D, NVIDIA GeForce RTX 4090 D
- Steps: `50`; batch per rank: `2`

## Results

| Backend | Steps | Loss first | Loss last | Loss min | Avg images/s | Avg e2e images/s | Avg dataloader wait ms | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 50 | 28.992 | 28.105 | 12.324 | 110.584 | 56.988 | 23.476 | 678.8 MiB |
| native_ddp | 50 | 24.002 | 27.520 | 14.686 | 120.901 | 77.379 | 23.639 | 728.7 MiB |

## Notes

- `images/s` is measured around the training step; `e2e images/s` includes dataloader wait.
- `native_ddp` uses `ddp_find_unused_parameters=true` because YOLO-World official loss leaves some parameters unused on a step.
- Full raw payload and per-step metrics are in the remote file `/home/wangjun/work/ParaScale/tests/benchmarks/reports/yolo_remote_20260622/yoloworld_s_objects365_50step_report.json`.
