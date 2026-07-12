# Dual RTX 4090 Functional and Performance Validation

- Suite: `dual_4090_full_validation`
- Hardware: dual RTX 4090D 24GB
- Image: `mixed: parascale-ci/vlm/yolo/grounding`
- Passed: True
- Input directory: `runs/benchmarks/dual_4090_full_validation`

## Model Summary

| Model | OK runs | Total runs | Best backend | Best throughput | Loss | Peak memory GB |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| clip | 3 | 3 | native_ddp | 131.603 | 2.801256 | 3.474 |
| ground | 1 | 1 | native | 1.344 | 74660.320312 | 3.240 |
| local | 1 | 1 | tests | 0.000 | n/a | 0.000 |
| tiny | 1 | 1 | smoke | 0.000 | n/a | 0.000 |
| vlm | 1 | 1 | native_ddp | 2843.336 | 9.190378 | 0.072 |
| yolo | 1 | 3 | native | 86.927 | 181.339515 | 0.658 |

## Runs

| Run | Model | Backend | OK | Runtime | Step | Throughput | Metric | Loss | Peak memory GB | Dataloader wait ms |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| clip_deepspeed | clip | deepspeed | True | real_local | 80 | 89.491 | stable_end_to_end_image_text_pairs_per_second | 2.814704 | 3.753 | 2.553 |
| clip_fsdp | clip | fsdp | True | real_local | 80 | 87.796 | stable_end_to_end_image_text_pairs_per_second | 2.642228 | 1.994 | 2.365 |
| clip_native_ddp | clip | native_ddp | True | real_local | 80 | 131.603 | stable_end_to_end_image_text_pairs_per_second | 2.801256 | 3.474 | 2.453 |
| ground_native | ground | native | True | real_local | 1 | 1.344 | end_to_end_image_text_pairs_per_second | 74660.320312 | 3.240 | 4.325 |
| local_tests | local | tests | True | n/a | 0 | 0.000 | n/a | n/a | 0.000 | 0.000 |
| tiny_smoke | tiny | smoke | True | n/a | 0 | 0.000 | n/a | n/a | 0.000 | 0.000 |
| vlm_native_ddp | vlm | native_ddp | True | real_local | 40 | 2843.336 | stable_end_to_end_image_text_pairs_per_second | 9.190378 | 0.072 | 0.221 |
| yolo_native | yolo | yolo_native | False | n/a | 0 | 0.000 | n/a | n/a | 0.000 | 0.000 |
| yolo_native_ddp | yolo | yolo_native_ddp | False | n/a | 0 | 0.000 | n/a | n/a | 0.000 | 0.000 |
| yolo_proxy_native | yolo | native | True | real_local | 80 | 86.927 | stable_end_to_end_image_text_pairs_per_second | 181.339515 | 0.658 | 9.277 |
