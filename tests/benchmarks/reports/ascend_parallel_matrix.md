# Ascend Parallel Training Matrix

- Suite: `ascend_parallel_matrix`
- Hardware: Ascend 910B4
- Image: `quay.io/ascend/llamafactory:latest-npu-a2`
- Steps: 80
- Warmup steps: 10
- Batch size: 8
- Passed: True
- Input directory: `tests/benchmarks/reports/ascend_parallel_matrix_raw_20260719`

## Scenario Summary

| Scenario | Containers | NPUs | OK | Aggregate pairs/s | Pairs/s/NPU | Loss | Peak memory GB sum | Peak memory GB max | Dataloader wait ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single_docker_2card | 1 | 2 | True | 130.348 | 65.174 | 2.119536 | 3.660 | 3.660 | 3.320 |
| two_docker_1card | 2 | 2 | True | 154.625 | 77.312 | 2.139495 | 6.492 | 3.246 | 3.267 |
| two_docker_2card | 2 | 4 | True | 265.301 | 66.325 | 2.119536 | 7.320 | 3.660 | 6.571 |

## Component Runs

| Run | Backend | OK | Runtime | Step | Throughput | Metric | Loss | Peak memory GB | Dataloader wait ms | Return code |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| single_docker_2card | native_ddp | True | real_local | 80 | 130.348 | stable_end_to_end_image_text_pairs_per_second | 2.119536 | 3.660 | 3.320 | n/a |
| two_docker_1card_a | ascend_native | True | real_local | 80 | 77.293 | stable_end_to_end_image_text_pairs_per_second | 2.139495 | 3.246 | 3.061 | n/a |
| two_docker_1card_b | ascend_native | True | real_local | 80 | 77.332 | stable_end_to_end_image_text_pairs_per_second | 2.139495 | 3.246 | 3.472 | n/a |
| two_docker_2card_a | native_ddp | True | real_local | 80 | 132.966 | stable_end_to_end_image_text_pairs_per_second | 2.119536 | 3.660 | 5.463 | n/a |
| two_docker_2card_b | native_ddp | True | real_local | 80 | 132.335 | stable_end_to_end_image_text_pairs_per_second | 2.119536 | 3.660 | 7.679 | n/a |
