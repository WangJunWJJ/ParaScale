# ParaScale Benchmark Report

This report is the review entrypoint for ParaScale benchmark evidence. Each section is generated from compact `summary.json` artifacts; detailed scenario reports remain linked for auditability.

## How to Update

1. Run or refresh a benchmark suite so its `summary.json` is current.
2. Regenerate this file:

```bash
python tests/benchmarks/tools/build_benchmark_report.py --report-root tests/benchmarks/reports --output tests/benchmarks/reports/BENCHMARK_REPORT.md
```

3. Commit the updated summary artifacts and this report together.

## Overview

| Suite | Status | Hardware | Image | Detail |
| --- | --- | --- | --- | --- |
| Dual 4090 full validation | passed | dual RTX 4090D 24GB | `mixed: parascale-ci/vlm/yolo/grounding` | [dual_4090_full_validation.md](dual_4090_full_validation.md) |
| Direct PyTorch/DeepSpeed comparison | passed | dual RTX 4090D 24GB | `parascale-ci:cu121-torch24` | [direct_pytorch_clip_comparison.md](direct_pytorch_clip_comparison.md) |
| Ascend functional validation | passed | Ascend 910B4 | `quay.io/ascend/llamafactory:latest-npu-a2` | [ascend_validation.md](ascend_validation.md) |
| Ascend parallel matrix | passed | Ascend 910B4 | `quay.io/ascend/llamafactory:latest-npu-a2` | [ascend_parallel_matrix.md](ascend_parallel_matrix.md) |
| Cross-hardware CLIP DataComp | passed | multiple | `n/a` | [cross_hardware_clip_datacomp.md](cross_hardware_clip_datacomp.md) |
| RTX 4090 precision comparison | recorded | dual RTX 4090D 24GB | `parascale-ci:cu121-torch24` | [rtx4090_clip_precision_datacomp.md](rtx4090_clip_precision_datacomp.md) |

## Dual RTX 4090 Validation

| Model | OK runs | Total runs | Best backend | Throughput | Loss | Peak memory GB |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| clip | 3 | 3 | native_ddp | 131.603 | 2.801256 | 3.474 |
| ground | 1 | 1 | native | 1.344 | 74660.320312 | 3.240 |
| local | 1 | 1 | tests | 0.000 | n/a | 0.000 |
| tiny | 1 | 1 | smoke | 0.000 | n/a | 0.000 |
| vlm | 1 | 1 | native_ddp | 2843.336 | 9.190378 | 0.072 |
| yolo | 1 | 3 | native | 86.927 | 181.339515 | 0.658 |

## Direct Baseline Comparison

| ParaScale backend | Direct baseline | ParaScale throughput | Direct throughput | Ratio |
| --- | --- | ---: | ---: | ---: |
| parascale_native_ddp | torch_ddp | 134.183 | 80.383 | 1.669x |
| parascale_fsdp | torch_fsdp | 88.843 | 65.494 | 1.357x |

| DeepSpeed backend | Baseline | DeepSpeed throughput | Baseline throughput | Ratio |
| --- | --- | ---: | ---: | ---: |
| parascale_deepspeed | parascale_native_ddp | 91.116 | 134.183 | 0.679x |
| parascale_deepspeed | parascale_fsdp | 91.116 | 88.843 | 1.026x |
| parascale_deepspeed | torch_ddp | 91.116 | 80.383 | 1.134x |
| parascale_deepspeed | torch_fsdp | 91.116 | 65.494 | 1.391x |

## Ascend Functional Validation

| Run | OK | Mode | Backend | Step | Throughput | Loss | torch_npu | NPU count |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| doctor | True | doctor | n/a | 0 | 0.000 | n/a | True | 2 |
| tiny_hccl | True | train | native_ddp | 5 | 1.670 | 1.415908 | None | 0 |
| tiny_single | True | train | ascend_native | 5 | 2.267 | 0.630087 | None | 0 |

## Ascend Parallel Matrix

| Scenario | Containers | NPUs | OK | Aggregate pairs/s | Pairs/s/NPU | Loss | Peak memory GB max | Dataloader wait ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| single_docker_2card | 1 | 2 | True | 130.348 | 65.174 | 2.119536 | 3.660 | 3.320 |
| two_docker_1card | 2 | 2 | True | 154.625 | 77.312 | 2.139495 | 3.246 | 3.267 |
| two_docker_2card | 2 | 4 | True | 265.301 | 66.325 | 2.119536 | 3.660 | 6.571 |

## Cross-Hardware CLIP DataComp

- Dataset: `datacomp_10k_wds`
- Model: `clip_medium`
- Precision: `fp32`

| Label | Hardware | Backend | Throughput | Relative to baseline | Loss | Peak memory GB | Dataloader wait ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| rtx4090 | dual RTX 4090D 24GB | native_ddp | 77.653 | 1.000x | 2.507681 | 3.705 | 2.596 |
| ascend | Ascend 910B4 | native_ddp | 121.507 | 1.565x | 2.577154 | 3.640 | 11.183 |

## RTX 4090 CLIP Precision

| Precision | Backend | Throughput | Relative to FP32 | Step time ms | Loss | Peak memory GB | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 | native_ddp | 77.653 | 1.000x | 203.453 | 2.507681 | 3.705 | strict rerun |
| bf16 | native_ddp | 131.603 | 1.695x | 118.955 | 2.801256 | 3.474 | existing full validation |
| fp16 | native_ddp | 79.593 | 1.025x | 198.657 | 2.649531 | 3.446 | strict rerun |

## Evidence Files

| Suite | Summary | Detail report |
| --- | --- | --- |
| dual_4090 | [dual_4090_full_validation/summary.json](dual_4090_full_validation/summary.json) | [dual_4090_full_validation.md](dual_4090_full_validation.md) |
| direct_pytorch | [direct_pytorch_clip_comparison/summary.json](direct_pytorch_clip_comparison/summary.json) | [direct_pytorch_clip_comparison.md](direct_pytorch_clip_comparison.md) |
| ascend_validation | [ascend_validation/summary.json](ascend_validation/summary.json) | [ascend_validation.md](ascend_validation.md) |
| ascend_matrix | [ascend_parallel_matrix/summary.json](ascend_parallel_matrix/summary.json) | [ascend_parallel_matrix.md](ascend_parallel_matrix.md) |
| cross_hardware | [cross_hardware_clip_datacomp/summary.json](cross_hardware_clip_datacomp/summary.json) | [cross_hardware_clip_datacomp.md](cross_hardware_clip_datacomp.md) |
| rtx4090_precision | [rtx4090_clip_precision_datacomp/summary.json](rtx4090_clip_precision_datacomp/summary.json) | [rtx4090_clip_precision_datacomp.md](rtx4090_clip_precision_datacomp.md) |
