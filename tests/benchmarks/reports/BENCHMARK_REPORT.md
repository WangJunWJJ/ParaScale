# ParaScale Benchmark Report

This report is the review entrypoint for ParaScale benchmark evidence. Each section is generated from compact `summary.json` artifacts; config snapshots remain linked where they are needed for auditability.

## How to Update

1. Run or refresh a benchmark suite so its `summary.json` is current.
2. Regenerate this file:

```bash
python tests/benchmarks/tools/build_benchmark_report.py --report-root tests/benchmarks/reports --output tests/benchmarks/reports/BENCHMARK_REPORT.md
```

3. Commit the updated summary artifacts and this report together.

## Overview

| Suite | Status | Hardware | Image | Summary |
| --- | --- | --- | --- | --- |
| Dual 4090 full validation | passed | dual RTX 4090D 24GB | `mixed: parascale-ci/vlm/yolo/grounding` | [dual_4090_full_validation/summary.json](dual_4090_full_validation/summary.json) |
| Direct PyTorch/DeepSpeed comparison | passed | dual RTX 4090D 24GB | `parascale-ci:cu121-torch24` | [direct_pytorch_clip_comparison/summary.json](direct_pytorch_clip_comparison/summary.json) |
| Ascend functional validation | passed | Ascend 910B4 | `quay.io/ascend/llamafactory:latest-npu-a2` | [ascend_validation/summary.json](ascend_validation/summary.json) |
| Ascend parallel matrix | passed | Ascend 910B4 | `quay.io/ascend/llamafactory:latest-npu-a2` | [ascend_parallel_matrix/summary.json](ascend_parallel_matrix/summary.json) |
| Cross-hardware CLIP DataComp | passed | multiple | `n/a` | [cross_hardware_clip_datacomp/summary.json](cross_hardware_clip_datacomp/summary.json) |
| RTX 4090 precision comparison | recorded | dual RTX 4090D 24GB | `parascale-ci:cu121-torch24` | [rtx4090_clip_precision_datacomp/summary.json](rtx4090_clip_precision_datacomp/summary.json) |
| A6000 native-DDP scaling | passed | 5x RTX A6000, measured with 1/2/4 visible GPUs | `parascale-ci:a6000-cu126-torch25` | [a6000_native_ddp_scaling/summary.json](a6000_native_ddp_scaling/summary.json) |

## Evidence Quality

| Suite | Run | Runtime status | Capability level | Warmup/measured |
| --- | --- | --- | --- | ---: |
| dual_4090 | clip_deepspeed | real_local | local_native_clip_contrastive_datacomp_wds | n/a |
| dual_4090 | clip_fsdp | real_local | local_native_clip_contrastive_datacomp_wds | n/a |
| dual_4090 | clip_native_ddp | real_local | local_native_clip_contrastive_datacomp_wds | n/a |
| dual_4090 | ground_native | real_local | local_native_real_torch | n/a |
| dual_4090 | vlm_native_ddp | real_local | local_native_vlm_lora_synthetic | n/a |
| dual_4090 | yolo_proxy_native | real_local | local_native_real_torch | n/a |
| ascend_validation | doctor | diagnostic | n/a | n/a |
| ascend_validation | tiny_hccl | synthetic | n/a | n/a |
| ascend_validation | tiny_single | synthetic | n/a | n/a |
| ascend_matrix | single_docker_2card | real_local | n/a | n/a |
| ascend_matrix | two_docker_1card_a | real_local | n/a | n/a |
| ascend_matrix | two_docker_1card_b | real_local | n/a | n/a |
| ascend_matrix | two_docker_2card_a | real_local | n/a | n/a |
| ascend_matrix | two_docker_2card_b | real_local | n/a | n/a |
| cross_hardware | rtx4090 | real_local | n/a | n/a |
| cross_hardware | ascend | real_local | n/a | n/a |
| a6000_native_ddp_scaling | data_4gpu_bf16_none_b8_w0 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | data_4gpu_bf16_none_b8_w2_p2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | data_4gpu_bf16_none_b8_w2_p4_persist | real_local | n/a | n/a |
| a6000_native_ddp_scaling | data_4gpu_bf16_none_b8_w4_p2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | data_4gpu_bf16_none_b8_w4_p4_persist | real_local | n/a | n/a |
| a6000_native_ddp_scaling | data_4gpu_bf16_none_b8_w8_p2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | data_4gpu_bf16_none_b8_w8_p4_persist | real_local | n/a | n/a |
| a6000_native_ddp_scaling | hook_2gpu_bf16_bf16_compress_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | hook_2gpu_fp16_fp16_compress_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | hook_4gpu_bf16_bf16_compress_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | hook_4gpu_fp16_fp16_compress_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | scale_1gpu_bf16_none_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | scale_1gpu_fp16_none_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | scale_1gpu_fp32_none_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | scale_2gpu_bf16_none_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | scale_2gpu_fp16_none_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | scale_2gpu_fp32_none_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | scale_4gpu_bf16_none_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | scale_4gpu_fp16_none_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | scale_4gpu_fp32_none_b8_w2 | real_local | n/a | n/a |

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

## A6000 Native-DDP Scaling

- Dataset: `/dataset/datacomp_subsets/final/datacomp_10k_wds`
- Model: `clip_medium`
- Steps: 120
- Warmup steps: 20

| Precision | 1 GPU | 2 GPU | 4 GPU | 1->2 | 2->4 | 1->4 | 4 GPU efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 108.328 | 109.833 | 166.887 | 1.014x | 1.519x | 1.541x | 0.385 |
| fp16 | 109.122 | 103.289 | 161.905 | 0.947x | 1.567x | 1.484x | 0.371 |
| bf16 | 107.664 | 104.758 | 163.892 | 0.973x | 1.564x | 1.522x | 0.381 |

| GPUs | Precision | Hook | Baseline throughput | Hook throughput | Relative |
| ---: | --- | --- | ---: | ---: | ---: |
| 2 | bf16 | bf16_compress | 104.758 | 150.030 | 1.432x |
| 4 | bf16 | bf16_compress | 163.892 | 235.913 | 1.439x |
| 2 | fp16 | fp16_compress | 103.289 | 141.748 | 1.372x |
| 4 | fp16 | fp16_compress | 161.905 | 230.336 | 1.423x |

Best dataloader candidate: `data_4gpu_bf16_none_b8_w2_p2` at 166.495 pairs/s, wait 5.823 ms.

## Evidence Files

| Suite | Summary | Config snapshots |
| --- | --- | --- |
| dual_4090 | [dual_4090_full_validation/summary.json](dual_4090_full_validation/summary.json) | n/a |
| direct_pytorch | [direct_pytorch_clip_comparison/summary.json](direct_pytorch_clip_comparison/summary.json) | n/a |
| ascend_validation | [ascend_validation/summary.json](ascend_validation/summary.json) | n/a |
| ascend_matrix | [ascend_parallel_matrix/summary.json](ascend_parallel_matrix/summary.json) | n/a |
| cross_hardware | [cross_hardware_clip_datacomp/summary.json](cross_hardware_clip_datacomp/summary.json) | [cross_hardware_clip_datacomp/ascend/ascend_clip_datacomp_native_ddp_fp32.config.json](cross_hardware_clip_datacomp/ascend/ascend_clip_datacomp_native_ddp_fp32.config.json), [cross_hardware_clip_datacomp/rtx4090/rtx4090_clip_datacomp_native_ddp_fp32.config.json](cross_hardware_clip_datacomp/rtx4090/rtx4090_clip_datacomp_native_ddp_fp32.config.json) |
| rtx4090_precision | [rtx4090_clip_precision_datacomp/summary.json](rtx4090_clip_precision_datacomp/summary.json) | [rtx4090_clip_precision_datacomp/fp16/rtx4090_clip_datacomp_native_ddp_fp16.config.json](rtx4090_clip_precision_datacomp/fp16/rtx4090_clip_datacomp_native_ddp_fp16.config.json) |
| a6000_native_ddp_scaling | [a6000_native_ddp_scaling/summary.json](a6000_native_ddp_scaling/summary.json) | n/a |
