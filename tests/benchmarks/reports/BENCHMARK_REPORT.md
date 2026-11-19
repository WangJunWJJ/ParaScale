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
| Llama reference PTQ/QAT smoke | passed | 1x NVIDIA RTX A6000 | `parascale-quantization:a6000-torch25-transformers451` | [llama_quantization/summary.json](llama_quantization/summary.json) |
| Llama online distillation smoke | passed | 1x NVIDIA RTX A6000 (156 host) | `parascale-quantization:a6000-torch25-transformers451` | [llama_distillation/summary.json](llama_distillation/summary.json) |
| A6000 architecture contracts | recorded | 2x RTX A6000, host 156, GPU 0/1 | `parascale-quantization:a6000-torch25-transformers451` | [architecture_contracts/summary.json](architecture_contracts/summary.json) |
| Cross-model compression portability | recorded | Local CPU and 1x RTX A6000, host 156 | `parascale-quantization:a6000-torch25-transformers451` | [compression_portability/summary.json](compression_portability/summary.json) |
| YOLO detection distillation | recorded | 192.168.8.156, single RTX A6000 (logical CUDA 0) | `parascale-detection:a6000-ultralytics83161` | [detection_distillation/summary.json](detection_distillation/summary.json) |

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
| a6000_native_ddp_scaling | bucket_4gpu_bf16_bf16_compress_bucket100_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | bucket_4gpu_bf16_bf16_compress_bucket200_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | bucket_4gpu_bf16_bf16_compress_bucket25_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | bucket_4gpu_bf16_bf16_compress_bucket50_b8_w2 | real_local | n/a | n/a |
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
| a6000_native_ddp_scaling | topo_4gpu_bf16_bf16_compress_bucket100_cuda0123_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | topo_4gpu_bf16_bf16_compress_bucket100_cuda0134_b8_w2 | real_local | n/a | n/a |
| a6000_native_ddp_scaling | topo_4gpu_bf16_bf16_compress_bucket100_cuda1234_b8_w2 | real_local | n/a | n/a |

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
| fp32 | 106.347 | 108.178 | 166.446 | 1.017x | 1.539x | 1.565x | 0.391 |
| fp16 | 106.761 | 105.147 | 162.224 | 0.985x | 1.543x | 1.520x | 0.380 |
| bf16 | 120.904 | 107.386 | 164.346 | 0.888x | 1.530x | 1.359x | 0.340 |

| GPUs | Precision | Hook | Baseline throughput | Hook throughput | Relative |
| ---: | --- | --- | ---: | ---: | ---: |
| 2 | bf16 | bf16_compress | 107.386 | 147.074 | 1.370x |
| 4 | bf16 | bf16_compress | 164.346 | 242.931 | 1.478x |
| 2 | fp16 | fp16_compress | 105.147 | 140.377 | 1.335x |
| 4 | fp16 | fp16_compress | 162.224 | 227.676 | 1.403x |

| Bucket cap MB | Throughput | Relative to default | Dataloader wait ms |
| ---: | ---: | ---: | ---: |
| 25 | 232.048 | 0.955x | 6.335 |
| 50 | 235.256 | 0.968x | 6.534 |
| 100 | 235.593 | 0.970x | 6.013 |
| 200 | 228.409 | 0.940x | 6.532 |

| CUDA_VISIBLE_DEVICES | Bucket cap MB | Throughput | Dataloader wait ms |
| --- | ---: | ---: | ---: |
| `0,1,2,3` | 100 | 234.117 | 7.136 |
| `0,1,3,4` | 100 | 262.403 | 6.210 |
| `1,2,3,4` | 100 | 235.049 | 6.438 |

Best dataloader candidate: `data_4gpu_bf16_none_b8_w2_p2` at 167.074 pairs/s, wait 6.754 ms.

## Llama Reference PTQ/QAT

Random two-layer Llama, synthetic tokens: functional validation only, not pretrained model quality evidence. Float dequantization compute; packed weights do not promise acceleration. Pretrained validation remains pending.

| Run | PPL (synthetic) | State bytes | Checkpoint bytes | Forward ms | Reload max error | Nonzero QAT steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-bit BASELINE | 64.128 | 90752 | 99248 | 1.934 | n/a | n/a |
| 4-bit PTQ | 64.126 | 30336 | 70809 | 3.829 | 0.000000 | n/a |
| 4-bit QAT | 64.089 | 30336 | 70809 | 4.931 | 0.000000 | 16 |
| 8-bit BASELINE | 64.128 | 90752 | 99248 | 1.951 | n/a | n/a |
| 8-bit PTQ | 64.129 | 39552 | 80025 | 2.644 | 0.000000 | n/a |
| 8-bit QAT | 64.097 | 39552 | 80025 | 2.655 | 0.000000 | 16 |

See [summary and full measurements](llama_quantization/summary.json) and the [quantization guide](../../../docs/quantization.md).

## Llama Online Distillation

Random Llama teacher/student, synthetic tokens: functional validation only, not pretrained model quality evidence. Both runs share initialization and training data; evaluation uses held-out batches. Pretrained validation remains pending.

| Run | Teacher/student parameters | Held-out CE | PPL (synthetic) | KL (teacher to student) | Step ms | Nonzero gradient steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Distillation | 82240/13408 | 4.168 | 64.593 | 0.021149 | 15.225 | 32 |
| CE baseline | 82240/13408 | 4.169 | 64.675 | 0.024388 | 7.087 | 32 |

Teacher unchanged: True; no teacher gradients: True; student-only reload max error: 0.000000.

Timing includes forward/loss/backward, gradient checks and optimizer; excludes batch transfer, with no warmup. This is verification-loop timing, not production throughput. Teacher remains resident in both runs.

See [full measurements](llama_distillation/summary.json) and the [distillation guide](../../../docs/distillation.md).

## YOLO Detection Distillation

short real-data functional and training-cost validation

| Mode | Step images/s (mean +/- sd) | End-to-end images/s | Peak allocated MiB | mAP50-95 (0-100) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 105.46 +/- 4.91 | 66.91 | 291.5 | 37.14 |
| kd | 88.72 +/- 2.22 | 61.70 | 356.1 | 37.31 |

Configuration: YOLOv8s teacher -> YOLOv8n student; native FP32; 320x320; batch=4; AdamW lr=1e-5, weight_decay=0.0005; feature_weight=0.1; 200 steps/run, first 30 excluded, three timing repeats per mode.
COCO128: sorted first 96 images for training, remaining 32 for evaluation. These subsets are disjoint for this fine-tuning run, but COCO-pretrained models have already seen this dataset; this is not an unseen-data generalization study.
Initial teacher mAP50-95=51.23 and initial student=45.04. Final means: supervised=37.14, KD=37.31. Both decreased; the +0.17-point difference does not establish a reliable accuracy benefit. Do not deploy this short-run recipe on the strength of this result.
End-to-end throughput includes measured data wait and is 7.78% lower with KD. Step-only throughput excludes data wait and checkpointing. Peak allocated memory rises from 291.5 to 356.1 MiB; this is allocated tensor memory, not total GPU reservation.
All repeats use seed 42; non-deterministic GPU kernels remain enabled for timing. These repeats measure timing variability, not multi-seed quality confidence. Exact resume tests separately enable deterministic kernels.
An earlier 40-step exploratory run using the generic default lr=1e-3 and weight_decay=0.01 caused severe accuracy degradation for both modes; its measurements are retained below, not presented as a successful training recipe.
The adapter uses square resize and ParaScale's optimizer, without the full Ultralytics augmentation/EMA/scheduler recipe. It matches all foreground/background features equally, not FGD or classification/box-distribution distillation.
Functional validation passed, but no accuracy-improvement claim is made. Raw configs, per-step metrics, input manifests, checkpoints and official validation output remain under the remote evidence root.

[Full measurements and validation scope](detection_distillation/summary.json)

## A6000 Architecture Contracts

Synthetic two-layer 1024-wide MLP, native DDP, per-rank microbatch 64, GAS 2, 20 warmup and 500 measured optimizer updates, three repeats. Resident GPU inputs exclude dataloader and storage. Short-window runtime overhead measurements only; not end-to-end model training or quality evidence.

| Precision | Baseline median samples/s | Fixed median samples/s | Change |
| --- | ---: | ---: | ---: |
| fp32 | 65645.2 | 63995.6 | -2.51% |
| bf16 | 59568.1 | 57198.5 | -3.98% |

Remote full suite: 707 passed, 1 skipped.
Two-rank DDP, FSDP and DeepSpeed Stage 0 FP32 numerical contracts passed; maximum parameter error 3.73e-9 against concatenated global-batch reference.
FP16 DDP checkpoint/resume: exact parameters, optimizer, GradScaler and rank-local RNG continuation.
DeepSpeed correctness image: parascale-ci:a6000-cu126-torch25 (DeepSpeed 0.19.3). No ZeRO-2/3 or Ascend validation in this run.
Performance clipping disabled in both revisions because the baseline ignored clipping. Synthetic FP32/BF16 throughput is not a real-workload speedup claim.

### Normalization Performance Optimization

Same 2x A6000 GPUs 0/1 and synthetic MLP as above; 20 warmup, 2000 measured updates, three repeats alternating async then source. Before is the correctness-fixed synchronous implementation, NOT original commit 9ce3737. Resident GPU inputs, no clipping, no checkpoints in timed region. Medians are observations, not statistically established speedups or real-model results.

| Precision | Before samples/s | After samples/s | Change |
| --- | ---: | ---: | ---: |
| fp32 | 62894.0 | 64331.9 | +2.29% |
| bf16 | 53471.1 | 57409.8 | +7.37% |

Implementation: async global denominator reduction, wait after forward, retain denominator on device; no assumed equal-batch fast path.
Correctness: DDP/FSDP/DeepSpeed Stage 0 unequal-batch tests passed, maximum parameter error 3.73e-9.
Local full suite: 709 passed, 3 skipped; torch-free suite: 422 passed, 4 skipped.
Per-run variance remains visible; do not compare these 2000-step medians directly to the earlier 500-step table.
Final remote full suite: 711 passed, 1 skipped.
Final async implementation: dual-GPU FP16 checkpoint continuation passed exactly for parameters, optimizer, scaler and per-rank RNG.

### Accumulation Statistics Optimization

Second round: async normalization baseline versus detached, pre-fence accumulated-loss statistics and per-window divisor reuse. Same synthetic MLP, 2x A6000 GPU0/1, microbatch64, GAS2, 20 warmup/2000 measured updates, three repeats alternating baseline and changed code. No profiler during throughput runs; medians are observations with visible variance, not real-model evidence.

| Precision | Before samples/s | After samples/s | Change |
| --- | ---: | ---: | ---: |
| fp32 | 63159.7 | 62987.1 | -0.27% |
| bf16 | 56090.3 | 56827.7 | +1.31% |

Separate 30-update profiling: aten::div 240 -> 210; aten::add 60 -> 30; cudaDeviceSynchronize 61 -> 61. Profiling elapsed time is excluded from throughput comparisons.
FP32 throughput is essentially unchanged; do not interpret sub-percent differences as a speedup.
DDP/FSDP/DeepSpeed Stage 0 unequal-batch correctness passed with maximum parameter error 3.73e-9.
Full suites: remote 712 passed, 1 skipped; local CPU Torch 710 passed, 3 skipped.
Benchmark scope remains synthetic; no quality or real-dataset scaling claim.
Dual-GPU FP16 checkpoint/resume passed exact parameter, optimizer, scaler and rank-local RNG continuation.

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
| llama_quantization | [llama_quantization/summary.json](llama_quantization/summary.json) | n/a |
| llama_distillation | [llama_distillation/summary.json](llama_distillation/summary.json) | n/a |
| architecture_contracts | [architecture_contracts/summary.json](architecture_contracts/summary.json) | n/a |
| compression_portability | [compression_portability/summary.json](compression_portability/summary.json) | n/a |
| detection_distillation | [detection_distillation/summary.json](detection_distillation/summary.json) | n/a |

## Cross-model Compression Portability

Random small Transformers architectures, single-device FP32 Python APIs. Not pretrained quality, inference acceleration, complete CLI workload coverage, or distributed compression evidence.

| Family | PTQ/QAT | Distillation |
| --- | --- | --- |
| BERT sequence classification | 4/8-bit PTQ/QAT and reload passed | Classification; different teacher/student widths passed |
| ViT image classification | 4/8-bit PTQ/QAT and reload passed | Classification; different teacher/student widths passed |
| ResNet image classification | 4/8-bit PTQ/QAT and reload passed | Classification; different teacher/student widths passed |
| Qwen2 causal LM | 4/8-bit PTQ/QAT and reload passed | Causal LM; different teacher/student widths passed |

Each QAT test asserts nonzero gradients, actual weight changes, frozen-observer conversion parity, reload parity and an unchanged source model; no convergence or accuracy claim.
Distillation checks student updates and unchanged teacher weights/buffers with no teacher gradients; class/token alignment remains the caller's responsibility.
GPT-2 default projection selection is explicitly rejected: Transformers Conv1D is unsupported. This does not imply GPT-2 logits are unsuitable for the separate causal distillation loss.
PTQ now accepts Mapping calibration inputs, including tokenizer-style batches, with finite-value validation preserved.
Storage-alias checking is indexed once. The 12-Linear regression reduces storage queries from 1152 to at most 48; this is preparation complexity evidence, not GPU inference throughput.
Reproduce: PARASCALE_PORTABILITY_DEVICE=cuda:0 python -m pytest tests/test_compression_portability.py -q (PyTorch and Transformers required).
