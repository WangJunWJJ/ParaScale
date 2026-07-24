# A6000 Native-DDP Scaling

- Hardware: `5x RTX A6000, measured with 1/2/4 visible GPUs`
- Image: `parascale-ci:a6000-cu126-torch25`
- Dataset: `/dataset/datacomp_subsets/final/datacomp_10k_wds`
- Model: `clip_medium`
- Steps: 120
- Warmup steps: 20
- Batch per GPU: 8

## Scaling

| Precision | 1 GPU | 2 GPU | 4 GPU | 1->2 | 2->4 | 1->4 | 4 GPU efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 108.328 | 109.833 | 166.887 | 1.014x | 1.519x | 1.541x | 0.385 |
| fp16 | 109.122 | 103.289 | 161.905 | 0.947x | 1.567x | 1.484x | 0.371 |
| bf16 | 107.664 | 104.758 | 163.892 | 0.973x | 1.564x | 1.522x | 0.381 |

## Communication Hooks

| GPUs | Precision | Hook | Baseline pairs/s | Hook pairs/s | Relative |
| ---: | --- | --- | ---: | ---: | ---: |
| 2 | bf16 | bf16_compress | 104.758 | 150.030 | 1.432x |
| 4 | bf16 | bf16_compress | 163.892 | 235.913 | 1.439x |
| 2 | fp16 | fp16_compress | 103.289 | 141.748 | 1.372x |
| 4 | fp16 | fp16_compress | 161.905 | 230.336 | 1.423x |

## Runs

| Run | Group | GPUs | Precision | Hook | Workers | OK | Throughput | Loss | Peak GB | Wait ms |
| --- | --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| data_4gpu_bf16_none_b8_w0 | data | 4 | bf16 | none | 0 | True | 158.035 | 2.093406 | 3.403 | 14.638 |
| data_4gpu_bf16_none_b8_w2_p2 | data | 4 | bf16 | none | 2 | True | 166.495 | 2.090334 | 3.403 | 5.823 |
| data_4gpu_bf16_none_b8_w2_p4_persist | data | 4 | bf16 | none | 2 | True | 164.377 | 2.090334 | 3.403 | 6.011 |
| data_4gpu_bf16_none_b8_w4_p2 | data | 4 | bf16 | none | 4 | True | 163.793 | 2.092310 | 3.403 | 6.492 |
| data_4gpu_bf16_none_b8_w4_p4_persist | data | 4 | bf16 | none | 4 | True | 166.216 | 2.092310 | 3.403 | 6.596 |
| data_4gpu_bf16_none_b8_w8_p2 | data | 4 | bf16 | none | 8 | True | 164.471 | 2.092354 | 3.403 | 6.360 |
| data_4gpu_bf16_none_b8_w8_p4_persist | data | 4 | bf16 | none | 8 | True | 157.999 | 2.092354 | 3.403 | 6.759 |
| hook_2gpu_bf16_bf16_compress_b8_w2 | hook | 2 | bf16 | bf16_compress | 2 | True | 150.030 | 2.108442 | 3.473 | 5.690 |
| hook_2gpu_fp16_fp16_compress_b8_w2 | hook | 2 | fp16 | fp16_compress | 2 | True | 141.748 | 2.114015 | 3.476 | 5.730 |
| hook_4gpu_bf16_bf16_compress_b8_w2 | hook | 4 | bf16 | bf16_compress | 2 | True | 235.913 | 2.094805 | 3.473 | 6.740 |
| hook_4gpu_fp16_fp16_compress_b8_w2 | hook | 4 | fp16 | fp16_compress | 2 | True | 230.336 | 2.096042 | 3.476 | 6.072 |
| scale_1gpu_bf16_none_b8_w2 | scale | 1 | bf16 | none | 2 | True | 107.664 | 2.138274 | 3.147 | 5.633 |
| scale_1gpu_fp16_none_b8_w2 | scale | 1 | fp16 | none | 2 | True | 109.122 | 2.146698 | 3.147 | 5.198 |
| scale_1gpu_fp32_none_b8_w2 | scale | 1 | fp32 | none | 2 | True | 108.328 | 2.145933 | 3.254 | 5.137 |
| scale_2gpu_bf16_none_b8_w2 | scale | 2 | bf16 | none | 2 | True | 104.758 | 2.107500 | 3.403 | 5.807 |
| scale_2gpu_fp16_none_b8_w2 | scale | 2 | fp16 | none | 2 | True | 103.289 | 2.114282 | 3.403 | 6.094 |
| scale_2gpu_fp32_none_b8_w2 | scale | 2 | fp32 | none | 2 | True | 109.833 | 2.118129 | 3.674 | 6.127 |
| scale_4gpu_bf16_none_b8_w2 | scale | 4 | bf16 | none | 2 | True | 163.892 | 2.090334 | 3.403 | 6.943 |
| scale_4gpu_fp16_none_b8_w2 | scale | 4 | fp16 | none | 2 | True | 161.905 | 2.095981 | 3.403 | 5.661 |
| scale_4gpu_fp32_none_b8_w2 | scale | 4 | fp32 | none | 2 | True | 166.887 | 2.091758 | 3.674 | 5.893 |

## Best Dataloader Candidate

- Run: `data_4gpu_bf16_none_b8_w2_p2`
- Throughput: 166.495
- Dataloader wait ms: 5.823

## Notes

- This suite does not sweep DDP bucket_cap_mb because ParaScale does not expose it in the native-DDP config yet.
- Use this evidence to decide whether bucket_cap_mb should become a production config field.
