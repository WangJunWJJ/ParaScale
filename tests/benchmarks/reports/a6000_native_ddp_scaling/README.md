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
| fp32 | 106.347 | 108.178 | 166.446 | 1.017x | 1.539x | 1.565x | 0.391 |
| fp16 | 106.761 | 105.147 | 162.224 | 0.985x | 1.543x | 1.520x | 0.380 |
| bf16 | 120.904 | 107.386 | 164.346 | 0.888x | 1.530x | 1.359x | 0.340 |

## Communication Hooks

| GPUs | Precision | Hook | Baseline pairs/s | Hook pairs/s | Relative |
| ---: | --- | --- | ---: | ---: | ---: |
| 2 | bf16 | bf16_compress | 107.386 | 147.074 | 1.370x |
| 4 | bf16 | bf16_compress | 164.346 | 242.931 | 1.478x |
| 2 | fp16 | fp16_compress | 105.147 | 140.377 | 1.335x |
| 4 | fp16 | fp16_compress | 162.224 | 227.676 | 1.403x |

## Bucket Cap

| Bucket cap MB | Throughput | Relative to default | Dataloader wait ms |
| ---: | ---: | ---: | ---: |
| 25 | 232.048 | 0.955x | 6.335 |
| 50 | 235.256 | 0.968x | 6.534 |
| 100 | 235.593 | 0.970x | 6.013 |
| 200 | 228.409 | 0.940x | 6.532 |

## Topology

| CUDA_VISIBLE_DEVICES | Bucket cap MB | Throughput | Dataloader wait ms | Peak GB |
| --- | ---: | ---: | ---: | ---: |
| `0,1,2,3` | 100 | 234.117 | 7.136 | 3.503 |
| `0,1,3,4` | 100 | 262.403 | 6.210 | 3.503 |
| `1,2,3,4` | 100 | 235.049 | 6.438 | 3.503 |

## Runs

| Run | Group | GPUs | Precision | Hook | Bucket MB | Visible devices | Workers | OK | Throughput | Loss | Peak GB | Wait ms |
| --- | --- | ---: | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| bucket_4gpu_bf16_bf16_compress_bucket100_b8_w2 | bucket | 4 | bf16 | bf16_compress | 100 | all | 2 | True | 235.593 | 2.091721 | 3.503 | 6.013 |
| bucket_4gpu_bf16_bf16_compress_bucket200_b8_w2 | bucket | 4 | bf16 | bf16_compress | 200 | all | 2 | True | 228.409 | 2.091550 | 3.554 | 6.532 |
| bucket_4gpu_bf16_bf16_compress_bucket25_b8_w2 | bucket | 4 | bf16 | bf16_compress | 25 | all | 2 | True | 232.048 | 2.092031 | 3.474 | 6.335 |
| bucket_4gpu_bf16_bf16_compress_bucket50_b8_w2 | bucket | 4 | bf16 | bf16_compress | 50 | all | 2 | True | 235.256 | 2.092947 | 3.481 | 6.534 |
| data_4gpu_bf16_none_b8_w0 | data | 4 | bf16 | none | 0 | all | 0 | True | 159.831 | 2.093406 | 3.403 | 14.722 |
| data_4gpu_bf16_none_b8_w2_p2 | data | 4 | bf16 | none | 0 | all | 2 | True | 167.074 | 2.090334 | 3.403 | 6.754 |
| data_4gpu_bf16_none_b8_w2_p4_persist | data | 4 | bf16 | none | 0 | all | 2 | True | 165.096 | 2.090334 | 3.403 | 6.138 |
| data_4gpu_bf16_none_b8_w4_p2 | data | 4 | bf16 | none | 0 | all | 4 | True | 163.250 | 2.092310 | 3.403 | 5.659 |
| data_4gpu_bf16_none_b8_w4_p4_persist | data | 4 | bf16 | none | 0 | all | 4 | True | 163.239 | 2.092310 | 3.403 | 6.792 |
| data_4gpu_bf16_none_b8_w8_p2 | data | 4 | bf16 | none | 0 | all | 8 | True | 164.575 | 2.092354 | 3.403 | 7.893 |
| data_4gpu_bf16_none_b8_w8_p4_persist | data | 4 | bf16 | none | 0 | all | 8 | True | 156.159 | 2.092354 | 3.403 | 9.183 |
| hook_2gpu_bf16_bf16_compress_b8_w2 | hook | 2 | bf16 | bf16_compress | 0 | all | 2 | True | 147.074 | 2.108442 | 3.473 | 5.671 |
| hook_2gpu_fp16_fp16_compress_b8_w2 | hook | 2 | fp16 | fp16_compress | 0 | all | 2 | True | 140.377 | 2.114015 | 3.476 | 6.105 |
| hook_4gpu_bf16_bf16_compress_b8_w2 | hook | 4 | bf16 | bf16_compress | 0 | all | 2 | True | 242.931 | 2.094805 | 3.473 | 5.622 |
| hook_4gpu_fp16_fp16_compress_b8_w2 | hook | 4 | fp16 | fp16_compress | 0 | all | 2 | True | 227.676 | 2.096042 | 3.476 | 5.328 |
| scale_1gpu_bf16_none_b8_w2 | scale | 1 | bf16 | none | 0 | all | 2 | True | 120.904 | 2.138274 | 3.147 | 5.268 |
| scale_1gpu_fp16_none_b8_w2 | scale | 1 | fp16 | none | 0 | all | 2 | True | 106.761 | 2.146698 | 3.147 | 5.791 |
| scale_1gpu_fp32_none_b8_w2 | scale | 1 | fp32 | none | 0 | all | 2 | True | 106.347 | 2.145927 | 3.254 | 5.379 |
| scale_2gpu_bf16_none_b8_w2 | scale | 2 | bf16 | none | 0 | all | 2 | True | 107.386 | 2.107500 | 3.403 | 5.364 |
| scale_2gpu_fp16_none_b8_w2 | scale | 2 | fp16 | none | 0 | all | 2 | True | 105.147 | 2.114282 | 3.403 | 6.510 |
| scale_2gpu_fp32_none_b8_w2 | scale | 2 | fp32 | none | 0 | all | 2 | True | 108.178 | 2.118158 | 3.674 | 6.377 |
| scale_4gpu_bf16_none_b8_w2 | scale | 4 | bf16 | none | 0 | all | 2 | True | 164.346 | 2.090334 | 3.403 | 6.806 |
| scale_4gpu_fp16_none_b8_w2 | scale | 4 | fp16 | none | 0 | all | 2 | True | 162.224 | 2.095981 | 3.403 | 6.480 |
| scale_4gpu_fp32_none_b8_w2 | scale | 4 | fp32 | none | 0 | all | 2 | True | 166.446 | 2.091754 | 3.674 | 6.558 |
| topo_4gpu_bf16_bf16_compress_bucket100_cuda0123_b8_w2 | topo | 4 | bf16 | bf16_compress | 100 | 0,1,2,3 | 2 | True | 234.117 | 2.091721 | 3.503 | 7.136 |
| topo_4gpu_bf16_bf16_compress_bucket100_cuda0134_b8_w2 | topo | 4 | bf16 | bf16_compress | 100 | 0,1,3,4 | 2 | True | 262.403 | 2.091721 | 3.503 | 6.210 |
| topo_4gpu_bf16_bf16_compress_bucket100_cuda1234_b8_w2 | topo | 4 | bf16 | bf16_compress | 100 | 1,2,3,4 | 2 | True | 235.049 | 2.091721 | 3.503 | 6.438 |

## Best Dataloader Candidate

- Run: `data_4gpu_bf16_none_b8_w2_p2`
- Throughput: 167.074
- Dataloader wait ms: 6.754

## Notes

- Bucket sweep uses bf16_compress on 4 visible GPUs.
- Topology sweep constrains CUDA_VISIBLE_DEVICES before torchrun.
