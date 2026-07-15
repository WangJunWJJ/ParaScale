# Direct Distributed Baselines vs ParaScale CLIP Comparison

- Suite: `direct_pytorch_clip_comparison`
- Hardware: dual RTX 4090D 24GB
- Image: `parascale-ci:cu121-torch24`
- Passed: True
- Input directory: `tests\benchmarks\reports\direct_pytorch_clip_comparison`

## Runs

| Label | Stack | Backend | OK | Step | Throughput | Metric | Loss | Peak memory GB | Dataloader wait ms |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| parascale_native_ddp | ParaScale | native_ddp | True | 80 | 134.183 | stable_end_to_end_image_text_pairs_per_second | 2.123793 | 3.473 | 0.459 |
| parascale_fsdp | ParaScale | fsdp | True | 80 | 88.843 | stable_end_to_end_image_text_pairs_per_second | 2.113393 | 1.991 | 0.460 |
| parascale_deepspeed | ParaScale | deepspeed | True | 80 | 91.116 | stable_end_to_end_image_text_pairs_per_second | 2.125000 | 3.753 | 0.431 |
| torch_ddp | Direct PyTorch | direct_ddp | True | 80 | 80.383 | stable_end_to_end_image_text_pairs_per_second | 2.125024 | 3.465 | 3.069 |
| torch_fsdp | Direct PyTorch | direct_fsdp | True | 80 | 65.494 | stable_end_to_end_image_text_pairs_per_second | 2.123469 | 2.836 | 3.200 |
| deepspeed | Direct DeepSpeed | deepspeed | False | 0 | 0.000 | n/a | n/a | 0.000 | 0.000 |

## DeepSpeed Backend Comparisons

| DeepSpeed backend | Baseline | DeepSpeed throughput | Baseline throughput | Ratio |
| --- | --- | ---: | ---: | ---: |
| parascale_deepspeed | parascale_native_ddp | 91.116 | 134.183 | 0.6790x |
| parascale_deepspeed | parascale_fsdp | 91.116 | 88.843 | 1.0256x |
| parascale_deepspeed | torch_ddp | 91.116 | 80.383 | 1.1335x |
| parascale_deepspeed | torch_fsdp | 91.116 | 65.494 | 1.3912x |

## Comparisons

| ParaScale | Direct baseline | ParaScale throughput | Direct throughput | Ratio |
| --- | --- | ---: | ---: | ---: |
| parascale_native_ddp | torch_ddp | 134.183 | 80.383 | 1.6693x |
| parascale_fsdp | torch_fsdp | 88.843 | 65.494 | 1.3565x |
