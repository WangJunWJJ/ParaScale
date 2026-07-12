# Direct PyTorch vs ParaScale CLIP Comparison

- Suite: `direct_pytorch_clip_comparison`
- Hardware: dual RTX 4090D 24GB
- Image: `parascale-ci:cu121-torch24`
- Passed: True
- Input directory: `runs/benchmarks/direct_pytorch_clip_comparison`

## Runs

| Label | Stack | Backend | Step | Throughput | Metric | Loss | Peak memory GB | Dataloader wait ms |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| parascale_native_ddp | ParaScale | native_ddp | 80 | 134.108 | stable_end_to_end_image_text_pairs_per_second | 2.123793 | 3.473 | 0.519 |
| parascale_fsdp | ParaScale | fsdp | 80 | 88.811 | stable_end_to_end_image_text_pairs_per_second | 2.113393 | 1.991 | 0.466 |
| torch_ddp | Direct PyTorch | torch_ddp | 80 | 80.485 | stable_end_to_end_image_text_pairs_per_second | 2.125024 | 3.465 | 3.053 |
| torch_fsdp | Direct PyTorch | torch_fsdp | 80 | 65.581 | stable_end_to_end_image_text_pairs_per_second | 2.123469 | 2.836 | 3.194 |

## Comparisons

| ParaScale | Direct PyTorch | ParaScale throughput | Direct throughput | Ratio |
| --- | --- | ---: | ---: | ---: |
| parascale_native_ddp | torch_ddp | 134.108 | 80.485 | 1.6662x |
| parascale_fsdp | torch_fsdp | 88.811 | 65.581 | 1.3542x |
