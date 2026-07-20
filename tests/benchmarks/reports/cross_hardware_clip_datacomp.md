# Cross-Hardware CLIP DataComp Comparison

- Suite: `cross_hardware_clip_datacomp_native_ddp_fp32`
- Dataset: `datacomp_10k_wds`
- Model: `clip_medium`
- Precision: `fp32`
- Steps: 80
- Warmup steps: 10
- Batch size: 8
- Passed: True

## Runs

| Label | Hardware | Image | Backend | OK | Runtime | Step | Throughput | Metric | Loss | Peak memory GB | Dataloader wait ms |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| rtx4090 | dual RTX 4090D 24GB | `parascale-ci:cu121-torch24` | native_ddp | True | real_local | 80 | 77.653 | stable_end_to_end_image_text_pairs_per_second | 2.507681 | 3.705 | 2.596 |
| ascend | Ascend 910B4 | `quay.io/ascend/llamafactory:latest-npu-a2` | native_ddp | True | real_local | 80 | 121.507 | stable_end_to_end_image_text_pairs_per_second | 2.577154 | 3.640 | 11.183 |

## Comparisons

| Label | Baseline | Throughput | Baseline throughput | Relative |
| --- | --- | ---: | ---: | ---: |
| rtx4090 | rtx4090 | 77.653 | 77.653 | 1.000 |
| ascend | rtx4090 | 121.507 | 77.653 | 1.565 |
