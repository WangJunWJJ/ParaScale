# RTX 4090D CLIP DataComp Precision Comparison

- Hardware: dual RTX 4090D 24GB
- Image: `parascale-ci:cu121-torch24`
- Dataset: `datacomp_10k_wds`
- Model: `clip_medium`
- Backend: `native_ddp`
- Steps: 80
- Warmup steps: 10
- Batch size: 8 per GPU

| Precision | Throughput pairs/s | Relative to FP32 | Step time ms | Loss | Peak memory GB | Dataloader wait ms | Hook | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| fp32 | 77.653 | 1.000 | 203.453 | 2.507681 | 3.705 | 2.596 | none | strict rerun |
| bf16 | 131.603 | 1.695 | 118.955 | 2.801256 | 3.474 | 2.698 | bf16_compress | existing full validation |
| fp16 | 79.593 | 1.025 | 198.657 | 2.649531 | 3.446 | 2.367 | none | strict rerun |
