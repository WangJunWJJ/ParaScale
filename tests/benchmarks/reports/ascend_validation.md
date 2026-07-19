# Ascend Functional Validation

- Suite: `ascend_validation`
- Hardware: Ascend 910B4
- Image: `quay.io/ascend/llamafactory:latest-npu-a2`
- Passed: True
- Blocked: False
- Input directory: `runs/benchmarks/ascend_validation`

## Runs

| Run | OK | Status | Mode | Backend | Step | Throughput | Loss | torch_npu | NPU count | Return code |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| doctor | True | ok | doctor | n/a | 0 | 0.000 | n/a | True | 2 | n/a |
| tiny_hccl | True | ok | train | native_ddp | 5 | 1.670 | 1.415908 | None | 0 | n/a |
| tiny_single | True | ok | train | ascend_native | 5 | 2.267 | 0.630087 | None | 0 | n/a |
