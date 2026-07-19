# Ascend Functional Validation

- Suite: `ascend_validation`
- Hardware: Ascend 910B4
- Image: `quay.io/ascend/llamafactory:latest-npu-a2`
- Passed: False
- Blocked: True
- Input directory: `tests\benchmarks\reports\ascend_validation`
- Blocked reason: SSH authentication failed for user1@47.107.62.29:30303: Permission denied (publickey,password).
- Missing required runs: `doctor, tiny_hccl, tiny_single`

## Runs

| Run | OK | Status | Mode | Backend | Step | Throughput | Loss | torch_npu | NPU count | Return code |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
