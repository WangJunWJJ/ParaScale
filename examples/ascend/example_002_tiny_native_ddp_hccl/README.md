# Ascend Example 002: Tiny Native-DDP HCCL

Run from the repository root inside an Ascend torch_npu environment:

```bash
bash examples/ascend/example_002_tiny_native_ddp_hccl/run.sh
```

Override the local process count with `NPROC_PER_NODE`, for example
`NPROC_PER_NODE=4 bash examples/ascend/example_002_tiny_native_ddp_hccl/run.sh`.

This example uses the unified runtime with NPU/HCCL hints and `native_ddp`.
