# ParaScale Remote Server Test Guide

This guide is for moving the current workspace to a remote Linux server and running the production-runtime smoke tests.

## 1. Copy The Project

Preferred path:

```bash
git clone https://github.com/WangJunWJJ/ParaScale.git /path/to/ParaScale
cd /path/to/ParaScale
git pull --ff-only
```

If Git access is unavailable, create a temporary archive from Windows
PowerShell:

```powershell
Compress-Archive -Path G:\2-ParaScale-master\* -DestinationPath G:\parascale.zip -Force
scp G:\parascale.zip user@server:/path/to/
```

On the server:

```bash
cd /path/to
unzip -o parascale.zip -d 2-ParaScale-master
cd 2-ParaScale-master
```

Zip upload is a fallback for temporary validation only. Keep local helper
scripts outside the repository; repeatable benchmark scripts belong under
`tests/benchmarks/scripts/`.

## 2. Create Python Environment

```bash
cd /path/to/2-ParaScale-master
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install the server-specific PyTorch build separately if it is not already installed. Match the CUDA/NPU driver stack on the server.

Optional packages:

```bash
python -m pip install deepspeed
```

For Ascend, install the matching `torch_npu` package for the server image.

## 3. Baseline No-Hardware Test

```bash
python tests/run_tests.py
```

Expected local baseline:

```text
46 passed or higher, depending on the current branch
```

If torch is not installed, torch-dependent tests should skip cleanly.

## 4. Runtime Doctor

```bash
parascale doctor
```

or:

```bash
python -m parascale.cli doctor
```

Check these fields:

- `dependencies.torch`
- `dependencies.torch_npu`
- `dependencies.deepspeed`
- `torch_runtime.cuda_available`
- `torch_runtime.cuda_device_count`
- `distributed_runtime.available`
- `distributed_runtime.recommended_backends`
- `ascend_runtime.available`
- `device_backends`
- `rank_env`

This command is the first thing to run after upload. It tells you whether the server can run native torch, CUDA/NCCL, Ascend/HCCL, or only CPU/no-torch smoke tests.

## 5. Plan Smoke

```bash
parascale plan --config configs/server_tiny_torch.json
```

This only builds strategy and dataloader plans. It does not train.

## 6. Native Tiny Torch Training

Requires PyTorch.

```bash
parascale train --config configs/server_tiny_torch.json
```

Expected output fields:

- `mode: train`
- `runtime_status: real_local`
- `capability_level: local_native_real_torch`
- `synthetic: false`
- `global_step: 2`
- `train_device: cuda:0` on a CUDA server, or `cpu` when CUDA is unavailable
- `checkpoint: ./runs/server_tiny_torch/step-00000002/manifest.json`

The checkpoint directory should contain:

```text
runs/server_tiny_torch/step-00000002/manifest.json
runs/server_tiny_torch/step-00000002/backend_state.pt
```

## 7. Resume Smoke

```bash
parascale train --config configs/server_tiny_torch.json --resume-step 2
```

Expected output:

- `resumed_from.global_step: 2`
- `resumed_from.metadata.backend_state_loaded: true`
- final `global_step` should advance beyond the resumed step.

## 8. Non-Mock Tiny Serving

Use the checkpoint path produced by training:

```bash
parascale serve --config configs/server_tiny_torch.json --checkpoint runs/server_tiny_torch/step-00000002/manifest.json
```

Expected output:

- `mode: serve`
- `runtime_status: real_local`
- `capability_level: local_tiny_torch_checkpoint`
- `mock: false`
- `result.mode: model`
- `result.outputs` is a numeric list.

This is a tiny tensor inference path. It validates checkpoint-to-serving plumbing; it is not an LLM text generation server.

## 9. Mock Manifest Load

If you only want to validate manifest loading without torch model execution, set `serving.mock: true` in a config or run with a small temporary config.

Expected output:

- `runtime_status: mock`
- `capability_level: manifest_load_validation`
- `mock: true`

## 10. Distributed Smoke

FSDP smoke requires PyTorch distributed support:

```bash
python tests/run_tests.py --distributed --backend fsdp
```

DeepSpeed smoke requires DeepSpeed:

```bash
python tests/run_tests.py --distributed --backend deepspeed
```

Current distributed tests are smoke-level. They should either pass in a configured environment or skip/fail with explicit dependency diagnostics.

The distributed path is launched through:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 tests/distributed_runtime_smoke.py --backend fsdp
```

The runner performs this invocation for you when `--distributed` is provided. If torch is missing, it skips before invoking torchrun. If CUDA is missing, the smoke script skips with an explicit message because the current FSDP/DeepSpeed smoke is intended for GPU server validation.

## 11. Suggested Server Test Order

1. `python tests/run_tests.py`
2. `parascale doctor`
3. `parascale plan --config configs/server_tiny_torch.json`
4. `parascale train --config configs/server_tiny_torch.json`
5. `parascale train --config configs/server_tiny_torch.json --resume-step 2`
6. `parascale serve --config configs/server_tiny_torch.json --checkpoint runs/server_tiny_torch/step-00000002/manifest.json`
7. `python tests/run_tests.py --distributed --backend fsdp`
8. `python tests/run_tests.py --distributed --backend deepspeed`

## 12. Server Smoke Report

To capture a compact JSON report for later comparison:

```bash
parascale smoke --config configs/server_tiny_torch.json --output runs/server_smoke_report.json
```

The report includes:

- doctor output
- plan output
- train result
- resume result
- serve result
- elapsed time per step
- any error type/message if a step fails

If torch is not installed and you only want environment and planning diagnostics:

```bash
parascale smoke --config configs/server_tiny_torch.json --output runs/server_smoke_report.json --skip-real
```

The legacy script entrypoint remains available for test-runner use:

## 13. Troubleshooting

If `parascale train` says torch is required:

- Install a PyTorch build matching the server.
- Run `parascale doctor` again.

If CUDA devices are missing:

- Check `nvidia-smi`.
- Check PyTorch CUDA compatibility.
- Check `torch_runtime.cuda_available` in doctor output.

If distributed fails:

- Check `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `LOCAL_RANK`, `WORLD_SIZE`.
- Start with single-node smoke before multi-node.

If DeepSpeed fails:

- Check `dependencies.deepspeed`.
- Confirm the installed package matches the torch/CUDA stack.

If Ascend/NPU fails:

- Check `dependencies.torch_npu`.
- Check `ascend_runtime`.
- Confirm server image and driver/runtime packages match.
