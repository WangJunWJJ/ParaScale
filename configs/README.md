# ParaScale 配置目录

配置按使用者视角分层，避免把历史 benchmark 配置暴露成默认入口。

## quickstart

`configs/quickstart/` 是用户首选入口：

- `tiny_torch.yaml`：最小真实训练、checkpoint、resume、serve smoke。
- `vision_synthetic.json`：视觉 synthetic workload。
- `clip_tiny.json`：tiny CLIP-style 图文对比学习。
- `vlm_lora_plan.yaml`：VLM LoRA 规划模板，不要求本地已有真实权重。

## benchmarks

`tests/benchmarks/configs/` 保存历史和专项 benchmark 配置。常规使用优先通过统一入口生成同口径矩阵：

```bash
python -m parascale.cli benchmark-matrix --scenario vlm-lora-hf-clip --dry-run
```

## validation

`tests/validation/configs/` 保存长训、resume stress、ZeRO-3、activation checkpointing 等稳定性验证配置。

## 根目录配置

根目录只保留少量通用示例和后端片段，后续新增配置应优先进入上述子目录。
