# ResolvedConfig 设计说明

## 目标

ParaScale 必须在训练、推理、benchmark 和远程复现实验前生成唯一、冻结、可打印、可保存的最终配置。该配置用于回答三个问题：

1. 最终字段值是什么。
2. 该字段来自哪里。
3. 是否被 CLI、策略规划、OOM retry 或后端默认值覆盖。

这可以避免用户配置、CLI override、workload 默认值、DeepSpeed 原生配置和硬件 profile 分散解释，导致长训和百卡规模问题难以复现。

## 输入与输出

输入层：

- `UserConfig`：用户配置文件。
- `CLI Override`：命令行或 benchmark-matrix 注入的覆盖项。
- `BackendConfig`：native/FSDP/DeepSpeed/Ascend 后端默认值与约束。
- `WorkloadConfig`：CLIP、YOLO、VLM LoRA 等 workload 默认值。
- `HardwareProfile`：world size、GPU/NPU 数量、显存、节点拓扑。

输出层：

- `ResolvedConfig`：冻结后的最终配置快照。
- `ResolvedField`：字段级值和来源。
- `config.resolved.json`：最终配置。
- `config.provenance.json`：字段来源与覆盖链。
- `backend.deepspeed.final.json`：DeepSpeed 最终配置。
- `config.warnings.json` / `config.errors.json`：冲突和风险说明。

## 字段来源优先级

优先级从低到高：

| 优先级 | 来源 |
|---:|---|
| 0 | built-in defaults |
| 20 | workload defaults |
| 30 | backend defaults |
| 35 | backend native config，例如 `deepspeed_config` |
| 40 | user config |
| 80 | strategy/tuner decision |
| 90 | CLI override |
| 100 | emergency/OOM retry override |

相同字段被高优先级来源覆盖时，`ResolvedField.overridden_by` 必须记录覆盖来源。

## DeepSpeed 冲突规则

P0 阶段必须检测：

- `precision=bf16` 与 `deepspeed_config.fp16.enabled=true` 冲突。
- `precision=fp16` 与 `deepspeed_config.bf16.enabled=true` 冲突。
- `deepspeed_config.optimizer` 与 ParaScale Python optimizer 同时存在时警告。
- `train_batch_size` 与 `train_micro_batch_size_per_gpu * gradient_accumulation_steps * world_size` 不一致时警告。
- `zero_stage=0` 必须保持为 0，不能被 DeepSpeed 后端隐式变成 2。

## P0 实施边界

P0 只做只读解析与报告接入：

- 新增 `parascale/configuration/resolved.py`。
- 新增 `parascale/configuration/resolver.py`。
- `plan`、`train --dry-run`、`benchmark --dry-run` 输出 `resolved_config`。
- DeepSpeed backend 使用同一份 final config 生成逻辑。

P0 不要求所有 runtime 代码立即改读 `ResolvedConfig`，但新增代码不得继续增加分散配置解释。

## P1 运行产物

每次训练、benchmark matrix 和 OOM retry 都使用独立运行目录。配置解析结果不写入
`parascale/` 源码或 `examples/`，而是与本次运行结果共同保存：

- `config.resolved.json`：最终值、字段来源、覆盖链、warning 和 error。
- `backend.deepspeed.final.json`：仅 DeepSpeed 运行生成，记录实际提交给
  DeepSpeed 的最终配置。

分布式执行仅允许 rank 0 原子写入。matrix 与 retry 报告必须透传
`config_artifacts`，使远程报告可追溯到本次运行的最终配置。OOM retry 的减 batch、
切后端与 ZeRO stage 变化记录为 `emergency override`，不能伪装成用户配置。
