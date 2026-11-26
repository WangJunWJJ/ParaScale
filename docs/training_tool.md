# ParaScale训练工具

交付范围以[冻结验收基线V1](production_acceptance.md)为准。本文描述开发候选版的入口，不表示全部真实模型指标已验收。与推理优化工具共享同一源码版本和wheel。

## 安装与检查

在独立Python 3.10+环境安装交付的wheel；下面文件名以实际交付版本为准：

```bash
python -m pip install "./parascale-0.2.0-py3-none-any.whl[training]"
parascale-train --version
parascale-train doctor --strict --require torch
parascale-train run --help
```

`training`配置固定Torch 2.9.1、Torchvision 0.24.1、Transformers 4.51.3和PEFT 0.17.0，其余依赖仍需在最终交付时锁定。它不是离线安装介质。DeepSpeed、YOLO、GroundingDINO和4bit优化器按相应功能额外安装；未安装依赖或不支持的组合不能视为验收通过。Ascend使用厂商匹配环境，不能直接套用CUDA依赖配置。

## 训练与恢复

配置文件及模型、数据资产由交付示例提供，不包含在wheel内。源码包中的tiny示例仅用于安装功能检查：

```bash
parascale-train config validate --config configs/quickstart/tiny_torch.yaml
parascale-train plan --config configs/quickstart/tiny_torch.yaml --json
parascale-train run --config configs/quickstart/tiny_torch.yaml --dry-run
parascale-train run --config configs/quickstart/tiny_torch.yaml
parascale-train run --config configs/quickstart/tiny_torch.yaml --resume-latest
```

恢复时保持模型、数据、优化器、调度器和执行策略身份不变。普通native训练的`max_steps`表示本次调用的更新预算；不能假定所有执行路径均把它视为全局累计目标。检查配置对应后端的语义和输出`global_step`/`resumed_from`。

分布式启动继续使用同一执行实现，例如已配置native_ddp的真实数据配置：

```bash
torchrun --standalone --nproc_per_node=4 --no-python parascale-train run --config train.json
```

产品命令与`python -m parascale.cli train`复用参数注册和处理函数。`prepare-text`、`checkpoint`、`config`、`plan`、`doctor`作为训练工具辅助命令保留；不包含在线服务。

## 压缩训练与模型交付

- QAT使用训练配置开启，导出的参考量化产物不自动代表整数内核加速。
- 蒸馏使用现有训练配置/适配器；必须与同结构同预算学生基线比较，不能把模型变小归因于蒸馏质量收益。
- 剪枝后的恢复训练必须创建新优化器，不复用原结构训练检查点。
- 交付推理时导出模型和tokenizer/processor，记录配置、压缩方式、模型/数据身份和环境；训练检查点并不自动等于部署产物。

真实用例见[GPU示例索引](../examples/gpu/README.md)。指定LLaMA-2-7B和各并行组合以独立验收记录为准，不用安装smoke替代。

## 错误与诊断

参数错误返回非零退出码；执行错误通过stderr输出结构化JSON。设置`PARASCALE_DEBUG=1`可获得原始异常栈。先检查`doctor`、配置和磁盘空间，再排查后端依赖及模型访问权。最终交付前仍需完成新环境安装和24小时候选版本稳定性门禁。
