# ParaScale推理优化工具

本工具用于离线批量推理和模型优化，不提供在线HTTP/gRPC服务承诺。交付范围以[冻结验收基线V1](production_acceptance.md)为准；功能存在不等于质量、性能和全部发布门禁通过。

## 安装与检查

在独立Python 3.10+环境安装交付wheel：

```bash
python -m pip install "./parascale-0.2.0-py3-none-any.whl[inference]"
parascale-infer --version
parascale-infer doctor --strict --require torch
parascale-infer run --help
parascale-infer quantize --help
```

`inference`配置固定Torch 2.9.1、Torchvision 0.24.1、Transformers 4.51.3、Diffusers 0.35.1；不要求DeepSpeed。其余传递依赖仍需在交付环境锁定，安装配置不等于可离线复现镜像。CUDA INT8融合执行需额外安装`inference-int8`，并满足对应硬件和形状约束。

## 离线推理

准备本地合法模型权重、tokenizer及输入JSONL，按模型场景配置：

```bash
parascale-infer config validate --config inference.json
parascale-infer run --config inference.json --output results/summary.json
```

因果语言模型示例见[生成推理示例](../examples/gpu/example_013_causal_lm_inference/README.md)，diffusion见[扩散推理示例](../examples/gpu/example_016_diffusion_inference/README.md)。模型和输入路径、run目录及输出路径不能冲突。生成任务默认不覆盖已有输出；失败的流式输出保留`.partial`后缀。

`run`复用统一CLI的`infer`执行路径，不训练模型，不加载优化器状态。任务可选择浮点、参考量化或满足约束的融合量化执行。配置中的synthetic场景仅用于功能检查，不能作为真实模型性能证据。

## 模型优化边界

```bash
parascale-infer quantize --config ptq.json --output model-int8.pt
```

该PTQ命令复用现有workload配置、校准数据和导出逻辑，当前要求单进程native、开启PTQ。输出是ParaScale参考量化产物，必须通过匹配模型模板重载；不自动产生标准HF目录或加速内核。真实Llama PTQ流程见[PTQ示例](../examples/gpu/example_005_llama_ptq/README.md)。

结构剪枝目前通过公共Python接口`parascale.pruning.prune_llama_mlp`使用，见[剪枝示例](../examples/gpu/example_017_llama_pruning/README.md)。它不是新提供的CLI子命令。剪枝产物需保存对应HF配置及tokenizer，并重新评估质量；恢复训练属于训练工具。

QAT、蒸馏在训练侧完成，本工具消费其可重载学生/量化模型。不能把在推理工具中重新训练教师当作部署要求。

## 指标与限制

- 区分模型预测吞吐与含输入准备、模型加载、预热、输出I/O的作业时间。
- 生成token吞吐不等于TTFT、逐token延迟或在线并发SLA。
- 量化、剪枝分别验证真实执行方式、质量、文件/显存大小和速度，不由压缩比例推导加速比。
- 指定7B、蒸馏/剪枝质量收益及diffusion独立质量仍须按冻结清单验收。

排障使用`doctor`、结构化错误和`PARASCALE_DEBUG=1`。配套模型、配置示例和离线依赖介质属于交付资产，不由wheel自动下载。
