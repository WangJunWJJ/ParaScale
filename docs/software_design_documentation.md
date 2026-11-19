# ParaScale 软件设计文档

文档基线：2026-11-19

状态：对应当前开发分支的架构修复、参考量化、在线蒸馏和检测蒸馏实现；不是软件发布版本号。

## 1. 定位与范围

ParaScale 是面向视觉与图文多模态任务、以训练为主的轻量控制层，复用 PyTorch native/DDP、FSDP、DeepSpeed 等后端。项目不替代这些后端，不是通用 RL/具身智能 actor-learner 框架，也不是 Megatron 级超大规模预训练系统。

训练和推理位于同一工程，复用部分配置、设备、检查点和指标基础设施，但当前不是完整的生产级训推一体平台。训练 checkpoint 到内置 `serve` 的直接加载仅覆盖 tiny 模型；真实 CLIP、YOLO-World 的 `infer` 与标准 YOLO 蒸馏导出是不同路径。

历史硬件数字、阶段性试验和原始摘要统一进入[测试报告](../tests/benchmarks/reports/BENCHMARK_REPORT.md)，不在设计文档重复维护。未来目标不得写成当前交付能力。

## 2. 模块边界

```text
CLI -> commands -> configuration / strategy
                    |
                    +-> workloads/specs + workload builders
                    |        |
                    |        +-> data/vision, data/multimodal, optimizers
                    |
                    +-> runtime/train_runner -> runtime/training -> runtime/backends
                    |        +-> quantization / distillation
                    |        +-> checkpoint
                    |
                    +-> workloads/inference -> runtime/inference
                    +-> workloads/serving -> runtime/serve_runner -> serving
                    +-> reporting -> tests/benchmarks/reports
```

| 模块 | 职责 | 不应承担的职责 |
| --- | --- | --- |
| `cli.py`、`commands/` | 根 parser、模块化命令注册、错误分类、启动编排 | 不实现模型训练循环 |
| `configuration/`、`config.py` | 配置来源解析、规范化、运行时配置投影 | 不在 step 热路径重复解析 |
| `contracts/` | batch、backend、workload、指标等跨模块协议 | 不引入逐步深拷贝与序列化 |
| `workloads/specs/` | 按 tiny、vision、CLIP、VLM LoRA、YOLO、GroundDINO 解析场景配置 | 不持有设备通信实现 |
| `workloads/` | 装配 model、optimizer、dataloader、loss callable | 不另起训练引擎 |
| `data/` | 预处理、采样、缓存、collator、数据等待观测 | 不拥有训练/服务生命周期 |
| `runtime/training/` | fit、累积、精度、指标、恢复协作 | 不实现另一个 DeepSpeed |
| `runtime/backends/` | native、DDP、FSDP、DeepSpeed、Ascend 执行适配 | 不反向拥有 CLI |
| `quantization/`、`distillation/` | 模型变换与目标函数 | 不承担分布式调度 |
| `checkpoint/` | manifest、路径、校验和、格式契约 | 不把推理权重当作可恢复训练状态 |
| `runtime/inference/`、`serving/` | 推理上下文、任务适配、请求调度和结果隔离 | 不等同于完整网络服务平台 |
| `reporting/`、`tests/benchmarks/` | 运行证据、对照实验、统一报告 | 不将实验脚本变成生产入口 |

设备能力由 `core/device/` 表达，collective/process group 由 `core/distributed/` 表达，DDP hook 与 bucket 规划属于 `communication/`。GPU/NPU 类型与训练策略是不同维度。

## 3. 配置解析

当前仍存在 `parascale` 配置入口以及职责分组后的顶层配置，不能描述为已完全删除平铺模型。训练和规划通过 `normalize_execution_config`、`resolve_config` 收敛为同一有效配置，`ParaScaleConfig` 承载运行时字段。

| 配置节 | 所有权 |
| --- | --- |
| `backend` | training_backend、precision、后端策略 |
| `training` | workload、max_steps、累积步数、checkpoint/resume 和训练输出 |
| `optimizer` | optimizer 类型、lr、betas、weight_decay 等 |
| `data` | batch_size、数据路径、worker、cache、processor 参数 |
| `model`、`task`、`lora` | workload 专属模型与任务语义 |
| `runtime`、`hardware_profile`、`launch` | 设备请求、硬件事实、启动拓扑 |
| `parascale.quantization` | 参考 PTQ/QAT 配置 |
| `distillation` | 蒸馏目标和教师来源，独立于量化配置 |
| `serving`、`inference` | 各自执行入口的工作负载与请求设置 |

resolver 记录字段来源，优先级从低到高为默认值（0）、DeepSpeed 原生字段（35）、用户配置（40）、strategy/tuner（80）、CLI（90）、emergency override（100）。由规范化生成的副本必须保留来源，不得在重复解析时被误当成新的用户覆盖。DeepSpeed 冲突在执行前报错，最终执行配置由后端契约消费。

训练使用规范化后的 workload 参数，避免 plan 的 batch、precision、学习率与实际执行不一致。PTQ 命令目前直接读取 `parascale`，不能据此宣称所有命令已共享完全相同的 resolver 链路。推荐量化配置仍按[量化指南](quantization.md)填写。

## 4. 训练执行与性能

`run_train_from_config` 的顺序为：

1. 规范化配置、检查后端/精度/并行和量化/蒸馏组合。
2. 解析恢复位置，扩展 workload 的 replay 窗口，派生 rank 数据配置。
3. 构建 workload；必要时进行 QAT prepare 或检测蒸馏包装。
4. 在模型变换之后创建最终 optimizer，确保量化参数或投影层被纳入。
5. 创建 TrainEngine，恢复检查点，执行训练并保存、校验训练状态。
6. 按任务导出量化模型或纯学生权重，输出指标及目标函数元数据。

`training.max_steps` 表示本次要求执行的 optimizer 更新预算，不是累计 microbatch 数。workload 供给量需覆盖累积窗口；恢复时还需覆盖已消费的数据位置。DeepSpeed 返回是否真正完成更新，global_step 只在更新边界递增。

累积逻辑在 `runtime/training/accumulation.py` 和 `loss_weight.py` 收敛。各 workload 明确 loss 的归一化权重；对可按该权重组合的损失，按有效样本/目标总量校正，而不是简单平均不同大小的 microbatch。该契约不意味着 CLIP 的跨 batch 负样本集合可由累积自动复现，也不自动使 token-level 蒸馏支持累积。

native 路径先完成反向传播和必要的 AMP unscale，再执行梯度裁剪、优化器更新。DeepSpeed 使用自身更新边界与配置。归一化归约尽量与计算重叠，累积窗口复用分母并在设备端汇总 detached loss；性能结论必须引用对照实验，不能仅依据减少同步的代码推断。

## 5. 数据契约

- `DistributedSampler` 的 rank 长度需要一致，尾部填充/丢弃按策略处理，避免 collective 次数不匹配。
- `workloads/specs/common.py` 统一换算训练 microbatch 供给量。
- 真实 HF CLIP 使用与模型匹配的 AutoProcessor，不使用仅在 tiny 模型中适用的任意 token ID 代替真实 tokenizer。
- 通用视觉预处理、cache 与 collator 属于 `data/vision/`；YOLO 标签和官方 batch 字段转换属于 `workloads/adapters/yolo.py`。
- `yolo_detection` 复用 YOLO 数据管道，但只接受官方检测损失，不允许 proxy loss。标准 Detect 的 DFL 固定积分卷积保持冻结。
- 当前 YOLO 图像路径采用 square resize，不等同于 Ultralytics 完整 augmentation/letterbox/EMA/scheduler 训练配方。
- 数据异常必须传播，不允许 DataLoader 失败后静默更换实现而继续输出原策略的性能结论。

## 6. 检查点与恢复

`CheckpointManager` 管理 `step-XXXXXXXX/manifest.json` 和 payload；`runtime/training/checkpointing.py` 协调后端状态、各 rank 的运行时侧文件及 manifest 发布。

恢复状态包括模型、optimizer、可选 scheduler、精度 scaler、随机数和数据消费位置。随机数处理集中在 `random_state.py`，rank-local 状态不得用 rank 0 的状态代替所有 rank。分布式写入/读取错误需要协调传播，校验和、world size、后端布局、optimizer 和训练目标契约必须在恢复时验证。

数据实现有 state_dict 协议时优先恢复该状态；否则明确警告并 replay-skip。恢复数据迭代位置时保护训练随机数状态。精确恢复还要求可重复的数据顺序、worker 行为和确定性算子，不能把恢复了 RNG 等同于任意输入管道均逐位一致。

检测蒸馏保存 student、projectors 和 optimizer，教师仅通过 checkpoint/config 哈希描述。纯学生导出不可覆盖源模型、教师 checkpoint 或训练目录下的 step 子目录；部署文件不包含 optimizer，不能用于恢复训练。

## 7. 参考 PTQ/QAT

| 对象 | 设计职责 |
| --- | --- |
| QuantizationConfig | 验证位宽、reference 后端及受支持组合 |
| PostTrainingQuantization | prepare、校准、convert、export |
| QuantizationAwareTraining | fake quant、真实梯度更新、observer 冻结与转换 |
| 量化层 / load_quantized_model | packed 存储、浮点反量化执行、匹配浮点模板的重载 |
| `core.py` | 层选择、存储别名索引、参数共享拒绝与序列化辅助 |

当前验证重点为 weight-only 4/8 位、FP32 reference 执行，不提供 GPTQ/AWQ 或融合 INT4/INT8 加速 kernel。精确支持的模块类型为 PyTorch Linear/Conv2d；共享的选中参数、自定义子类和 GPT-2 Conv1D 等边界需拒绝或单独适配。

校准接收 Tensor、约定的输入/标签对和 Mapping（含 BatchEncoding）。存储别名预索引避免逐个选中参数再次扫描全模型。QAT optimizer 必须在 prepare 后建立。

单进程 native FP32 QAT 使用完整 checkpoint；AMP、FSDP/DeepSpeed、分布式 QAT 以及 QAT+KD 不属于已验收组合。4-bit optimizer、模型权重量化和 DDP 梯度通信压缩是三种不同能力。

## 8. 在线蒸馏

### 8.1 分类与因果语言模型

`DistillationConfig` 不导入 torch；`OnlineDistillationLoss` 持有独立教师，冻结参数、清理梯度并以 eval/no-grad 前向。教师不注册为学生子模块。

```text
total = (1-alpha) * CE + alpha * T^2 * KL(P_teacher(T) || P_student(T))
```

分类对齐类别语义与次序；causal_lm 对齐 token ID、位置和 shifted next-token 标签，并过滤无效位置。不同隐藏维度不构成障碍，但不同 tokenizer 或类别空间不能直接蒸馏。

### 8.2 检测专用适配

`DetectionDistillationConfig` 仅表达 detection 目标及非负 feature_weight。`create_detection_distillation` 返回训练模型和独立 loss callable：

| 对象 | 持有内容 |
| --- | --- |
| DetectionDistillationModel | student、三尺度 1x1 projectors；进入 optimizer 和训练 checkpoint |
| DetectionDistillationObjective | 冻结教师、配置快照、官方损失 callable、最近指标；不进入学生 state_dict |
| DetectionDistillationOutput | 官方前向结果和投影后的多尺度特征 |

通过 Detect 的临时 pre-hook 提取 neck 特征，先将输入列表转为 tuple，防止 head 原地替换列表造成捕获失效；无论成功失败都清理 hook。投影层在 optimizer 前初始化，保护 CPU/设备 RNG，避免改变采样或 dropout 序列。

```text
normalize(F) = F / sqrt(mean_CHW(F^2) + 1e-6)
feature_loss = mean_scales MSE(normalize(project(F_student)), normalize(F_teacher))
total = official_detection_loss + feature_weight * batch_size * feature_loss
```

保留官方 batch-summed 损失尺度。教师看到同一预处理图像，并延迟放置到对应设备。前景和背景等权；这不是 FGD，也未实现分类 logit 或框回归分布蒸馏。

只接受同系列标准 DetectionModel、三个 Detect 尺度、相同类别映射和 stride。检查师生共享存储，拒绝 YOLO-World、pose/segment、end-to-end 和跨系列头。YOLOv8、YOLO11 已有实际训练与恢复测试，验证上游版本为 Ultralytics 8.3.161。

feature_weight=0 跳过特征捕获、投影计算和教师推理。训练结束默认导出 checkpoint_dir/student.pt，其中仅含原始学生 DetectionModel state_dict；用匹配的 Ultralytics 模型加载，不是直接注册到 ParaScale infer/serve。

### 8.3 Runtime 集成

`runtime/distillation.py` 负责执行组合校验、教师加载、哈希和 objective metadata；不同目标的数学逻辑留在 `distillation/`。CLI 限定单进程 native、FP32、累积为 1、完整检查点；拒绝与量化联合启用。检测任务只接受 yolo_detection 学生和教师 workload。

## 9. 推理与服务

| 路径 | 当前实现 | 边界 |
| --- | --- | --- |
| `infer` | CLIP embedding、YOLO-World 的合成/真实适配与指标 | 不自动消费所有训练 workload 的检查点 |
| `serve` | 校验 manifest，加载 tiny 模型，提交配置中的请求并 drain | 非常驻 HTTP/gRPC 服务 |
| YOLO KD 导出 | 纯学生权重与重载预测一致性验证 | 使用 Ultralytics 推理；不是已完成一键上线 |
| PTQ/QAT 导出 | 匹配模板的 reference 重载 | 不是低比特推理服务后端 |

`runtime/inference/execution.py` 统一 eval/no-grad 执行并恢复原训练模式。服务批结果只接受明确的逐请求映射/序列合同，不把不匹配的单个结果广播给多请求；失败按请求隔离。

scheduler、batcher、KV cache 等工程模块不代表已实现大模型 continuous token batching、paged attention 或生成式 streaming。未来应先完成目标模型的标准化产物与 infer 适配，再讨论网络协议、部署、版本切换和回滚。

## 10. 测试与性能证据

统一入口为 `python tests/run_tests.py`，覆盖语法与 pytest。no-torch 环境验证配置和协议，CPU/GPU 环境验证真实张量行为。功能正确性、预训练精度和吞吐是独立验收维度。

| 设计域 | 主要回归入口 |
| --- | --- |
| 配置/预算/累积 | `tests/test_config_execution_contract_no_torch.py`、`tests/test_training_budget_no_torch.py`、`tests/test_training_execution_contract.py` |
| 数据/分布式 | `tests/test_distributed_data_contracts_torch.py`、`tests/benchmarks/scripts/distributed_training_contracts.py` |
| RNG/scaler/resume | `tests/test_checkpoint_runtime_state.py`、`tests/benchmarks/scripts/distributed_checkpoint_contracts.py` |
| 推理结果合同 | `tests/test_inference_execution_contract.py` |
| 压缩跨架构 | `tests/test_compression_portability.py` |
| 检测 KD | `tests/test_detection_distillation.py`、`tests/test_detection_training.py`、`tests/test_detection_distillation_runtime_no_torch.py` |

检测实际数据对照使用 YOLOv8s -> YOLOv8n、COCO128 固定 96/32 划分、单 A6000 FP32、三次 200 步。该子集在 COCO 预训练中已出现，不能视为独立泛化数据。两种微调结果均低于初始学生，不能宣称精度提升。数值和失败的高学习率探测保存在统一报告及 detection_distillation/summary.json。

Llama、BERT、ViT、ResNet、Qwen2 的部分测试使用小型随机模型，验证 API、参数更新及序列化，不证明预训练质量，也不自动增加 CLI workload 注册。

## 11. 发布与工程约束

- 新功能必须有对应回归；无硬件/依赖应明确 skip，不以 mock 冒充实测。
- 提交格式遵循 CONTRIBUTING.md 的 `<type>(<scope>): <subject>`；开发分支通过 PR 合并，不直接提交或推送主分支。
- 已发布 tag/release 不移动；文档基线更新不等于软件版本升级。
- 低频配置验证不得进入 step 热路径；不为保留旧 API 复制第二套执行逻辑。
- 远程密码、密钥、临时运行目录和大权重不纳入版本控制。
- Ascend、GPU 单机和多机的验收按实际报告分开；旧环境验证不能自动扩展到新增量化/蒸馏能力。

## 12. 文档关联

- [软件需求](software_requirements_specification.md)：编号、能力分级、验收要求。
- [用户手册](user_manual.md)：安装、配置、训练、压缩与推理操作。
- [量化指南](quantization.md)、[蒸馏指南](distillation.md)：数学目标、支持边界及完整示例。
- [统一测试报告](../tests/benchmarks/reports/BENCHMARK_REPORT.md)：历史环境和当前实测证据。
