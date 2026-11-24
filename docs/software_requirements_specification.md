# ParaScale 软件需求规格说明书

文档基线：2026-11-24

状态：当前开发分支需求基线；不是软件发布版本号。2026-11-24 新增的目标功能和验收边界以[生产工具验收基线](production_acceptance.md)为准，尚未实现的目标不得作为当前能力声明。

## 1. 文档目标

本文定义 ParaScale 当前代码版本的软件需求边界、验收范围和能力分级，与[软件设计文档](software_design_documentation.md)保持一致，用于指导开发、审查和测试。硬件和性能证据统一引用[测试报告](../tests/benchmarks/reports/BENCHMARK_REPORT.md)，历史验证不等于本次提交重新验收。

ParaScale 的目标定位为：

```text
面向 Transformer、扩散模型及已支持视觉/图文任务的并行训练与推理优化工具。
```

现有视觉与图文训练能力是工程基础，不限制新增的 LLaMA-7B、模型剪枝和 diffusion 验收目标。本文后续章节中的“当前版本”描述实现边界；新增目标不意味着相关功能已经完成。核心价值链为：

```text
Config -> RuntimePlan -> Workload Adapter -> Data Pipeline -> Training/Inference Runtime
       -> Backend Adapter -> Profile -> Tuner Evidence -> Report -> Checkpoint/Resume
```

## 2. 产品定位

ParaScale 当前版本应支持：

- 纯视觉模型训练、评测、profile 和推理 runtime 骨架。
- 图文多模态训练，重点覆盖 VLM LoRA、CLIP-style 对比学习、DataComp WDS、YOLO/视觉检测适配。
- Nvidia CUDA/NCCL 路线下的 native-DDP、FSDP、DeepSpeed 训练后端选择、benchmark 和 fallback。
- Huawei Ascend NPU/HCCL 设备和通信抽象；已测场景以统一报告为准，不外推为全部功能生产可用。
- 训练、checkpoint、resume 和 benchmark 闭环；推理与 serving 只覆盖已实现的模型入口。
- 单设备 FP32 参考 PTQ/QAT，以及分类、因果语言建模和标准 YOLO 检测在线蒸馏。
- 基于 profile/tuner evidence 的自动后端选择、批处理策略建议和 OOM retry 规划。

## 3. 明确不纳入范围

本工程不是 DeepSpeed/PyTorch DDP/FSDP 替代品，不是通用 RL 或具身智能 actor-learner 框架，也不是 Megatron 级超大规模预训练系统。当前不交付通用生产级训推一体平台、HTTP/gRPC 模型服务或高性能 INT4/INT8 推理内核。模型量化、优化器状态量化和通信压缩是不同功能，不得混同。

当前多模态能力仅聚焦图文多模态：

- 文本字段：`input_ids`、`attention_mask`、`labels`、caption/prompt metadata。
- 图像字段：`pixel_values`、`image_sizes`、`num_images`、`patch_tokens`、image/text pair metadata。
- 训练目标：VLM LoRA、CLIP-style image-text contrastive、视觉检测/YOLO 适配。

## 4. 术语

| 术语 | 定义 |
|---|---|
| DeviceBackend | CPU/CUDA/Ascend 设备能力抽象 |
| CollectiveBackend | Gloo/NCCL/HCCL/mock 通信抽象 |
| RuntimePlan | 由配置、硬件、数据、策略、后端、推理和 checkpoint 子计划组成的运行计划 |
| Workload Adapter | 将 CLIP/DataComp/VLM LoRA/YOLO 等场景映射到统一 runtime contract 的薄适配层 |
| TrainEngine | 训练 runtime 编排入口，负责 fit loop、step、metrics、checkpoint 和 backend 协作 |
| InferenceEngine / ServingEngine | 通用推理 runtime 与上层服务编排入口，负责批处理、调度、KV cache、模型加载和 serving contract |
| TrainingBackend | native、FSDP、DeepSpeed、Ascend native 等训练执行后端契约 |
| Patch Token | 视觉模型中由图像分辨率和 patch size 推导出的计算 token |
| Token Budget | 按真实 token/patch-token 成本而非样本数组织 batch |
| RuntimeTuner | 根据 warmup profile、显存峰值、padding ratio、dataloader wait 和 OOM 反馈给出调优建议 |
| Capability Level | 能力声明等级，例如 interface ready、smoke verified、single-node verified、multi-node smoke、production verified |

## 5. 当前代码结构需求

当前代码应以 `parascale/` 为唯一主包，采用 rebuild 后的模块边界：

```text
parascale/
  cli.py
  config.py
  configuration/
  commands/
  contracts/
  core/device/
  core/distributed/
  data/vision/
  data/multimodal/
  workloads/
  workloads/specs/
  runtime/training/
  runtime/inference/
  runtime/backends/
  runtime/launcher/
  strategy/
  communication/
  checkpoint/
  reporting/
  serving/
  parallel/
  optimizers/
  quantization/
  distillation/
```

若代码结构、CLI 输出、能力边界或验收目标变化，本文、软件设计文档和用户手册必须同步更新。当前仍支持 `parascale` 平铺配置与按职责分组的配置，不能将分层设计方向写成已经完成的单一 schema。

## 6. 能力分级

所有功能声明必须带能力等级：

| 等级 | 含义 | 可声明能力 |
|---|---|---|
| interface ready | 接口、配置、plan 或 skeleton 已存在 | 可以声明架构已预留，不能声明可用 |
| smoke verified | 本地或远程少量 step/smoke 通过 | 可以声明链路可启动，不能声明性能或生产稳定 |
| single-node verified | 单机单卡或单机多卡真实训练/benchmark 已通过 | 可以声明单机能力可用 |
| multi-node smoke | 多机启动、通信初始化、少量 step smoke 通过 | 可以声明多机链路可启动 |
| production verified | 长窗口训练、checkpoint/resume、benchmark matrix、故障恢复均通过 | 可以声明该场景生产可用 |

禁止将 dry-run、mock、synthetic、tiny smoke 结果包装为真实性能优势或生产能力。

## 7. 功能需求

### 7.1 硬件抽象

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-HW-001 | 提供 `DeviceBackend` 抽象 | interface ready | CPU/CUDA/Ascend 后端实现同一设备能力接口 |
| FR-HW-002 | 支持 Nvidia GPU 能力发现 | smoke/single-node verified，依赖远程 CUDA 环境 | 可获取 device、显存、bf16、同步和 CUDA 相关能力 |
| FR-HW-003 | 支持 Huawei Ascend 抽象 | interface ready | 可表达 NPU device、显存、同步、HCCL 能力；实机验收另列，不声明当前生产可用 |
| FR-HW-004 | 设备能力进入策略系统 | interface ready | `StrategyPlan`/`RuntimePlan` 可读取设备类型、显存、world size、topology |
| FR-HW-005 | 支持集群拓扑表达 | interface ready | 可表达 node、rank、device、backend、node group 和异构约束 |

### 7.2 通信抽象

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-COMM-001 | 提供 `CollectiveBackend` | interface ready | 支持 all_reduce、reduce_scatter、all_gather、all_to_all、broadcast、barrier |
| FR-COMM-002 | 训练和推理复用通信层 | interface ready | TP、SP、sharding、serving TP 均通过统一通信接口表达 |
| FR-COMM-003 | 支持 Gloo/NCCL/HCCL/mock | selected scenarios | 各通信后端分别验收；HCCL 历史实机结果不能替代新增量化/蒸馏链路验收 |
| FR-COMM-004 | 支持 process group 管理 | interface ready | 可按 DP/TP/PP/SP/shard group 创建通信组 |
| FR-COMM-005 | 支持 DDP 通信规划 | single-node verified for selected scenarios | bf16 compression hook、bucket policy、no_sync、LoRA sync 等策略可被 plan/report 记录 |

### 7.3 训练 runtime

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-TR-001 | 提供 `TrainEngine` | smoke/single-node verified | 支持 setup、fit loop、train step、metrics、save/load checkpoint |
| FR-TR-002 | 支持 native-DDP | single-node verified for CLIP/DataComp/YOLO selected paths | 单机多卡中小视觉/图文场景可通过 native-DDP 训练和 benchmark |
| FR-TR-003 | 兼容 FSDP/DeepSpeed baseline | smoke verified | 可通过配置选择 FSDP 或 DeepSpeed，作为大模型/显存压力 fallback |
| FR-TR-004 | 支持 runtime tuning | smoke verified | warmup/profile 后输出后端、batch、checkpointing、offload、prefetch 等建议 |
| FR-TR-005 | 支持 OOM retry plan | smoke verified | OOM 类失败后生成 retry plan，记录 batch、fallback、日志和原因 |
| FR-TR-006 | native ZeRO Stage 1 | interface/smoke ready | 仅声明 optimizer-state sharding，不声明 native ZeRO Stage 2/3 |
| FR-TR-007 | 禁止虚假 native ZeRO Stage 2/3 | required | Stage 2/3 必须明确拒绝或 fallback 到 DeepSpeed/FSDP |
| FR-TR-008 | 统一优化器步数预算 | regression verified | `max_steps` 表示本次调用新增的优化器更新数；累积和恢复跳过数据后仍供应足够 microbatch |
| FR-TR-009 | 按有效样本权重归一化 | regression verified | 区分 batch-sum/mean，处理不等 microbatch 和 DDP 全局分母；不承诺 CLIP 小批累积等价于大批对比学习 |
| FR-TR-010 | 正确执行更新边界和裁剪 | regression verified | DeepSpeed 更新边界正确；AMP 梯度先 unscale 再裁剪；指标聚合不改变训练语义 |

### 7.4 推理 runtime

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-INF-001 | 提供推理 runtime | smoke verified | `runtime/inference` 和 `serving` 提供 engine、batcher、scheduler、KV cache skeleton |
| FR-INF-002 | 推理不依赖 DeepSpeed | smoke verified | 移除 DeepSpeed 后仍可执行 tiny/real-mode 边界 smoke |
| FR-INF-003 | 支持 continuous batching | smoke verified | scheduler 可合批请求，记录队列、批次和失败隔离指标 |
| FR-INF-004 | 支持有界 KV cache | smoke verified | KV cache 支持容量约束、近似 LRU、请求清理 |
| FR-INF-005 | 支持 tokenizer/HF/vLLM/TGI adapter | not complete | 后续需要接入 adapter，当前不得声明生产 serving |
| FR-INF-006 | 支持 prefill/decode 分离与 streaming | not complete | 后续实现后以 serving benchmark 验收 |
| FR-INF-007 | 支持视觉/图文推理 batching | interface ready | 当前表达 runtime 边界，生产支持需真实模型 adapter 验收 |
| FR-INF-008 | 推理状态与请求隔离 | regression verified | 推理采用 eval/no-grad，结束后恢复原训练状态；输出数量与请求不匹配时显式失败，不广播给错误请求 |

当前 `infer` 是一次性推理入口；`serve` 消费配置中的请求并排空队列，不是驻留网络服务。真实 CLIP/YOLO-World 推理与 tiny checkpoint serving 为不同路径；KV cache/scheduler 的骨架测试不等于生产生成引擎。

### 7.5 视觉训练与数据 runtime

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-VIS-001 | 提供 `data/vision` 子包 | smoke/single-node verified | 包含 dataset、preprocessor、transforms、sampler、cache、collator、profiler |
| FR-VIS-002 | 支持 resolution bucket | partial | 可按分辨率降低 padding/resize 浪费，并在 profile 中输出证据 |
| FR-VIS-003 | 支持 patch-token batch | smoke verified | 可按 `H / patch_size * W / patch_size` 控制 batch budget |
| FR-VIS-004 | 支持视觉增强 pipeline | partial | RandomResizedCrop 等基础增强可组合，复杂增强按场景逐步补充 |
| FR-VIS-005 | 支持视觉吞吐 profile | smoke verified | 输出 images/sec、patch_tokens/sec、decode、augment、H2D/transfer 等指标 |
| FR-VIS-006 | 支持 YOLO/视觉检测 workload adapter | smoke/single-node verified for selected paths | YOLO/Objects365 适配进入 workload 层，不承载通用 runtime 逻辑 |
| FR-VIS-007 | 支持 sample-based baseline 对照 | partial | benchmark 报告必须说明同硬件、同数据、同 batch budget 或差异原因 |

### 7.6 图文多模态训练与数据 runtime

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-MM-001 | 固化图文 batch schema | smoke verified | 覆盖 input_ids、attention_mask、pixel_values、labels、metadata、num_images、patch_tokens、token_count |
| FR-MM-002 | 支持 modality-aware token estimator | smoke verified | 估计 text token、image/patch token、image-text pair token cost |
| FR-MM-003 | 支持 dynamic token/patch budget | partial | 按文本长度、图像分辨率和 patch-token 压力动态组 batch |
| FR-MM-004 | 支持 processor adapter | partial | VLM LoRA processor adapter 已接入，packing 和长训练 loop 继续完善 |
| FR-MM-005 | 支持 padding ratio profile | smoke verified | 输出 padding ratio 并反馈给 RuntimeTuner |
| FR-MM-006 | 支持 image cache/prompt cache | partial | 图像 decode/cache 和 prompt cache 可用于降低数据管线等待 |
| FR-MM-007 | 支持 VLM LoRA | smoke verified | 支持 VLM LoRA 主路径、adapter-only checkpoint、native/FSDP/DeepSpeed/ZeRO 矩阵配置 |
| FR-MM-008 | 支持 CLIP-style 对比学习 | single-node verified for selected paths | DataComp WDS/CLIP-style image-text contrastive 可进入 train/benchmark |
| FR-MM-009 | 不支持视频、音频、agent 数据协议 | required | 代码、配置、文档和验收均不得把视频/音频/agent protocol 纳入当前路线 |

### 7.7 LLM 相关能力

当前版本不以通用 LLM 训练框架为主要交付目标。Llama/Qwen2 小模型已有参考量化或因果语言蒸馏 Python API 验证，但不等于新增所有对应 CLI workload，也不证明预训练精度保持：

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-LLM-001 | LLM token budget 抽象 | interface ready | 可在配置/plan 中表达 token budget，不声明生产 LLM 训练优势 |
| FR-LLM-002 | sequence packing | not in current acceptance | 当前版本不作为验收项 |
| FR-LLM-003 | consumed token resume | not in current acceptance | 当前版本不作为验收项，checkpoint 以 step/sample/backend state 为主 |
| FR-LLM-004 | 100B 以下策略规划 | roadmap only | 仅保留策略分层思想，不作为当前版本验收项 |

### 7.8 策略系统

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-ST-001 | 提供静态 planner | smoke verified | 根据 config、model/workload、hardware、task type 输出 backend、batch policy、parallel policy |
| FR-ST-002 | 提供 profile 数据结构 | smoke verified | 收集 tokens/sec、images/sec、patch_tokens/sec、peak memory、dataloader wait、padding ratio |
| FR-ST-003 | 提供 RuntimeTuner | smoke verified | 基于 evidence 输出 action、reason、threshold、config_updates |
| FR-ST-004 | 支持 OOM retry plan | smoke verified | 自动降低 batch 或启用 activation checkpointing/fallback，并记录 retry lineage |
| FR-ST-005 | 支持任务差异化策略 | partial | vision、multimodal、serving 有差异化策略；通用 LLM 不作为当前重点验收 |
| FR-ST-006 | 支持可解释 plan/report | smoke verified | plan 和 benchmark report 必须包含 reasons、warnings、evidence |

### 7.9 Checkpoint

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-CKPT-001 | 提供 `CheckpointManager` | smoke verified | 统一管理 model、optimizer、scheduler、rng、backend state 和 data state |
| FR-CKPT-002 | 提供 manifest | smoke verified | checkpoint 包含 global_step、backend、parallel plan、payload metadata、validation status、last metrics |
| FR-CKPT-003 | 支持 checkpoint validate | smoke verified | smoke 和 benchmark 可自动校验 checkpoint 结构和 manifest |
| FR-CKPT-004 | 支持 resume | smoke/single-node verified for selected paths | native/DataComp/YOLO 等已验证路径支持 save/resume |
| FR-CKPT-005 | 支持 adapter-only checkpoint | smoke verified | VLM LoRA 可只保存 adapter、optimizer、scheduler、RNG 等状态 |
| FR-CKPT-006 | 支持格式转换 | not complete | HF/FSDP/DeepSpeed/Ascend converter 真实转换仍为后续工作 |
| FR-CKPT-007 | 训练到推理加载 | partial | 内置 tiny checkpoint serving 已覆盖；YOLO 检测学生导出由匹配架构的 Ultralytics 加载，不自动接入 ParaScale infer |
| FR-CKPT-008 | rank-local 运行状态恢复 | regression verified | 各 rank 保存和恢复 RNG/AMP scaler；检查点错误传播到所有参与 rank，避免单 rank 失败后挂起 |
| FR-CKPT-009 | 恢复数据与目标一致性 | regression verified | replay-skip 不额外消耗模型 RNG；蒸馏校验教师与目标配置指纹；导出权重不能充当完整续训检查点 |

### 7.10 CLI

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-CLI-001 | `doctor` | smoke verified | 输出环境、依赖、设备、配置诊断 |
| FR-CLI-002 | `plan` | smoke verified | 生成 runtime、launch、benchmark、strategy、dataloader 子计划 |
| FR-CLI-003 | `train` / `run` | smoke verified | 从 JSON/YAML config 启动训练或 smoke train |
| FR-CLI-004 | `benchmark` | smoke verified | 输出 benchmark_result、comparison contract 和关键指标 |
| FR-CLI-005 | `benchmark-matrix` | smoke/single-node verified for selected paths | 统一执行 native/FSDP/DeepSpeed 同口径矩阵 |
| FR-CLI-006 | `checkpoint validate` | smoke verified | 校验 checkpoint manifest 和 payload |
| FR-CLI-007 | `serve`/`infer` 命名边界 | implemented with limited scope | `infer` 为一次性推理，`serve` 为 tiny 请求队列执行；不得宣传为已提供网络服务 |
| FR-CLI-008 | 输出可审计 | required | JSON 输出必须显式标记 dry_run、mock、runtime_status、capability_level |

### 7.11 Benchmark 与 Reporting

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-BENCH-001 | 统一 BenchmarkResult | smoke verified | 记录 backend、config、hardware、strategy、batch policy、throughput、peak memory |
| FR-BENCH-002 | 支持后端矩阵对比 | smoke/single-node verified for selected paths | native-DDP/FSDP/DeepSpeed 在同配置下输出 JSON summary 和 Markdown report |
| FR-BENCH-003 | 支持同口径性能声明 | required | 同硬件、同数据、同 batch budget、同 warmup/measurement window，才允许声明优势 |
| FR-BENCH-004 | 支持 tuner evidence 报告 | smoke verified | 报告 action、reason、evidence、threshold、config_updates |
| FR-BENCH-005 | 支持中文报告 | smoke verified | reporting/markdown 可生成面向审查的中文报告 |

### 7.12 异构资源

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-HET-001 | 支持集群设备发现 | interface ready | 能识别 rank、node、device type、device model、memory、communication backend |
| FR-HET-002 | 支持 NodeGroup | interface ready | 可按 Nvidia/Ascend、GPU 型号、显存等级创建 node group |
| FR-HET-003 | 支持多型号 GPU 规划 | interface ready | A100/H100/L20/4090 等可进入 placement plan |
| FR-HET-004 | 支持 weighted DP | roadmap/interface | 不同 GPU 可按算力或 profile 吞吐分配不同 batch/token 量 |
| FR-HET-005 | 支持 GPU + Ascend 资源池规划 | interface ready | 同 planner 管理，但默认分组执行，不做高频同步混用 |
| FR-HET-006 | 禁止高频同步异构混用 | required | 默认禁止 TP/ZeRO/FSDP 高频同步组跨 GPU+Ascend |
| FR-HET-007 | 支持异构 PP 研究路径 | roadmap only | 必须显式配置并记录 conversion/communication cost，不作为当前生产验收 |

### 7.13 模型参考量化

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-QNT-001 | weight-only PTQ | API/selected CLI verified | 精确类型的 Linear/Conv2d 支持 4/8-bit；校准接受 tensor、输入标签对和关键字 Mapping |
| FR-QNT-002 | QAT 真实参数更新 | API/selected CLI verified | fake quantization 后学生权重确实更新；冻结 observer 后转换和重新加载结果一致 |
| FR-QNT-003 | 明确量化边界 | required | 仅 reference FP32；拒绝激活量化、融合、已选择的共享参数和自定义层子类；GPT-2 Conv1D 不在支持范围 |
| FR-QNT-004 | 原生训练集成约束 | guarded | QAT 为单进程 native FP32/full checkpoint；拒绝 AMP、FSDP、DeepSpeed、分布式及 QAT+KD 组合 |
| FR-QNT-005 | 压缩与性能分开验收 | required | packed 权重只证明存储格式；测量磁盘体积、执行显存和延迟，禁止直接声称推理加速或精度保持 |

### 7.14 在线知识蒸馏

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-KD-001 | 分类与因果语言蒸馏 | API/selected CLI verified | hard CE 与 teacher-to-student KL 混合；正确温度平方、因果位移、有效标签和 mask；调用方保证类别/token 语义对齐 |
| FR-KD-002 | 教师隔离 | regression verified | 教师冻结、eval/no-grad，不进入学生 optimizer/checkpoint/export；共享存储被拒绝 |
| FR-KD-003 | 标准 YOLO 特征蒸馏 | A6000 functional verified | 同家族、同类别/stride、三尺度 Detect；训练学生 1x1 投影，RMS 归一化特征 MSE 加官方检测损失 |
| FR-KD-004 | 检测适用边界 | guarded | 已测 YOLOv8/YOLO11、Ultralytics 8.3.161；拒绝跨家族、YOLO-World、分割、姿态、end-to-end head；不是 FGD 或框分布蒸馏 |
| FR-KD-005 | 续训与独立导出 | regression verified | 续训保留投影、优化器和教师指纹；学生导出不含教师/投影；禁止导出覆盖源权重或 step 检查点 |
| FR-KD-006 | 运行范围与质量声明 | guarded/quality unverified | CLI 只支持单进程 native FP32、GAS=1；COCO128 smoke 不证明未见数据泛化或精度提升 |

### 7.15 配置执行契约

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-CFG-004 | 计划与训练共享解析结果 | regression verified | 执行前规范化配置，保留默认、用户、策略、CLI 等来源和优先级；重复规范化不改变有效值 |
| FR-CFG-005 | 配置所有权明确 | implemented with exceptions | workload spec 管理场景字段，backend 管理执行参数；PTQ 命令仍直接读取 `parascale`，不得宣称所有命令已统一分层解析 |

## 8. 非功能需求

### 8.1 性能

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| NFR-PERF-001 | 图文多模态吞吐优于通用 baseline | partial/single-node selected | 仅在同硬件、同数据、同 batch budget 的报告中声明具体场景优势 |
| NFR-PERF-002 | 纯视觉吞吐优化 | partial | patch_tokens/sec、images/sec、dataloader wait 可观测，并与 sample-based baseline 对照 |
| NFR-PERF-003 | DeepSpeed/FSDP 对照 | smoke/single-node selected | benchmark matrix 比较 step time、吞吐、显存、稳定性 |
| NFR-PERF-004 | 大模型显存压力 fallback | smoke | DeepSpeed/FSDP/ZeRO-3/offload 作为 fallback，不声明 native 全替代 |
| NFR-PERF-005 | 禁止 smoke 结果冒充性能优势 | required | tiny/synthetic/smoke 报告必须明确标注能力等级 |

### 8.2 稳定性

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| NFR-REL-001 | checkpoint 可恢复 | smoke/single-node selected | kill/restart 或 resume suite 能从最近 checkpoint 恢复 |
| NFR-REL-002 | OOM 可恢复/可规划 | smoke | OOM 后输出 retry plan，不静默成功 |
| NFR-REL-003 | 长时间训练稳定 | partial | 生产声明前需长窗口训练、显存曲线、checkpoint/resume stress |
| NFR-REL-004 | 错误隔离 | smoke | serving batch 失败不污染 cache，训练失败保留日志和 runtime status |
| NFR-REL-005 | 多硬件一致性 | interface | CUDA/Ascend 配置差异必须通过 device/backend/communication plan 表达 |

### 8.3 易用性

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| NFR-USE-001 | 配置简单 | smoke | quickstart 和 configs 提供 tiny、vision、CLIP、VLM LoRA、FSDP、DeepSpeed 示例 |
| NFR-USE-002 | 错误清晰 | required | backend 缺失、显存不足、配置错误必须给出可操作提示 |
| NFR-USE-003 | CLI 统一 | partial | 用户入口收敛到 `python -m parascale.cli ...`，脚本仅放 tests/benchmarks 或专项验收 |
| NFR-USE-004 | 文档中文、代码注释英文 | required | 主文档中文，代码注释/docstring 英文，统一 UTF-8 |

### 8.4 可测试性

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| NFR-TEST-001 | no-torch 测试可运行 | smoke verified historically | 配置、策略、schema、CLI plan、architecture reset 不依赖 torch |
| NFR-TEST-002 | 单机 GPU smoke | single-node selected | 远程 CUDA 容器运行 native/FSDP/DeepSpeed smoke |
| NFR-TEST-003 | benchmark 可复现 | partial | 输出机器、配置、模型、数据、backend、measurement window 和指标 |
| NFR-TEST-004 | 硬件测试 clean skip | required | 无对应硬件时必须 skip 并说明原因，不允许假成功 |
| NFR-TEST-005 | 文档与代码路径同步 | required | SRS、设计文档、测试入口必须使用 rebuild 后路径 |

## 9. 交付与后续验收顺序

以下是当前基线之后的验收方向，不表示已经交付，也不构成某个版本的发布时间承诺。

1. 保持现有契约正确：配置、优化器更新边界、rank 数据供给、checkpoint/resume、推理请求隔离持续回归，CLI 保持薄分发。
2. 扩展真实训练证据：CLIP/DataComp/VLM LoRA/YOLO 在同模型、同数据、同精度和更新预算下比较后端；报告数据等待、吞吐、显存和恢复结果。
3. 验证压缩质量：PTQ/QAT 使用预训练模型和独立评测集；检测蒸馏补足未见数据、多 seed、足够训练预算和学生基线，先确定质量，再讨论加速。
4. 扩展压缩运行范围：只有数值、恢复和质量验收后，才逐项开放 AMP、梯度累积、分布式或 QAT+KD；不能仅删除 guard 就宣称支持。
5. 推理标准化：先打通目标模型产物和真实 infer 适配，再考虑网络服务、tokenizer/生成、流式输出和资源隔离。
6. 研究型能力：TP/PP、native ZeRO Stage 2/3、异构 weighted DP 等需独立设计和实机验收，不混入当前可用能力声明。

## 10. 当前实现状态

当前代码版本已经重整到 rebuild 包边界，具备以下能力：

- `commands/`：doctor、plan、train/run、benchmark、benchmark-matrix、checkpoint、smoke、vision、stability 等命令实现分离。
- `contracts/`：backend、batch、checkpoint、metrics、plan、workload 等跨模块协议。
- `core/device/`：CPU/CUDA/Ascend 设备抽象与 registry。
- `core/distributed/`：collective、process group、registry。
- `data/vision/`：image folder、preprocessor、transforms、sampler、cache、collator、profiler。
- `data/multimodal/`：batch、cache、processor、prompt、profiler。
- `workloads/`：tiny、vision、CLIP、DataComp、VLM LoRA、YOLO、serving 等薄适配器。
- `runtime/training/`：engine、fit loop、step、precision、memory、accumulation、prefetch、metrics、checkpointing。
- `runtime/inference/`：engine、batcher、scheduler、memory 和 tasks 边界。
- `runtime/backends/`：native、FSDP、DeepSpeed、Ascend native、registry。
- `communication/`：DDP hook、communication plan、profiler。
- `checkpoint/`：manifest、manager、adapter-only、converter、validator。
- `reporting/`：benchmark、matrix、profile、tuner、markdown。
- `serving/`：API、engine、KV cache、sampler、scheduler。
- `configuration/`：配置规范化、来源与优先级、计划/训练契约。
- `quantization/`：Linear/Conv2d 的 reference weight-only PTQ/QAT 与导出重载。
- `distillation/`：分类/因果语言目标和检测专用特征目标；教师与学生状态隔离。
- `tests/benchmarks/` 与 `tests/validation/` 已成为 benchmark 和验证资产的统一目录。

当前仍未完成或不得声明为生产可用的能力：

- 新增量化/蒸馏的 Ascend NPU/HCCL 验收；已有历史训练证据不自动覆盖新增功能。
- PTQ/QAT 优化整数内核、预训练质量保持及分布式/AMP 集成。
- 蒸馏分布式/AMP/梯度累积组合，以及检测未见数据集上的精度收益。
- native ZeRO Stage 2/3。
- 分布式 TP/PP 真实通信执行。
- HuggingFace/vLLM/TGI serving adapter。
- prefill/decode 分离、paged KV、streaming HTTP serving。
- checkpoint converter 的真实格式转换矩阵。
- 通用 LLM 训练、sequence packing、consumed token resume 和 100B 级策略验收。
- 视频、音频、多模态 agent 数据协议：明确不扩展、不验收。

## 11. 需求追踪矩阵摘要

| 需求域 | 对应设计文档章节 | 主要代码路径（相对 parascale/） | 主要测试/验证入口 |
|---|---|---|---|
| 产品与硬件边界 | 1、2、11 | `core/device/`、`core/distributed/`、`contracts/` | 统一报告中各硬件场景；不跨场景外推 |
| 配置执行 | 3 | `configuration/`、`commands/run.py` | `tests/test_config_execution_contract_no_torch.py` |
| 训练预算与归一化 | 4 | `runtime/training/`、`runtime/backends/` | `tests/test_training_budget_no_torch.py`、`tests/test_training_execution_contract.py` |
| 数据与处理器 | 5 | `data/`、`workloads/` | `tests/test_distributed_data_contracts_torch.py`、`tests/test_pretrained_clip_inputs.py` |
| 检查点与恢复 | 6 | `checkpoint/`、`runtime/training/checkpointing.py` | `tests/test_checkpoint_runtime_state.py`、`tests/benchmarks/scripts/distributed_checkpoint_contracts.py` |
| PTQ/QAT | 7 | `quantization/` | `tests/test_compression_portability.py`、量化专项指南 |
| 在线蒸馏 | 8 | `distillation/`、`runtime/distillation.py` | `tests/test_detection_distillation.py`、`tests/test_detection_distillation_runtime_no_torch.py` |
| 推理与请求隔离 | 9 | `runtime/inference/`、`serving/` | `tests/test_inference_execution_contract.py` |
| 性能与证据 | 10 | `reporting/`、`strategy/` | `tests/benchmarks/reports/BENCHMARK_REPORT.md` |

## 12. 文档清理要求

`docs/` 目录应保留少量长期有效文档：

- `software_requirements_specification.md`：当前 SRS。
- `software_design_documentation.md`：当前实现的模块边界和执行契约。
- `user_manual.md`：安装、配置、训练、恢复、压缩和推理操作。
- `quantization.md`、`distillation.md`：专项 API、完整示例和验证边界。
- `remote_server_test_guide.md`：远程服务器测试指南。

## 13. 风险与约束

- 当前路线聚焦视觉和图文多模态，不能重新膨胀为视频/音频/agent 通用多模态平台。
- DeepSpeed/FSDP 仍是大模型和显存压力场景的成熟 fallback。
- native-DDP 性能优势只对已验证场景有效，不能泛化到所有模型和数据。
- Ascend 抽象必须保留，但实机能力必须单独验证。
- 所有 benchmark 和 serving 结论必须明确能力等级、硬件、数据、配置、batch budget、measurement window。
- 文档必须避免引用已删除或已迁移的旧路径。

### 13.1 试用版发布门禁

- 包版本必须由单一源码定义，wheel metadata、`parascale.__version__` 和发布文档保持一致。
- Python 3.10、3.11、3.12 必须执行源码测试；Python 3.11 必须执行 wheel clean-install 闭环。
- `doctor --strict` 必须根据 core、Torch 和显式 `--require` 能力返回可靠退出码与 JSON 证据。
- CLI 退出码固定为：2 配置/环境要求、3 依赖、4 runtime、5 checkpoint、6 benchmark、70 internal error。
- CI 的 extras resolution 只证明依赖声明可解析，CUDA/NPU/DeepSpeed 生产能力必须由远程实机门禁确认。
- FR-CFG-001：仓库配置必须显式声明当前 schema v1；缺失版本按 legacy v0 读取并支持非破坏性迁移。
- FR-CFG-002：高于 runtime 支持范围的配置必须在执行前失败，不得静默忽略未知 schema。
- FR-CFG-003：CLI 必须提供 `config validate` 与 `config migrate`，迁移不得修改原文件。
- NFR-API-001：根包公共 API 必须声明独立版本并由快照测试约束；破坏性变化必须升级公共 API 版本。
- NFR-REL-006：已发布 tag 不得移动或重建；发布后修复必须使用新的版本并重新执行发布门禁。

每次发布应针对变更影响列明 wheel clean-install、源码测试、目标硬件矩阵和 checkpoint/resume 门禁。暂不可执行的硬件验收记录为 `pending`；历史结果、dry-run 或 mock 不得替代当前变更的验收。例外必须明确风险接受人和未验证范围。代码通过开发分支和 PR 合并，不直接向主分支提交或推送。

## 14. 总结

ParaScale 当前需求核心是：保持视觉与图文多模态训练控制层的清晰边界，通过 native-DDP/FSDP/DeepSpeed、数据管线、profile/tuner、checkpoint/resume 和统一报告形成可审查闭环。参考量化和在线蒸馏复用训练主路径，功能正确性、质量收益和部署性能分别验收；推理服务与更大规模能力按后续验收顺序推进。

视频、音频和复杂多模态 agent 数据协议不纳入当前版本，也不作为后续扩展路线。
