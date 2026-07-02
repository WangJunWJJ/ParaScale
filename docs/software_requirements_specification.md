# ParaScale 软件需求规格说明书

版本：v1.1-rebuild-sync
日期：2026-06-25
状态：与 `docs/software_design_documentation.md` v1.1-rebuild 同步后的当前版本需求基线

## 1. 文档目标

本文定义 ParaScale 当前代码版本的软件需求边界、验收范围和能力分级。本文与 `docs/software_design_documentation.md` 的最新 rebuild 版本保持一致，用于指导后续开发、代码审查、测试验收和文档清理。

ParaScale 当前定位不再是泛化覆盖所有 AI 训练/推理形态，而是聚焦为：

```text
面向百卡以下视觉与多模态图文场景的分布式训练和推理控制层。
```

核心价值链为：

```text
Config -> RuntimePlan -> Workload Adapter -> Data Pipeline -> Training/Inference Runtime
       -> Backend Adapter -> Profile -> Tuner Evidence -> Report -> Checkpoint/Resume
```

## 2. 产品定位

ParaScale 当前版本应支持：

- 纯视觉模型训练、评测、profile 和推理 runtime 骨架。
- 图文多模态训练，重点覆盖 VLM LoRA、CLIP-style 对比学习、DataComp WDS、YOLO/视觉检测适配。
- Nvidia CUDA/NCCL 路线下的 native-DDP、FSDP、DeepSpeed 训练后端选择、benchmark 和 fallback。
- Huawei Ascend NPU/HCCL 的一等架构预留与接口级抽象，但当前不声明实机生产可用。
- 训练、checkpoint、resume、benchmark、serving 的统一闭环。
- 基于 profile/tuner evidence 的自动后端选择、批处理策略建议和 OOM retry 规划。

## 3. 明确不纳入范围

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
  commands/
  contracts/
  core/device/
  core/distributed/
  data/vision/
  data/multimodal/
  workloads/
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
```

设计文档继续沿用 `docs/software_design_documentation.md` 的 v1.1-rebuild 版本作为主设计依据。若代码结构、CLI 输出、能力边界或验收目标变化，本文和设计文档必须同步更新。

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
| FR-COMM-003 | 支持 Gloo/NCCL/HCCL/mock | partial | mock/Gloo/NCCL 路线可测试；HCCL 仅保留接口级路线 |
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

当前版本不以通用 LLM 训练框架为主要交付目标。LLM 相关能力仅作为配置、策略和 serving 扩展预留：

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
| FR-CKPT-007 | 训练到推理加载 | partial | tiny/real serving smoke 已覆盖，生产模型需后续 adapter 验收 |

### 7.10 CLI

| 编号 | 需求 | 当前状态 | 验收标准 |
|---|---|---|---|
| FR-CLI-001 | `doctor` | smoke verified | 输出环境、依赖、设备、配置诊断 |
| FR-CLI-002 | `plan` | smoke verified | 生成 runtime、launch、benchmark、strategy、dataloader 子计划 |
| FR-CLI-003 | `train` / `run` | smoke verified | 从 JSON/YAML config 启动训练或 smoke train |
| FR-CLI-004 | `benchmark` | smoke verified | 输出 benchmark_result、comparison contract 和关键指标 |
| FR-CLI-005 | `benchmark-matrix` | smoke/single-node verified for selected paths | 统一执行 native/FSDP/DeepSpeed 同口径矩阵 |
| FR-CLI-006 | `checkpoint validate` | smoke verified | 校验 checkpoint manifest 和 payload |
| FR-CLI-007 | `serve`/`infer` 命名边界 | open alignment | 用户级推理服务命令应统一为 `serve`；如保留 `infer`，需定义为一次性推理入口 |
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

## 9. 交付优先级

### P0：rebuild 主干稳定

- `commands/`、`contracts/`、`core/device/`、`core/distributed/`、`runtime/training/`、`runtime/inference/`、`runtime/backends/`、`strategy/`、`checkpoint/`、`reporting/` 包边界稳定。
- `cli.py` 保持薄分发。
- `RuntimePlan`、`DevicePlan`、`BackendPlan`、`CommunicationPlan`、`DataPlan`、`InferencePlan` 可审查。
- mock/dry-run 显式标记。

### P1：生产训练最小闭环

- plan -> train -> checkpoint -> validate -> resume -> benchmark report。
- native-DDP、FSDP、DeepSpeed smoke 和 selected single-node benchmark。
- DataComp WDS、CLIP-style、YOLO selected paths 验收。
- checkpoint manifest 和 adapter-only checkpoint 验收。

### P2：视觉与图文多模态优势路径

- vision resolution bucket、patch-token budget、profile。
- multimodal token estimator、padding ratio profile、processor adapter。
- DataComp WDS、VLM LoRA、YOLO/Objects365 selected matrix。
- RuntimeTuner evidence 与 recommended_config_updates。

### P3：推理 runtime

- `runtime/inference` 和 `serving` 生产化。
- tokenizer/HF/vLLM/TGI adapter。
- prefill/decode 分离、paged KV、streaming、admission control。
- train checkpoint 到 serving benchmark。

### P4：Native 高性能后端与异构研究

- native ZeRO Stage 2/3 仅在具备真实、可测 kernel 后推进。
- communication overlap。
- 分布式 TP/PP 真实执行。
- 多型号 GPU weighted DP。
- GPU 训练 + Ascend 推理/评估协同。

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
- `tests/benchmarks/` 与 `tests/validation/` 已成为 benchmark 和验证资产的统一目录。

当前仍未完成或不得声明为生产可用的能力：

- Ascend NPU/HCCL 实机验证。
- native ZeRO Stage 2/3。
- 分布式 TP/PP 真实通信执行。
- HuggingFace/vLLM/TGI serving adapter。
- prefill/decode 分离、paged KV、streaming HTTP serving。
- checkpoint converter 的真实格式转换矩阵。
- 通用 LLM 训练、sequence packing、consumed token resume 和 100B 级策略验收。
- 视频、音频、多模态 agent 数据协议：明确不扩展、不验收。

## 11. 需求追踪矩阵摘要

| 需求域 | 对应设计文档章节 | 主要代码路径 | 主要测试/验证入口 |
|---|---|---|---|
| 产品定位 | 1、2、20、附录 A | README、config、cli | 文档审查 |
| 硬件抽象 | 6、15、附录 A.3 | `core/device/`、`core/cluster.py`、`contracts/plan.py` | `tests/test_core_architecture_no_torch.py`、`tests/test_architecture_boundaries_no_torch.py` |
| 通信抽象 | 6.2、12、附录 A.2 | `core/distributed/`、`communication/`、`parallel/` | `tests/test_contracts_communication_no_torch.py`、`tests/test_strategy_no_torch.py` |
| 训练 runtime | 9、16、17 | `runtime/training/`、`runtime/backends/`、`commands/run.py` | `tests/test_train_no_torch.py`、`tests/test_backend_smoke.py` |
| 推理 runtime | 11、17 P4 | `runtime/inference/`、`serving/` | `parascale smoke`、`tests/test_inference_runtime_no_torch.py` |
| 视觉数据 | 8.2、17 P2 | `data/vision/`、`workloads/vision.py`、`workloads/yolo.py` | `tests/test_vision_*`、`tests/test_yolo_*` |
| 图文多模态 | 8.3、17 P3 | `data/multimodal/`、`workloads/clip.py`、`workloads/datacomp.py`、`workloads/vlm_lora.py` | `tests/test_clip_contrastive_workload.py`、`tests/test_vlm_lora_workload.py` |
| 策略/tuner | 7、13 | `strategy/`、`reporting/matrix.py` | `tests/test_strategy_*`、P2 tuner validation |
| Checkpoint | 10 | `checkpoint/`、`runtime/training/checkpointing.py` | checkpoint validate、resume suites |
| CLI | 14 | `cli.py`、`commands/` | `tests/test_cli_no_torch.py`、`tests/test_benchmark_matrix_cli_no_torch.py` |
| Benchmark/reporting | 13 | `reporting/`、`commands/benchmark*.py`、`tests/benchmarks/` | benchmark matrix scripts/reports |
| 异构资源 | 15、附录 A.6 | `core/cluster.py`、`strategy/hetero.py` | architecture reset/no-torch tests |

## 12. 文档清理要求

`docs/` 目录应保留少量长期有效文档：

- `software_requirements_specification.md`：当前 SRS。
- `software_design_documentation.md`：最新 rebuild 主设计文档。
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

## 14. 总结

ParaScale 当前版本的需求核心是：以最新 rebuild 架构为基线，建立视觉与图文多模态分布式训练/推理控制层，短期通过 native-DDP/FSDP/DeepSpeed、DataComp/CLIP/VLM LoRA/YOLO、profile/tuner、checkpoint/resume 和 benchmark/reporting 形成可审查闭环；中长期再推进 serving 生产化、native 高性能后端和异构资源协同。

视频、音频和复杂多模态 agent 数据协议不纳入当前版本，也不作为后续扩展路线。
