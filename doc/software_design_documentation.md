# ParaScale 软件设计文档

版本：v1.1-rebuild  
日期：2026-06-10  
状态：重构后主设计文档，用于指导后续架构实现、代码审查和功能验收。

## 0. P0 已验证执行路径

本阶段优先把已经在远程双卡 4090 环境中验证过的有效路径产品化，而不是泛化实现所有并行形态。

已固化策略如下：

- CLIP/DataComp WDS 与 VLM-style 对比学习：在单机多卡、中小模型、无显存压力场景下，`training_backend=auto` 应优先选择 `native_ddp`，并在 bf16 精度下启用 `bf16_compress` DDP communication hook。
- YOLO-World/Object365 与中小视觉检测训练：在单机多卡、中小模型、无显存压力场景下，`training_backend=auto` 应优先选择 `native_ddp`，默认不启用梯度压缩 hook，避免在 detection 场景中过早引入未验证通信压缩。
- FSDP/DeepSpeed 继续作为大模型、显存压力、ZeRO-3/offload 与生产 fallback 路径；所有性能优势声明必须通过同硬件、同数据、同 batch budget 的 benchmark 报告验证。
- `python -m parascale.cli benchmark-matrix` 是当前推荐的统一矩阵入口，负责运行 VLM LoRA HF CLIP 与 YOLO-World native-DDP/FSDP/DeepSpeed 同口径 benchmark，并生成吞吐、peak GPU memory、推荐后端和配置更新建议。

## 0.1 P1 已验证功能闭环

P1 的核心目标不是继续追求单点吞吐，而是证明 ParaScale 的最小生产训练闭环可重复执行：`plan -> train -> checkpoint -> validate -> resume -> serve/benchmark report`。

当前验收入口为：

```bash
bash tests/benchmarks/scripts/run_p1_functional_validation_suite.sh
```

2026-06-12 远程双卡 4090 容器验收结果：

- `server_tiny_torch` smoke 通过 doctor、plan、train、checkpoint validate、resume 和 real serve。
- DataComp WDS CLIP native-DDP 通过 checkpoint 校验和 resume 验证，resume 从 step 4 恢复到 step 5，backend state 成功加载，吞吐约 129.46 pairs/s。
- YOLO-World Object365 native-DDP 通过 checkpoint 校验和 resume 验证，resume 从 step 3 恢复到 step 4，backend state 成功加载，吞吐约 88.57 images/s。
- `parascale benchmark` 支持 `training.validate_resume=true`，可在功能验收场景下把 resume 结果写入 `validation.resume`，性能基准默认不启用该额外步骤。

P1 之后的优化应继续围绕真实生产风险推进：更长训练的 checkpoint/resume stress、多 worker dataloader 稳定性、异常中断恢复、配置可复现性和报告可审计性。

### 0.1.1 P1 真实 VLM LoRA 升级状态

真实 VLM LoRA 已从 “HF CLIP frozen encoder + adapter” 推进到独立主路径：

- 模型入口：`hf_vlm_lora`、`qwen2_vl_lora`、`llava_onevision_lora`、`internvl_lora`。
- 默认首选小规格场景：LLaVA-OneVision 0.5B，用于在双卡 4090 上建立真实 VLM LoRA smoke 和 benchmark。
- Processor adapter：支持 HuggingFace `AutoProcessor`，将 image processor、tokenizer 和 conversation template 合并为训练 batch。
- LoRA 注入：优先使用 PEFT 对 `q_proj/k_proj/v_proj/o_proj` 等 target modules 注入 LoRA，而不是外接 fusion adapter。
- Checkpoint：新增 `adapter_only_checkpoint`，native/DDP 可只保存 LoRA adapter state、optimizer state、scheduler state 和 RNG state。
- Benchmark：`benchmark-matrix --scenario vlm-lora-real` 支持 native-DDP、FSDP、DeepSpeed ZeRO-2、DeepSpeed ZeRO-3 同口径矩阵。
- Batch-size sweep：`benchmark-matrix --batch-size-sweep ...` 必须自动展开 batch 维度，run_id 中保留 batch 标识，保证报告可审查。
- OOM retry：`benchmark-matrix --oom-retry` 必须捕获 launcher 日志，在 OOM 类失败后自动降低 batch、启用 activation checkpointing，并按同后端、FSDP、DeepSpeed ZeRO-2、DeepSpeed ZeRO-3 顺序尝试 fallback。
- 指标：输出 tokens/s、images/s、loss、peak memory、adapter params、trainable params、trainable ratio 和 LoRA rank。

当前远程已构建 `parascale-vlm:cu121-torch24-transformers451-peft` 镜像，并完成 tiny VLM LoRA adapter-only checkpoint smoke。真实 LLaVA/Qwen/InternVL 权重尚未在远程缓存，实权重训练矩阵需在权重放入 `/home/wangjun/work/vlm_models` 后执行。

## 0.2 P2 已验证 Profile/Tuner 可解释调优

P2 的核心目标是让自动策略具备可审计解释能力。`parascale plan` 当前输出 `explain` 字段，包含：

- `selected_backend`：最终后端选择。
- `static_reasons`：静态策略选择依据。
- `runtime_decisions`：runtime profile 驱动的结构化调优决策。
- `recommended_config_updates`：建议写回配置的字段。

每条 runtime decision 必须包含：

- `action`：建议动作。
- `reason`：人可读原因。
- `evidence`：观测到的指标。
- `threshold`：触发阈值。
- `config_updates`：建议配置变更。

当前验收入口为：

```bash
bash tests/benchmarks/scripts/run_p2_profile_tuner_validation.sh
```

2026-06-12 远程容器验收结果：

- memory pressure 场景通过：当 peak memory 超出 memory budget 时，输出 `reduce_memory_pressure` 决策，并建议 `enable_activation_checkpointing=true`。
- padding + dataloader wait 场景通过：当 padding ratio=0.5 且 dataloader_wait_ms=60 高于阈值时，输出 `switch_to_token_budget_batching` 和 `increase_dataloader_parallelism` 决策。
- P2 验收报告见 `tests/reports/archive/p2_profile_tuner_validation_report.md`。

## 0.3 P3 已验证混合精度、ZeRO-3 与真实权重稳定性

P3 的核心目标是验证更接近生产训练的稳定性边界：bf16/AMP、activation checkpointing、DeepSpeed ZeRO-3、checkpoint/resume stress，以及真实 HF CLIP 离线预训练权重加载。

当前验收入口为：

```bash
bash tests/benchmarks/scripts/run_p3_checkpoint_resume_stress.sh
```

2026-06-15 远程双卡 4090 容器验收结果：

- native bf16 + activation checkpointing train 通过，global_step=4，checkpoint 校验通过。
- native bf16 + activation checkpointing resume 通过，从 step 2 恢复并继续到 global_step=6。
- DeepSpeed ZeRO-3 bf16 train 通过，global_step=4，checkpoint 校验通过。
- DeepSpeed ZeRO-3 bf16 resume 通过，从 step 2 恢复并继续到 global_step=6。
- DeepSpeed ZeRO-3 + activation checkpointing train-only smoke 通过。
- HF `openai/clip-vit-base-patch32` 离线真实权重 + DataComp WDS one-step train 通过，checkpoint 校验通过。

P3 后续边界：

- DeepSpeed ZeRO-3 + activation checkpointing 仍只声明 train-only smoke，不作为长跑 resume 默认路径。
- HF 权重 smoke 已覆盖 native，FSDP/DeepSpeed 下的真实 HF 权重长跑仍需后续单独验收。
- LoRA/QLoRA adapter 保存、恢复和继续训练尚未纳入本阶段验收。

## 1. 设计定位

ParaScale 面向百卡以下的小规模生成环境，目标是成为一个高效、易用、可扩展的分布式训练与推理框架。核心场景是视觉、多模态和中小规模生成式模型，首批重点包括 VLM LoRA、CLIP-style 对比学习、ViT/vision tower 训练和训练到推理闭环。

ParaScale 不应只是 DeepSpeed、FSDP 或 torchrun 的薄封装。短期必须可靠复用这些成熟后端作为 baseline 和 fallback；中长期应在通用框架不敏感的场景形成优势：

- 视觉训练中的 resolution bucket、patch-token batch 调度和数据管线 profile。
- 多模态训练中的 modality-aware token estimator、padding ratio profile 和 processor adapter。
- 基于 warmup profile、OOM retry、显存峰值和 dataloader wait 的闭环调优。
- 训练、checkpoint、恢复、benchmark、推理共用一套 runtime contract。
- 在 Nvidia CUDA/NCCL 和 Huawei Ascend NPU/HCCL 之间保持一致的工程抽象。

设计原则是极简、真实、可测、渐进。先把骨干接口打稳，再逐步实现真实子模块；不为了展示功能而堆叠无法验证的抽象。

## 2. 三层目标

### 2.1 P0/P1：不输 baseline

ParaScale 必须先做到能稳定运行 native、FSDP、DeepSpeed 基线路径，并具备 plan、train、checkpoint、resume、serve、benchmark 的最小闭环。

验收要求：

- 本地 `python tests/run_tests.py` 全量通过。
- 远程双卡 GPU 容器中 FSDP 和 DeepSpeed smoke 通过。
- 所有 dry-run、mock、synthetic 输出显式标记。
- checkpoint manifest 可保存、校验、恢复，并可被 serving runtime 读取。
- 错误信息可操作，不能静默 fallback 到虚假成功。

### 2.2 P2/P3：赢目标场景

ParaScale 的差异化优先来自数据与 runtime，而不是重新实现所有通用分布式能力。

视觉路径：

- 按 patch token 而不是 sample 数组织 batch。
- 支持真实 ImageFolder profile，输出 images/sec、patch_tokens/sec、decode、augment、host-to-device 指标。
- 让 RuntimeTuner 能根据吞吐、显存和数据等待时间调整 batch budget。

图文多模态路径：

- 标准 batch schema 覆盖 text、image、label、metadata。
- 支持 VLM LoRA 和 CLIP-style 对比学习优先场景。
- 不扩展视频理解、音频理解和复杂多模态 agent 数据协议。
- processor adapter、token estimator、padding ratio profile 必须逐步接入真实模型和数据。

### 2.3 P4/P5：赢系统闭环

长期目标是在训练、checkpoint、推理、评估、benchmark 的系统效率上超过通用训练框架。

重点能力：

- 训练产物可直接进入 serving runtime。
- 推理 runtime 不依赖 DeepSpeed。
- 支持连续 batching、有界 KV cache、后续 paged KV、流式输出和服务端 admission control。
- 逐步建设 ParaScale native shard backend，覆盖 optimizer-state sharding、gradient sharding、parameter sharding 和通信计算 overlap。
- Ascend 路线先做独立同构 NPU 训练，最后再做 GPU 训练 + Ascend 推理/评估协同。

## 3. 当前工程结构

当前工程以 `parascale/` 为唯一主包。旧 `Engine/ParaEngine`、旧 eager `DataParallel`、`ModelParallel`、`TensorParallel`、`PipelineParallel`、`HybridParallel` 不再作为兼容目标。

```text
parascale/
  config.py              # 统一配置模型
  cli.py                 # plan / train / serve / benchmark / smoke / doctor
  core/                  # device、collective、cluster topology
  data/                  # text、vision、multimodal 数据 schema 与 pipeline
  strategy/              # 静态规划、profile、tuner、异构规划
  runtime/               # context、launcher、benchmark、train、infer、backend、factory
  checkpoint/            # manifest、manager、validator、converter
  serving/               # request/response、scheduler、KV cache、sampling、engine
  parallel/              # ParallelPlan、TP/PP/SP、communication compression
  optimizers/            # optimizer 兼容、4bit optimizer、native ZeRO Stage 1
  quantization/          # QAT/PTQ、fake quant、observer、quantized layers
  utils/                 # logging、rank、distributed init
tests/
  run_tests.py           # 一键验证入口
```

主干对象：

- `ParaScaleConfig`
- `RuntimeContext`
- `StrategyPlan`
- `LaunchPlan`
- `BenchmarkPlan`
- `BenchmarkResult`
- `TrainEngine`
- `ServeEngine`
- `TrainingBackend`
- `CheckpointManifest`
- `ParallelPlan`

新能力必须优先进入这些主干对象或其清晰下游，不能旁路形成第二套框架。

## 4. 架构总览

ParaScale v1 采用“配置事实源 + 策略规划 + runtime 执行 + profile 反馈”的闭环架构。

```text
User Config / CLI
        |
        v
RuntimeContext
  - workload descriptor
  - model and data profile
  - hardware and topology
  - strategy plan
  - budget constraints
        |
        +--> LaunchPlan
        +--> DataPlan
        +--> BenchmarkPlan
        +--> ParallelPlan
        |
        v
Runtime Execution
  - TrainEngine
  - ServeEngine
  - TrainingBackend
  - CheckpointManager
        |
        v
Profile / Metrics / Tuning
  - throughput
  - memory
  - padding ratio
  - dataloader wait
  - decode / augment / transfer
  - checkpoint / restore / serve latency
```

## 5. 配置模型

`ParaScaleConfig` 是配置入口，必须保持小而明确。

关键字段：

- `task_type`: `generic`、`llm`、`vision`、`multimodal`
- `model_family`: `vit`、`clip`、`vlm`、`llm` 等
- `target_scale`: `local`、`single_node`、`small_cluster`、`sub_100_gpus`
- `optimize_for`: `throughput`、`memory`、`latency`、`balanced`
- `training_backend`: `native`、`fsdp`、`deepspeed`、`auto`
- `tensor_parallel_size`
- `pipeline_parallel_size`
- `zero_stage`
- `max_tokens_per_batch`
- `max_patch_tokens_per_batch`
- `resolution_buckets`

约束：

- 配置必须可由 JSON/YAML 表达。
- CLI 输出必须可 JSON 序列化。
- 默认配置必须保守可运行。
- 新字段必须有明确使用者和测试。
- 不允许为了未来想象添加无调用者字段。

## 6. Core 层

`core/` 屏蔽硬件和通信基础能力。

### 6.1 DeviceBackend

目标是统一 CPU、CUDA、NPU 的设备能力访问。

职责：

- 发现设备数量、名称、显存和能力。
- 设置当前设备。
- 查询 bf16、flash attention 等能力。
- 提供 CPU fallback，保证本地轻量测试可运行。

后续重点：

- 完善 Nvidia CUDA capability 与 torch.cuda 诊断。
- 完善 Ascend capability 与 torch_npu 诊断。
- 将设备能力接入 `StrategyPlan`，避免策略层散落硬件分支。

### 6.2 CollectiveBackend

训练和推理必须共用通信抽象。

接口能力：

- `all_reduce`
- `reduce_scatter`
- `all_gather`
- `all_to_all`
- `broadcast`
- `barrier`
- `new_group`

规则：

- `MockCollectiveBackend` 保持无 torch 测试可用。
- `TorchDistributedCollectiveBackend` 对接 Gloo/NCCL。
- HCCL 通过 torch_npu 或独立 adapter 接入。
- TP、SP、sharding、serving TP 都必须复用这一层。

### 6.3 ClusterTopology

目标是显式表达节点、设备、rank、通信 backend 和异构分组。

设计规则：

- 高频同步组优先同构，例如 TP、FSDP/ZeRO shard group、高频 all-reduce DP group。
- 多型号 GPU 可进入 weighted DP，但 batch/token 分配必须受慢卡或小显存卡约束。
- GPU + Ascend 默认分组协同，不默认混入同一高频同步组。
- 异构 pipeline stage 必须显式配置并由 profile 验证。

## 7. Strategy 层

`strategy/` 是 ParaScale 从“能跑”走向“跑得好”的关键层。

模块：

- `plan.py`: `StrategyPlan`
- `planner.py`: 静态策略规划
- `profiler.py`: runtime profile 数据结构
- `tuner.py`: profile-driven tuning 与 OOM retry
- `hetero.py`: 异构规划

规划输入：

- `ParaScaleConfig`
- model profile
- hardware profile
- topology
- task type

规划输出：

- backend
- DP/TP/PP/SP 建议
- ZeRO/FSDP stage
- offload
- precision
- activation checkpointing
- batch policy
- checkpoint policy
- reasons / warnings

策略必须可解释。每一项自动决策都应写入 `reasons` 或 `warnings`。

## 8. Data 层

数据层是 ParaScale 相对通用分布式框架的主要优势来源。

### 8.1 通用数据合同

标准 batch schema 应覆盖：

- `input_ids`
- `attention_mask`
- `pixel_values`
- `labels`
- `metadata`
- `num_images`
- `patch_tokens`
- `token_count`

### 8.2 视觉数据

当前已有：

- `ImageFolderDataset`
- `PatchTokenBatchSampler`
- `VisionCollator`
- `ImageDecodeCache`
- `VisionThroughputProfiler`
- real image profile smoke
- tensor-ready vision collator，支持不同分辨率 padding、label tensor 化、`image_sizes`、`num_images` 和 `patch_tokens` 指标输出

目标能力：

- 按分辨率分桶，减少 resize/padding 浪费。
- 按 patch-token budget 控制 batch。
- 输出 images/sec、patch_tokens/sec、decode、augment、transfer 等指标。
- 与 sample-based batching 做可复现 benchmark。

### 8.3 图文多模态数据

当前版本多模态边界收敛为图文多模态，不再扩展视频、音频和复杂多模态 agent 数据协议；这些内容不作为当前版本验收项，也不作为后续主线扩展方向。

优先目标：

- VLM LoRA/QLoRA 风格 finetune。
- CLIP-style 图文对比学习。
- 先以 tiny CLIP-style synthetic image-text contrastive workload 验证 schema、collator、loss、metrics、checkpoint 和 benchmark contract。
- DataComp WDS 作为第一条真实图文数据路径：从 `.tar` shard 读取 image bytes、caption text 和 JSON metadata，直接进入 CLIP-style train/benchmark。
- DataComp parquet metadata 作为真实元数据入口：先验证 parquet/json 元数据读取和 text schema，后续再扩展到可复现采样、过滤、去重和 image retrieval/cache。
- ImageFolder+caption 或 HuggingFace datasets 作为后续通用 adapter，而不是当前首要闭环。

明确不扩展、不验收：

- 视频理解数据协议。
- 音频理解数据协议。
- 复杂多模态 agent 数据协议。

## 9. Runtime 层

`runtime/` 是 v1 主干。

### 9.1 RuntimeContext

`RuntimeContext` 是 plan、train、serve、benchmark 的共享事实源。所有 CLI plan 输出和 benchmark 报告都应围绕它组织。

### 9.2 LaunchPlan

`LaunchPlan` 给出保守启动建议：

- `python`
- `torchrun`
- `deepspeed`
- `manual`

它只输出命令、环境变量、world size、nproc、reasons、warnings，不直接启动进程，便于本地、远程和容器环境审查。

### 9.3 TrainEngine

`TrainEngine` 只做编排，不内嵌大量 backend 细节。

职责：

- 初始化策略。
- 调用 backend setup。
- 执行 train step。
- 记录 metrics。
- 保存和加载 checkpoint。
- 暴露 strategy plan。

真实分布式执行由 `TrainingBackend` 负责。

### 9.4 TrainingBackend

当前后端：

- `NativeTrainingBackend`
- `FSDPTrainingBackend`
- `DeepSpeedTrainingBackend`

规则：

- FSDP/DeepSpeed 是 baseline 和 fallback。
- native backend 可用于本地真实训练和 ZeRO Stage 1。
- native ZeRO Stage 2/3 不做虚假支持。
- 后端必须暴露统一 `setup`、`backward`、`step`、`state_dict`、`save_checkpoint`、`load_checkpoint` 契约。

## 10. Checkpoint 设计

`checkpoint/` 必须独立成体系，不散落在 engine/backend 中。

Manifest 应包含：

- step / global_step
- backend
- parallel plan
- file entries
- payload metadata
- validation status
- last metrics

目标：

- 训练 checkpoint 可直接被 serving runtime 校验和加载。
- 支持 Native/FSDP/DeepSpeed/HF/Ascend 格式转换。
- checkpoint validation 能在 smoke 和 benchmark 中自动运行。

## 11. Serving 设计

推理 runtime 不依赖 DeepSpeed。训练和推理继续放在同一个工程内，因为二者需要复用 device、collective、topology、checkpoint、profile、data schema 和配置系统。

当前已有：

- `ServeEngine` real/mock mode 边界。
- `ServingEngine` batched request execution。
- request-level error isolation。
- `ContinuousBatchScheduler` 队列与指标。
- `KVCacheManager` 有界缓存和 LRU 行为。
- `SamplingConfig` 结构化导出。
- tiny torch checkpoint 到 real serving smoke。

下一步：

- tokenizer 与 HuggingFace/vLLM/TGI adapter。
- prefill/decode 分离。
- paged KV cache 和 GPU block allocator。
- HTTP server lifecycle、admission control、streaming token output。
- train checkpoint 到 serving benchmark。

## 12. Parallel 与 Optimizer 设计

旧 eager parallel wrapper 不再恢复。并行能力通过 `ParallelPlan` 表达，通过 runtime backend 或显式 adapter 执行。

当前已有：

- `ParallelPlan`
- `TensorParallelAdapter`
- `column_parallel_linear`
- `row_parallel_linear`
- `build_pipeline_stages`
- `LocalPipelineExecutor`
- `SequenceParallelAdapter`
- gradient compression helpers
- native ZeRO Stage 1
- 4bit optimizer 兼容模块

生产边界：

- 分布式 TP/PP 不默认静默启用。
- 模型必须显式集成 TP/PP adapter。
- TP/PP 必须通过 benchmark 验证后才能声明生产支持。
- native ZeRO 当前只支持 Stage 1 optimizer-state sharding。
- gradient/parameter sharding 继续委托 DeepSpeed/FSDP，直到 ParaScale 有真实 native kernel。

## 13. Benchmark 设计

性能声明必须通过 benchmark 支撑。

当前已有：

- `BenchmarkPlan`
- `BenchmarkScenario`
- `BenchmarkResult`
- `BenchmarkComparison`
- `compare_benchmark_results`
- CLI benchmark 输出 `benchmark_result` 和 comparison contract
- CLI `benchmark-matrix` 统一执行 native-DDP/FSDP/DeepSpeed 矩阵，不再为每个场景维护单独执行脚本

Benchmark 必须记录：

- 配置文件或配置快照。
- 硬件 profile。
- backend。
- strategy plan。
- batch policy。
- throughput。
- peak memory。
- dataloader wait。
- checkpoint/restore/serve latency。

对比要求：

- 同硬件。
- 同配置。
- 同数据口径。
- 同 batch budget 或明确说明差异。
- 不允许用 smoke 结果宣称性能优势。

## 14. CLI 设计

当前 CLI 命令：

- `doctor`
- `plan`
- `train`
- `serve`
- `benchmark`
- `benchmark-matrix`
- `smoke`
- `vision-profile`
- `checkpoint validate`

要求：

- 所有命令输出 JSON 或可写 JSON 文件。
- dry-run 必须明确标记 `dry_run`、`runtime_status`、`capability_level`。
- mock 结果必须显式标记 `mock: true`。
- plan 输出必须包含 strategy、launch、benchmark、dataloader。
- benchmark 必须记录硬件、配置、backend、策略和指标。
- benchmark-matrix 必须自动生成派生配置、执行同口径后端矩阵、输出 JSON summary 与中文 markdown report。
- benchmark-matrix 的 OOM retry 必须显式记录 retry_of、batch_size、log path、失败原因和最终采用的 fallback，不允许静默降级。

## 15. 异构资源设计

支持两类异构资源：

- 多型号 Nvidia GPU，例如 H100、A100、L20、4090。
- Nvidia GPU 与 Huawei Ascend NPU 协同。

默认策略：

| 并行维度 | 多型号 GPU | GPU + Ascend |
| --- | --- | --- |
| DP | 可 weighted DP | 默认分组，不做高频同步混用 |
| TP | 同型号、同节点优先 | 默认禁止混用 |
| PP | 可跨型号，必须 profile | 可显式配置，必须记录转换成本 |
| FSDP/ZeRO shard | 同型号、同 backend 优先 | 默认禁止混用 |
| 推理 TP | 同型号优先 | 默认禁止混用 |
| 数据预处理 | 可混用 | 可混用 |
| 评估/推理协同 | 可混用 | 推荐支持 |

优先级：

1. 同构 Nvidia / 同构 Ascend 稳定训练。
2. 多型号 GPU node group 与 weighted DP。
3. GPU 训练 + Ascend 推理/评估协同。
4. 异构 pipeline stage。
5. 更细粒度 GPU+Ascend 混训仅作为研究路径。

Ascend 实机验证本阶段暂缓，但架构抽象必须保留。

## 16. 当前实现状态

已具备：

- `ParaScaleConfig` 基础字段与校验。
- `RuntimeContext`、`LaunchPlan`、`BenchmarkPlan`。
- `parascale plan` 输出 runtime、launch、benchmark、strategy、dataloader。
- `TrainEngine` 本地训练、checkpoint 保存、恢复。
- `FSDPTrainingBackend` 和 `DeepSpeedTrainingBackend` smoke。
- `ServeEngine` mock/real mode 边界。
- `ServingEngine` 连续微批处理、错误隔离、KV cache、metrics。
- `vision_synthetic` 工作负载与 patch-token 指标。
- 真实 ImageFolder profile smoke。
- `ParallelPlan`、TP/PP 本地原语、sequence/compression 实验资产。
- native ZeRO Stage 1。
- `BenchmarkResult` 与 `BenchmarkComparison`。
- no-torch 测试、本地 smoke 测试和远程双卡容器 smoke。
- 代码与文档统一 UTF-8；代码注释/docstring 使用英文；主文档使用中文。

尚未完成：

- HuggingFace/vLLM/TGI serving adapter。
- paged KV cache 和流式输出。
- 分布式 TP/PP 真实通信执行。
- native ZeRO Stage 2/3。
- VLM LoRA 真实 processor adapter 已接入，packing 与更长训练 loop 继续完善。
- CLIP-style DataComp WDS 真实数据 adapter 已接入 benchmark 路径，生产长跑与数据治理仍需继续完善。
- checkpoint converter 的真实格式转换。
- Ascend NPU 实机验证。
- 与 DeepSpeed/FSDP 的系统化性能对比报告已形成统一入口，后续继续扩大模型、数据和长跑规模。

## 17. 开发路线

### P0：主干稳定

- 固化 `RuntimeContext`、`LaunchPlan`、`BenchmarkPlan` schema。
- 保持 `tests/run_tests.py` 全量通过。
- 所有 mock/dry-run 明确标记。
- 代码注释英文、文档中文、UTF-8 编码。

### P1：生产训练最小闭环

- 本地 native 与远程 2-GPU FSDP/DeepSpeed smoke。
- checkpoint save/resume/validate。
- benchmark 输出统一 JSON。
- remote container 流程可重复执行。

### P2：视觉优势路径

- 完善 resolution bucket。
- 完善 patch-token batch。
- 接入真实 image decode/cache/augment。
- 加入 dataloader wait、decode、augment、transfer profile。
- 与 sample-based baseline 做 benchmark。

### P3：多模态优势路径

- 固化 multimodal batch schema。
- 实现 modality-aware token estimator。
- 接入 VLM LoRA processor adapter。
- 接入 CLIP-style 对比学习数据流。
- 实现 padding ratio profile。

### P4：推理 runtime

- HuggingFace/vLLM/TGI adapter。
- prefill/decode 分离。
- paged KV cache。
- HTTP server 与 streaming。
- train checkpoint 到 serve 的端到端 benchmark。

### P5：Native 高性能后端

- native ZeRO Stage 2/3。
- 通信计算 overlap。
- 分布式 TP/PP 执行。
- 异构 weighted DP 与 pipeline stage。

## 18. 设计约束

- 新接口必须有测试或 smoke。
- 新模块必须说明服务哪个主干目标。
- 不新增无调用者的抽象。
- 不让旧兼容 API 成为新能力入口。
- 不把 mock 输出包装成真实能力。
- 硬件相关测试必须可 clean skip，并给出原因。
- benchmark 结果必须可复现，至少包含配置、硬件、backend、策略、指标。
- 删除冗余文件优先于维护过期文档和示例。

## 19. 审查问题

后续审查建议重点看：

- 真实 VLM LoRA 首个模型适配选择 Qwen-VL、LLaVA 还是更小的内部 toy VLM。
- CLIP-style 对比学习先接 ImageFolder+caption，还是 HuggingFace datasets。
- serving adapter 先接 HuggingFace generate，还是直接对接 vLLM/TGI。
- benchmark runner 已收敛到 CLI 子命令；后续应继续减少一次性远程脚本，保留专项验收脚本即可。
- Ascend 同构 NPU 训练选择 torch_npu 原生路径还是先做 HCCL smoke。

## 20. 总结

ParaScale v1 的核心不是堆叠所有分布式训练功能，而是建立一个可审查、可测试、可演进的 runtime 主干。短期用 FSDP/DeepSpeed/native 建立可靠 baseline；中期用视觉 patch-token、多模态 token-aware 数据 runtime 和训练到推理闭环形成差异化；长期再通过 native sharding、通信重叠、异构规划和独立推理 runtime 建立系统级优势。

本文档是后续代码开发的主设计依据。若代码结构、CLI 输出、能力边界或验收目标发生变化，应同步更新本文档。
## 附录 A：2026-06-23 架构重整基线

本附录与 `doc/parascale_architecture_reset_plan.md` 保持一致，作为后续代码重构、人工审查和功能验收的当前基线。ParaScale 尚未上线运行，因此本轮重整不要求兼容早期 `Engine`、旧并行 wrapper 或历史 CLI 形态，只保留已经验证有价值的能力，并以当前最优工程结构重新组织。

### A.1 产品边界

ParaScale 的定位是面向百卡以下视觉与多模态场景的分布式训练和推理控制层。短中期不重写 DeepSpeed/FSDP 的全部能力，而是把 DeepSpeed/FSDP 作为成熟 baseline 和 fallback，并围绕真实数据管线、profile、自动后端选择、可解释 benchmark、checkpoint/resume 闭环形成差异化。

核心价值链为：

```text
Config -> RuntimePlan -> Workload Adapter -> Data Pipeline -> Training/Inference Runtime
       -> Backend Adapter -> Profile -> Tuner Evidence -> Report -> Checkpoint/Resume
```

### A.2 分层架构

重整后的主包边界如下：

- `commands/`：CLI 命令实现，`cli.py` 只做薄分发。
- `contracts/`：跨模块稳定协议，包含 batch、backend、checkpoint、metrics、plan、workload 等核心 contract。
- `core/device/`：CPU、CUDA、Ascend 设备能力抽象。
- `core/distributed/`：Gloo、NCCL、HCCL、mock collective 和 process group 抽象。
- `data/vision/`：通用视觉预处理、cache、collator、sampler、profile。
- `data/multimodal/`：processor、tokenizer、conversation template、prompt cache、modality-aware profile。
- `workloads/`：YOLO、CLIP、DataComp、VLM LoRA 等薄适配器。
- `runtime/training/`：训练执行 runtime，包含 engine、fit loop、step、precision、memory、prefetch、checkpointing。
- `runtime/inference/`：推理 runtime，包含 engine、batcher、scheduler 和 vision/multimodal/embedding task。
- `runtime/backends/`：训练执行后端，包含 native、FSDP、DeepSpeed、Ascend native。
- `strategy/`：RuntimePlan、DevicePlan、BackendPlan、CommunicationPlan、DataPlan、InferencePlan、profiler、tuner。
- `communication/`：DDP hook、bucket policy、no_sync、LoRA sync、overlap profile。
- `checkpoint/`：manifest、manager、adapter-only、converter、validator。
- `reporting/`：benchmark、matrix、profile、tuner 和 markdown 报告。

### A.3 硬件和训练后端分离

CUDA 和 Ascend 是硬件设备能力，不应与 native、FSDP、DeepSpeed 等训练策略混在同一层。设备层使用：

```text
core/device/cpu.py
core/device/cuda.py
core/device/ascend.py
```

训练策略层使用：

```text
runtime/backends/native.py
runtime/backends/fsdp.py
runtime/backends/deepspeed.py
runtime/backends/ascend_native.py
```

CUDA 路线默认组合为 `CudaDeviceBackend + NCCL + native/FSDP/DeepSpeed`。Ascend 路线默认组合为 `AscendDeviceBackend + HCCL + ascend_native`。Ascend 在没有实机验证前只能声明 interface ready 或 smoke ready，不能声明 production ready。

### A.4 Contracts 性能约束

`contracts/` 是跨模块工程协议，不是高频训练热路径包装层。`RuntimePlan`、`BackendPlan`、`CommunicationPlan`、`DataPlan` 等计划对象应在训练前生成，运行中只读使用。默认训练模式不做 per-step 深度 schema 校验，不因 contracts 产生 batch 深拷贝、tensor clone 或频繁 JSON 序列化。严格校验只在 debug、doctor、smoke 或 profile 模式启用。

### A.5 Plan 产物和 Examples 组织

每次训练、推理或 benchmark 配置解析后会生成 RuntimePlan 和相关子计划。默认情况下 plan 只在内存中流转；显式 `--output` 时才保存到 `runs/`、`tests/` 或对应 `examples/example_xxx/` 子目录，不能写入 `parascale/` 源码目录。

`examples/` 按一次完整运行一个子目录组织，例如：

```text
examples/example_001_clip_datacomp_native_ddp/
  README.md
  config.yaml
  run.sh
  runtime_plan.example.json
  expected_metrics.example.json
```

真实运行产物、checkpoint、模型权重、机器专属路径和历史 benchmark 结果不进入 examples。

### A.6 多机验收分级

多机分布式是 ParaScale 的一等架构目标，但必须按验收级别声明能力：

| 等级 | 含义 | 可声明能力 |
| --- | --- | --- |
| `single-node verified` | 单机单卡或单机多卡真实训练/benchmark 已通过 | 可以声明单机能力可用 |
| `multi-node smoke` | 多机启动、通信初始化、少量 step smoke 通过 | 可以声明多机链路可启动，但不能声明生产性能 |
| `multi-node production` | 多机长窗口训练、checkpoint/resume、benchmark matrix、故障恢复均通过 | 可以声明该场景具备生产可用性 |

所有 CLI 输出、benchmark report 和文档必须标明当前结果属于哪一档。
