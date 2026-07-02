# ParaScale 训练试用版闭环设计

## 1. 目标

下一里程碑将 ParaScale 收敛为可由外部用户独立试用的视觉与图文多模态分布式训练控制层。首发范围不扩展新的模型门类，而是确保用户在干净环境中能够完成：

```text
install -> doctor -> resolve -> plan -> train -> interrupt -> resume
        -> benchmark matrix -> explain -> report
```

首批真实黄金路径为 DataComp WDS + CLIP-B/ViT-B，以及小规格真实 VLM + LoRA。native-DDP、FSDP 和 DeepSpeed 保持统一配置与同口径比较。

## 2. 产品原则

- 统一 CLI 和配置文件是主要用户接口，不新增 workload 专用入口脚本。
- workload 只负责模型、processor、loss 和指标适配；设备、数据迁移、训练循环、checkpoint、profile 和报告属于框架核心。
- dry-run、synthetic、smoke、single-node verified 和 production verified 必须显式区分。
- 任意 fallback、OOM retry 或执行路径变化必须进入结果与报告，不能静默降级。
- 性能结论必须控制硬件、数据、模型、全局 batch、精度、warmup、measurement window 和 checkpoint 设置。
- DeepSpeed/FSDP 是成熟 baseline 与显存 fallback；没有同口径证据时不自研 ZeRO-2/3。
- contracts 只用于模块边界和低频控制流，不进入训练 step 高频热路径。

## 3. 发布架构

### 3.1 配置与计划

所有用户配置、CLI override、workload 默认值、backend 配置和 hardware profile 必须先进入解析层，生成唯一、冻结、可打印和可保存的 ResolvedConfig。RuntimePlan 及 BackendPlan、CommunicationPlan、DataPlan、CheckpointPlan 只能从 ResolvedConfig 生成。

每个最终字段至少记录 `value`、`source` 和覆盖链。训练、benchmark、OOM retry 和远程报告均保存同一份 `config.resolved.json`；DeepSpeed 额外保存 `backend.deepspeed.final.json`。

### 3.2 训练 runtime

训练 runtime 继续由 backend setup、FitLoopRunner、StepRunner、AccumulationController、CheckpointController 和统一 DeviceRuntime 组成。黄金路径不得在 workload 内自行迁移 batch、保存 checkpoint 或实现训练循环。

每次运行统一产生：

- 运行身份与环境摘要；
- ResolvedConfig 和 RuntimePlan；
- step 与稳定窗口指标；
- checkpoint manifest 与校验结果；
- backend/profile/tuner evidence；
- 成功、失败或 fallback 的最终状态。

### 3.3 Workload 适配

CLIP 黄金路径负责真实 WDS 数据、image/text processor、contrastive loss 和 images/tokens 指标。VLM LoRA 黄金路径负责 processor、conversation template、LoRA target 注入、trainable ratio 和 adapter-only checkpoint。

共享的 image decode、cache、collator、worker preprocess、prefetch、H2D 和 profile 必须复用 `parascale/data`，不能在两个 workload 中复制。

### 3.4 Benchmark 与 tuner

benchmark matrix 使用同一个 resolved base config 展开 native-DDP、FSDP 和 DeepSpeed 变体。报告同时展示吞吐、peak memory、dataloader wait、loss 稳定性、失败率、checkpoint/resume 和 fallback 轨迹。

tuner 的每项建议必须包含 action、evidence、threshold、expected trade-off 和 config update。推荐结果必须允许用户追溯“为什么选择该后端”和“为什么没有选择其他后端”。

## 4. 错误与恢复语义

- 配置冲突、缺少必需依赖和没有 trainable parameter 必须在启动前失败。
- worker crash、数据损坏、collate 错误、device mismatch 和 checkpoint 写入失败必须传播为非零退出码。
- 只有明确识别的 OOM 可以进入自动重试；每次重试必须生成新的 attempt 记录并保留原始错误。
- rank0-only checkpoint、per-rank shard、barrier、world size 和 state dict type 必须写入协议。
- resume 前必须校验 manifest、payload、checksum、world size 兼容性和 backend state。
- 自动 fallback 后的性能结果不得与原配置混为同一条成功记录。

## 5. 开发阶段

### P0：发布阻断项

- 统一包版本来源，消除 `pyproject.toml` 与 `parascale.__version__` 不一致。
- 建立 CPU、CUDA、DeepSpeed、VLM 的 clean-install 验证矩阵。
- 增加严格环境诊断，明确缺失依赖、驱动、设备和 backend。
- 清理 quickstart/examples 中的开发机路径和隐式依赖。
- 建立 Python 3.10/3.11/3.12 CI，覆盖安装、CLI 和 no-torch 测试。
- 统一结构化错误与退出码，禁止静默 fallback。

验收：全新环境仅按 README 可完成 tiny 训练和 checkpoint 校验。

### P1：真实训练黄金路径

- 固化 DataComp WDS + CLIP-B/ViT-B。
- 固化小规格真实 VLM + LoRA。
- 统一输出 loss、images/s、tokens/s、peak memory、dataloader wait、trainable ratio 和 ResolvedConfig。

验收：双卡 RTX 4090 D 上运行 500 step，执行中断恢复，恢复前后 loss 与稳定吞吐无异常跳变。

### P2：分布式可靠性

- native-DDP、FSDP 和 DeepSpeed 统一配置校验。
- torchrun 单机双卡与双容器 multi-node smoke。
- 覆盖 kill/restart、损坏 checkpoint、worker crash 和 OOM retry。
- 将 launcher 日志、rank 失败和最终退出原因纳入统一报告。

验收：故障只能明确恢复或明确失败，不允许成功状态掩盖执行路径变化。

### P3：自动选型可信化

- profile 进入 RuntimePlan 和 tuner evidence。
- 自动 batch-size sweep、OOM fallback 和后端矩阵。
- 输出统一 JSON 与 Markdown 选择依据报告。

验收：报告能解释选择结果、证据、阈值、备选方案及其代价。

### P4：试用版发布

- 固化 0.1.x 配置 schema 和公开 API 边界。
- 构建 wheel 并执行 clean-install 测试。
- 发布 changelog、已知限制和能力等级。
- 在远程双卡 GPU 容器执行最终回归并建立可复现发布记录。

验收：发布产物可由未参与开发的用户从空环境独立安装和完成黄金路径。

## 6. 测试矩阵

本地 CI 覆盖 Python 版本、配置解析、CLI、contracts、无设备逻辑和 wheel 安装。远程 GPU 覆盖真实 Torch、native-DDP、FSDP、DeepSpeed、真实数据、真实权重和 checkpoint/resume。双容器测试只声明 multi-node orchestration smoke，不替代真实多机网络及故障验收。

每次发布候选至少记录：

- commit、wheel、容器镜像和依赖版本；
- GPU、world size、backend 和通信配置；
- 模型、数据、精度、batch 与 gradient accumulation；
- warmup、measurement steps、吞吐、显存和 dataloader wait；
- checkpoint/resume、OOM/fallback 和失败统计。

## 7. 非目标

本里程碑不实现 native ZeRO-2/3、生产 TP/PP、通用 LLM 训练、视频/音频/agent 数据协议、paged KV cache 或生产 HTTP serving。这些能力不能阻塞训练试用版发布，也不能通过占位接口进入当前黄金路径。
