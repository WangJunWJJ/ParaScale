# ParaScale 版本历史

本文记录 ParaScale 对外发布版本的用户可见变化。ParaScale `0.1.x` 为试用版本，适用于架构评估、功能验证和受控训练实验。

## 0.1.0 - 2026-07-03

### Added

- 提供统一的 `parascale` CLI，覆盖环境诊断、计划生成、训练、benchmark、checkpoint 校验和推理入口。
- 提供 native-DDP、FSDP 和 DeepSpeed 训练后端，以及 CUDA 与 Ascend 设备抽象入口。
- 提供 DataComp WDS、CLIP 对比学习、VLM LoRA 和 YOLO-World workload 适配能力。
- 提供 ResolvedConfig、RuntimePlan、BackendPlan、CommunicationPlan 和可解释 backend matrix 报告。
- 提供 batch-size sweep、OOM retry、checkpoint/resume、kill/restart 和数据管线 profile 能力。
- 提供 PEP 517 wheel、可选依赖、strict doctor、稳定 CLI 退出码和 Python 3.10-3.12 CI。

### Changed

- runtime 按 training、backend、device、workload、reporting 和 commands 职责拆分，CLI 仅负责命令分发。
- workload 默认输出 CPU batch，设备迁移统一由 backend/device prefetch 链路负责。
- benchmark matrix 排除最终 full-state checkpoint I/O，checkpoint 持久化由独立稳定性测试验收。
- backend 推荐同时输出候选评估、落选原因、置信度、阈值、预期权衡和配置建议。

### Fixed

- 修复分布式 checkpoint rank-gating、manifest 并发写入和失败被错误验证为成功的问题。
- 修复梯度累积绕过设备迁移、忽略后续 micro-batch 和不完整累积窗口的问题。
- 修复 DataLoader 异常静默降级、bytes 图像 cache key 冲突和冻结参数进入 LoRA optimizer 的问题。
- 修复 OOM fallback 对非 OOM 错误继续重试、重试元数据未持久化和报告证据不完整的问题。

### Validation

- 本地源码测试覆盖配置、CLI、contracts、runtime、checkpoint、reporting、workload 和 packaging。
- wheel 已在不挂载源码的一次性容器中完成 clean-install，并通过 public CLI 执行 `parascale --version`、strict doctor、plan、tiny train 和 checkpoint validate。
- CUDA 发布门禁使用双 RTX 4090 D、PyTorch 2.4.0+cu121、真实 DataComp WDS 与预训练 CLIP-B/32，完成 20-step native-DDP、FSDP 和 DeepSpeed 矩阵；三后端均成功，报告以高置信度推荐 FSDP。
- CUDA checkpoint smoke 从 step 2 恢复到 step 4，最终 manifest 校验通过。
- Ascend 服务器可连接且 8 张 910B4 健康，但验收时全部 AICore 100% 并被现有任务占用；为避免干扰，实机 smoke 标记为 pending，正式 tag 尚未创建。

### Known Limitations

- 当前生产主路径依赖 PyTorch native-DDP/FSDP 或 DeepSpeed；native ZeRO、生产级 TP/PP 尚未提供。
- 单机多卡已完成真实验证；双容器 multi-node 仅属于编排 smoke，不能替代真实多服务器网络与故障验收。
- Serving 和推理 runtime 仍为试用能力，不声明生产级 HTTP 服务、paged KV cache 或通用大语言模型 serving。
- Ascend 以独立同构 NPU 路径为目标，GPU 训练与 Ascend 推理协同不属于本版本验收范围。
- 性能结果只对报告中明确记录的硬件、数据、模型、精度、batch 和测量窗口有效。
