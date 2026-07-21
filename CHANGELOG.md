# ParaScale 版本历史

本文记录 ParaScale 对外发布版本的用户可见变化。ParaScale `0.1.x` 为试用版本，适用于架构评估、功能验证和受控训练实验。

## Unreleased

暂无。

## 0.2.0 - 2026-07-21

### Added

- 引入配置 schema v1、`config validate` 与 `config migrate`，旧 v0 配置可显式迁移，未来版本配置会被拒绝。
- 冻结 `PUBLIC_API_VERSION = "0.2"` 的根包公共 API，并通过快照测试防止无意破坏。
- 增加 bug、性能回归和 workload 请求 Issue 表单。
- 增加架构边界守门测试，约束 no-torch 测试文件规模，并防止 capability 模块反向拥有训练编排。

### Changed

- 仓库内配置统一声明 `schema_version: 1`，wheel clean-install 增加配置校验和迁移验证。
- CI action 升级到 Node 24 运行时版本，并收紧为只读仓库权限。
- 已发布 tag 采用不可变策略；发布后的修复必须使用新的 patch 版本，不再移动历史 tag。
- 将 workload specs 按 tiny、vision、clip、vlm_lora、yolo、ground_dino 场景拆分，原 `parascale.runtime.specs` 模块不再作为配置汇聚点。
- 将 CLI parser 注册下沉到 `parascale/commands/*`，保持 `parascale/cli.py` 为薄入口。
- 将 train、serve、benchmark runner 按执行模式拆分，移除旧 `parascale.runtime.orchestrator` 聚合入口，并集中 torch device selection。
- 将 no-torch 测试覆盖按 runtime、checkpoint、training、config、workload 边界拆分，降低后续维护冲突。
- 将 benchmark 查阅入口收敛到 `tests/benchmarks/reports/BENCHMARK_REPORT.md`，并刷新 README 的架构、CLI、benchmark 与测试说明。

### Validation

- 本地通过 `python -m ruff check parascale tests setup.py`。
- 本地通过 `python tests/run_tests.py`，共 `289 passed`。
- README 链接与 packaging 相关测试通过，版本源仍由 `parascale._version.__version__` 单点定义。

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
- Ascend 服务器可连接且 8 张 910B4 健康，但验收时全部 AICore 100% 并被现有任务占用；为避免干扰，未执行实机 smoke。发布负责人已明确豁免该门禁，因此 `0.1.0` 的发布等级为 `GPU-verified / Ascend-unverified`。

### Known Limitations

- 当前生产主路径依赖 PyTorch native-DDP/FSDP 或 DeepSpeed；native ZeRO、生产级 TP/PP 尚未提供。
- 单机多卡已完成真实验证；双容器 multi-node 仅属于编排 smoke，不能替代真实多服务器网络与故障验收。
- Serving 和推理 runtime 仍为试用能力，不声明生产级 HTTP 服务、paged KV cache 或通用大语言模型 serving。
- Ascend 以独立同构 NPU 路径为目标，GPU 训练与 Ascend 推理协同不属于本版本验收范围。
- 性能结果只对报告中明确记录的硬件、数据、模型、精度、batch 和测量窗口有效。
