# ParaScale 架构收口设计

## 目标

在不重新制造巨型模块的前提下，删除重构后遗留的兼容壳、空占位和纯重导出文件，统一跨模块协议与推理入口，使源码目录直接表达真实产品能力。

## 设计原则

1. `contracts/` 是跨模块稳定数据协议的唯一归属。
2. `runtime/` 负责执行生命周期，不定义 workload 或硬件专用协议。
3. `communication/` 负责通信策略构建与执行，不重复定义 contract。
4. `serving/` 是推理服务编排层，不重复实现通用推理 runtime。
5. 不保留旧接口兼容壳；仓库尚未上线，内部引用一次性迁移到正式路径。
6. 保留 training、backends、device、workloads、commands 的有效职责边界。
7. 保留 workload 与 inference task 扩展命名空间，但其中必须承载真实协议或实现，不能只有占位常量。

## 协议收口

`parascale.contracts.plan.CommunicationPlan` 是唯一通信计划结构，吸收当前通信优化所需字段：

- `backend`
- `ddp_hook`
- `bucket_cap_mb`
- `use_no_sync`
- `adapter_only_sync`
- `overlap_h2d`
- `reasons`
- `evidence`

`parascale.communication.plan` 只保留 `build_communication_plan()`，并返回 contracts 中的 `CommunicationPlan`。删除 `strategy/` 下五个 plan 重导出文件，调用方直接依赖 contracts。

## 推理入口收口

- `InferenceEngine`：唯一通用推理 runtime，负责设备执行、批处理调用、内存指标和 collective 协作。
- `ServingEngine`：上层服务编排，负责 scheduler、KV cache、请求队列和调用 `InferenceEngine`。
- 删除 `ServeEngine` 名称和相关导出。
- `runtime/inference/tasks/` 保留视觉、文本、多模态任务模块，并新增统一 task protocol 与 registry。
- task adapter 负责输入准备、batch 约束、后处理、静态图提示和指标扩展；registry 只在初始化阶段解析，执行热路径直接持有 adapter。
- 现有三个仅包含 `TASK_TYPE` 常量的实现将替换为真实 task adapter，而不是删除任务边界。
- 推理复用通用 runtime memory 能力时直接导入正式实现，不保留重导出文件。

## Workload Adapter 边界

保留 `parascale/workloads/adapters/`，作为模型和外部生态接入的扩展命名空间：

- `contracts/workload.py` 定义唯一的 `WorkloadAdapter` protocol，`workloads/adapters/` 提供 adapter registry 与模型专用实现。
- 内置 CLIP、YOLO、VLM workload 仍由 `workloads/` 顶层模块负责装配。
- 模型专用输入和目标转换可放在 `workloads/adapters/`，不得进入通用 runtime。
- adapter 注册发生在构建阶段，不在训练 step 热路径动态查找。

## 删除范围

确定删除：

- `parascale/runtime/backend.py`
- `parascale/runtime/factory.py`
- `parascale/checkpoint/manifest.py`
- `parascale/checkpoint/validator.py`
- `parascale/runtime/inference/memory.py`
- `parascale/strategy/backend_plan.py`
- `parascale/strategy/communication_plan.py`
- `parascale/strategy/data_plan.py`
- `parascale/strategy/device_plan.py`
- `parascale/strategy/inference_plan.py`
- 无实际能力且无近期设计用途的 `data/text/`
- 经引用检查确认未进入生产链路的 reporting 占位模块

删除前必须将生产代码、测试和公开导出迁移到正式路径。不能通过新增其他 facade 维持旧导入。

## 保留边界

- `runtime/training/`：训练生命周期组件。
- `runtime/backends/`：Native、DDP、FSDP、DeepSpeed、Ascend 后端。
- `core/device/`：CUDA、Ascend、CPU 设备操作。
- `workloads/`：视觉和多模态薄适配器。
- `workloads/adapters/`：模型专用及第三方 workload 扩展协议与实现。
- `commands/`：统一 CLI 的命令实现。
- `data/`：通用数据预处理、缓存、collator 和 profile。
- `runtime/inference/tasks/`：视觉、文本和多模态推理任务适配器。

## 测试与验收

1. 架构测试确认旧 facade 和占位路径不能再被导入。
2. `CommunicationPlan` 全工程只有一个类定义。
3. 公共推理入口只导出 `InferenceEngine` 和上层 `ServingEngine`。
4. workload adapter 和 inference task adapter 均有真实 protocol、registry 和行为测试。
5. 训练、benchmark、checkpoint、配置和无 torch 测试保持通过。
6. `python -m ruff check parascale tests setup.py` 无错误。
7. `python tests/run_tests.py` 全部通过。
8. 统计收口后的 Python 文件数量，并确认没有新增等价转发壳。

## 非目标

- 本轮不改变训练算法、后端性能策略或模型行为。
- 本轮不补充新的推理特性。
- 本轮不为旧导入路径保留兼容层。
- 本轮不以减少文件数为由合并职责独立的有效模块。
