# ParaScale

ParaScale 是面向视觉与多模态场景的分布式训练和推理控制层，重点服务百卡以下集群中的 CLIP-style 对比学习、VLM LoRA、视觉模型训练及视觉/多模态推理。

ParaScale 不重新实现 PyTorch、FSDP 或 DeepSpeed。它在成熟计算后端之上统一配置、运行计划、数据管线、训练与推理、性能分析以及 checkpoint/resume，让一次实验更容易运行、比较和复现。

[特性](https://github.com/WangJunWJJ/ParaScale#%E7%89%B9%E6%80%A7) · [安装](https://github.com/WangJunWJJ/ParaScale#%E5%AE%89%E8%A3%85) · [快速开始](https://github.com/WangJunWJJ/ParaScale#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B) · [文档](https://github.com/WangJunWJJ/ParaScale#%E6%96%87%E6%A1%A3) · [示例](https://github.com/WangJunWJJ/ParaScale#%E7%A4%BA%E4%BE%8B) · [API 参考](https://github.com/WangJunWJJ/ParaScale#api%E5%8F%82%E8%80%83) · [架构设计](https://github.com/WangJunWJJ/ParaScale#%E6%9E%B6%E6%9E%84%E8%AE%BE%E8%AE%A1) · [测试](https://github.com/WangJunWJJ/ParaScale#%E6%B5%8B%E8%AF%95) · [版本历史](https://github.com/WangJunWJJ/ParaScale#%E7%89%88%E6%9C%AC%E5%8E%86%E5%8F%B2) · [许可证](https://github.com/WangJunWJJ/ParaScale#%E8%AE%B8%E5%8F%AF%E8%AF%81)

> ParaScale 当前为试用版本，适合架构评估、功能验证和受控训练实验。生产使用前，请在目标模型、数据、硬件和训练窗口上完成同口径验收。

## 特性

- **统一入口**：通过 `doctor`、`plan`、`train`、`infer`、`benchmark` 和 `checkpoint` 完成主要工作流。
- **统一配置**：合并用户配置、命令行覆盖、workload、backend 与硬件信息，生成可追溯的 ResolvedConfig 和 RuntimePlan。
- **多训练后端**：支持 native、native-DDP、FSDP 和 DeepSpeed；FSDP/DeepSpeed 作为大模型及显存压力场景的成熟 fallback。
- **视觉与多模态数据管线**：提供 batching、collation、缓存、worker 预处理、异步 prefetch 和 dataloader profile。
- **可解释调优**：根据静态配置和 runtime profile 推荐后端、精度、batch、通信与数据加载参数，并说明选择依据。
- **可靠恢复**：提供 checkpoint manifest、完整性校验、rank-aware 保存、resume 和 adapter-only checkpoint。
- **同口径对比**：统一比较 native-DDP、FSDP 与 DeepSpeed 的吞吐、显存和稳定性。
- **GPU/NPU 一体化架构**：CUDA/NCCL 与 Ascend NPU/HCCL 共用 runtime 边界，环境与示例分别组织。

当前推荐原则：

- 中小规模视觉/多模态任务可将 native-DDP 作为吞吐优先候选，但必须以同口径 benchmark 为依据；
- 大模型、ZeRO、offload 或显存压力场景优先使用 FSDP/DeepSpeed；
- Ascend 已具备统一架构入口，当前仍需在目标 CANN/`torch_npu` 环境完成生产级实机验收；
- TP、PP 和 native ZeRO 仅按当前文档中的实现及验证等级声明，不作为试用版生产主路径。

## 安装

### 环境要求

- Python 3.10+
- PyTorch 2.4+
- CUDA 训练需匹配的 NVIDIA 驱动、CUDA 与 NCCL
- Ascend 训练需匹配的 CANN、`torch_npu` 与 HCCL

建议先安装与目标硬件匹配的 PyTorch，再安装 ParaScale，避免 pip 自动选择不匹配的设备 wheel。

### 从源码安装

```bash
git clone https://github.com/WangJunWJJ/ParaScale.git
cd ParaScale

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

按任务安装可选依赖：

```bash
pip install -e ".[deepspeed]"  # DeepSpeed
pip install -e ".[datacomp]"   # DataComp / WebDataset
pip install -e ".[vlm]"        # Transformers / PEFT / VLM LoRA
pip install -e ".[yolo]"       # YOLO / detection
pip install -e ".[ascend]"     # Ascend NPU
pip install -e ".[dev]"        # tests / lint
```

## 快速开始

以下流程使用内置 tiny workload，不需要下载模型权重或数据集。

### 1. 检查环境

```bash
python -m parascale.cli doctor
```

发布或正式训练前使用严格诊断，并按任务声明必需能力：

```bash
parascale doctor --strict
parascale doctor --require cuda --require distributed
parascale doctor --require deepspeed
parascale doctor --require npu
```

普通 `doctor` 只报告事实并返回 0；`--strict` 默认要求 core 与 Torch，`--require` 指定的能力不可用时返回 2，并在 JSON 中给出证据。

### 2. 查看运行计划

```bash
python -m parascale.cli plan \
  --config configs/quickstart/tiny_torch.yaml
```

使用 `--json` 查看完整 RuntimePlan，或使用 `--output runs/plan.json` 保存结果。

### 3. 预览并执行训练

```bash
# 只解析配置和执行计划
python -m parascale.cli train \
  --config configs/quickstart/tiny_torch.yaml \
  --dry-run

# 执行 2 个训练 step
python -m parascale.cli train \
  --config configs/quickstart/tiny_torch.yaml
```

### 4. 校验 checkpoint

```bash
python -m parascale.cli checkpoint validate \
  --checkpoint runs/quickstart/tiny_torch
```

至此完成最小闭环：

```text
doctor -> plan -> train -> checkpoint validate
```

### 5. 预览后端对比矩阵

```bash
python -m parascale.cli benchmark-matrix \
  --scenario vlm-lora-hf-clip \
  --backends native_ddp fsdp deepspeed \
  --dry-run
```

`--dry-run` 只生成配置和启动命令。正式性能结论必须使用相同模型、数据、全局 batch、精度、硬件、warmup 和 measurement window。

## 文档

- [软件设计文档](docs/software_design_documentation.md)：产品目标、总体架构、模块边界和路线图。
- [软件需求规格说明](docs/software_requirements_specification.md)：功能需求、能力等级和验收标准。
- [ResolvedConfig 设计](docs/resolved_config_design.md)：配置来源、覆盖规则和审计输出。
- [架构收口设计](docs/architecture_closure_design.md)：核心职责和推理入口边界。
- [远程服务器测试指南](docs/remote_server_test_guide.md)：容器化 GPU 测试与远程验证。
- [统一测试与 benchmark 报告](tests/UNIFIED_TEST_BENCHMARK_REPORT.md)：历史测试方法、口径和结果。

## 示例

`examples/` 按 GPU 和 Ascend 环境组织。每个示例目录包含完整配置、`run.sh` 和独立说明，运行产物不会写入示例目录。

```bash
# GPU tiny CLIP 训练
bash examples/gpu/example_001_clip_tiny_native/run.sh

# GPU 视觉 synthetic 训练
bash examples/gpu/example_002_vision_synthetic_native/run.sh

# GPU 真实 CLIP / YOLO-World 推理
bash examples/gpu/example_003_clip_real_inference/run.sh
bash examples/gpu/example_004_yolo_world_real_inference/run.sh

# Ascend 单卡与 HCCL 分布式训练
bash examples/ascend/example_001_tiny_ascend_native/run.sh
bash examples/ascend/example_002_tiny_native_ddp_hccl/run.sh
```

Ascend CLIP、YOLO-World 推理示例位于 [`examples/ascend/`](examples/ascend/)。真实任务运行前，请根据服务器环境配置模型、数据集和输出路径。完整约定见 [examples/README.md](examples/README.md)。

<a id="api参考"></a>

## API 参考

### CLI

```text
doctor                检查设备、PyTorch 和可选依赖
plan                  解析配置并生成 RuntimePlan
train                 启动训练或执行 dry-run
infer                 执行一次性推理 workload
serve                 启动 serving runtime
benchmark             执行单项 benchmark
benchmark-matrix      执行统一后端对照矩阵和 OOM retry
benchmark-stability   执行长窗口及恢复稳定性测试
vision-profile        分析真实图像目录的数据管线
checkpoint validate   校验 checkpoint manifest 和 payload
```

查看完整参数：

```bash
python -m parascale.cli --help
python -m parascale.cli <command> --help
```

CLI 使用稳定退出码：`2` 表示配置或环境要求失败，`3` 表示依赖缺失，`4` 表示运行失败，`5` 表示 checkpoint 失败，`6` 表示 benchmark 子任务失败，`70` 表示未预期的内部错误。预期失败会向 stderr 输出一条 JSON；设置 `PARASCALE_DEBUG=1` 可在开发环境重新抛出原始异常。

### Python API

```python
from parascale import (
    CheckpointManager,
    ParaScaleConfig,
    RuntimePlan,
    TrainEngine,
    build_strategy_plan,
)
```

顶层包导出配置、计划、设备、数据、训练、推理和 checkpoint 的主要公共类型。试用阶段 Python API 仍可能调整，用户工作流应优先使用统一 CLI 与配置文件。

## 架构设计

ParaScale 将硬件能力与训练策略分层：CUDA/Ascend 负责设备能力，native-DDP/FSDP/DeepSpeed 负责训练执行；workload 只保留模型和任务适配，通用数据处理、设备迁移、profile 与 checkpoint 位于框架核心。

```text
User config / CLI overrides
            |
     Config resolver
            |
       RuntimePlan
       /    |    \
 device  backend  communication/data/checkpoint plans
       \    |    /
       Unified runtime
        /          \
 training          inference
    |                  |
 native/FSDP/DS   vision/text/multimodal adapters
        \              /
       profile / benchmark / checkpoint evidence
```

核心目录：

```text
parascale/
  commands/          CLI 命令实现
  config/            配置加载、解析和追踪
  contracts/         跨模块稳定协议
  core/              device、collective、topology
  data/              通用视觉与多模态数据能力
  runtime/           training、inference、backends
  workloads/         薄 workload adapter
  strategy/          planner、profile feedback、OOM retry
  checkpoint/        checkpoint contract 与管理器
  serving/           推理服务编排
```

详细设计和能力边界见 [软件设计文档](docs/software_design_documentation.md)。

## 测试

```bash
pip install -e ".[dev]"
python tests/run_tests.py
python -m ruff check parascale tests setup.py
python -m build
```

CI 在 Python 3.10、3.11 和 3.12 上运行源码测试，并在独立 Python 3.11 环境安装 wheel，执行 `doctor -> plan -> train -> checkpoint validate`。GPU/NPU 和 DeepSpeed 的真实执行仍属于远程硬件发布门禁，不能由依赖解析结果代替。

GPU/NPU、真实数据和真实权重验证应在隔离容器或测试节点执行，并记录 commit、镜像、模型、数据、精度、全局 batch、吞吐、显存、dataloader wait 和 checkpoint/resume 结果。

benchmark 资产统一位于 [`tests/benchmarks/`](tests/benchmarks/)，稳定性验证位于 [`tests/validation/`](tests/validation/)。正式长训容器建议配置充足共享内存，例如 `--shm-size=8g`。

## 版本历史

### 0.1.0 - 试用版

- 建立 runtime-first 架构和统一 CLI；
- 提供 ResolvedConfig、RuntimePlan 和 backend registry；
- 接入 native-DDP、FSDP、DeepSpeed 及 Ascend 架构入口；
- 建立视觉/多模态数据管线、profile/tuner 和 benchmark matrix；
- 完成选定路径的 checkpoint/resume、推理和远程 GPU smoke；
- 将可运行示例按 GPU 与 Ascend 环境组织。

详细记录见 [统一测试与 benchmark 报告](tests/UNIFIED_TEST_BENCHMARK_REPORT.md)。

## 许可证

ParaScale 使用 [MIT License](LICENSE)。你可以在遵守许可证条款的前提下使用、修改和分发本项目。
