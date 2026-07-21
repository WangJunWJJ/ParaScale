# ParaScale

<div align="center">

**面向视觉与多模态任务的分布式训练控制层**

统一配置、运行计划、数据管线、后端选择、benchmark、checkpoint/resume 与 serving，让 CLIP、VLM LoRA、YOLO、GroundingDINO 等任务可以用同一套口径训练、对比和复现。

[特性](#特性) · [快速开始](#快速开始) · [架构](#架构) · [CLI](#cli) · [Benchmark](#benchmark-与验证) · [测试](#测试) · [文档](#文档)

</div>

---

## 项目定位

ParaScale 不是 PyTorch、FSDP 或 DeepSpeed 的替代品。它是一个轻量的训练与推理控制层，负责把成熟后端、数据管线、配置解析、运行计划、性能证据和恢复闭环组织到同一条产品化路径里。

核心闭环：

```text
config -> plan -> train/smoke -> benchmark -> profile/tune -> checkpoint/resume -> serve
```

适用场景：

| 场景 | ParaScale 负责 | 成熟后端负责 |
| --- | --- | --- |
| CLIP / DataComp | 数据管线、profile、后端矩阵、同口径报告 | DDP/FSDP/DeepSpeed 执行 |
| VLM LoRA | trainable ratio、adapter-only checkpoint、LoRA 同步策略 | FSDP/DeepSpeed 显存缩放 |
| YOLO / GroundingDINO | workload adapter、真实数据 smoke、checkpoint 证据 | 原生模型和训练库 |
| Ascend / CUDA | 统一 device/backend 边界和配置入口 | CANN/HCCL 或 CUDA/NCCL |

> 当前版本适合架构评估、功能验证和受控训练实验。任何性能结论都必须基于相同硬件、数据、模型、batch budget、精度、warmup 和测量窗口。

## 特性

| 能力 | 当前实现 |
| --- | --- |
| 统一 CLI | `doctor`、`config`、`plan`、`train`、`infer`、`serve`、`benchmark`、`checkpoint` 等命令集中在 `python -m parascale.cli` |
| 分层配置 | `LayeredParaScaleConfig` 作为权威配置模型，`ParaScaleConfig` 作为运行时平铺视图 |
| 后端选择 | native、native-DDP、FSDP、DeepSpeed、Ascend native 后端在 `parascale/runtime/backends/` 注册 |
| 数据管线 | `parascale/data/vision` 和 `parascale/data/multimodal` 承载通用预处理、cache、collator 和 profile |
| workload adapter | CLIP、VLM LoRA、YOLO、GroundingDINO、tiny workload 的 spec 已按场景拆入 `parascale/workloads/specs/` |
| 训练运行时 | train、serve、benchmark runner 已按执行模式拆分，避免单一 orchestrator 膨胀 |
| checkpoint/resume | manifest、checksum、rank-aware 保存、world-size 校验、converter 和 serving manifest |
| benchmark 证据 | 所有正式结果收敛到 `tests/benchmarks/reports/BENCHMARK_REPORT.md` |
| 架构守门 | no-torch 测试按行为边界拆分，并约束 capability 模块不反向拥有训练编排 |

## 安装

### 环境要求

- Python 3.10+
- PyTorch 2.4+
- CUDA 训练需要匹配的 NVIDIA Driver、CUDA、NCCL
- Ascend 训练需要匹配的 CANN、`torch_npu`、HCCL

### 源码安装

```bash
git clone https://github.com/WangJunWJJ/ParaScale.git
cd ParaScale

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

按场景安装可选依赖：

```bash
python -m pip install -e ".[deepspeed]"
python -m pip install -e ".[datacomp]"
python -m pip install -e ".[vlm]"
python -m pip install -e ".[yolo]"
python -m pip install -e ".[grounding-dino]"
python -m pip install -e ".[ascend]"
```

确认版本：

```bash
parascale --version
```

当前包版本为 `0.1.0`，公共 API 版本为 `0.1`。

## 快速开始

以下流程使用内置 tiny workload，不需要下载真实模型或数据。

### 1. 检查环境

```bash
python -m parascale.cli doctor
python -m parascale.cli doctor --strict
```

按目标能力检查：

```bash
python -m parascale.cli doctor --require cuda --require distributed
python -m parascale.cli doctor --require deepspeed
python -m parascale.cli doctor --require npu
```

### 2. 校验配置

```bash
python -m parascale.cli config validate \
  --config configs/quickstart/tiny_torch.yaml
```

迁移旧配置：

```bash
python -m parascale.cli config migrate \
  --config legacy.json \
  --output migrated.v1.json
```

### 3. 查看运行计划

```bash
python -m parascale.cli plan \
  --config configs/quickstart/tiny_torch.yaml \
  --json
```

### 4. 预览并执行训练

```bash
python -m parascale.cli train \
  --config configs/quickstart/tiny_torch.yaml \
  --dry-run

python -m parascale.cli train \
  --config configs/quickstart/tiny_torch.yaml
```

### 5. 校验 checkpoint

```bash
python -m parascale.cli checkpoint validate \
  --checkpoint runs/quickstart/tiny_torch
```

### 6. 生成后端矩阵预览

```bash
python -m parascale.cli benchmark-matrix \
  --scenario vlm-lora-hf-clip \
  --backends native_ddp fsdp deepspeed \
  --dry-run
```

`--dry-run` 只生成配置和启动命令，不代表真实性能。正式 benchmark 必须记录模型、数据、硬件、精度、global batch、warmup、测量窗口、吞吐、显存、dataloader wait 和 checkpoint/resume 状态。

## 架构

ParaScale 当前按产品职责拆分，而不是按单个模型脚本堆叠：

```text
parascale/
  commands/          CLI parser 与命令实现，cli.py 只保留薄入口
  configuration/     配置读取、环境变量解析、ResolvedConfig
  config.py          layered schema 与 runtime flat config
  contracts/         batch、metric、checkpoint、backend、workload、plan 协议
  core/              device、collective、cluster topology
  data/              vision/multimodal 通用数据管线
  runtime/           train/serve/benchmark runners 与 runtime context
  runtime/backends/  native、FSDP、DeepSpeed、Ascend native 后端
  runtime/training/  fit loop、accumulation、precision、memory、checkpointing
  runtime/inference/ inference engine、batcher、task adapters
  communication/     DDP hook、bucket、no_sync、adapter-only sync 计划
  strategy/          planner、profile feedback、tuner、heterogeneous plan
  checkpoint/        manifest、manager、converter、adapter checkpoint
  workloads/         场景 adapter 与 model/data wiring
  workloads/specs/   tiny、vision、clip、vlm_lora、yolo、ground_dino specs
  serving/           serving orchestration、scheduler、KV cache
  reporting/         benchmark aggregation、matrix、markdown report
```

运行路径：

```text
User config / CLI overrides
          |
  configuration resolver
          |
   LayeredParaScaleConfig
          |
      RuntimePlan
   /    |      |     \
device backend data checkpoint
   \    |      |     /
    mode-specific runner
      /       |       \
   train    infer    serve
      \       |       /
   benchmark / profile / checkpoint evidence
```

关键边界：

| 模块 | Ownership |
| --- | --- |
| `commands/*` | 注册命令参数，调用 runtime 公共入口，不导入私有 runner 细节 |
| `runtime/*_runner.py` | train、serve、benchmark 的执行入口，各自只负责一种运行模式 |
| `runtime/backends/*` | 后端 setup、state dict、checkpoint payload，不拥有 workload 逻辑 |
| `workloads/specs/*` | workload 配置解析，按场景拆分 |
| `data/*` | 可复用 preprocessing、cache、sampler、collator、profile |
| `parallel/`、`quantization/`、`serving/` | capability 模块，不反向拥有训练编排 |

## CLI

```text
config               Validate or migrate a ParaScale configuration.
plan                 Build an auto strategy and dataloader plan.
doctor               Diagnose dependencies and devices.
smoke                Run the compact server smoke flow.
train                Validate and launch a training run.
infer                Run an inference workload.
serve                Validate and launch serving runtime.
benchmark            Validate and launch one benchmark run.
benchmark-matrix     Run native-DDP/FSDP/DeepSpeed benchmark matrix.
benchmark-stability  Run long-window stability and resume stress benchmarks.
vision-profile       Profile a real image folder data pipeline.
checkpoint           Validate or convert checkpoints.
```

查看完整参数：

```bash
python -m parascale.cli --help
python -m parascale.cli <command> --help
```

稳定退出码：

| Exit code | 含义 |
| ---: | --- |
| `0` | 成功 |
| `2` | 配置或环境要求不满足 |
| `3` | 依赖缺失 |
| `4` | 运行失败 |
| `5` | checkpoint 失败 |
| `6` | benchmark 子任务失败 |
| `70` | 未预期内部错误 |

## Benchmark 与验证

统一报告入口：

- [tests/benchmarks/reports/BENCHMARK_REPORT.md](tests/benchmarks/reports/BENCHMARK_REPORT.md)

更新报告：

```bash
python tests/benchmarks/tools/build_benchmark_report.py \
  --report-root tests/benchmarks/reports \
  --output tests/benchmarks/reports/BENCHMARK_REPORT.md
```

当前报告覆盖：

| Suite | 内容 |
| --- | --- |
| Dual 4090 full validation | 多模型功能与吞吐验证 |
| Direct PyTorch/DeepSpeed comparison | 与直接 PyTorch DDP/FSDP、DeepSpeed 同任务对比 |
| Ascend functional validation | Ascend 910B4 功能验证 |
| Ascend parallel matrix | 单 docker 2 卡、2 docker 单卡、2 docker 2 卡并行吞吐 |
| Cross-hardware CLIP DataComp | RTX 4090 与 Ascend 同口径对比 |
| RTX 4090 precision comparison | FP32/BF16/FP16 对比 |

Benchmark 资产位置：

```text
tests/benchmarks/configs/    benchmark 配置
tests/benchmarks/scripts/    远程或容器运行脚本
tests/benchmarks/tools/      汇总与报告工具
tests/benchmarks/reports/    summary.json 与统一报告
tests/validation/configs/    长窗口与稳定性验证配置
```

## 示例

```bash
# GPU tiny CLIP 训练
bash examples/gpu/example_001_clip_tiny_native/run.sh

# GPU vision synthetic 训练
bash examples/gpu/example_002_vision_synthetic_native/run.sh

# GPU 真实 CLIP / YOLO-World 推理
bash examples/gpu/example_003_clip_real_inference/run.sh
bash examples/gpu/example_004_yolo_world_real_inference/run.sh

# Ascend tiny 训练和 HCCL 分布式 smoke
bash examples/ascend/example_001_tiny_ascend_native/run.sh
bash examples/ascend/example_002_tiny_native_ddp_hccl/run.sh
```

更多说明：

- [examples/README.md](examples/README.md)
- [examples/gpu/README.md](examples/gpu/README.md)
- [examples/ascend/README.md](examples/ascend/README.md)

## Python API

试用阶段建议优先使用 CLI 与配置文件。需要嵌入时，可使用公共导出：

```python
from parascale import (
    CheckpointManager,
    ParaScaleConfig,
    RuntimePlan,
    TrainEngine,
    build_strategy_plan,
)
```

## 测试

```bash
python -m pip install -e ".[dev]"
python -m ruff check parascale tests setup.py
python tests/run_tests.py
python -m build
```

测试组织已经按边界拆分：

| 测试范围 | 文件示例 |
| --- | --- |
| 核心架构边界 | `tests/test_architecture_boundaries_no_torch.py` |
| runtime/topology/registry | `tests/test_core_architecture_no_torch.py` |
| checkpoint runtime/controller/converter | `tests/test_checkpoint_*_no_torch.py` |
| training loop/device/checkpoint | `tests/test_training_*_no_torch.py` |
| config/optimizer/workload/deepspeed | `tests/test_*_config_no_torch.py` |
| benchmark 报告与矩阵 | `tests/test_benchmark_*_no_torch.py` |

CI 覆盖 Python 3.10、3.11、3.12 的源码测试，并在独立 Python 3.11 环境安装 wheel 后执行 `doctor -> plan -> train -> checkpoint validate`。GPU/NPU 和真实数据长窗口验证属于远程硬件门禁，不能用依赖解析或 dry-run 替代。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/software_design_documentation.md](docs/software_design_documentation.md) | 产品目标、总体架构、模块边界、路线图 |
| [docs/software_requirements_specification.md](docs/software_requirements_specification.md) | 功能需求、能力等级、验收标准 |
| [docs/remote_server_test_guide.md](docs/remote_server_test_guide.md) | 远程 GPU/NPU 容器测试指南 |
| [configs/README.md](configs/README.md) | 配置文件组织与示例 |
| [tests/benchmarks/reports/BENCHMARK_REPORT.md](tests/benchmarks/reports/BENCHMARK_REPORT.md) | 统一 benchmark 证据入口 |

## 版本历史

### 0.1.0 - GPU-verified 试用版

- 建立 runtime-first 架构与统一 CLI。
- 接入 native-DDP、FSDP、DeepSpeed 和 Ascend 架构入口。
- 建立视觉/多模态数据管线、profile/tuner、benchmark matrix。
- 完成 checkpoint/resume、serving 和远程 GPU smoke 路径。
- 将 workload specs、runtime runners、CLI parser、config schema、no-torch 测试按边界拆分。
- benchmark 报告收敛到 `tests/benchmarks/reports/BENCHMARK_REPORT.md`。

详细变化见 [CHANGELOG.md](CHANGELOG.md)。

## 许可证

ParaScale 使用 [MIT License](LICENSE)。
