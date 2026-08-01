# ParaScale 用户手册

本文面向第一次使用 ParaScale 的训练工程师和研究人员，重点说明如何安装、检查环境、编写配置、启动训练、比较后端、查看 benchmark 证据以及处理常见问题。

ParaScale 当前是一个面向视觉与多模态任务的轻量级分布式训练控制层。它统一配置、运行计划、数据管线、后端选择、benchmark、checkpoint/resume 与 serving，但不替代 PyTorch DDP/FSDP、DeepSpeed、Megatron-LM 或通用 RL 框架。

## 1. 适用范围

ParaScale 适合以下工作：

| 场景 | 推荐用法 |
| --- | --- |
| 本地功能验证 | 使用 `configs/quickstart/tiny_torch.yaml` 跑通配置、训练和 checkpoint |
| CLIP / DataComp | 比较 native DDP、FSDP、DeepSpeed 的吞吐、显存和 dataloader wait |
| VLM LoRA | 验证 LoRA adapter-only 训练、trainable ratio、checkpoint 和通信策略 |
| YOLO / GroundingDINO | 使用真实视觉数据做 workload smoke、profile 和 checkpoint 验证 |
| CUDA / Ascend | 通过统一 device/backend 配置入口保持跨硬件运行口径一致 |

当前不建议把 ParaScale 直接当作以下系统使用：

- DeepSpeed 替代品。
- PyTorch DDP/FSDP 替代品。
- Megatron/NeMo 级别的超大规模 LLM 预训练系统。
- 在线 RL、具身智能 rollout 或 actor-learner 框架。
- 不限定硬件、数据、模型和 batch budget 的通用性能加速器。

## 2. 安装

### 2.1 基础环境

推荐环境：

- Python 3.10+
- PyTorch 2.4+
- CUDA 训练需要匹配的 NVIDIA Driver、CUDA、NCCL
- Ascend 训练需要匹配的 CANN、`torch_npu`、HCCL

源码安装：

```bash
git clone https://github.com/WangJunWJJ/ParaScale.git
cd ParaScale

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell 激活虚拟环境时使用：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2.2 可选依赖

按任务安装额外依赖：

```bash
python -m pip install -e ".[deepspeed]"
python -m pip install -e ".[datacomp]"
python -m pip install -e ".[vlm]"
python -m pip install -e ".[yolo]"
python -m pip install -e ".[grounding-dino]"
python -m pip install -e ".[ascend]"
```

确认安装：

```bash
parascale --version
python -m parascale.cli --help
```

## 3. 环境检查

使用 `doctor` 检查本机依赖、设备和分布式能力：

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

`doctor` 适合在正式训练前运行。若 `--strict` 返回非零退出码，优先根据输出修复缺失依赖或设备不可见问题。

## 4. 快速开始

以下流程使用内置 tiny workload，不需要真实模型或真实数据。

### 4.1 校验配置

```bash
python -m parascale.cli config validate \
  --config configs/quickstart/tiny_torch.yaml
```

### 4.2 查看运行计划

```bash
python -m parascale.cli plan \
  --config configs/quickstart/tiny_torch.yaml \
  --json
```

运行计划会展示后端选择、数据加载计划、并行策略和 checkpoint 策略。正式训练前建议先看计划，确认 batch、precision、backend 和 world size 符合预期。

### 4.3 预览训练

```bash
python -m parascale.cli train \
  --config configs/quickstart/tiny_torch.yaml \
  --dry-run
```

`--dry-run` 只验证配置和启动路径，不代表真实性能。

### 4.4 执行训练

```bash
python -m parascale.cli train \
  --config configs/quickstart/tiny_torch.yaml
```

### 4.5 校验 checkpoint

```bash
python -m parascale.cli checkpoint validate \
  --checkpoint runs/quickstart/tiny_torch
```

## 5. 配置文件指南

ParaScale 推荐从 `configs/quickstart/` 开始：

| 文件 | 用途 |
| --- | --- |
| `configs/quickstart/tiny_torch.yaml` | 最小真实训练、checkpoint、resume、serve smoke |
| `configs/quickstart/vision_synthetic.json` | 视觉 synthetic workload |
| `configs/quickstart/clip_tiny.json` | tiny CLIP-style 图文对比学习 |
| `configs/quickstart/vlm_lora_plan.yaml` | VLM LoRA 规划模板 |

历史 benchmark 和专项验证配置位于：

```text
tests/benchmarks/configs/
tests/validation/configs/
```

### 5.1 常见配置层

常见配置可以按职责理解：

| 配置层 | 典型内容 |
| --- | --- |
| `backend` | 后端类型、DeepSpeed/FSDP/native-DDP 参数、通信策略 |
| `training` | workload、batch、precision、steps、learning rate |
| `data` | 数据路径、num workers、prefetch、pin memory、cache |
| `workload` | 模型类型、模型路径、LoRA、任务专属参数 |
| `checkpoint` | 保存路径、resume、校验和、rank-aware 保存策略 |
| `hardware` | world size、GPU/NPU 数量、拓扑和设备约束 |

不同示例配置可能使用 JSON 或 YAML。建议新实验优先复制 quickstart 或 benchmark 配置，再做小范围修改。

### 5.2 环境变量引用

配置支持环境变量解析时，应把机器相关路径放在环境变量里，例如：

```bash
export PARASCALE_DATA_ROOT=/data/datasets
export PARASCALE_MODEL_ROOT=/data/models
```

这样同一份配置可以在 4090、A6000、Ascend 或 CI 环境中复用，减少硬编码路径。

## 6. 启动训练

### 6.1 单配置训练

```bash
python -m parascale.cli train \
  --config configs/quickstart/tiny_torch.yaml
```

如果当前配置需要分布式启动，ParaScale 会根据配置和后端选择合适的 launcher。你也可以先通过 `plan` 确认启动方式。

### 6.2 Benchmark 训练

```bash
python -m parascale.cli benchmark \
  --config tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp.json
```

`benchmark` 更适合正式性能记录，会关注吞吐、显存、dataloader wait、checkpoint 证据和运行元数据。

### 6.3 后端矩阵

```bash
python -m parascale.cli benchmark-matrix \
  --scenario vlm-lora-hf-clip \
  --backends native_ddp fsdp deepspeed \
  --dry-run
```

使用 `--dry-run` 先生成启动命令和配置预览。正式跑矩阵前，确认每个后端的依赖、模型路径、数据路径和 global batch 一致。

## 7. Workload 使用入口

### 7.1 Tiny Torch

用于最小功能验证：

```bash
python -m parascale.cli train \
  --config configs/quickstart/tiny_torch.yaml
```

### 7.2 CLIP / DataComp

常用配置：

```bash
python -m parascale.cli benchmark \
  --config tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp.json

python -m parascale.cli benchmark \
  --config tests/benchmarks/configs/benchmark_datacomp_medium_fsdp.json

python -m parascale.cli benchmark \
  --config tests/benchmarks/configs/benchmark_datacomp_medium_deepspeed.json
```

CLIP/DataComp 是比较 native DDP、FSDP、DeepSpeed 的主要场景。正式对比时要保持相同模型、数据、global batch、精度、warmup 和测量窗口。

### 7.3 VLM LoRA

```bash
python -m parascale.cli benchmark \
  --config tests/benchmarks/configs/benchmark_vlm_lora_datacomp_native_ddp.json
```

VLM LoRA 场景重点关注 trainable ratio、adapter-only checkpoint、LoRA 参数同步和 ZeRO-2/ZeRO-3 fallback。

### 7.4 YOLO-World

```bash
python -m parascale.cli benchmark \
  --config tests/benchmarks/configs/benchmark_yolo_world_objects365_official_native_ddp.json
```

YOLO-World 依赖对应模型库、真实检测数据和标签缓存。建议先跑 smoke，再扩展到长窗口 benchmark。

### 7.5 GroundingDINO

```bash
python -m parascale.cli benchmark \
  --config tests/benchmarks/configs/benchmark_ground_dino_objects365_native.json
```

GroundingDINO 场景用于验证视觉 grounding workload adapter、数据加载、tensor cache 和 checkpoint 路径。

## 8. 后端选择建议

| 后端 | 推荐场景 | 注意事项 |
| --- | --- | --- |
| `native` | 单卡 smoke、最小验证、CPU/CUDA/NPU 基础路径 | 不提供多卡梯度同步 |
| `native_ddp` | 单机多卡视觉/多模态训练，尤其是 CLIP/DataComp 中等规模任务 | 需要关注通信 hook、bucket、拓扑和 dataloader wait |
| `fsdp` | 参数量较大、显存压力明显、需要 shard state dict 的任务 | 配置更复杂，checkpoint 格式要提前确认 |
| `deepspeed` | ZeRO-2/ZeRO-3、offload、成熟大模型显存缩放 | DeepSpeed JSON 应由 backend 层统一生成或校验 |
| `ascend_native` | Ascend NPU 功能验证和 HCCL 分布式路径 | 需要匹配 CANN、`torch_npu` 和镜像环境 |

经验上，小中型视觉/多模态任务可以优先尝试 native DDP，并用 FSDP/DeepSpeed 做基线。大模型、显存压力和 ZeRO 能力优先使用 DeepSpeed 或 FSDP。

## 9. 数据管线

ParaScale 的数据层主要覆盖：

- vision preprocessing。
- multimodal processor。
- collator。
- sampler。
- tensor cache / processor cache / prompt cache。
- dataloader profile。

常见调优项：

| 项 | 作用 |
| --- | --- |
| `num_workers` | 提高 CPU 侧数据加载并行度 |
| `prefetch_factor` | 增加 DataLoader 预取深度 |
| `persistent_workers` | 避免每个 epoch 反复创建 worker |
| `pin_memory` | CUDA 场景下加速 host-to-device 传输 |
| `device_prefetch` / `cuda_prefetch` | 尝试把 H2D 传输和计算重叠 |
| tensor cache | 缓存昂贵图像预处理结果 |
| processor cache | 缓存多模态 processor 结果 |

如果 benchmark 中 `dataloader_wait_ms` 偏高，优先检查数据路径、磁盘吞吐、worker 数、cache 命中率和 batch 形状分布。

## 10. Benchmark 与报告

统一报告入口：

[tests/benchmarks/reports/BENCHMARK_REPORT.md](../tests/benchmarks/reports/BENCHMARK_REPORT.md)

更新报告：

```bash
python tests/benchmarks/tools/build_benchmark_report.py \
  --report-root tests/benchmarks/reports \
  --output tests/benchmarks/reports/BENCHMARK_REPORT.md
```

正式 benchmark 需要记录：

- Git commit。
- 硬件型号和卡数。
- CUDA/NCCL 或 CANN/HCCL 版本。
- 容器或 Python 环境。
- 模型路径和版本。
- 数据路径和样本规模。
- 后端、精度、global batch、step 数。
- warmup 和测量窗口。
- 吞吐、峰值显存、`dataloader_wait_ms`。
- checkpoint/resume 状态。

不要把 `--dry-run`、依赖解析或 synthetic 输出当作真实性能证据。

## 11. Checkpoint 与 Resume

校验 checkpoint：

```bash
python -m parascale.cli checkpoint validate \
  --checkpoint runs/quickstart/tiny_torch
```

常见关注点：

- manifest 是否存在。
- checksum 是否匹配。
- rank-aware shard 是否完整。
- world size 是否和 resume 目标兼容。
- FSDP/DeepSpeed checkpoint 格式是否和后端一致。

建议每个正式 benchmark 至少保存一次 checkpoint，并在报告中记录 validate 结果。

## 12. Inference 与 Serving

推理入口：

```bash
python -m parascale.cli infer \
  --config examples/gpu/example_003_clip_real_inference/config.json
```

Serving 入口：

```bash
python -m parascale.cli serve \
  --config configs/server_tiny_torch.yaml
```

当前 serving 更适合功能 smoke 和架构验证。生产部署前需要结合具体模型、batcher、KV cache、服务协议和资源隔离策略做专项压测。

## 13. 远程 GPU/NPU 验证

远程硬件验证建议参考：

[docs/remote_server_test_guide.md](remote_server_test_guide.md)

基本原则：

- 每次测试前确认实际 worktree、数据、模型、镜像和驱动版本。
- 数据密集任务建议容器增加 `--shm-size=8g`。
- 4090/A6000/Ascend 结果必须使用相同任务口径才能比较。
- 不要把本地 dry-run 结果替代远程 GPU/NPU 真实测试。

## 14. 常见问题

### 14.1 `doctor --strict` 失败怎么办？

先看缺失项。如果是 CUDA/NPU 不可见，优先检查驱动、容器挂载、`CUDA_VISIBLE_DEVICES`、`ASCEND_VISIBLE_DEVICES`、CANN 和 `torch_npu`。如果是 DeepSpeed 缺失，安装 `.[deepspeed]` 或切换到 native/FSDP 后端。

### 14.2 `dataloader_wait_ms` 很高怎么办？

优先检查数据是否在本地盘、是否启用合适的 `num_workers`、`prefetch_factor`、`persistent_workers`、`pin_memory` 和 cache。多卡训练中 dataloader wait 可能直接拉低 scaling 效率。

### 14.3 native DDP、FSDP、DeepSpeed 怎么选？

小中型视觉/多模态任务先用 native DDP 建立强基线；显存压力增大时尝试 FSDP；需要 ZeRO、offload 或成熟大模型后端时使用 DeepSpeed。最终选择必须以同口径 benchmark 为准。

### 14.4 为什么 dry-run 不能代表真实性能？

`--dry-run` 只验证配置和启动命令，不执行真实训练 step，不包含数据加载、模型计算、通信、显存峰值和 checkpoint 成本。

### 14.5 如何新增 workload？

优先把通用能力放到 `parascale/data`、`parascale/runtime`、`parascale/strategy`、`parascale/communication` 或 `parascale/contracts`。workload 自身只保留轻量 adapter、spec 解析和 model/data wiring。

### 14.6 如何判断性能结论可信？

可信结论至少需要同一硬件、同一数据、同一模型、同一 batch budget、同一精度、同一 warmup 和同一测量窗口。报告中还应包含原始配置、运行日志、吞吐、显存、dataloader wait 和 checkpoint 验证结果。

## 15. 下一步阅读

| 文档 | 内容 |
| --- | --- |
| [README.md](../README.md) | 项目定位、快速开始、架构和示例总览 |
| [configs/README.md](../configs/README.md) | 配置目录组织与推荐入口 |
| [software_design_documentation.md](software_design_documentation.md) | 软件设计、模块边界和路线图 |
| [software_requirements_specification.md](software_requirements_specification.md) | 功能需求和验收标准 |
| [remote_server_test_guide.md](remote_server_test_guide.md) | 远程 GPU/NPU 容器测试指南 |
| [BENCHMARK_REPORT.md](../tests/benchmarks/reports/BENCHMARK_REPORT.md) | 统一 benchmark 证据入口 |
