# ParaScale 用户手册

文档基线：2026-11-19，对应当前开发分支；具体发布包能力以其版本文档为准。

本文面向第一次使用 ParaScale 的训练工程师和研究人员，重点说明如何安装、检查环境、编写配置、启动训练、比较后端、查看 benchmark 证据以及处理常见问题。

ParaScale 当前是一个面向视觉与多模态任务的轻量级分布式训练控制层，统一配置、运行计划、数据管线、后端选择、benchmark 和 checkpoint/resume，并提供有限的推理入口。它不替代 PyTorch DDP/FSDP、DeepSpeed、Megatron-LM 或通用 RL 框架，也不是完整的生产级训推一体平台。

## 1. 适用范围

ParaScale 适合以下工作：

| 场景 | 推荐用法 |
| --- | --- |
| 本地功能验证 | 使用 `configs/quickstart/tiny_torch.yaml` 跑通配置、训练和 checkpoint |
| CLIP / DataComp | 比较 native DDP、FSDP、DeepSpeed 的吞吐、显存和 dataloader wait |
| VLM LoRA | 验证 LoRA adapter-only 训练、trainable ratio、checkpoint 和通信策略 |
| YOLO / GroundingDINO | 使用真实视觉数据做 workload smoke、profile 和 checkpoint 验证 |
| CUDA / Ascend | 通过统一 device/backend 配置入口保持跨硬件运行口径一致 |
| PTQ / QAT | 单设备 FP32 的 4/8-bit 权重参考量化；不承诺整数内核推理加速 |
| 在线蒸馏 | 分类/因果语言模型 API，以及标准 YOLO 同家族特征蒸馏 |

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
| `backend` | `training_backend`、`precision`、后端执行和通信参数 |
| `training` | `workload`、`max_steps`、`checkpoint_dir`、`checkpoint_interval` |
| `data` | 数据路径、batch size、num workers、prefetch、pin memory、cache |
| `model` / `task` / `lora` | 模型路径及由具体 workload spec 解释的任务参数 |
| `optimizer` | `type`、`lr`、`weight_decay` 等优化器参数 |
| `runtime` / `hardware_profile` | 实际设备选择，以及规划使用的卡数、显存、拓扑信息 |
| `parascale` | 现有平铺配置入口；`parascale.quantization` 配置参考 PTQ/QAT |
| `distillation` | 学习目标、教师完整配置与教师检查点 |

不同示例配置可能使用 JSON 或 YAML。建议新实验优先复制 quickstart 或 benchmark 配置，再做小范围修改。

`ParaScaleConfig` 平铺模型与分组配置目前并存。计划/训练会解析优先级并记录配置来源；避免在两个位置重复指定同一字段。PTQ 命令目前直接读取 `parascale`，量化时应在该节点显式设置 native/FP32，不要假定每个命令都已共享同一配置解析路径。

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

单次 `train` 调用不会自动创建所需的分布式进程。多卡时先确认配置的后端、设备数和 batch，再通过 launcher 启动，例如在具备两张 GPU 的环境中：

```bash
torchrun --standalone --nproc_per_node=2 -m parascale.cli train \
  --config tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp.json
```

示例中的真实数据和模型路径需按本机修改。`training.max_steps` 是本次调用的优化器更新次数，不是 microbatch 数，也不是恢复后训练总步数上限。

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

### 7.6 标准 YOLO 检测与蒸馏

标准闭集检测使用 `training.workload: yolo_detection`，与 `yolo_world` 分开。它复用官方检测损失；数据采用 YOLO 的 `images/`、`labels/` 目录，空标签文件表示背景，缺少对应标签文件的图像会被跳过。当前方形 resize 管线不等于 Ultralytics 完整增强配方。

安装已验证版本：

```bash
python -m pip install ultralytics==8.3.161
```

在完整学生配置中增加以下片段，详细可运行结构和教师权重准备见[蒸馏指南](distillation.md#yolo-detection-distillation)：

```yaml
distillation:
  enabled: true
  task: detection
  feature_weight: 0.1
  teacher:
    checkpoint: /models/teacher_state.pt
    config: {}  # Replace with the full matching teacher workload configuration.
```

`teacher.config` 必须填写完整的嵌套配置字典，不能使用配置文件路径或空字典。教师检查点必须可加载，不会回退为随机教师。以常规训练命令执行填写完整后的配置：

```bash
python -m parascale.cli train --config /configs/yolo_kd.yaml
```

支持同家族三尺度 Detect、相同类别名称/顺序和 stride，已测 YOLOv8、YOLO11。不同宽度通过可训练 1x1 投影对齐；目标为官方检测损失加 RMS 归一化特征 MSE，并非 FGD 或框/类别分布蒸馏。拒绝 YOLO-World、分割、姿态、end-to-end head、跨家族和 GroundingDINO。仅支持单进程 native FP32、梯度累积为 1、无 QAT。

结束后默认导出 `checkpoint_dir/student.pt`，可通过 `training.student_output` 改路径。该文件只含学生原始模型参数，不含教师、投影和优化器；使用匹配架构的 Ultralytics 加载做验证/预测，不是 ParaScale `infer` 自动可识别的新模型格式，也不能用于完整续训。

### 7.7 PTQ 与 QAT

从已有完整 workload 配置开始，在 `parascale` 中合入以下字段。此处只是配置片段，不是完整训练配置：

```yaml
parascale:
  training_backend: native
  precision: fp32
  quantization:
    enabled: true
    mode: ptq
    bits: 4
    backend: reference
    fuse_modules: false
    per_channel: true
    quantize_activations: false
    exclude_modules: [lm_head]
```

PTQ 使用已有 workload 的模型和校准 dataloader：

```bash
python -m parascale.cli quantize --config /configs/model_ptq.yaml --output /outputs/model_ptq.pt
```

QAT 将 `mode` 改为 `qat`，继续使用 `train --config /configs/model_qat.yaml`。默认导出 `checkpoint_dir/quantized.pt`，可由 `training.quantized_output` 指定。QAT 使用 full-model checkpoint，不能设置 adapter-only、混合精度、FSDP/DeepSpeed、分布式或与蒸馏同时启用。

仅精确类型的 PyTorch Linear/Conv2d 被转换；4-bit/8-bit 是权重存储格式，运行时反量化为浮点计算，不是 GPTQ/AWQ 或优化的 INT4 推理。共享权重、自定义子类及 GPT-2 Conv1D 需要额外适配。量化导出重载需要匹配的浮点模型模板，不能直接当作完整续训检查点。

Llama、BERT、ViT、ResNet、Qwen2 的小型随机模型 API 测试不代表这些家族所有模型都已接入 CLI，也不证明预训练精度保持。Python 调用和重载示例见[量化指南](quantization.md)。

### 7.8 分类与因果语言模型蒸馏

Python API 支持分类 `[B,C]` 和因果语言预测 `[B,L,V]`，采用 hard CE 与带温度平方的 teacher-to-student KL。教师和学生可不同深度/宽度，但类别语义或 token-to-ID 必须对应；相同词表大小不足以证明对齐。因果语言目标会处理下一 token 位移和 padding/ignore mask，不能直接用于 masked LM、分割或连续动作回归。

CLI 在完整学生配置中设置顶层 `distillation`，选择 `task: classification` 或 `causal_lm`，提供 `temperature`、`alpha`、嵌套教师配置和 tensor 权重检查点。仅支持已有 workload 构造路径、单进程 native FP32、梯度累积为 1。续训会校验教师权重和目标配置指纹，不能不经说明更换教师。完整示例见[蒸馏指南](distillation.md)。

### 7.9 本地因果语言模型训练

`training.workload: causal_lm` 将本地 Hugging Face 因果语言模型接入统一 `train`、检查点和 PTQ/QAT 流程。模型目录必须包含匹配的 fast tokenizer；不会隐式下载模型或执行远程模型代码。

分布式产品入口要求启动器提供完整的 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`，并与已初始化的进程组一致。多 rank 不允许使用单进程 `native`/`ascend_native` 后端。DP 规模以实际进程组或启动器分配为准，而不是机器 GPU 总数；显式设置的 `hardware_profile.world_size` 或拓扑必须与执行规模一致，恢复前会跨 rank 校验计划。

训练输出的 `execution` 按 rank 记录实际 `world_size`、通信后端、最终 optimizer 类、是否为 PyTorch `ZeroRedundancyOptimizer` 及 scheduler 是否绑定该 optimizer。它用于交叉核验 `strategy_plan`，不代表性能或质量验收。scheduler 绑定为 `null` 表示未配置；ZeRO 标记为 `null` 表示可选类无法导入、未完成类型验证。Native ZeRO-1 由 `zero_stage: 1` 控制，不依赖旧的 `zero_optimization` 开关。

先用 `python -m parascale.cli prepare-text --input train.txt --tokenizer /data/model --output /data/tokens/train` 生成不可变 token 语料，再设置 `data.type: token_corpus` 和 `data.path`。每个非空、去除首尾空白的 UTF-8 行作为一篇文档，末尾追加 EOS；默认拒绝超过 1 MiB 的行，不截断。对照实验必须使用同一份预处理 token，不能与整文件一次性分词混用。

数据通过只读内存映射读取，`data.num_workers` 必须为 0。各 DP rank 分配互不重叠的完整块，共同舍弃不足一个全局 microbatch 的尾部。`data.shuffle_window_blocks` 默认 65536，仅在固定顺序的窗口内打乱，不等同于全语料随机排列。恢复时校验模型配置、训练目标、数据身份和并行拓扑；任一 rank 校验失败，各 rank 在应用模型、优化器和 RNG 状态之前共同拒绝恢复。

完整配置及单卡、2/4 卡 DDP 启动命令见 [causal-LM 示例](../examples/gpu/example_012_causal_lm_training/README.md)。离线小型 Llama 的 CPU、双进程 Gloo 恢复及 PTQ/QAT 导出已纳入回归测试；真实预训练模型质量、A6000 吞吐、长跑稳定性及 FSDP/DeepSpeed 仍需独立验收。该入口尚未接入 TP/PP，示例关闭 CUDA 预取；不能据此宣称已完成生产性能验收。

### 7.10 学习率调度与恢复

在 `training.scheduler` 中设置 `type: cosine` 或 `linear`，并显式指定 `total_steps`；`warmup_steps` 默认 0，`min_lr_ratio` 默认 0。例如 `{"type": "cosine", "total_steps": 1000, "warmup_steps": 50, "min_lr_ratio": 0.1}` 表示先从零学习率 warmup，完成 50 次更新后达到 `optimizer.lr`，再衰减至第 1000 次更新时的 10%，之后保持该下限。未配置或设为 `null` 时不创建调度器。

调度单位为成功的 optimizer update，不是 microbatch；梯度累积及 AMP 跳过更新不会多推进调度。`total_steps` 是全程固定计划，与每次调用的追加训练步数 `max_steps` 独立，恢复时不要改写它。训练结果的 `next_learning_rates` 是下次更新将使用的学习率。

目前配置式调度支持 native/native DDP（含 native ZeRO-1），其他后端显式拒绝，不能静默忽略。检查点保存调度状态及其配置身份；恢复时禁止增删调度器或修改类型、总步数、warmup、下限及基础学习率。所有 rank 在训练/保存前协调校验调度器，单 rank 创建失败或身份不一致时共同失败。

SDK 使用 `TrainEngine.fit(..., scheduler_builder=factory)` 将调度器绑定到后端最终 optimizer；恢复时向 `load_checkpoint` 传递同一配置的 factory，后续 `fit` 不再重复传 factory。引擎保留调度器，最终 `save_checkpoint` 自动包含其状态。直接传入的 scheduler 必须绑定最终 optimizer，不能在训练中替换。调度状态支持有限值 JSON 与 PyTorch Tensor；跨 rank 比较 Tensor 的 dtype、shape 和数值，不比较各 rank 的设备编号。自定义 LambdaLR 等函数型调度器（包括嵌套调度）必须通过 `TrainEngine(scheduler_metadata=...)` 提供稳定配置身份，否则拒绝执行；函数语义不会被 PyTorch 保存，不能仅依赖相同的序列化状态证明函数未改变。

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

`training.checkpoint_interval: 0` 关闭训练过程中的周期性保存；结束时是否保存由
`training.skip_final_checkpoint` 独立控制。纯吞吐测试若要完全禁止检查点写入，需同时设置
`checkpoint_interval: 0` 和 `skip_final_checkpoint: true`。这类测试不能作为检查点恢复验收。

恢复 quickstart 的 step 1 检查点：

```bash
python -m parascale.cli train \
  --config configs/quickstart/tiny_torch.yaml --resume-step 1
```

使用配置中的同一 `checkpoint_dir`；该配置 `max_steps: 2` 表示再更新两步，预期到 global step 3。训练保存 rank-local RNG 和 AMP scaler，恢复包含数据 replay-skip。精确重现还要求相同数据顺序、数据处理行为和确定性算子，不能仅因 RNG 恢复就承诺任意数据管线位级一致。

蒸馏续训使用完整训练检查点，保留投影参数和优化器状态；独立的 `student.pt` 或量化导出文件不是替代品。

### 11.1 最新完整检查点恢复

`native` / `native_ddp` 可以显式使用 `--resume-latest`：

```bash
python -m parascale.cli train \
  --config configs/quickstart/tiny_torch.yaml --resume-latest
```

分布式任务保留原来的 `torchrun` 启动参数和卡数，只在训练命令末尾增加此选项。
该选项与 CLI 或配置内的 `resume_step` 互斥。没有完整检查点时直接失败，不会从头训练。
检查点按步数倒序检查目录身份、native 后端、模型状态文件、所有 rank 的运行状态，以及文件大小和 SHA256；
会跳过缺失、损坏、结构不完整或允许 benchmark 忽略错误的记录。结果 JSON 的 `checkpoint_recovery`
列出选中步数和每个跳过原因；分布式运行由 rank 0 选择并广播结果。

这只是文件完整性选择，不是模型、数据、优化器配置的自动兼容性搜索，也不是针对不可信文件的安全扫描。
选中后仍执行既有恢复校验，若后端、world size、模型或优化器不匹配应修正输入，不会再次自动回退。
`--resume-step` 保持严格指定步数的行为。`--dry-run` 不读取检查点，不能作为恢复成功的证据。

native 模型状态、各 rank 运行状态和 manifest 使用同目录临时文件，完成写入并同步后替换目标文件。
替换前的序列化或磁盘同步失败不会截断原文件；POSIX 系统还会同步父目录。
这不是多文件事务：覆盖同一步时，中途失败仍可能造成文件集合不一致，必须重新校验并使用较早完整步骤恢复。
不支持多个作业同时写同一目录，不承诺所有文件系统断电恢复，也未给 FSDP/DeepSpeed 后端文件增加事务保证。
检查点应存放在可信、独占的运行目录；同步和校验耗时也应计入训练成本。

### 11.2 CLIP / DataComp 首条生产验收链路

候选入口：[A6000 单机 1/2/4 卡配置与操作说明](../configs/production/clip_datacomp/README.md)。
当前交付包括保守配置、恢复入口及 CPU 故障注入测试，不代表生产认证。
GPU 真实数据恢复、独立验证集质量、数据身份校验、进程故障退出时限和 24 小时稳定性仍是上线前门禁。
CLIP 当前通过可微特征汇聚使用数据并行组内每个微批的全局负样本；SigLIP 使用逐对 sigmoid 损失。
运行时负责绑定纯数据并行组及按样本数归一化。直接使用 workload builder 时，损失默认仅在本地计算；
推荐通过 `TrainEngine.fit` 或 `train` 入口运行，手动 DDP 调用者必须显式绑定损失的 DP group 并处理不同 rank 的样本数加权。
梯度累积不跨微批共享负样本，比较时既要匹配 optimizer global batch，也要匹配每个对比微批的全局样本。
历史各卡本地负样本结果与当前目标不等价；不能直接用于新目标的加速或质量结论。旧检查点也不能绕过目标身份检查继续使用。

### 11.3 长跑资源与数据恢复边界

训练配置提供以下默认上限（放在 `training` 层）：

```json
{
  "training": {
    "checkpoint_keep_last": 3,
    "metrics_history_limit": 1000,
    "max_resume_replay_batches": 10000
  }
}
```

- native/native_ddp 新检查点发布后保留最近 3 个通过完整恢复契约及文件校验的受管检查点，最小允许值为 2。
  不自动删除历史未受管目录、损坏目录、软链接或外部引用；这些文件仍需人工检查。
  被其他检查点引用的目录也会保留；若存在无法解析的清单，暂停该目录的自动清理。
  每个任务必须独占检查点目录。FSDP/DeepSpeed 不使用此自动清理策略。
  保存期间还需容纳一个正在写入的完整检查点，保留数量不是磁盘配额。
- 内存只保留最近 1000 条指标；输出中的 `metrics_history_dropped` 明确报告省略数量。
  `metrics_history` 是本次调用的尾部窗口，不代表全部训练过程，不得用其长度作为训练总步数。
  benchmark/stability 会扣除已被截断的预热步骤，不会在尾部窗口重复执行预热排除。
- 真实 HuggingFace CLIP/SigLIP 的零 worker、无 CUDA 预取、未压缩 JPG/TXT WDS 路径
  使用分片偏移索引和可保存的读取位置。恢复时重建索引、核对分片 SHA256 与 rank/批大小身份，
  然后直接读取后续样本，不重放之前的图片解码和模型预处理。源文件必须在训练期间保持不变。
  每个 rank 的读取位置独立保存到 runtime state，不共享 rank 0 的读取位置。
  索引占用随数据集样本数量增长，而非随训练步数增长；大数据集启动扫描仍需单独评估。
- 其他无状态加载器使用有上限的 replay-skip；超过 `max_resume_replay_batches` 会报错，
  不自动忽略数据位置。通用训练运行时对有状态加载器按已完成训练的批次保存游标，
  不把 CUDA 预取领先的批次算作已消费；连续调用 `fit()` 时也会恢复未消费的预取位置。
  多 worker 加载器必须实现 loader-level `state_dict()/load_state_dict()`，并保证恢复后
  下一批等于原本下一待交付批次；只有 sampler 状态会被拒绝，不能据此宣称精确恢复。
  随机增强必须使用独立且随数据状态保存的 RNG，或按样本确定性生成。
  与模型共享全局 RNG 的预取增强不保证逐位重现；上述 WDS 工作负载的预取限制未改变。
- benchmark matrix/stability 为每个 run/backend 生成独立运行目录；调用者仍须为独立实验使用新输出根目录。

这些保护不等于完成生产认证，仍须执行真实故障恢复、质量评估及长时间稳定性测试。

## 12. Inference 与 Serving

推理入口：

```bash
python -m parascale.cli infer \
  --config examples/gpu/example_003_clip_real_inference/config.json
```

Serving 入口：

```bash
python -m parascale.cli serve \
  --config configs/quickstart/tiny_torch.yaml \
  --checkpoint runs/quickstart/tiny_torch
```

请先完成 quickstart 训练。当前 `serve` 仅内置 tiny 模型检查点路径，执行配置中的 `serving.requests` 后排空队列并返回，不监听 HTTP/gRPC 端口。真实 CLIP/YOLO-World 的一次性推理使用 `infer`；`clip_real`、`yolo_world_real` 是真实模型入口，而 `clip`、`yolo_world` 等部分别名是 synthetic 路径，使用前核对配置。

推理执行使用 eval/no-grad，调用后模型保持 eval 状态；继续训练前应显式切回 train 模式。Serving 响应的输出数量必须与请求匹配，批量 `infer` 的结果保留策略见下一节。生产 tokenizer、流式生成、paged KV 和通用模型部署不在当前交付范围。标准 YOLO 检测蒸馏导出通过 Ultralytics 使用，不能与 YOLO-World 推理入口混淆。

### 12.1 批量推理的结果保留与消费

`infer` 的 `inference.max_outputs` 默认是 `32`：只将前 32 个测量批次的结果放进返回的
`outputs`，但仍处理全部输入并统计全部测量批次。`output_retention` 返回 `retained`、
`dropped` 和配置上限，不能用 `len(outputs)` 推算总请求数。设为 `0` 时不保留结果，
显式设为 JSON `null` 才会完整收集；后者仅适合已知大小的有限输入。

Python API 可逐批消费结果，例如将 `runner` 的后处理结果写为 JSON Lines：

```python
import json

with open("predictions.jsonl", "w", encoding="utf-8") as stream:
    summary = runner.run(
        batches,
        warmup_steps=0,
        max_outputs=0,
        on_output=lambda output: stream.write(json.dumps(output) + "\n"),
    )
```

回调同步执行，为下游处理提供背压；回调异常或设备同步失败会终止调用，不返回虚假的成功结果。
回调不接收预热批次，返回值被忽略，不支持异步回调。回调与保留列表使用相同的结果对象；
不要原地修改需要保留的结果。示例要求结果可 JSON 序列化，张量需要由任务后处理转换。
这不是 token 流式生成，也不会自动释放模型缓存、上游已收集的数据或回调自行保存的对象。
上限按批次数而非字节数计算，大结果或长跑优先使用 `max_outputs=0`。

`latency_ms_*` 和相应吞吐只衡量预测及后处理，排除预热、batch 准备/传输和回调耗时；
不等于含数据读取、排队和输出写入的端到端服务吞吐。内存指标仍覆盖本次调用的预热和测量阶段。
长跑只累计计数和延迟总量，不保留逐批延迟历史。若每条输入都必须交付，使用
`warmup_steps=0`，将预热放到正式数据流之外，并提供可靠的结果消费回调。

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
| [quantization.md](quantization.md) | PTQ/QAT API、导出重载和支持边界 |
| [distillation.md](distillation.md) | 分类、因果语言和 YOLO 检测蒸馏完整示例 |
| [BENCHMARK_REPORT.md](../tests/benchmarks/reports/BENCHMARK_REPORT.md) | 统一 benchmark 证据入口 |
