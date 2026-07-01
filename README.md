# ParaScale

ParaScale 是面向视觉与多模态训练场景的轻量级分布式训练控制层。它不试图重写 PyTorch、FSDP 或 DeepSpeed，而是在这些成熟后端之上提供更简洁的工程入口：

- 用小配置描述 workload、硬件预算和训练偏好；
- 用 `plan` 查看后端、精度、batch、dataloader 与 checkpoint 策略；
- 用 `train` / `smoke` 验证最小训练闭环；
- 用 `benchmark-matrix` 在同一场景下对比 native-DDP、FSDP、DeepSpeed；
- 用 profile/tuner 输出解释“为什么这样选”。

当前工程仍处于预发布阶段。`dry-run`、mock、synthetic 路径必须显式标记，不能作为真实生产 benchmark 证据。

## 快速开始

从仓库根目录运行：

```bash
python -m parascale.cli doctor
python -m parascale.cli plan --config configs/quickstart/tiny_torch.yaml
python -m parascale.cli train --config configs/quickstart/tiny_torch.yaml --dry-run
python -m parascale.cli smoke --config configs/quickstart/tiny_torch.yaml --skip-real
```

安装 PyTorch 后，可运行最小真实训练闭环：

```bash
python -m parascale.cli train --config configs/quickstart/tiny_torch.yaml
python -m parascale.cli smoke --config configs/quickstart/tiny_torch.yaml
```

同口径 benchmark 预览：

```bash
python -m parascale.cli benchmark-matrix \
  --scenario yolo-world-large \
  --variants m \
  --backends native_ddp fsdp deepspeed \
  --dry-run
```

`plan` 默认输出人类可读摘要；需要机器可读完整计划时使用 `--json`：

```bash
python -m parascale.cli plan --config configs/quickstart/vision_synthetic.json --json
python -m parascale.cli plan --config configs/quickstart/vision_synthetic.json --output runs/plan.json
```

## 统一入口

ParaScale 的用户入口应尽量保持少而稳定：

```bash
python -m parascale.cli doctor
python -m parascale.cli plan --config configs/quickstart/tiny_torch.yaml
python -m parascale.cli train --config configs/quickstart/tiny_torch.yaml
python -m parascale.cli smoke --config configs/quickstart/tiny_torch.yaml
python -m parascale.cli benchmark-matrix --scenario vlm-lora-hf-clip --dry-run
```

历史阶段脚本已归入 `tests/benchmarks/scripts/`，后续新能力优先进入 CLI，而不是继续增加一次性执行脚本。

## 配置分层

```text
configs/
  quickstart/      # 面向用户首跑的最小模板
  *.json, *.yaml   # 少量通用示例与后端片段
```

首选 quickstart 配置：

- `configs/quickstart/tiny_torch.yaml`：最小 train/resume/serve smoke。
- `configs/quickstart/vision_synthetic.json`：视觉 synthetic 规划与训练 smoke。
- `configs/quickstart/clip_tiny.json`：tiny CLIP-style 图文对比学习。
- `configs/quickstart/vlm_lora_plan.yaml`：不依赖真实权重的 VLM LoRA 规划模板。

## 工程结构

```text
parascale/
  cli.py          # doctor / plan / train / serve / benchmark 统一入口
  config.py       # 配置模型
  runtime/        # 训练运行时、launcher、workload factory、benchmark matrix
  workloads/      # vision、CLIP、DataComp、VLM LoRA、VLM cache、YOLO 等实现
  strategy/       # 静态 planner 与 runtime tuner
  data/           # 数据 schema、sampler、collator、vision/multimodal 工具
  checkpoint/     # manifest、保存、校验、恢复
  serving/        # batched serving 与本地 checkpoint 加载
  parallel/       # 声明式 TP/PP/SP 规划原语
docs/
  software_design_documentation.md       # 主设计文档
  software_requirements_specification.md # 需求说明
  remote_server_test_guide.md           # 远程验证指南
tests/
  ITERATION_TEST_REPORT.md              # 合并版迭代测试报告
  benchmarks/                           # benchmark 配置、脚本、工具和后续报告
  validation/                           # 长训与稳定性验证配置
  reports/archive/                      # 历史原始报告归档
```

## ParaScale 的定位

ParaScale 当前最有价值的方向不是替代 DeepSpeed/FSDP，而是在视觉和多模态场景中形成更好用的控制层：

- 数据管线 profile 与缓存策略；
- 自动后端选择与可解释 tuner；
- 同口径 native-DDP/FSDP/DeepSpeed benchmark；
- checkpoint/resume/serve 闭环；
- 面向 CLIP、YOLO、VLM LoRA 的实际训练模板。

目标用户体验是：

```text
config -> plan -> train/smoke -> benchmark -> profile-driven plan update
```

## 远程验证

远程验证建议使用隔离的 GPU 或 Ascend 测试节点，并通过环境变量或部署系统管理
主机、用户和凭据。CUDA/PyTorch 验证可基于以下项目镜像：

```text
parascale-ci:cu121-torch24
parascale-vlm:cu121-torch24-transformers451-peft
```

## 开发检查

```bash
python tests/run_tests.py
python -m pytest tests/test_cli_no_torch.py tests/test_config_no_torch.py tests/test_strategy_feedback_no_torch.py
```

真实 CUDA 验证应在远程容器中执行同样的重点回归，并补充 smoke 或 benchmark-matrix。

## 设计边界

- 性能优于主流框架的结论必须来自同硬件、同数据、同 batch budget 的 benchmark。
- mock、dry-run、synthetic 都必须显式标注。
- native ZeRO 当前只表示 Stage 1 optimizer-state sharding；Stage 2/3 走 DeepSpeed/FSDP。
- TP/PP 原语可测试，但完整分布式 TP/PP 必须由具体模型路径集成并通过 benchmark。
- Ascend 路线保留架构抽象，实机验证后再进入推荐路径。
## Examples 组织

`examples/` 只承载用户可直接参考的一次运行目录，不承载测试产物、checkpoint、模型权重或历史 benchmark 报告。框架代码保持一套 runtime，硬件差异只体现在示例配置和启动方式中：

```text
examples/
  gpu/
    example_001_clip_tiny_native/
    example_002_vision_synthetic_native/
  ascend/
    example_001_tiny_ascend_native/
    example_002_tiny_native_ddp_hccl/
```

GPU 示例使用 CUDA/NCCL 配置提示，Ascend 示例使用 NPU/HCCL 配置提示；二者都通过统一入口运行：

```bash
python -m parascale.cli train --config examples/gpu/example_001_clip_tiny_native/config.json
python -m parascale.cli train --config examples/ascend/example_001_tiny_ascend_native/config.json
```
