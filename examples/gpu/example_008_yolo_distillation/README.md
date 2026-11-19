# YOLO 检测蒸馏：真实图像与预训练模型

本案例使用 YOLOv8s 教师、YOLOv8n 学生、官方检测损失和三尺度特征蒸馏。
同初始学生与数据预算下，对比监督训练和蒸馏训练，并检查检查点保存、教师冻结及学生导出。
调用已有验证程序，该程序通过 ParaScale 标准训练入口运行，不另建训练引擎。

## 准备资产

使用匹配的 PyTorch/CUDA 环境，并安装：

```bash
python -m pip install -e . ultralytics==8.3.161
```

准备可信的本地 `yolov8n.pt`、`yolov8s.pt`，以及 YOLO 格式的真实图像和标签。
Ultralytics `.pt` 可能包含序列化 Python 对象，不加载不可信来源的权重。
程序禁止运行中下载模型；缺文件会失败，不会改用随机权重。

已有 `coco128.zip`、`yolov8n.pt`、`yolov8s.pt` 时，可将它们放入同一资产目录，运行：

```bash
python tests/benchmarks/tools/prepare_coco128_detection.py --assets /datasets/yolo_example
```

此步骤生成固定的 96 张训练图像、32 张验证图像和 `evaluation.yaml`，不会下载数据。
COCO128 来自 COCO 训练集，预训练模型已见过它；此划分不是未见数据泛化评测。
资产准备和后续训练必须使用相同容器挂载路径，因为生成的列表包含绝对路径。

## 运行

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda OUTPUT_DIR=runs/examples/yolo_kd_real \
bash examples/gpu/example_008_yolo_distillation/run.sh \
  --student-model /datasets/yolo_example/yolov8n.pt \
  --teacher-model /datasets/yolo_example/yolov8s.pt \
  --image-dir /datasets/yolo_example/train96/images \
  --label-dir /datasets/yolo_example/train96/labels \
  --evaluation-data /datasets/yolo_example/evaluation.yaml \
  --image-size 320 --batch-size 4 --num-samples 96 \
  --steps 40 --warmup 10 --repeats 3 --learning-rate 0.00001
```

先用以上有界验证，再以 `--steps 200 --warmup 30` 做较长对照。三次重复是相同 seed
的运行波动测量，不是多 seed 的质量研究。每次使用空输出目录。
自有数据可直接替换 `--image-dir`、`--label-dir` 和验证 YAML，无需 COCO128 准备步骤。
空标签文件表示背景，缺少标签的图像会被跳过；必须核对实际样本量。

CPU 验证应隐藏 CUDA 并指定 `DEVICE=cpu`；默认选择单卡逻辑 `cuda:0`。
PowerShell 可直接调用 `python tests/benchmarks/scripts/validate_detection_distillation.py`
并传入上述参数，同时显式指定 `--device`、`--output-dir`。

## 结果与边界

- `summary.json` 应为 `status: passed`；保留模型/数据哈希、软件版本和重复配置。
- `runs` / `aggregates`：监督与蒸馏的 step 吞吐、含数据等待的吞吐和峰值显存。
- `evaluation`：初始师生及训练后模型的 mAP；只有提供验证 YAML 才执行。
- 每次运行的 `validation.json`：梯度、教师状态、检查点内容和导出验证。
- 训练检查点保留投影/优化器；纯学生导出只用于匹配架构的 Ultralytics 推理。

该对照程序校验保存的检查点，但不执行恢复训练；恢复链路由检测训练回归测试覆盖，
需要另用标准 `train --resume-step` 流程验收，不把本案例通过当作 resume 实测。

功能通过不等于 mAP 提升。短程微调可能低于初始学生；已有 COCO128 结果不能证明
蒸馏收益。这里使用 square resize，不能与完整 Ultralytics 训练配方直接等同。
只支持同家族、同类别定义、三尺度标准 Detect；不支持 YOLO-World、分割、姿态、
端到端头或跨家族蒸馏。仅单进程 native FP32、GAS=1，不支持 QAT+KD。

完整配置与产物加载见[蒸馏指南](../../../docs/distillation.md)，实测记录见
[统一测试报告](../../../tests/benchmarks/reports/BENCHMARK_REPORT.md)。
