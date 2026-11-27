# Llama 在线蒸馏：教师、学生与 CE 基线

复用 ParaScale `OnlineDistillationLoss`，执行 hard CE 与温度缩放的
teacher-to-student KL，并与相同初始学生、相同步数的纯 CE 训练对照。
验证教师不变、学生更新和纯学生权重重载，不保证蒸馏后精度提升。
依赖 PyTorch、Transformers 4.51.3，安装方法同 [PTQ 案例](../example_005_llama_ptq/README.md)。

## 离线功能验证

```bash
OUTPUT_DIR=runs/examples/llama_kd_cpu bash examples/gpu/example_007_llama_distillation/run.sh \
  --tiny --steps 16 --temperature 2 --alpha 0.5 --learning-rate 0.0001
```

这里的教师与学生均为随机初始化 Llama，学生更浅、更窄，token ID 定义相同。
随机教师不能用来证明知识迁移质量。可用 `DEVICE=cuda` 在单 GPU 运行。
输出目录必须为空；重复执行需换目录。

## 预训练教师和学生

准备两个本地 Llama 模型与 tokenizer，以及独立的训练/评测 UTF-8 文本。
学生通常选择更小的架构；两个 tokenizer 的 token-to-ID 和特殊 token 定义必须一致，
仅词表大小相同不够。以下路径需替换：

```bash
DEVICE=cuda OUTPUT_DIR=runs/examples/llama_kd_pretrained \
bash examples/gpu/example_007_llama_distillation/run.sh \
  --teacher-model /models/llama_teacher --student-model /models/llama_student \
  --local-files-only \
  --training-text /datasets/text/train.txt --evaluation-text /datasets/text/heldout.txt \
  --seqlen 128 --batch-size 1 --batches 16 --steps 100 \
  --temperature 2 --alpha 0.5 --learning-rate 0.00001
```

每份文本至少提供 2048 个 token；评测语料不得与训练语料重复。
程序同时加载教师和学生，FP32 显存不足时应选择更小模型，不要改为未经验证的 AMP。

## 结果与检查

- `report.json` 的 `passed` 应为 `true`，核对 `model_source.kind`。
- `distillation` 与 `ce_baseline`：同预算对照的质量、训练耗时、内存和更新证据。
- `teacher_state`：教师参数与 buffer 的前后哈希应一致。
- `student_export`：纯学生导出与重新加载校验。
- `student_state_dict.pt`：仅学生参数；需匹配的学生架构才能加载，不能完整续训。

蒸馏增加训练期教师计算，学生推理速度取决于学生架构，不能由训练 loss 推断。
支持范围为单进程 native FP32；不同 tokenizer、分布式、AMP、QAT+KD 不在本案例内。
CLI 集成与 checkpoint 恢复要求见[蒸馏指南](../../../docs/distillation.md)。

## 产品训练后的独立部署

通过 `python -m parascale.cli train --config student-kd.json` 训练得到的完整
native 因果语言模型检查点，可导出为不依赖教师的 Hugging Face 模型目录：

```bash
python -m parascale.cli checkpoint export-causal-lm \
  --checkpoint /runs/student/step-00000256 \
  --model /models/student-initial \
  --output /releases/student-256
```

路径需要替换为实际训练产物。`--model` 指向学生原始配置和 tokenizer，
无需保留原始预训练权重；输出包含学生 safetensors、配置、tokenizer 和校验清单。
将[因果语言模型推理案例](../example_013_causal_lm_inference/README.md)中的
`model.pretrained_model_name_or_path` 改为 `/releases/student-256` 即可独立推理。
教师、训练数据和优化器不进入发布目录。导出不等于质量验收，部署前仍需独立评测。

复用已有推理配置，单张 A6000 上运行：

```bash
export PARASCALE_MODEL=/releases/student-256
export PARASCALE_PROMPTS=/datasets/prompts.jsonl
export PARASCALE_RUN_DIR=/runs/student-infer-new
python -m parascale.cli infer \
  --config examples/gpu/example_013_causal_lm_inference/float.json \
  --output "$PARASCALE_RUN_DIR/summary.json"
```

该配置使用 BF16、动态 KV cache，输出到新运行目录；提示词 JSONL 格式、
输入长度与质量/吞吐测量边界见对应推理案例。

本页离线验证脚本产生的 `student_state_dict.pt` 缺少产品 checkpoint manifest，
不能直接传给此命令。输出必须是新目录；仅完整 native/native-DDP 检查点支持，
不转换 FSDP/TP/PP 分片、adapter-only 或 QAT 状态。普通 DDP 导出能力不意味着
在线蒸馏已支持分布式。具体边界及 CPU 内存要求见上述蒸馏指南。

PowerShell 对应入口：
`python tests/benchmarks/scripts/validate_llama_distillation.py --tiny --steps 16 --device cpu --output runs/examples/llama_kd_windows`。

实测记录见[统一测试报告](../../../tests/benchmarks/reports/BENCHMARK_REPORT.md)。
