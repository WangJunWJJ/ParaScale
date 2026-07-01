# ParaScale 统一测试与 Benchmark 盘查报告

更新时间：2026-06-18

## 1. 报告定位

本文件统一汇总 ParaScale 重构以来的代码场景测试、远程实机 benchmark、稳定性测试、profile/tuner 验证和数据管线优化结果。后续人工盘查测试方法与历史结论时，以本文件为入口。

历史零散报告、临时 `runs/` 输出、pytest 缓存和中间产物已清理；保留必要的测试配置、测试脚本、模型权重和数据依赖。

## 2. 测试资产布局

```text
tests/
  UNIFIED_TEST_BENCHMARK_REPORT.md   # 当前统一测试盘查报告
  run_tests.py                       # 本地回归入口
  test_*.py                          # 单元与场景回归
  benchmarks/
    configs/                         # benchmark 配置
    scripts/                         # 历史 benchmark 复现实验脚本
    tools/                           # benchmark 聚合和数据缓存工具
    reports/                         # 后续新增 benchmark 报告输出目录
  validation/
    configs/                         # 长训、resume、ZeRO-3 等稳定性配置
```

## 3. 常用测试方法

### 3.1 本地基础回归

```bash
python tests/run_tests.py
```

覆盖内容：

- 源码语法检查。
- no-torch 配置、planner、CLI、tuner、benchmark-matrix、checkpoint、serving 基础回归。
- workload registry、strategy、data schema、parallel plan、quantization、runtime smoke 等测试。

最新结果：

```text
82 passed
```

### 3.2 CLI 与 benchmark 重点回归

```bash
python -m pytest \
  tests/test_config_no_torch.py \
  tests/test_cli_no_torch.py \
  tests/test_clip_contrastive_workload.py \
  tests/test_benchmark_matrix_cli_no_torch.py -q
```

最新结果：

```text
34 passed, 1 skipped
```

### 3.3 Benchmark matrix dry-run

```bash
python -m parascale.cli benchmark-matrix \
  --scenario yolo-world-large \
  --variants m \
  --backends native_ddp fsdp deepspeed \
  --dry-run

python -m parascale.cli benchmark-matrix \
  --scenario vlm-lora-real \
  --backends native_ddp \
  --dry-run
```

最新结果：均通过，能生成同口径命令计划和配置。

### 3.4 远程 CUDA 容器验证

远程环境：

- 机器：内部单节点双 GPU 验证服务器
- GPU：双 RTX 4090 / 4090D
- 常用镜像：
  - `parascale-ci:cu121-torch24`
  - `parascale-vlm:cu121-torch24-transformers451-peft`
  - `parascale-yolo:cu121-torch24-ultralytics83161`

已验证 smoke：

```text
doctor: True
plan: True
train: True
checkpoint_validate: True
resume: True
serve: True
```

## 4. P0-P3 阶段验证结论

### P0：同口径后端 benchmark

目标：建立 native-DDP、FSDP、DeepSpeed 在同硬件、同数据、同 batch budget 下的公平对照。

代表性结论：

- DataComp WDS CLIP-B style b8：native-DDP + bf16 compression 优于 FSDP 和 DeepSpeed。
- YOLO-World/Object365 official loss：native-DDP 在双卡 4090 上优于 FSDP 和 DeepSpeed。
- 性能优势声明只接受同硬件、同数据、同 batch budget 的 benchmark 结果。

### P1：功能闭环

目标：验证最小可用训练闭环，不只看吞吐。

已验证：

- `doctor`
- `plan`
- `train`
- checkpoint manifest validate
- resume
- serve
- CLIP/DataComp WDS native-DDP smoke
- YOLO-World native-DDP smoke

### P2：profile/tuner 可解释调优

目标：让 ParaScale 输出“为什么这样选”，而不是只输出“选了什么”。

已验证：

- step time、dataloader wait、H2D、processor、cache hit、peak memory 等 profile 字段。
- tuner 输出 decision、reason、evidence、threshold、recommended config updates。
- 针对 dataloader wait、padding、H2D、memory pressure 给出配置建议。

### P3：长训与稳定性

目标：从 smoke/短步 benchmark 推进到生产可用稳定窗口。

已验证：

- bf16 / AMP。
- activation checkpointing。
- DeepSpeed ZeRO-2 / ZeRO-3。
- checkpoint/resume stress。
- VLM LoRA 500-1000 step 稳定窗口。
- YOLO m/l/x 稳定性窗口。
- dataloader workers 0/2/4/8 对比。

## 5. 代表性 benchmark 结果

### 5.1 DataComp WDS CLIP-B style b8

测试方法：

- 数据：DataComp WDS 真实图文 shard。
- 模型：CLIP-B style contrastive。
- 后端：native-DDP + bf16 hook、FSDP、DeepSpeed。
- 主指标：end-to-end image-text pairs/s。
- 显存指标：CUDA peak allocated memory。

结果：

| 后端 | 图文对/s | Tokens/s | Patch tokens/s | Peak GB | Dataloader wait ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| native-DDP bf16 hook | 132.264 | 34835.019 | 25923.735 | 3.475 | 2.954 |
| FSDP | 84.166 | 22167.148 | 16496.482 | 2.675 | 2.690 |
| DeepSpeed | 91.039 | 23977.455 | 17843.687 | 3.754 | 3.004 |

结论：

- native-DDP / FSDP：1.5715x。
- native-DDP / DeepSpeed：1.4528x。

### 5.1.1 YOLO-World Vision Adapter Refactor Smoke

测试方法：

- 数据：远程 Object365 tiny YOLO cache。
- 模型：`yolov8s-worldv2.pt`。
- 目标：验证通用 `VisionPreprocessor` / `VisionBatchCollator` 与 YOLO official-loss adapter 的真实训练链路。
- 容器：`parascale-yolo:cu121-torch24-ultralytics83161`，`--shm-size=8g`。

结果：

| 指标 | 数值 |
| --- | ---: |
| measured steps | 4 |
| stable end-to-end images/s | 106.903 |
| stable dataloader wait ms | 0.087 |
| peak memory bytes | 711078912 |
| stable pipeline cache hit | 1.0 |

结论：

- YOLO 数据缓存与 profile 已从 workload patch 下沉为通用 vision 数据层能力。
- YOLO 特化逻辑保留在 adapter 中，主 workload 只负责模型加载、样本枚举和 DataLoader 组装。

### 5.2 DataComp P2 Medium CLIP-B/ViT-B style

测试方法：

- 数据：DataComp WDS streaming IterableDataset，支持 rank/worker shard。
- 模型：image size 224、patch size 16、hidden dim 768、vision layers 12、text layers 6、heads 12。
- 后端：native、FSDP、DeepSpeed ZeRO-2。
- 容器：`unified-torch-distributed:cu121-torch24`。

结果：

| 后端 | 图文对/s | Tokens/s | Patch tokens/s | Peak memory bytes | Dataloader wait ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 61.108 | 15826.848 | 11977.074 | 3373275136 | 0.864 |
| FSDP | 16.426 | 4221.539 | 3219.539 | 3710194688 | 2.392 |
| DeepSpeed | 12.046 | 3095.868 | 2361.051 | 4364514304 | 13.644 |

结论：

- native / FSDP：3.7201x。
- native / DeepSpeed：5.0728x。
- native peak memory 低于 FSDP 和 DeepSpeed。

### 5.3 DataComp P2+ Medium 1000-step 稳定窗口

测试方法：

- 训练步数：1000 steps。
- warmup：50 steps。
- 稳定统计窗口：950 steps。
- 主指标：stable end-to-end image-text pairs/s。

结果：

| 后端 | measured steps | 稳定图文对/s | 稳定 tokens/s | 稳定 patch tokens/s | Dataloader wait ms | Peak memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 950 | 68.869 | 18106.686 | 13498.419 | 1.189 | 3373275136 |
| FSDP | 950 | 16.451 | 4326.537 | 3224.466 | 1.410 | 3711180800 |
| DeepSpeed | 950 | 12.761 | 3356.033 | 2501.179 | 1.376 | 4364515328 |

结论：

- native / FSDP：4.1862x。
- native / DeepSpeed：5.3968x。
- DataComp WDS dataloader wait 已降到 1-1.4 ms，数据侧不是主要瓶颈。

### 5.4 YOLO-World Object365 official detection loss

测试方法：

- 数据：Object365 Tiny cached images + YOLO labels。
- 模型：`yolov8s-worldv2`，official detection loss。
- 后端：native、native-DDP、FSDP、DeepSpeed。
- 主指标：stable end-to-end images/s。

结果：

| 后端 | images/s | Peak GB | Dataloader wait ms |
| --- | ---: | ---: | ---: |
| native | 67.763 | 0.666 | 13.100 |
| native-DDP | 90.936 | 0.712 | 13.976 |
| FSDP | 82.817 | 0.791 | 13.621 |
| DeepSpeed | 84.465 | 2.602 | 13.352 |

结论：

- native-DDP / FSDP：1.0980x。
- native-DDP / DeepSpeed：1.0766x。
- native-DDP 显存低于 FSDP，显著低于 DeepSpeed。

### 5.5 Native-DDP 与 WDS sharded dataloader 综合测试

YOLO-World official loss 80-step 稳定窗口：

| 后端 | GPU | Stable images/s | Compute images/s | Dataloader wait ms | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 1 | 68.294 | 122.232 | 12.946 | 715 MB |
| native-DDP | 2 | 91.445 | 117.249 | 9.618 | 768 MB |
| FSDP | 2 | 82.197 | 113.644 | 13.427 | 850 MB |
| DeepSpeed | 2 | 81.649 | 113.494 | 13.732 | 2794 MB |

DataComp WDS CLIP medium 80-step 稳定窗口：

| 后端 | GPU | Stable pairs/s | Compute pairs/s | Dataloader wait ms | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 1 | 73.308 | 76.629 | 1.187 | 3377 MB |
| native-DDP | 2 | 20.445 | 20.569 | 1.172 | 3371 MB |
| FSDP | 2 | 22.657 | 22.839 | 1.411 | 2870 MB |
| DeepSpeed | 2 | 23.584 | 23.751 | 1.195 | 3696 MB |

结论：

- YOLO 场景 native-DDP 优于 FSDP/DeepSpeed。
- CLIP medium 小 batch 场景 native-DDP 慢于 FSDP/DeepSpeed，瓶颈从数据管线转为 DDP 全量梯度同步。

### 5.6 CUDA prefetch 大 batch 复测

测试方法：

- 场景：真实 VLM LoRA + DataComp WDS + DeepSpeed ZeRO-2。
- 模型：LLaVA-OneVision Qwen2 0.5B。
- dataloader：workers=8, persistent_workers=true, prefetch_factor=2, pin_memory=true。
- 对照：`cuda_prefetch` off/on。

结果：

| Batch | Prefetch | images/s | Wait ms | H2D ms | Prefetch wait ms | Cache hit | Peak GB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | off | 20.036 | 0.335 | 0.227 | 0.000 | 1.000 | 4.234 |
| 1 | on | 20.016 | 0.217 | 0.000 | 0.014 | 1.000 | 4.237 |
| 2 | off | 19.590 | 0.537 | 0.434 | 0.000 | 1.000 | 5.335 |
| 2 | on | 19.617 | 0.236 | 0.000 | 0.015 | 1.000 | 5.342 |
| 4 | off | 20.502 | 0.913 | 0.811 | 0.000 | 1.000 | 8.930 |
| 4 | on | 20.492 | 0.277 | 0.000 | 0.015 | 1.000 | 8.944 |

结论：

- Batch 4 wait 从 0.913 ms 降到 0.277 ms，下降约 69.6%。
- H2D 等待被 CUDA stream prefetch 显著隐藏。
- 吞吐未明显提升，说明该短窗口主要瓶颈仍在模型计算/后端路径。

### 5.7 VLM LoRA 长窗口稳定性

测试方法：

- 模型：LLaVA-OneVision Qwen2 0.5B LoRA。
- 后端：DeepSpeed ZeRO-2、ZeRO-3、FSDP。
- 验证：train + resume、500-1000 step 或 300-step 窗口。

DeepSpeed ZeRO-2 1000-step：

| Workers | Phase | Step | Throughput | Peak GB | Wait ms | Jitter | Failure |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | train | 500 | 18.890 | 5.492 | 21.146 | 0.820 | 0.000 |
| 0 | resume | 1000 | 18.871 | 5.492 | 21.334 | 0.658 | 0.000 |
| 8 | train | 500 | 18.102 | 5.570 | 17.344 | 0.401 | 0.000 |
| 8 | resume | 1000 | 17.980 | 5.569 | 17.849 | 0.127 | 0.000 |

DeepSpeed ZeRO-3 300-step：

| Workers | Phase | Step | Throughput | Peak GB | Wait ms | Jitter | Failure |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | train | 150 | 8.726 | 6.436 | 20.597 | 0.427 | 0.000 |
| 0 | resume | 300 | 8.638 | 6.431 | 20.796 | 0.413 | 0.000 |
| 8 | train | 150 | 7.482 | 6.511 | 17.569 | 0.162 | 0.000 |
| 8 | resume | 300 | 7.467 | 6.506 | 17.597 | 0.175 | 0.000 |

FSDP 300-step：

| Workers | Phase | Step | Throughput | Peak GB | Wait ms | Jitter | Failure |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | train | 150 | 4.271 | 9.382 | 90.075 | 0.106 | 0.000 |
| 0 | resume | 300 | 4.319 | 9.386 | 89.084 | 0.064 | 0.000 |
| 8 | train | 150 | 4.252 | 9.405 | 75.498 | 0.044 | 0.000 |
| 8 | resume | 300 | 4.259 | 9.405 | 76.194 | 0.042 | 0.000 |

结论：

- 当前 VLM LoRA 推荐 DeepSpeed ZeRO-2。
- ZeRO-3 可作为大模型/OOM fallback。
- 当前 FSDP 吞吐最低、显存最高，不建议默认。

### 5.8 YOLO-L workers=8 1000-step

测试方法：

- 模型：YOLO-World L。
- 后端：native-DDP。
- workers：8。
- 验证：train 500 step + resume 到 1000 step。

结果：

| Phase | Step | Throughput | Peak GB | Wait ms | Jitter | Failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 500 | 47.506 | 1.915 | 13.046 | 0.055 | 0.000 |
| resume | 1000 | 46.698 | 1.913 | 12.977 | 0.089 | 0.000 |

结论：

- workers=8 的高 jitter 未在 1000-step 长窗口复现。
- YOLO-L native-DDP 训练与 resume 稳定。

## 6. 当前后端推荐

| 场景 | 推荐后端 | 依据 |
| --- | --- | --- |
| YOLO-World / detection 小中模型 | native-DDP | 吞吐优于 FSDP/DeepSpeed，显存低，长窗口稳定 |
| DataComp CLIP 中等模型单卡可承载 | native 或 native-DDP + batch/comm 优化 | native 路径吞吐好；native-DDP 小 batch 会受 all-reduce 影响 |
| VLM LoRA 小中模型 | DeepSpeed ZeRO-2 | 长窗口吞吐和显存综合最优，resume 已验证 |
| VLM LoRA 大模型或 OOM | DeepSpeed ZeRO-3 | 吞吐较低，但显存扩展潜力更强 |
| FSDP | fallback / 对照 baseline | 当前 VLM LoRA 性能不占优，optimizer state 恢复仍需硬化 |

## 7. 已知边界与风险

- 多数 benchmark 关注工程吞吐、显存和稳定性，不代表模型收敛质量。
- DataComp CLIP 部分结果使用随机初始化 CLIP-B/ViT-B style，不等同真实预训练 CLIP/SigLIP 收敛评估。
- VLM LoRA 已接真实小规格权重，但尚未覆盖更大 VLM、QLoRA、长周期收敛质量评估。
- Ascend 路线保留架构抽象，尚未实机验证。
- native-DDP 优势依赖模型规模、batch、通信比例和数据管线；不能泛化为所有分布式训练场景。

## 8. 后续测试规范

1. 新 benchmark 优先通过 `python -m parascale.cli benchmark-matrix ...` 执行。
2. 新 benchmark 配置放入 `tests/benchmarks/configs/`。
3. 新 benchmark 输出报告放入 `tests/benchmarks/reports/`。
4. 一次性复现实验脚本放入 `tests/benchmarks/scripts/`，成熟后合并进 CLI。
5. 产品设计文档 `docs/` 不再堆叠阶段性测试记录。
6. 数据集、模型权重、wheelhouse 依赖缓存不得作为中间产物清理。
