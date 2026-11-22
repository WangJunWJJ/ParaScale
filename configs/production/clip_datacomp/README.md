# CLIP / DataComp 有界候选

本目录仅为历史候选环境上的待验收配置，非生产认证，不代表可直接上线。
2026-11-19，remote156 两次连接超时；2026-11-20 已恢复连接并开始真实 CLIP/COCO WebDataset 有界恢复验证。
这不是 DataComp 数据集验收，最新结果以[统一报告](../../../tests/benchmarks/reports/BENCHMARK_REPORT.md)为准。

2026-11-20 修正：必须显式设置 `training.workload: clip_contrastive`。
此前仅设置 `task.workload` 会落入 synthetic_regression，不能作为 CLIP 验收证据；
本次增加实际 workload、参数量和非合成数据断言，防止类似假通过。

## 范围与语义

| 配置 | 后端 | 单节点卡数 | 每 rank batch / GAS | global batch | 输出目录 |
| --- | --- | --- | --- | --- | --- |
| `native_1gpu.json` | native | 1 | 8 / 1 | 8 | `runs/production_candidate/1gpu` |
| `native_ddp_2gpu.json` | native_ddp | 2 | 8 / 1 | 16 | `runs/production_candidate/2gpu` |
| `native_ddp_4gpu.json` | native_ddp | 4 | 8 / 1 | 32 | `runs/production_candidate/4gpu` |

均使用 schema_version=1、FP32、AdamW lr=1e-6、1000 个优化器步、每 100 步保存检查点并保留最终检查点，数据 worker=0。
world_size=1 的 native_ddp 当前被拒绝，因此单卡使用 native。无量化、KD 或通信压缩。
当前 CLIP 目标使用同一个数据并行组内的可微特征汇聚，每卡计算本地 query 对全组 key 的损失，即 global-negative CLIP。
SigLIP 使用独立的逐对 sigmoid 损失，不再复用 CLIP 的 softmax 目标。
固定每卡 batch 仍是**弱扩展**，global batch 和每个 query 的负样本数随卡数变化，不能据此声称质量或收敛等价。
同口径强扩展需固定每个微批的全局图文对及其顺序，并随卡数调整每卡 batch。
GAS 只累积梯度，不跨微批汇聚负样本，因此相同 optimizer global batch 不代表相同对比学习目标。

历史报告使用 **rank-local negatives**，不能与当前目标直接计算优化加速比，旧长跑与恢复结果也不构成新目标的验收。
检查点记录目标版本与负样本语义；不要绕过目标身份检查继续旧目标的训练。
可微汇聚允许不同 rank 上非空尾批大小不同；运行时按样本数加权，空特征批会报错。
当前普通训练入口仅绑定纯数据并行组；不能将本功能解释为 CLIP 已支持 TP/PP。

## 离线环境前置检查

历史已核验 A6000 环境：torch `2.5.0a0+872d972e41.nv24.08`、Transformers `4.51.3`，
镜像标签 `parascale-detection:a6000-ultralytics83161`，本地 image ID
`sha256:043573ecd623390f83f2a7dadac4d1ae55c178fb07e3b6122dd8606e6b323493`。
这是历史身份记录，**非公开可部署 digest**；不要把 image ID 当作 registry digest 拉取。
其他依赖版本未知，不作推断。验收前必须重新检查实际镜像，并在实际运行容器内保存完整 `pip freeze`、驱动/GPU 信息及版本输出。

宿主机先执行，核对 ID 后再使用已有本地镜像启动容器，禁止隐式拉取：

```bash
docker image inspect parascale-detection:a6000-ultralytics83161 --format '{{.Id}} {{json .RepoDigests}}'
```

以下命令在已准备好的 Linux GPU 容器仓库根目录执行。提前只读挂载完整本地 CLIP 权重、配置、tokenizer/processor 到 `/models/clip`，将真实 DataComp WDS tar 分片挂载到 `/datasets/datacomp`；确保 `runs` 可写且磁盘充足。
不安装依赖、不下载模型或数据；缺失文件应停止排查，不能改为在线标识或取消离线开关。

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
test -d /models/clip
test -d /datasets/datacomp
mkdir -p runs/production_candidate/evidence
python -m pip freeze > runs/production_candidate/evidence/pip-freeze.txt
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__); assert torch.__version__ == "2.5.0a0+872d972e41.nv24.08"; assert transformers.__version__ == "4.51.3"'
nvidia-smi > runs/production_candidate/evidence/nvidia-smi.txt
git rev-parse HEAD > runs/production_candidate/evidence/revision.txt
git diff > runs/production_candidate/evidence/worktree.patch
find /models/clip /datasets/datacomp -type f -print0 | sort -z | xargs -0 -r sha256sum > runs/production_candidate/evidence/inputs.sha256
sha256sum configs/production/clip_datacomp/*.json > runs/production_candidate/evidence/configs.sha256

CUDA_VISIBLE_DEVICES=0 python -m parascale.cli train --config configs/production/clip_datacomp/native_1gpu.json --output runs/production_candidate/1gpu/result.json
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m parascale.cli train --config configs/production/clip_datacomp/native_ddp_2gpu.json --output runs/production_candidate/2gpu/result.json
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 -m parascale.cli train --config configs/production/clip_datacomp/native_ddp_4gpu.json --output runs/production_candidate/4gpu/result.json
```

只运行符合实际卡数的命令；每次独立实验先归档旧输出，避免覆盖。检查退出码并保存完整日志，任何 I/O 错误均视为失败，不增加忽略错误或跳过最终检查点的选项。
现有 DataComp 读取器可能跳过部分坏样本并循环读取分片，配置本身不能提供严格完整性保证；必须预检图片/文本配对、解码、有效样本数及 rank 分片覆盖，不能把步数当作唯一真实样本数。

## 待验收证据

- 真实模型、数据和配置的 SHA256 清单及来源，实际 image inspect 输出、完整依赖冻结、运行代码版本和本地差异；以上命令产物需归档，历史版本记录不能替代当前检查。
- 1/2/4 卡分别记录退出码、实际 world size、有限 loss、吞吐和峰值显存；不能将无 Torch 配置测试或 dry-run 当作 GPU 成功证据。
- 周期与最终检查点的 manifest、文件哈希、模型/优化器/RNG/数据进度保存和恢复证据。归档完整运行后，在独立验收副本中使用相同配置及 `--resume-step 100` 复跑；DDP 保持相同 torchrun 卡数，与不中断轨迹比较并实际验证恢复，不能只检查文件存在。
- 独立验证集质量阈值、真实数据质量检查和 **24h** 稳定性 soak 均待完成。1000 步只是有界候选，不等于 24h；长跑需另行审批配置和记录故障/恢复、内存及 I/O 趋势，不能据此宣称生产就绪。

## 故障恢复验收

先归档原始检查点，在独立实验副本中注入故障，不要破坏唯一可恢复的训练状态。
正常中断可使用原来的启动命令加 `--resume-latest`，由 rank 0 选择最近的完整 native 检查点并广播。
若最新步骤的文件不完整或哈希失配，结果 `checkpoint_recovery.skipped` 应包含该步骤和原因，
`selected_step` 应指向较早完整步骤；没有完整步骤必须失败。指定 `--resume-step` 时不得自动回退。
`max_steps` 表示本次调用再执行的步数，恢复后并非以 1000 为绝对停止步数。

必须用相同数据、模型、卡数和优化器配置复跑；当前文件完整性选择不会替你验证数据身份。
在两卡和四卡分别测试写入失败、损坏最新文件、任一 rank 退出，并记录退出时限及恢复轨迹。
本次补充的本地 CPU 和 Gloo 回归不能替代 NCCL、GPU 或真实 CLIP 质量验收。
原子写只保证单文件替换，不保证完整检查点目录事务；多个任务不得共用写入目录。
更多边界见[用户手册](../../../docs/user_manual.md#111-最新完整检查点恢复)。

## 当前本地验证记录

### 全局目标数值与运行时开销对照

先用本地真实 CLIP 权重和 WDS 图文运行数值门禁，再测试吞吐。以下命令在仓库根目录执行：

```bash
torchrun --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=4 \
  --module tests.benchmarks.scripts.validate_contrastive_training \
  --model /models/clip --data-dir /datasets/datacomp --global-batch 32 \
  --output /runs/clip-parity-w4.json

torchrun --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=4 \
  --module tests.benchmarks.scripts.benchmark_contrastive_training \
  --implementation parascale --model /models/clip --data-dir /datasets/datacomp \
  --global-batch 32 --output /runs/clip-parascale-w4.json
```

将 `--implementation` 改为 `pytorch` 并使用新输出路径可运行独立 DDP 对照；
卡数分别设为 1、2、4，保持同一份数据和模型。脚本允许全局 batch=32 或 128，固定 FP32、
eager attention、SGD、无通信压缩，使用相同样本及权重哈希。
数值门禁比较原生 HF 全局损失、全部参数梯度及一次更新，而不是仅检查 loss 有限。
吞吐脚本预热 10 步，再测 3 个各 50 步的窗口，取最慢 rank 的窗口耗时计算全局吞吐。
输入是预处理后常驻 GPU 的同一组图文（数量为选定的全局 batch），计时不含解码、tokenization、H2D、加载及窗口边界有限性扫描；
它用于隔离运行时开销，不代表完整数据管线吞吐、训练收敛或独立验证集质量。
两种实现串行测试，不与其他 GPU 作业重叠；保留所有失败输出，不覆盖结果或调整通过阈值。
完整的单卡/四卡同口径操作见 [Example 014](../../../examples/gpu/example_014_clip_strong_scaling/README.md)。
增加 `--profile-dir /runs/traces` 可采集预热后的三个更新步；带 profiler 的报告显式标记为诊断，不能用于吞吐结论。

### 已有图文数据复测

可复用已校验的 JPG/TXT WebDataset，不要求必须是 DataComp：

```bash
python tests/benchmarks/scripts/validate_clip_recovery.py \
  --wds /datasets/image_text/train --model /models/clip \
  --scratch /runs/recovery-state --output /runs/recovery-evidence \
  --find-unused-parameters --keep-checkpoints
```

该有界脚本使用物理 GPU `0`、`0,1`、`0,1,3,4`，仅在对应设备可用时执行。
输出和状态目录必须为空；预检每张图片解码、图文配对及重复键，并记录分片/模型哈希。
它执行 40 步训练、损坏最新检查点、回退到第 20 步再续训 20 步，不是长期吞吐或质量评估。
2026-11-20 从 145 同步至 156 的 Flickr8k 标记数据含 6,000 对图文、4 个分片，传输哈希一致。
其原始下载来源与训练/验证划分尚未恢复，不能用于宣称独立验证集质量。
另有 E 盘下载的 DataComp 333 对可用子集，来源和变换记录位于下载清单；未达到计划的 512 对。
具体结果及边界见[统一报告](../../../tests/benchmarks/reports/BENCHMARK_REPORT.md)。

### 四卡精确恢复对照（2026-11-20）

在 156 的预训练 CLIP ViT-B/32、COCO 512 对图文、FP32、40 步有界测试中，
默认 DDP 两次连续训练完全一致，但恢复训练的模型最大绝对误差为 `9.29e-7`。
将 `parascale.ddp_find_unused_parameters` 设置为 `true` 后，连续与恢复训练的模型及优化器状态完全一致。
这支持分桶/归约路径相关的解释，但尚未记录 reducer 内部跟踪，不把推断当作已证明的唯一原因。

需要优先验收位级恢复时，可在初始训练和恢复配置中同时开启该选项，再重新验证。
不能只在旧检查点恢复时临时切换，不能推广到所有模型/版本。
该选项有额外计算图遍历开销；这次短测平均训练步耗时增加约 8.4%，因此未改为全局默认。
原始对照结果见[统一报告](../../../tests/benchmarks/reports/BENCHMARK_REPORT.md#four-gpu-recovery-investigation)。

### CPU 回归

2026-11-19，Windows、Python 3.12.8、PyTorch 2.15.0+cpu 环境，以下聚焦回归 **42 通过**：

```bash
python -m pytest -q tests/test_checkpoint_durability_no_torch.py tests/test_checkpoint_recovery.py tests/test_checkpoint_runtime_state.py
```

覆盖单文件序列化/fsync/替换失败、manifest 结构与哈希校验、显式步骤失败、latest 回退、
真实 CPU 训练恢复、双进程 Gloo 恢复，以及单个 rank 加载失败时的错误汇聚。
这是合成训练与故障注入证据，不包含真实 CLIP/DataComp 质量或 GPU 吞吐结果。
CI 新增 Linux/Python 3.11/PyTorch 2.5.1 CPU 的独立门禁，远程 CI 是否通过以 PR 检查结果为准。
