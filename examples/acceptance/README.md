# 项目验收案例

本目录是[冻结验收清单V1](../../docs/production_acceptance.md)的执行入口。第一阶段接通产品入口检查、真实Llama并行训练/新进程恢复、Transformer和diffusion生成；尚未接通的质量、性能对照及长跑明确返回`BLOCKED`。**当前不是一键通过全部项目验收的套件。**

## 八组案例

| 分组 | 当前可运行内容 | 正式验收缺口 |
| --- | --- | --- |
| [installation](installation/README.md) | 两工具版本和帮助检查 | 新环境、离线介质和完整安装闭环 |
| [parallel_training](parallel_training/README.md) | 单卡、DP/TP/PP、四卡二维组合；保存后新进程恢复 | 原生对照、独立质量、严格恢复等价；八卡3D资源 |
| [llama7b_finetuning](llama7b_finetuning/README.md) | 明确阻断原因 | 指定7B及优化器状态显存降低至少40%的完整协议 |
| [quantization](quantization/README.md) | 链接已有PTQ/QAT用例，未自动执行 | 校准、独立质量、真实INT8内核联合门禁 |
| [distillation](distillation/README.md) | 链接已有蒸馏用例，未自动执行 | 同初始化同预算CE对照及独立质量收益 |
| [pruning](pruning/README.md) | 链接已有剪枝用例，未自动执行 | 原始/剪枝/恢复训练模型质量与速度对照 |
| [generative_inference](generative_inference/README.md) | 真实Transformer/diffusion离线生成 | 三组配对原生对照和独立质量 |
| [stability](stability/README.md) | 明确阻断原因 | 最终候选24小时和故障注入 |

## 准备环境

使用包含本目录的源码包和匹配版本的已安装ParaScale。`examples`不随wheel提供；仅安装wheel不足以运行本套件。激活对应Python环境，使`parascale-train`、`parascale-infer`在PATH上。GPU执行以Linux/CUDA为目标，本轮不声明Ascend验收。

额外安装编排依赖：`python -m pip install -r examples/acceptance/requirements.txt`。psutil用于跟踪并清理独立会话中的torchrun子进程；不能仅杀死启动器进程组。

编辑`environment.example.json`的副本，填入合法本地资产路径。相对资产路径相对于该JSON文件解析，不相对于工作目录。空路径不触发下载，而是`BLOCKED`。Llama模型须有完整HF权重和tokenizer；`token_corpus`须由现有`prepare-text`生成，且至少容纳两段训练预算；提示词为产品要求的JSONL。

GPU ID由操作者按实际拓扑填写，最多四卡。不要与其他训练任务重叠运行。并行案例在计划和执行阶段均读取模型配置，提前阻断不兼容的TP/PP布局；详见[模型与布局要求](parallel_training/README.md)。默认预算是短程功能验证，不是推荐的正式质量训练预算；diffusion默认两步、256像素尤其不能作为图像质量达标证据。

`gpu_ids`使用运行环境内的CUDA逻辑编号。Docker只暴露部分宿主机GPU后，逻辑编号可能重新从0排列；例如宿主机0、1、3、4映射成容器内0、1、2、3时，应填写后者。运行前核对容器内可见卡数，不要将宿主机物理编号直接重复用于筛选。

训练前根据浮点safetensors头部计算参数元素上界，预算FP32参数和AdamW两份状态、DP副本，以及TP中未分片的embedding/head/norm等参数副本。再计入两份保留检查点、一份临时载荷和1GiB余量。按套件顺序累计保留输出，每条命令前复查剩余空间。无法预估的权重格式或空间不足返回BLOCKED，不自动删除旧结果。该数值是保守容量规划，不是实测显存/磁盘指标；其他进程仍可能占用磁盘，必须避免并发争用。

## 执行

从仓库根目录运行，输出必须是仓库和所有输入资产之外的**新目录**：

```bash
python examples/acceptance/run.py --list
python examples/acceptance/run.py --config /configs/acceptance.json \
  --output /runs/acceptance-plan --group parallel_training --plan

python examples/acceptance/run.py --config /configs/acceptance.json \
  --output /runs/acceptance-entries --case installation.entries --mode functional

python examples/acceptance/run.py --config /configs/acceptance.json \
  --output /runs/acceptance-dp2 --case parallel_training.dp2 --mode functional

python examples/acceptance/run.py --config /configs/acceptance.json \
  --output /runs/acceptance-generation --group generative_inference --mode functional

python examples/acceptance/run.py --config /configs/acceptance.json \
  --output /runs/acceptance-all --mode acceptance
```

`--case`或`--group`可重复，但不能同时使用。省略两者执行全目录清单；`--plan`不启动进程、不创建输出目录，退出0只表示生成了计划。入口也支持使用绝对脚本路径从其他目录启动。

## 状态与证据

- `functional`：仅断言所选功能路径成立。`PASS`不表示质量提高、比PyTorch更快或项目验收完成。
- `acceptance`（默认）：功能成功后仍检查已知验收缺口；第一阶段这些缺口未闭合，因此返回`BLOCKED`，不会静默降级到tiny/synthetic。
- `FAIL`：启动失败、超时、错误退出、非有限数值、缺失产物或恢复步数错误。`BLOCKED`：缺少资产/入口/卡数，或正式门禁未接通。
- 退出码：0为所选范围通过，1为执行/核验失败，2为阻断或配置错误。完整项目标志`full_acceptance_passed`当前始终为false。

输出包含冻结的`protocol.json`（预算、命令、资产SHA256、编排源码身份）、逐命令日志、产品原始JSON和模型产物、`summary.json`、`report.md`。文件哈希在执行前计算，大模型首次准备可能耗时；失败证据保留，不自动删除或覆盖。超时或中断时清理启动器及跟踪到的子进程；清理失败会停止后续案例。没有自动重试以隐藏失败。

并行吞吐沿用产品的`timing_scope`，包含数据和检查点，不含模型初始化；推理指标保留产品的计时定义。当前没有原生基线、配对重复或质量统计，因此`performance_claim_eligible`始终为false。正式阈值必须在后续对照适配器运行前冻结，不能根据结果回填。
