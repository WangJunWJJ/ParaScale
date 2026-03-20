# ParaScale

<p align="center">
  <strong>基于 PyTorch 的高性能分布式训练框架</strong>
</p>

<p align="center">
  <a href="#特性">特性</a> •
  <a href="#安装">安装</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#文档">文档</a> •
  <a href="#示例">示例</a> •
  <a href="#api参考">API</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python 3.7+">
  <img src="https://img.shields.io/badge/PyTorch-1.9+-orange.svg" alt="PyTorch 1.9+">
  <img src="https://img.shields.io/badge/CUDA-10.2+-green.svg" alt="CUDA 10.2+">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</p>

---

## 📋 目录

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [文档](#文档)
- [示例](#示例)
- [API 参考](#api参考)
- [架构设计](#架构设计)
- [测试](#测试)
- [版本历史](#版本历史)
- [许可证](#许可证)

## ✨ 特性

### 🚀 智能自动并行引擎（ParaEngine）
- 根据模型规模、硬件状态自适应决策并行优化策略组合
- 自动分析模型参数量、层结构、内存需求
- 实时监测 GPU 内存、计算能力、通信带宽
- 智能选择最优并行配置（数据并行、张量并行、流水线并行）
- 支持小模型（<1B）、中等模型（1B-10B）、大模型（>10B）的自动策略选择

### 🔄 多维度并行支持
| 并行策略 | 描述 | 适用场景 |
|---------|------|---------|
| **数据并行 (DP)** | 每个 GPU 持有完整模型副本，处理不同数据子集 | 模型小，数据量大 |
| **模型并行 (MP)** | 将模型不同层分配到不同设备 | 模型大，单卡放不下 |
| **张量并行 (TP)** | 将权重矩阵分割到不同 GPU，支持行/列并行 | 线性层多，矩阵运算密集 |
| **流水线并行 (PP)** | 将模型按层分割，支持微批次处理 | 层数多，需要高吞吐 |
| **3D 混合并行** | DP + TP + PP 的灵活组合 | 超大规模模型训练 |

### 🎯 量化训练
- **量化感知训练 (QAT)**：支持 INT8/INT4 量化，对称/非对称量化
- **训练后量化 (PTQ)**：无需重新训练，快速部署
- 伪量化层模拟量化误差，逐通道量化支持

### 🌐 多节点分布式训练
- 自动检测 torchrun、SLURM、OpenMPI、MPICH 环境
- 自动初始化分布式环境
- 支持多节点多 GPU 训练

### ⚡ ZeRO 优化器
- 支持 ZeRO Stage 0/1/2/3
- 支持 CPU Offload
- 减少内存使用，支持大规模模型训练

### 🔧 4-bit 优化器
- **FourBitAdamW**：4-bit AdamW 优化器，节省 75% 内存
- **FourBitSGD**：4-bit SGD with Momentum
- 可配置的 group_size 和误差补偿

## 📦 安装

### 环境要求

- Python >= 3.7
- PyTorch >= 1.9.0
- CUDA >= 10.2（用于 GPU 训练）

### 从源码安装

```bash
git clone https://github.com/yourusername/ParaScale.git
cd ParaScale
pip install -e .
```

### 依赖安装

```bash
pip install torch torchvision numpy
```

## 🚀 快速开始

### 1. 自动并行模式（推荐）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from parascale import ParaEngine

# 定义模型
model = MyModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 创建 ParaEngine（自动选择最优并行策略）
engine = ParaEngine(
    model=model,
    optimizer=optimizer,
    auto_parallel=True
)

# 查看自动选择的策略
strategy = engine.get_strategy()
print(f"策略类型：{strategy.strategy_type}")
print(f"并行配置：DP={strategy.dp_size}, TP={strategy.tp_size}, PP={strategy.pp_size}")

# 开始训练
engine.train(dataloader, epochs=10)
```

### 2. 手动配置并行策略

```python
from parascale import Engine, ParaScaleConfig

# 配置并行策略
config = ParaScaleConfig(
    data_parallel_size=2,
    tensor_parallel_size=2,
    batch_size=32
)

# 创建引擎
engine = Engine(model, optimizer, config)
engine.train(trainloader, epochs=10)
```

### 3. 命令行启动训练

```bash
# 单节点多 GPU（自动并行）
torchrun --nproc_per_node=8 examples/para_engine_example.py

# 数据并行（2 GPU）
torchrun --nproc_per_node=2 examples/basic_parallel_examples.py --example 1

# 多节点训练
# Node 0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
    --master_addr=<node0_ip> --master_port=29500 \
    examples/multi_node_example.py

# Node 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
    --master_addr=<node0_ip> --master_port=29500 \
    examples/multi_node_example.py
```

## 📚 文档

- [API 文档](docu/api_documentation.md)
- [ParaEngine 指南](docu/para_engine_guide.md)
- [软件设计文档](docu/software_design_documentation.md)
- [软件需求规格](docu/software_requirements_specification.md)
- [用户手册](docu/user_manual.md)

## 💡 示例

| 示例 | 描述 | 命令 |
|-----|------|------|
| [基础并行示例](examples/basic_parallel_examples.py) | 数据/模型/张量/流水线并行 | `python examples/basic_parallel_examples.py --example 1` |
| [ParaEngine 示例](examples/para_engine_example.py) | 自动并行策略选择 | `python examples/para_engine_example.py` |
| [量化训练示例](examples/quantization_examples.py) | QAT/PTQ 量化训练 | `python examples/quantization_examples.py --example 1` |
| [4-bit 优化器示例](examples/fourbit_optimizer_example.py) | 内存优化训练 | `python examples/fourbit_optimizer_example.py` |
| [多节点示例](examples/multi_node_example.py) | 分布式训练 | `torchrun --nproc_per_node=4 examples/multi_node_example.py` |

## 🔌 API 参考

### 核心类

```python
# 引擎
from parascale import ParaEngine      # 自动并行引擎
from parascale import Engine          # 手动策略引擎

# 配置
from parascale import ParaScaleConfig     # 主配置
from parascale import QuantizationConfig  # 量化配置

# 并行策略
from parascale import (
    DataParallel,      # 数据并行
    ModelParallel,     # 模型并行
    TensorParallel,    # 张量并行
    PipelineParallel,  # 流水线并行
    HybridParallel,    # 3D 混合并行
)

# 优化器
from parascale import ZeroOptimizer   # ZeRO 优化器
from parascale import AdamW           # AdamW 优化器
from parascale import FourBitAdamW    # 4-bit AdamW
from parascale import FourBitSGD      # 4-bit SGD

# 量化
from parascale import (
    QuantizationAwareTraining,    # QAT
    PostTrainingQuantization,     # PTQ
    ptq_quantize,                 # PTQ 便捷函数
)

# 分布式工具
from parascale import (
    initialize_distributed,   # 初始化分布式环境
    get_rank,                 # 获取 rank
    get_world_size,           # 获取 world size
    is_main_process,          # 判断是否主进程
)
```

### 配置选项

#### ParaScaleConfig

| 配置项 | 类型 | 默认值 | 描述 |
|-------|------|-------|------|
| `data_parallel_size` | int | 1 | 数据并行大小 |
| `tensor_parallel_size` | int | 1 | 张量并行大小 |
| `pipeline_parallel_size` | int | 1 | 流水线并行大小 |
| `tensor_parallel_mode` | str | "row" | 张量并行模式（row/column） |
| `zero_optimization` | bool | False | 是否启用 ZeRO |
| `zero_stage` | int | 0 | ZeRO 阶段（0/1/2/3） |
| `batch_size` | int | 32 | 批次大小 |
| `learning_rate` | float | 1e-3 | 学习率 |

#### QuantizationConfig

| 配置项 | 类型 | 默认值 | 描述 |
|-------|------|-------|------|
| `enabled` | bool | False | 是否启用量化 |
| `mode` | str | "qat" | 量化模式（qat/ptq） |
| `bits` | int | 8 | 量化位数（8/4） |
| `scheme` | str | "symmetric" | 量化方案（symmetric/asymmetric） |
| `qat_epochs` | int | 10 | QAT 训练轮数 |

## 🏗️ 架构设计

```
ParaScale/
├── parascale/
│   ├── engine/              # 引擎模块
│   │   ├── engine.py        # 手动策略引擎
│   │   └── para_engine.py   # 自动并行引擎
│   ├── parallel/            # 并行策略
│   │   ├── data_parallel.py
│   │   ├── tensor_parallel.py
│   │   ├── pipeline_parallel.py
│   │   └── hybrid_parallel.py
│   ├── optimizers/          # 优化器
│   │   ├── optimizers.py
│   │   └── fourbit_optimizer.py
│   ├── quantization/        # 量化训练
│   │   ├── qat.py
│   │   ├── ptq.py
│   │   └── fake_quantize.py
│   └── utils/               # 工具函数
├── examples/                # 示例程序
├── tests/                   # 测试文件
└── docu/                    # 文档
```

## 🧪 测试

```bash
# 运行所有测试
python3 -m pytest tests/ -v

# 特定测试
python3 tests/test_engine.py
python3 tests/test_quantization.py
python3 tests/test_fourbit_optimizer.py

# 多 GPU 测试
torchrun --nproc_per_node=2 tests/test_all_parallel.py
torchrun --nproc_per_node=4 tests/test_hybrid_parallel.py
```

## 📈 版本历史

### v0.1.0 (当前版本)
- 初始版本发布
- 支持多种并行策略：数据并行、模型并行、张量并行、流水线并行
- 支持3D混合并行（DP+TP+PP）
- 支持量化感知训练（QAT）和训练后量化（PTQ）
- 支持ParaEngine自动并行引擎
- 支持ZeRO优化器和4-bit优化器
- 支持多节点分布式训练

**历史版本参考**: 本版本为初始版本(v0.1.0)，后续版本的详细变更记录请参阅文档更新日志。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

<p align="center">
  Made with ❤️ for the PyTorch community
</p>
