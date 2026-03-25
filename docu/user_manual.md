# ParaScale 用户手册

本文档提供了 ParaScale 框架的详细使用指南。

## 目录

1. [安装指南](#1-安装指南)
2. [快速开始](#2-快速开始)
3. [配置系统](#3-配置系统)
4. [并行策略](#4-并行策略)
   - 4.1 [数据并行](#41-数据并行)
   - 4.2 [模型并行](#42-模型并行)
   - 4.3 [张量并行](#43-张量并行)
   - 4.4 [流水线并行](#44-流水线并行)
   - 4.5 [序列并行](#45-序列并行)
   - 4.6 [3D混合并行](#46-3d混合并行)
5. [优化器](#5-优化器)
6. [量化训练](#6-量化训练)
7. [多节点训练](#7-多节点训练)
8. [检查点管理](#8-检查点管理)
9. [故障排除](#9-故障排除)

---

## 1. 安装指南

### 1.1 环境要求

- Python >= 3.7
- PyTorch >= 1.9.0
- CUDA >= 10.2（用于 GPU 训练）

### 1.2 从源码安装

```bash
git clone <repository-url>
cd ParaScale
pip install -e .
```

### 1.3 依赖安装

```bash
pip install torch torchvision numpy
```

---

## 2. 快速开始

### 2.1 基本使用流程

```python
import torch
import torch.nn as nn
import torch.optim as optim
from ParaScale import Engine, ParaScaleConfig

# 1. 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 创建模型和优化器
model = SimpleModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 3. 配置 ParaScale
config = ParaScaleConfig(
    data_parallel_size=2,
    batch_size=32
)

# 4. 创建引擎
engine = Engine(model, optimizer, config)

# 5. 训练
engine.train(trainloader, epochs=10)

# 6. 评估
loss, accuracy = engine.evaluate(testloader)
```

### 2.2 单节点多 GPU 训练

```bash
# 使用2个GPU进行数据并行训练（带量化）
torchrun --nproc_per_node=2 examples/basic_parallel_examples.py --example 1

# 使用2个GPU进行模型并行训练（带量化）
torchrun --nproc_per_node=2 examples/basic_parallel_examples.py --example 2

# 使用4个GPU进行流水线并行训练（带量化）
torchrun --nproc_per_node=4 examples/basic_parallel_examples.py --example 3

# 使用2个GPU进行张量并行训练（带量化）
torchrun --nproc_per_node=2 examples/basic_parallel_examples.py --example 4

# 使用8个GPU进行3D混合并行训练
torchrun --nproc_per_node=8 examples/hybrid_parallel_example.py

# 使用4个GPU进行序列并行训练 (SP=2, TP=2)
torchrun --nproc_per_node=4 examples/sequence_parallel_example.py

# 使用8个GPU进行序列并行训练 (SP=4, TP=2)
torchrun --nproc_per_node=8 examples/sequence_parallel_example.py
```

---

## 3. 配置系统

### 3.1 ParaScaleConfig 配置

```python
from ParaScale import ParaScaleConfig

config = ParaScaleConfig(
    # 并行策略配置
    data_parallel_size=2,           # 数据并行大小
    model_parallel_size=1,          # 模型并行大小
    tensor_parallel_size=1,         # 张量并行大小
    tensor_parallel_mode="row",     # 张量并行模式
    pipeline_parallel_size=1,       # 流水线并行大小
    pipeline_parallel_chunks=1,     # 流水线并行分块数
    sequence_parallel_size=1,       # 序列并行大小
    sequence_parallel_mode="standard",  # 序列并行模式
    
    # ZeRO 优化器配置
    zero_optimization=False,        # 是否启用 ZeRO
    zero_stage=0,                   # ZeRO 阶段
    zero_offload=False,             # 是否启用 offload
    
    # 训练配置
    batch_size=32,                  # 批次大小
    gradient_accumulation_steps=1,  # 梯度累积步数
    learning_rate=1e-3,             # 学习率
    
    # 检查点配置
    checkpoint_save_path="./checkpoints",
    checkpoint_save_interval=1000
)
```

### 3.2 配置验证

配置类会自动验证参数的合法性：

```python
# 这会抛出 ValueError，因为 batch_size 不能小于 1
config = ParaScaleConfig(batch_size=0)

# 这会抛出 ValueError，因为 zero_stage 必须是 0, 1, 2, 或 3
config = ParaScaleConfig(zero_stage=4)
```

### 3.3 配置更新

```python
config = ParaScaleConfig()

# 使用 update 方法更新配置
config.update({
    'batch_size': 64,
    'learning_rate': 1e-4
})

# 配置序列化
config_dict = config.to_dict()

# 从字典创建配置
new_config = ParaScaleConfig.from_dict(config_dict)
```

---

## 4. 并行策略

### 4.1 数据并行

数据并行是最常用的并行策略，每个 GPU 持有完整的模型副本，处理不同的数据子集。

```python
from ParaScale import ParaScaleConfig, Engine

config = ParaScaleConfig(
    data_parallel_size=4,  # 使用4个GPU
    batch_size=32
)

engine = Engine(model, optimizer, config)
```

**启动命令：**

```bash
torchrun --nproc_per_node=4 examples/data_parallel_test.py
```

### 4.2 模型并行

模型并行将模型的不同层分配到不同设备，适用于单个 GPU 无法容纳整个模型的情况。

```python
config = ParaScaleConfig(
    model_parallel_size=2  # 将模型分割到2个GPU
)

engine = Engine(model, optimizer, config)
```

**支持的模型结构：**
- 带有 `layers` 属性的模型（如 nn.ModuleList）
- 带有 `encoder`/`decoder` 属性的模型
- 普通 Sequential 模型

### 4.3 张量并行

张量并行将模型的权重矩阵分割到不同 GPU，支持两种并行模式：

**行并行（row）：**
```python
config = ParaScaleConfig(
    tensor_parallel_size=2,
    tensor_parallel_mode="row"  # 按输出维度分割
)
```

**列并行（column）：**
```python
config = ParaScaleConfig(
    tensor_parallel_size=2,
    tensor_parallel_mode="column"  # 按输入维度分割
)
```

**特性：**
- 使用 `torch.autograd.Function` 实现支持梯度的分布式通信
- 自动并行化模型中的线性层

### 4.4 流水线并行

流水线并行将模型按层分割到不同设备，支持微批次处理。

```python
config = ParaScaleConfig(
    pipeline_parallel_size=2,    # 2个流水线阶段
    pipeline_parallel_chunks=4   # 4个微批次
)

engine = Engine(model, optimizer, config)
```

**特性：**
- 支持动态形状传输
- 支持微批次并行处理
- 自动层提取，支持多种模型结构

### 4.5 序列并行

序列并行（Sequence Parallelism）将序列维度切分到不同GPU，主要用于减少LayerNorm、Dropout等层的激活内存占用。通常与张量并行结合使用。

#### 4.5.1 概述

**序列并行原理：**
- 沿序列维度（sequence dimension）切分输入数据
- 每个GPU只处理部分序列
- LayerNorm、Dropout等层的激活内存减少 `sp_size` 倍

**与张量并行结合：**
```
输入: [B, S, H]
    ↓ (序列并行切分)
[B, S/SP_SIZE, H]
    ↓ (张量并行切分)
[B, S/SP_SIZE, H/TP_SIZE]
    ↓ (计算)
[B, S/SP_SIZE, H/TP_SIZE]
    ↓ (张量并行收集)
[B, S/SP_SIZE, H]
    ↓ (序列并行收集)
[B, S, H]
```

#### 4.5.2 基本使用

**方法1：使用配置对象**
```python
from parascale.parallel import SequenceParallel, SequenceParallelConfig

# 创建配置
config = SequenceParallelConfig(
    sp_size=4,                      # 4路序列并行
    tp_size=2,                      # 2路张量并行
    mode="standard",                # 标准模式（Megatron风格）
    scatter_input=True,             # 自动切分输入
    gather_output=True,             # 自动收集输出
    enable_for_layernorm=True,      # 为LayerNorm启用序列并行
    enable_for_dropout=True,        # 为Dropout启用序列并行
)

# 创建序列并行实例
sp = SequenceParallel(
    model=model,
    rank=rank,
    world_size=8,
    config=config
)

# 训练
output = sp(inputs)
```

**方法2：使用便捷函数**
```python
from parascale.parallel import enable_sequence_parallel

# 快速启用序列并行
sp = enable_sequence_parallel(
    model=model,
    sp_size=4,
    tp_size=2
)

output = sp(inputs)
```

#### 4.5.3 内存优化效果

| SP Size | LayerNorm内存 | Dropout内存 | 节省倍数 |
|---------|---------------|-------------|---------|
| 1 (标准) | 32.00 MB | 8.00 MB | 1x |
| 2 | 16.00 MB | 4.00 MB | 2x |
| 4 | 8.00 MB | 2.00 MB | 4x |
| 8 | 4.00 MB | 1.00 MB | 8x |

#### 4.5.4 序列并行模式

**标准模式（Standard）：**
```python
config = SequenceParallelConfig(
    mode="standard",  # Megatron-LM风格
    sp_size=4,
)
```
- 使用All-gather/All-reduce通信
- 适合中等长度序列（< 100K tokens）

**Ulysses模式：**
```python
config = SequenceParallelConfig(
    mode="ulysses",  # DeepSpeed-Ulysses风格
    sp_size=8,
)
```
- 使用All-to-all通信切换并行维度
- 支持超长序列（1M+ tokens）

#### 4.5.5 启动序列并行训练

```bash
# 使用8个GPU进行序列并行训练 (SP=4, TP=2)
torchrun --nproc_per_node=8 examples/sequence_parallel_example.py

# 使用4个GPU进行序列并行训练 (SP=2, TP=2)
torchrun --nproc_per_node=4 examples/sequence_parallel_example.py
```

#### 4.5.6 完整训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from parascale.parallel import SequenceParallel, SequenceParallelConfig

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, hidden_size=512, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Attention with residual
        x = x + self.dropout(self.attn(self.ln1(x), x, x)[0])
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x

# 初始化分布式环境
torch.distributed.init_process_group(backend='nccl')
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(rank)

# 创建模型
model = TransformerModel(hidden_size=512, num_layers=6)

# 配置序列并行
config = SequenceParallelConfig(
    sp_size=4,          # 4路序列并行
    tp_size=2,          # 2路张量并行
    enable_for_layernorm=True,
    enable_for_dropout=True,
)

sp = SequenceParallel(
    model=model,
    rank=rank,
    world_size=world_size,
    config=config
)

# 创建优化器
optimizer = optim.AdamW(sp.model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(epochs):
    for inputs, targets in dataloader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        # 前向传播
        outputs = sp(inputs)
        
        # 计算损失
        loss = nn.functional.cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

#### 4.5.7 配置选择建议

| 场景 | 推荐配置 | 说明 |
|------|---------|------|
| 长序列训练 | SP=4, TP=2 | 序列并行减少激活内存 |
| 超长序列 | SP=8, mode="ulysses" | Ulysses模式支持1M+ tokens |
| 大模型 | SP=2, TP=4, DP=2 | 4D并行组合 |
| 内存受限 | SP=4, TP=2 | 最大化内存节省 |

#### 4.5.8 注意事项

1. **与张量并行结合**: 序列并行通常与张量并行一起使用，SP组与TP组互补
2. **序列长度**: 序列长度必须能被 `sp_size` 整除
3. **LayerNorm**: 序列并行会自动替换模型中的LayerNorm为SequenceParallelLayerNorm
4. **Dropout**: 序列并行会自动替换模型中的Dropout为SequenceParallelDropout
5. **梯度同步**: 序列并行不需要额外的梯度同步（与张量并行共享）

#### 4.5.9 运行测试

```bash
# 运行单进程测试
python tests/test_sequence_parallel.py

# 使用多进程运行测试（需要4个GPU）
torchrun --nproc_per_node=4 tests/test_sequence_parallel.py
```

---

### 4.6 3D混合并行

3D混合并行将数据并行、张量并行和流水线并行组合在一起，实现更高效的分布式训练。

#### 4.5.1 概述

3D并行配置由三个维度组成：
- **DP (Data Parallel)**: 数据并行大小
- **TP (Tensor Parallel)**: 张量并行大小
- **PP (Pipeline Parallel)**: 流水线并行大小

**总进程数 = DP × TP × PP**

#### 4.5.2 配置示例

**2D并行 (DP+TP)**：
```python
from parascale.parallel.hybrid_parallel import HybridParallel

# 使用4个GPU: DP=2, TP=2, PP=1
hp = HybridParallel(
    model=model,
    rank=rank,
    world_size=4,
    dp_size=2,
    tp_size=2,
    pp_size=1,
    tensor_parallel_mode="row"
)
```

**2D并行 (DP+PP)**：
```python
# 使用4个GPU: DP=2, TP=1, PP=2
hp = HybridParallel(
    model=model,
    rank=rank,
    world_size=4,
    dp_size=2,
    tp_size=1,
    pp_size=2,
    pipeline_chunks=2
)
```

**3D并行 (DP+TP+PP)**：
```python
# 使用8个GPU: DP=2, TP=2, PP=2
hp = HybridParallel(
    model=model,
    rank=rank,
    world_size=8,
    dp_size=2,
    tp_size=2,
    pp_size=2,
    tensor_parallel_mode="row",
    pipeline_chunks=2
)
```

#### 4.5.3 使用HybridParallel进行训练

```python
import torch
import torch.nn as nn
import torch.optim as optim
from parascale.parallel.hybrid_parallel import HybridParallel

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ])
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

# 初始化分布式环境
torch.distributed.init_process_group(backend='nccl')
torch.cuda.set_device(rank)

# 创建模型和3D并行实例
model = MyModel()
hp = HybridParallel(
    model=model,
    rank=rank,
    world_size=8,
    dp_size=2,
    tp_size=2,
    pp_size=2,
    tensor_parallel_mode="row",
    pipeline_chunks=2
)

# 打印并行信息
info = hp.get_parallel_info()
print(f"Rank {rank}: DP={info['dp_rank']}, TP={info['tp_rank']}, PP={info['pp_rank']}")

# 创建优化器
optimizer = optim.AdamW(hp.stage_layers.parameters(), lr=1e-3)

# 训练循环
for epoch in range(epochs):
    for inputs, targets in dataloader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        # 前向传播
        outputs = hp.forward(inputs)
        
        # 计算损失和反向传播（仅在最后一个流水线阶段）
        if hp.is_last_stage and outputs is not None:
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
        
        # 收集梯度（数据并行）
        hp.gather_gradients()
        
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
```

#### 4.5.4 启动3D并行训练

**使用torchrun启动：**

```bash
# 使用8个GPU进行3D并行训练 (DP=2, TP=2, PP=2)
torchrun --nproc_per_node=8 examples/hybrid_parallel_example.py

# 使用4个GPU进行2D并行训练 (DP=2, TP=2, PP=1)
torchrun --nproc_per_node=4 examples/hybrid_parallel_example.py
```

#### 4.5.5 配置选择建议

| 场景 | 推荐配置 | 说明 |
|------|---------|------|
| 小模型，大数据 | DP=8, TP=1, PP=1 | 充分利用数据并行 |
| 大模型，单节点 | DP=1, TP=8, PP=1 | 张量并行减少内存 |
| 深层模型 | DP=1, TP=1, PP=8 | 流水线并行处理深层模型 |
| 超大模型 | DP=2, TP=4, PP=2 | 3D并行组合 |
| 多节点训练 | DP=4, TP=2, PP=2 | 跨节点数据并行 |

#### 4.5.6 注意事项

1. **进程数匹配**: `dp_size × tp_size × pp_size` 必须等于 `world_size`
2. **层数要求**: 模型层数必须大于等于 `pp_size`
3. **维度可分割**: 张量并行的维度必须能被 `tp_size` 整除
4. **内存平衡**: 合理分配各并行维度以平衡内存使用
5. **CUDA可用**: 多进程训练需要CUDA支持

#### 4.5.7 实现细节

**进程组拓扑**:

总进程数 = DP_SIZE × TP_SIZE × PP_SIZE

例如: DP=2, TP=2, PP=2, 总进程数=8
- 8个进程被分成2个流水线阶段 (PP组)
- 每个流水线阶段有4个进程，被分成2个数据并行组 (DP组)
- 每个数据并行组有2个进程，进行张量并行 (TP组)

**进程ID计算**:
```
global_rank = dp_rank × (tp_size × pp_size) + tp_rank × pp_size + pp_rank
```

**模型分割**:
1. **流水线分割**: 首先按流水线并行分割模型层
2. **张量并行**: 然后在每个阶段的线性层上应用张量并行

**通信机制**:
- **张量并行通信**: 在张量并行组内进行all-gather或all-reduce
- **流水线并行通信**: 使用点对点通信(send/recv)在流水线阶段间传输数据
- **数据并行通信**: 在数据并行组内进行all-reduce同步梯度

**支持的模型结构**:
- 带有 `layers` 属性的模型（如 nn.ModuleList）
- 带有 `features` 属性的模型
- Encoder-Decoder结构的模型
- 普通Sequential模型

#### 4.5.8 性能优化建议

1. **选择合适的并行配置**: 根据模型大小和硬件配置选择最优的DP/TP/PP组合
2. **使用微批次**: 设置 pipeline_chunks > 1 来提高流水线利用率
3. **张量并行模式**:
   - 行并行("row")适合输出维度较大的层
   - 列并行("column")适合输入维度较大的层

#### 4.5.9 运行测试

```bash
# 运行单进程测试
python tests/test_hybrid_parallel.py

# 使用多进程运行测试（需要4个GPU）
torchrun --nproc_per_node=4 tests/test_hybrid_parallel.py
```

---

## 5. 优化器

### 5.1 ZeRO 优化器

ZeRO（Zero Redundancy Optimizer）优化器可以减少内存使用。

```python
config = ParaScaleConfig(
    zero_optimization=True,
    zero_stage=2,       # ZeRO 阶段（0, 1, 2, 3）
    zero_offload=True   # 启用 offload 到 CPU
)

engine = Engine(model, optimizer, config)
```

**ZeRO 阶段说明：**
- Stage 0：禁用 ZeRO
- Stage 1：优化器状态分片
- Stage 2：优化器状态 + 梯度分片
- Stage 3：优化器状态 + 梯度 + 参数分片

---

## 6. 量化训练

### 6.1 量化训练概述

量化感知训练（Quantization Aware Training, QAT）是在训练过程中模拟低精度计算，使模型能够适应量化后的推理环境。

### 6.2 配置量化训练

```python
from ParaScale import ParaScaleConfig, QuantizationConfig

# 配置量化
config = ParaScaleConfig()
config.quantization = QuantizationConfig(
    enabled=True,                   # 启用量化
    bits=8,                         # 8位量化
    scheme="symmetric",             # 对称量化
    per_channel=True,               # 逐通道量化
    observer_type="minmax",         # MinMax观察器
    fuse_modules=True,              # 融合模块
    qat_epochs=10                   # QAT训练轮数
)

engine = Engine(model, optimizer, config)
```

### 6.3 量化训练流程

```python
# 阶段1：正常训练（可选）
engine.train(trainloader, epochs=5)

# 阶段2：量化感知训练
engine.train(trainloader, epochs=10)

# 阶段3：冻结观察器，微调
engine.freeze_quantization_observer()
engine.train(trainloader, epochs=2)

# 导出量化模型
engine.export_quantized_model("model_int8.pt")
```

### 6.4 量化配置选项

| 配置项 | 类型 | 默认值 | 描述 |
|-------|------|-------|------|
| `enabled` | bool | False | 是否启用量化 |
| `mode` | str | "qat" | 量化模式（"qat"或"ptq"） |
| `bits` | int | 8 | 量化位数（8或4） |
| `scheme` | str | "symmetric" | 量化方案（"symmetric"或"asymmetric"） |
| `per_channel` | bool | True | 是否逐通道量化 |
| `observer_type` | str | "minmax" | 观察器类型（"minmax"或"moving_average"） |
| `moving_average_ratio` | float | 0.9 | 移动平均比例 |
| `fuse_modules` | bool | True | 是否融合Conv+BN+ReLU |
| `qat_epochs` | int | 10 | QAT训练轮数 |
| `calib_batches` | int | 100 | PTQ校准批次数量 |
| `backend` | str | "fbgemm" | 量化后端（"fbgemm"或"qnnpack"） |

---

## 7. 多节点训练

### 7.1 多节点训练概述

ParaScale 支持多节点分布式训练，自动检测并初始化 torchrun、SLURM、MPI 等多种分布式环境。

### 7.2 支持的环境

| 环境 | 启动方式 | 自动检测 |
|------|---------|---------|
| torchrun | `torchrun --nnodes=N ...` | ✅ |
| SLURM | `sbatch --nodes=N ...` | ✅ |
| OpenMPI | `mpirun -np N ...` | ✅ |
| MPICH | `mpiexec -n N ...` | ✅ |
| 手动 | `python train.py --rank=R ...` | ✅ |

### 7.3 torchrun 启动

**单节点多 GPU：**

```bash
torchrun --nproc_per_node=4 examples/multi_node_example.py --epochs=5
```

**多节点（2 节点，每节点 4 GPU）：**

```bash
# Node 0 (主节点)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
    --master_addr=<node0_ip> --master_port=29500 \
    examples/multi_node_example.py --epochs=5

# Node 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
    --master_addr=<node0_ip> --master_port=29500 \
    examples/multi_node_example.py --epochs=5
```

### 7.4 SLURM 启动

**创建作业脚本 `multi_node_job.sh`：**

```bash
#!/bin/bash
#SBATCH --job-name=parascale_train
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00

# 激活环境
source activate parascale_env

# 运行训练
srun python examples/multi_node_example.py --epochs=10
```

**提交作业：**

```bash
sbatch multi_node_job.sh
```

### 7.5 手动初始化

```python
from ParaScale import initialize_distributed, Engine

# 手动初始化分布式环境
rank, world_size, local_rank = initialize_distributed(
    backend="nccl",
    rank=0,
    world_size=8,
    local_rank=0,
    master_addr="192.168.1.100",
    master_port=29500
)

# 创建引擎（禁用自动初始化）
engine = Engine(model, optimizer, config, auto_init_distributed=False)
```

### 7.6 多节点工具函数

```python
from ParaScale import (
    get_rank,           # 获取全局rank
    get_world_size,     # 获取进程总数
    get_local_rank,     # 获取本地rank
    get_node_rank,      # 获取节点编号
    get_num_nodes,      # 获取节点总数
    is_main_process,    # 判断是否为主进程
)

# 使用示例
print(f"Global rank: {get_rank()}")
print(f"World size: {get_world_size()}")
print(f"Local rank: {get_local_rank()}")
print(f"Node rank: {get_node_rank()}")
print(f"Num nodes: {get_num_nodes()}")
```

### 7.7 多节点训练最佳实践

**1. 数据加载：**

```python
# 使用 DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    trainset, shuffle=True
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, sampler=train_sampler
)

# 每个 epoch 设置 epoch
for epoch in range(epochs):
    train_sampler.set_epoch(epoch)
    engine.train(trainloader, epochs=1)
```

**2. 检查点保存：**

```python
# 只在主进程保存检查点
if engine.is_main_process():
    engine.save_checkpoint()
```

**3. 日志打印：**

```python
from ParaScale import print_rank_0

# 只在 rank 0 打印
print_rank_0(f"Epoch {epoch}, Loss: {loss:.4f}")
```

---

## 8. 检查点管理

### 8.1 保存检查点

```python
# 自动保存（根据配置中的 checkpoint_save_interval）
engine.train(trainloader, epochs=10)

# 手动保存
engine.save_checkpoint("./my_checkpoints")
```

### 8.2 加载检查点

```python
# 加载检查点继续训练
engine.load_checkpoint("./checkpoints/checkpoint_1000.pt")
engine.train(trainloader, epochs=10)
```

### 8.3 检查点内容

检查点包含以下信息：
- 模型状态（model_state_dict）
- 优化器状态（optimizer_state_dict）
- 全局步数（global_step）
- 配置（config）

---

## 9. 故障排除

### 9.1 量化训练问题

**问题：** 量化后精度下降严重
**解决方案：**
- 增加 QAT 训练轮数
- 使用逐通道量化（per_channel=True）
- 尝试非对称量化（scheme="asymmetric"）
- 使用移动平均观察器（observer_type="moving_average"）

**问题：** 量化训练速度慢
**解决方案：**
- 量化训练确实比 FP32 慢，这是正常的
- 确保只在最后几个 epoch 启用 QAT
- 使用混合精度训练加速

### 9.2 多节点训练问题

**问题：** 分布式初始化失败
**解决方案：**
- 检查 MASTER_ADDR 和 MASTER_PORT 是否正确
- 确保所有节点可以访问主节点
- 检查防火墙设置
- 使用 `initialize_distributed()` 的调试输出

**问题：** 多节点训练速度慢
**解决方案：**
- 检查网络带宽（建议使用 InfiniBand）
- 减少梯度同步频率（增加 gradient_accumulation_steps）
- 使用异步数据加载（num_workers > 0）
- 检查是否有节点成为瓶颈

**问题：** 显存不足
**解决方案：**
- 启用 ZeRO 优化器
- 使用模型并行或流水线并行
- 使用3D混合并行（DP+TP+PP组合）
- 减小批次大小
- 使用梯度累积

**问题：** 3D并行配置错误
**解决方案：**
- 确保 `dp_size × tp_size × pp_size = world_size`
- 检查模型层数是否大于等于 `pp_size`
- 确保张量并行维度能被 `tp_size` 整除
- 使用 `get_parallel_info()` 检查并行配置

**问题：** 3D并行训练速度慢
**解决方案：**
- 调整 DP/TP/PP 比例，找到最优配置
- 增加 `pipeline_chunks` 提高流水线利用率
- 检查通信带宽是否成为瓶颈
- 尝试不同的张量并行模式（row/column）

### 9.3 调试技巧

**查看分布式信息：**

```python
from ParaScale import print_distributed_info

print_distributed_info()
```

**检查环境变量：**

```bash
echo $RANK
echo $WORLD_SIZE
echo $MASTER_ADDR
echo $MASTER_PORT
```

**测试多节点连接：**

```bash
# 在主节点
python -c "import socket; s=socket.socket(); s.bind(('', 29500)); s.listen(1); print('Listening...')"

# 在其他节点
python -c "import socket; s=socket.socket(); s.connect(('<master_ip>', 29500)); print('Connected!')"
```
