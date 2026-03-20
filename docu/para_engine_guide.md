# ParaEngine 使用指南

## 概述

ParaEngine 是 ParaScale 框架中的智能并行训练引擎，能够根据模型规模、硬件状态自适应决策并行优化策略组合，无需手动配置复杂的并行参数。

**核心特性**:
- 自动并行策略选择
- 模型结构智能分析
- 硬件资源实时监控
- 支持 3D 混合并行自动配置

---

## 一分钟上手

```python
from parascale import ParaEngine
import torch.optim as optim

# 创建模型和优化器
model = MyModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 创建 ParaEngine（自动并行）
engine = ParaEngine(model, optimizer, auto_parallel=True)

# 查看自动选择的策略
strategy = engine.get_strategy()
print(f"DP={strategy.dp_size}, TP={strategy.tp_size}, PP={strategy.pp_size}")

# 开始训练
engine.train(dataloader, epochs=10)
```

---

## 核心组件

### 1. ModelAnalyzer（模型分析器）

分析模型参数量、层结构、内存需求，检测模型类型（Transformer、RNN、CNN、MLP）。

```python
def analyze(self) -> ModelProfile:
    """分析模型并生成配置文件"""
    self._count_parameters()      # 统计参数量
    self._analyze_layers()        # 分析层结构
    self._estimate_memory()       # 估算内存
    self._detect_model_type()     # 检测模型类型
```

### 2. HardwareMonitor（硬件监控器）

监控 GPU 数量、内存、计算能力，测量通信带宽。

```python
def monitor(self) -> HardwareProfile:
    """监控硬件资源并生成配置文件"""
    self._detect_gpus()           # 检测 GPU 信息
    self._measure_memory()        # 测量内存
    self._detect_compute_capability()  # 检测计算能力
    self._estimate_bandwidth()    # 估算带宽
```

### 3. StrategyDecider（策略决策器）

基于启发式规则决策最优并行策略：
- 小模型（<1B）: 纯数据并行
- 中等模型（1B-10B）: DP+TP 混合并行
- 大模型（>10B）: 3D 混合并行（DP×TP×PP）

```python
def decide(self) -> ParallelStrategy:
    """决策最优并行策略"""
    if world_size == 1:
        return self._single_gpu_strategy()
    elif model_params > 10B:
        return self._large_model_strategy(world_size)
    elif model_params > 1B:
        return self._medium_model_strategy(world_size)
    else:
        return self._small_model_strategy(world_size)
```

---

## 自动策略选择

| 模型规模 | 参数量 | 策略 | 配置 |
|---------|--------|------|------|
| 小模型 | < 1B | 数据并行 | DP=N, TP=1, PP=1 |
| 中等模型 | 1B-10B | DP+TP 混合 | DP×TP=N, PP=1 |
| 大模型 | > 10B | 3D 混合 | DP×TP×PP=N |

### 策略选择逻辑

**小模型策略（< 1B 参数）**:
```python
def _small_model_strategy(self, world_size: int):
    return ParallelStrategy(
        dp_size=world_size,
        tp_size=1,
        pp_size=1,
        strategy_type='data',
        reason='Small model, use pure data parallel',
        estimated_speedup=world_size * 0.9
    )
```

**中等模型策略（1B-10B 参数）**:
```python
def _medium_model_strategy(self, world_size: int):
    if model_memory > available_memory * 0.5:
        tp_size = min(4, world_size)
        dp_size = world_size // tp_size
        return ParallelStrategy(
            dp_size=dp_size,
            tp_size=tp_size,
            pp_size=1,
            strategy_type='hybrid',
            reason='Medium model with memory constraints'
        )
```

**大模型策略（> 10B 参数）**:
```python
def _large_model_strategy(self, world_size: int):
    if model_memory > available_memory:
        pp_size = min(8, world_size)
        remaining = world_size // pp_size
        tp_size = min(4, remaining)
        dp_size = remaining // tp_size
        return ParallelStrategy(
            dp_size=dp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            strategy_type='hybrid',
            reason='Large model, use 3D parallel'
        )
```

---

## 核心 API

### ParaEngine

```python
ParaEngine(model, optimizer, config=None, auto_parallel=True)
  - train(dataloader, epochs)     # 训练模型
  - evaluate(dataloader)          # 评估模型
  - get_strategy() -> ParallelStrategy  # 获取当前策略
  - get_parallel_info() -> dict   # 获取并行信息
```

### ParallelStrategy

```python
@dataclass
class ParallelStrategy:
    dp_size: int = 1                # 数据并行大小
    tp_size: int = 1                # 张量并行大小
    pp_size: int = 1                # 流水线并行大小
    strategy_type: str = 'data'     # 策略类型
    reason: str = ''                # 选择原因
    estimated_memory_saving: float = 0.0  # 内存节省
    estimated_speedup: float = 1.0  # 预估加速比
```

### ModelProfile

```python
@dataclass
class ModelProfile:
    total_params: int = 0           # 总参数量
    total_memory: int = 0           # 总内存（字节）
    num_layers: int = 0             # 层数
    max_layer_memory: int = 0       # 最大层内存
    layer_types: Dict[str, int] = None  # 层类型计数
    embedding_size: int = 0         # 嵌入层大小
    hidden_size: int = 0            # 隐藏层维度
    vocab_size: int = 0             # 词表大小
    model_type: str = 'unknown'     # 模型类型
```

### HardwareProfile

```python
@dataclass
class HardwareProfile:
    num_gpus: int = 0               # GPU 数量
    gpu_memory: int = 0             # GPU 内存（字节）
    gpu_compute_capability: float = 0.0  # 计算能力
    available_memory: int = 0       # 可用内存
    communication_bandwidth: float = 0.0  # 通信带宽
    num_nodes: int = 1              # 节点数
    gpus_per_node: int = 1          # 每节点 GPU 数
```

---

## 配置选项

### 自动模式（推荐）

```python
engine = ParaEngine(model, optimizer, auto_parallel=True)
```

### 手动模式

```python
config = ParaScaleConfig(
    data_parallel_size=2,
    tensor_parallel_size=2,
    pipeline_parallel_size=2
)
engine = ParaEngine(model, optimizer, config=config, auto_parallel=False)
```

---

## 启动命令

```bash
# 自动并行训练（8 GPU）
torchrun --nproc_per_node=8 your_script.py

# 运行示例
torchrun --nproc_per_node=8 examples/para_engine_example.py

# 运行测试
torchrun --nproc_per_node=8 tests/test_para_engine.py
```

---

## 常用场景

### 场景 1: 小模型快速训练

```python
model = MLPModel(hidden_size=256, num_layers=3)
engine = ParaEngine(model, optimizer, auto_parallel=True)
# 自动选择：纯数据并行
```

### 场景 2: Transformer 训练

```python
model = TransformerModel(d_model=768, num_layers=12)
engine = ParaEngine(model, optimizer, auto_parallel=True)
# 自动选择：DP+TP 混合并行
```

### 场景 3: 大模型训练

```python
model = TransformerModel(d_model=2048, num_layers=24)
engine = ParaEngine(model, optimizer, auto_parallel=True)
# 自动选择：3D 混合并行
```

---

## 性能调优

### 增加梯度累积

```python
config = ParaScaleConfig(gradient_accumulation_steps=4)
engine = ParaEngine(model, optimizer, config=config)
```

### 调整流水线分块

```python
config = ParaScaleConfig(pipeline_parallel_chunks=2)
engine = ParaEngine(model, optimizer, config=config)
```

---

## 故障排除

### OOM（内存不足）

- 减小 batch_size
- 增加 pipeline_parallel_chunks
- 启用梯度检查点

### 训练速度慢

- 检查并行策略是否合理
- 增加梯度累积步数
- 使用混合精度训练

### 分布式初始化失败

- 检查 CUDA_VISIBLE_DEVICES
- 确保端口未被占用
- 检查防火墙设置

---

## 测试覆盖

### 单元测试

1. **ModelAnalyzer 测试**: 简单模型分析、中等模型分析、模型配置文件转换
2. **HardwareMonitor 测试**: 无 CUDA 环境监控、硬件配置文件转换
3. **StrategyDecider 测试**: 单 GPU 策略、小/中/大模型策略、策略验证
4. **ParaEngine 测试**: 自动/手动模式初始化、单步训练、模型评估
5. **自动并行场景测试**: 小模型自动配置、中等模型自动配置

### 使用示例

1. 小模型自动并行配置
2. 中等模型自动并行配置
3. 大模型自动并行配置（模拟）
4. 手动配置并行策略
5. 完整训练流程
6. CNN 图像分类
7. Transformer 语言模型

---

## 文件结构

```
parascale/
├── engine/
│   └── para_engine.py      # ParaEngine 核心实现 (1060 行)
├── __init__.py              # 导出 ParaEngine
examples/
├── para_engine_example.py   # 使用示例 (365 行)
tests/
├── test_para_engine.py      # 单元测试 (383 行)
docu/
├── Para_Engine_guide.md     # 本文件
```

---

## 版本历史

### v0.1.0 (2026-03-19)

- 初始版本发布
- ParaEngine自动并行训练引擎
- ModelAnalyzer模型分析器
- HardwareMonitor硬件监控器
- StrategyDecider策略决策器
- 支持小/中/大模型自动策略选择
- 支持3D混合并行自动配置

**历史版本参考**: 本版本为初始版本(v0.1.0)，后续版本的详细变更记录请参阅文档更新日志。

---

## 技术亮点

1. **智能自动并行**: 无需手动配置复杂的并行参数，自动选择最优策略
2. **模型感知**: 自动分析模型结构和资源需求
3. **硬件感知**: 实时监控硬件状态和资源可用性
4. **启发式决策**: 基于规则和性能模型的智能决策
5. **灵活扩展**: 支持手动配置和自动配置两种模式
6. **完整测试**: 覆盖所有核心组件和使用场景
7. **详细文档**: 提供完整的用户指南和 API 参考

---

## 未来工作

1. 强化学习-based 策略优化
2. 运行时动态调整策略
3. 更精细的性能建模
4. 支持更多并行模式（序列并行等）
5. 自动调优功能
6. 可视化工具

---

## 总结

ParaEngine 成功实现了自动并行策略选择功能，大大降低了分布式训练的门槛。用户无需深入了解各种并行策略的优缺点，只需使用 ParaEngine，系统会自动根据模型和硬件情况选择最优的并行配置。这对于大规模模型训练尤其有价值。
