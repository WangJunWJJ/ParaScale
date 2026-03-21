# ParaScale 软件设计文档

**版本**: v0.1.0
**日期**: 2026-03-19
**状态**: 已更新

本文档详细描述了 ParaScale 框架的软件架构和设计决策。

***

## 版本历史

| 版本     | 日期         | 更新内容   |
| ------ | ---------- | ------ |
| v0.1.0 | 2026-03-19 | 初始版本发布 |

**历史版本参考**: 本版本为初始版本(v0.1.0)，后续版本的详细变更记录请参阅文档更新日志。

***

## 目录

1. [概述](#1-概述)
2. [架构设计](#2-架构设计)
3. [核心模块](#3-核心模块)
   - 3.1 [ParaScaleEngine](#31-parascaleengine)
   - 3.2 [配置系统](#32-配置系统)
   - 3.3 [ParaEngine](#33-paraengine)
4. [并行策略设计](#4-并行策略设计)
5. [量化训练设计](#5-量化训练设计)
   - 5.1 [整体架构](#51-整体架构)
   - 5.2 [伪量化层](#52-伪量化层)
   - 5.3 [观察器](#53-观察器)
   - 5.4 [QAT管理](#54-qat-管理)
   - 5.5 [PTQ设计](#55-ptq-设计)
   - 5.6 [PTQ vs QAT对比](#56-ptq-vs-qat-对比)
   - 5.7 [量化模块架构分析](#57-量化模块架构分析)
6. [分布式训练设计](#6-分布式训练设计)
7. [配置系统设计](#7-配置系统设计)
8. [测试设计](#8-测试设计)
9. [部署计划](#9-部署计划)

***

## 1. 概述

### 1.1 设计目标

ParaScale 框架的设计目标包括：

- **易用性**：提供简洁的 API，与 PyTorch 无缝集成
- **灵活性**：支持多种并行策略的组合使用
- **可扩展性**：模块化设计，易于扩展新的并行策略
- **性能**：高效的分布式通信，最小化通信开销
- **类型安全**：完整的类型注解支持

### 1.2 技术栈

- **深度学习框架**：PyTorch
- **分布式通信**：torch.distributed
- **配置管理**：Python dataclass
- **类型系统**：Python typing

***

## 2. 架构设计

### 2.1 整体架构

```
ParaScale/
├── __init__.py                 # 包入口，导出所有公共 API
├── engine/                     # 引擎模块（双引擎架构）
│   ├── __init__.py             # 引擎模块导出
│   ├── engine.py               # ParaScaleEngine（手动策略引擎）
│   └── para_engine.py          # ParaEngine（自动调度引擎）
├── config.py                   # 配置管理（ParaScaleConfig, QuantizationConfig）
├── utils.py                    # 工具函数
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── utils.py                # 通用工具函数
│   └── distributed_utils.py    # 分布式工具
├── optimizers/                 # 优化器模块（统一扩展模块）
│   ├── __init__.py             # 优化器模块导出
│   └── optimizers.py           # 优化器实现（ZeroOptimizer, AdamW）
├── parallel/                   # 并行策略模块
│   ├── __init__.py
│   ├── base.py                 # 并行策略基类（BaseParallel）
│   ├── data_parallel.py        # 数据并行（DataParallel）
│   ├── model_parallel.py       # 模型并行（ModelParallel）
│   ├── tensor_parallel.py      # 张量并行（TensorParallel）
│   ├── pipeline_parallel.py    # 流水线并行（PipelineParallel）
│   └── hybrid_parallel.py      # 3D混合并行（HybridParallel）
├── quantization/               # 量化训练模块
│   ├── __init__.py
│   ├── base.py                 # 量化配置（QuantizationConfig）
│   ├── observers.py            # 观察器
│   ├── fake_quantize.py        # 伪量化层
│   ├── qat.py                  # 量化感知训练
│   ├── ptq.py                  # 训练后量化
│   └── utils.py                # 量化工具函数
├── examples/                   # 示例程序
└── tests/                      # 测试文件
```

### 2.2 模块依赖关系

```
ParaScaleEngine (手动策略引擎)
    ├── ParaScaleConfig
    ├── BaseParallel (DataParallel, ModelParallel, TensorParallel, PipelineParallel, HybridParallel)
    ├── ZeroOptimizer
    ├── QuantizationAwareTraining
    └── distributed_utils

ParaEngine (自动策略引擎)
    ├── ParaScaleConfig
    └── BaseParallel

BaseParallel
    └── torch.nn.Module

QuantizationAwareTraining (QAT)
    ├── QuantizationConfig
    ├── FakeQuantize
    └── observers

PostTrainingQuantization (PTQ)
    ├── QuantizationConfig
    └── observers
```

***

## 3. 核心模块

### 3.1 ParaScaleEngine

**职责**：协调训练过程，管理所有并行策略和训练状态。

**主要功能**：

- 自动初始化分布式环境
- 根据配置创建并行策略实例
- 管理训练循环（train/evaluate）
- 检查点保存和加载
- 量化训练管理

**关键设计决策**：

- 使用组合模式管理并行策略
- 使用 local\_rank 而不是 rank 设置 GPU 设备
- 支持自动和手动分布式初始化

**代码结构**：

```python
class ParaScaleEngine:
    def __init__(self, model, optimizer, config, auto_init_distributed=True):
        # 初始化分布式环境
        # 获取分布式信息（rank, world_size, local_rank, node_rank, num_nodes）
        # 配置量化
        # 配置并行策略
        # 配置优化器
    
    def train(self, dataloader, epochs=1):
        # 训练循环
        # 梯度累积
        # 定期保存检查点
    
    def evaluate(self, dataloader):
        # 评估模型
        # 计算损失和准确率
    
    def save_checkpoint(self, save_path=None):
        # 保存模型状态、优化器状态、全局步数、配置
    
    def load_checkpoint(self, checkpoint_path):
        # 加载检查点
```

### 3.2 配置系统

**ParaScaleConfig**：主配置类，使用 dataclass 实现。

**设计特点**：

- 自动参数验证（__post\_init__）
- 支持配置更新（update 方法）
- 支持序列化/反序列化（to\_dict/from\_dict）

**QuantizationConfig**：量化配置类。

**配置项**：

- 基础配置：enabled, mode, bits
- 量化方案：scheme, per\_channel
- 观察器配置：observer\_type, moving\_average\_ratio
- 训练配置：qat\_epochs, calib\_batches
- 后端配置：backend

### 3.3 ParaEngine

**职责**：智能并行策略选择引擎，自动分析模型结构并推荐最优并行配置。

**主要功能**：

- 自动分析模型结构特征
- 估计训练内存需求
- 推荐最优并行策略组合
- 生成ParaScaleConfig配置
- 实时监控硬件状态（GPU 内存、计算能力、通信带宽）

**关键设计决策**：

- 基于模型大小和GPU内存自动选择并行策略
- 支持速度、内存、平衡三种优化优先级
- 提供一键式配置生成功能
- 集成实时硬件监控，为性能优化提供数据支持

**代码结构**：

```python
class ParaEngine:
    def __init__(self, model=None, batch_size=32, world_size=None, 
                 gpu_memory_gb=None, network_bandwidth_gbps=10.0):
        # 初始化引擎参数
        # 自动检测GPU数量和显存
        # 初始化实时硬件监控器
    
    def analyze_model(self, model):
        # 分析模型结构
        # 返回模型信息字典
    
    def estimate_memory(self, model, sequence_length=512):
        # 估计内存需求
        # 返回内存需求字典
    
    def recommend_strategy(self, model=None, priority="balanced"):
        # 推荐并行策略
        # 返回配置字典
    
    def get_optimal_config(self, model=None, priority="balanced"):
        # 获取最优配置
        # 返回ParaScaleConfig实例
    
    def train(self, dataloader, epochs=1):
        # 训练模型
        # 在训练过程中调用实时硬件监控
        # 定期打印硬件监控摘要
    
    def _monitor_hardware(self):
        # 实时监控硬件状态
        # 收集 GPU 内存、计算能力、通信带宽指标
```

**实时硬件监控集成**：

- 在训练过程中每 100 个 batch 监控一次
- 每 1000 个 batch 打印硬件监控摘要
- 支持历史记录和平均值/峰值统计
- 为性能优化和资源管理提供实时数据支持

***

## 4. 并行策略设计

### 4.1 基类设计

**BaseParallel**：所有并行策略的抽象基类。

**职责**：

- 参数验证
- 设备管理
- 定义前向传播接口

**代码结构**：

```python
class BaseParallel(ABC):
    def __init__(self, model, rank, world_size):
        self._validate_params(model, rank, world_size)
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def forward(self, inputs):
        pass
    
    def gather_gradients(self):
        # 默认空实现，子类可覆盖
        pass
```

### 4.2 数据并行

**DataParallel**：每个 GPU 持有完整模型副本，处理不同数据子集。

**核心操作**：

- 广播模型参数（broadcast）
- 收集并平均梯度（all-reduce）

**代码结构**：

```python
class DataParallel(BaseParallel):
    def broadcast_model(self):
        for param in self.model.parameters():
            dist.broadcast(param, src=0)
    
    def gather_gradients(self):
        for param in self.model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= self.world_size
    
    def prepare_dataloader(self, dataset, batch_size, shuffle=True, num_workers=0):
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
```

### 4.3 模型并行

**ModelParallel**：将模型不同层分配到不同设备。

**支持的模型结构**：

- 带有 layers 属性的模型（nn.ModuleList）
- 带有 encoder/decoder 属性的模型
- 普通 Sequential 模型

**通信方式**：点对点通信（send/recv）

**代码结构**：

```python
class ModelParallel(BaseParallel):
    def _split_model(self):
        # 根据模型结构分割层
        # 支持 layers、encoder/decoder、普通模型
    
    def forward(self, inputs):
        # 按顺序执行各个阶段
        # 通过 send/recv 传递中间结果
        # 只有最后一个 rank 返回输出
```

### 4.4 张量并行

**TensorParallel**：将权重矩阵分割到不同 GPU。

**并行模式**：

- 行并行（row）：按输出维度分割，使用 all-gather 拼接
- 列并行（column）：按输入维度分割，使用 all-reduce 求和

**梯度传播**：使用 torch.autograd.Function 实现支持梯度的分布式通信。

**代码结构**：

```python
class AllGatherWithGradient(Function):
    @staticmethod
    def forward(ctx, tensor, world_size):
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=-1)
    
    @staticmethod
    def backward(ctx, grad_output):
        split_size = grad_output.size(-1) // ctx.world_size
        grad_input = grad_output[:, split_size * ctx.rank : split_size * (ctx.rank + 1)]
        return grad_input, None

class TensorParallel(BaseParallel):
    def _parallelize_row(self, linear_layer):
        # 按输出维度分割权重
    
    def _parallelize_column(self, linear_layer):
        # 按输入维度分割权重
    
    def forward(self, inputs):
        # 行并行：all-gather 拼接输出
        # 列并行：all-reduce 求和输出
```

### 4.5 流水线并行

**PipelineParallel**：将模型按层分割，支持微批次处理。

**核心特性**：

- 动态形状传输
- 微批次并行处理
- 自动层提取

**代码结构**：

```python
class PipelineParallel(BaseParallel):
    def _partition_model(self):
        # 将模型分割到不同阶段
    
    def _extract_layers(self):
        # 从模型中提取所有层
        # 支持多种模型结构
    
    def _forward_single_batch(self, input_data):
        # 单批次前向传播
    
    def _forward_micro_batches(self, input_data):
        # 微批次前向传播
    
    def _send_tensor(self, tensor, dst):
        # 发送张量（支持动态形状）
    
    def _recv_tensor(self, src):
        # 接收张量（支持动态形状）
```

### 4.6 3D混合并行

**HybridParallel**：将数据并行、张量并行和流水线并行组合在一起，实现3D并行。

**核心特性**：

- 支持三种并行策略的任意组合（DP+TP、DP+PP、TP+PP、DP+TP+PP）
- 多进程组管理（TP组、PP组、DP组）
- 支持行并行和列并行两种张量并行模式
- 支持微批次流水线处理
- 1F1B调度优化
- 进程ID映射：global\_rank = dp\_rank × (tp\_size × pp\_size) + tp\_rank × pp\_size + pp\_rank

**架构层次**：

```
Layer 3: Data Parallel
    - 数据分割到不同节点
    - 梯度同步

Layer 2: Tensor Parallel
    - 每层内部张量切分
    - 使用 TensorParallel 的实现

Layer 1: Pipeline Parallel
    - 模型按层分割到不同 GPU
    - 1F1B 调度优化
```

**1F1B 调度算法**：

```
调度顺序 (以 4 个 stage, 8 个 micro-batch 为例):

Stage 0: F0 F1 F2 F3 F4 F5 F6 F7 B7 B6 B5 B4 B3 B2 B1 B0
Stage 1:    F0 F1 F2 F3 F4 F5 F6 B7 B6 B5 B4 B3 B2 B1 B0
Stage 2:       F0 F1 F2 F3 F4 B7 B6 B5 B4 B3 B2 B1 B0
Stage 3:          F0 F1 F2 B7 B6 B5 B4 B3 B2 B1 B0

Fn = Forward of micro-batch n
Bn = Backward of micro-batch n
```

**进程组拓扑**：

```
总进程数 = DP_SIZE × TP_SIZE × PP_SIZE

示例: DP=2, TP=2, PP=2, 总进程数=8
- 8个进程被分成2个流水线阶段 (PP组)
- 每个流水线阶段有4个进程，被分成2个数据并行组 (DP组)
- 每个数据并行组有2个进程，进行张量并行 (TP组)

Global Ranks:    [0, 1, 2, 3, 4, 5, 6, 7]
                  │  │  │  │  │  │  │  │
DP Groups:       [0, 4] [1, 5] [2, 6] [3, 7]
TP Groups:       [0, 1] [2, 3] [4, 5] [6, 7]
PP Groups:       [0, 2] [1, 3] [4, 6] [5, 7]

坐标计算:
- pp_rank = rank % pp_size
- tp_rank = (rank // pp_size) % tp_size
- dp_rank = rank // (pp_size * tp_size)
```

**配置选项**：

```python
class HybridParallelConfig:
    dp_size: int                    # 数据并行大小
    tp_size: int                    # 张量并行大小
    pp_size: int                    # 流水线并行大小
    schedule: PipelineSchedule      # 1f1b / fill_drain / interleaved
    num_micro_batches: int          # 微批次数量
    enable_overlap: bool            # 是否启用通信重叠
```

**代码结构**：

```python
class HybridParallel(BaseParallel):
    def __init__(self, model, rank, world_size, dp_size, tp_size, pp_size, ...):
        # 验证配置：dp_size × tp_size × pp_size == world_size
        # 初始化进程组
        # 分割模型（流水线分割 + 张量并行分割）
    
    def _init_process_groups(self):
        # 创建张量并行组（相同 dp_rank 和 pp_rank）
        # 创建流水线并行组（相同 dp_rank 和 tp_rank）
        # 创建数据并行组（相同 tp_rank 和 pp_rank）
    
    def _partition_model(self):
        # 首先按流水线并行分割模型层
        # 然后在每个阶段的线性层上应用张量并行
    
    def _apply_tensor_parallel(self, layers):
        # 在层列表上应用张量并行
        # 支持行并行和列并行
    
    def forward(self, inputs):
        # 第一个流水线阶段接收输入
        # 张量并行column模式：切分输入
        # 执行当前阶段的层
        # 张量并行row模式：all-gather输出
        # 如果不是最后一个阶段，发送给下一阶段
    
    def _tensor_all_gather(self, tensor):
        # 在张量并行组内执行all-gather
    
    def _tensor_all_reduce(self, tensor):
        # 在张量并行组内执行all-reduce
    
    def gather_gradients(self):
        # 在张量并行组内对梯度进行all-reduce
```

***

## 5. 实时硬件监控设计

### 5.1 整体架构

```
RealTimeHardwareMonitor
    ├── HardwareMetrics
    ├── GPU 内存监控
    ├── 计算能力监控
    ├── 通信带宽监控
    └── 历史记录和统计
```

### 5.2 HardwareMetrics

**HardwareMetrics**：硬件指标数据类，存储单个时间点的硬件状态信息。

**主要属性**：

- `timestamp`：时间戳
- `gpu_memory_used`：已使用的 GPU 内存（字节）
- `gpu_memory_peak`：峰值 GPU 内存（字节）
- `gpu_memory_percent`：内存使用百分比
- `gpu_utilization`：GPU 利用率（0-100）
- `gpu_temperature`：GPU 温度（摄氏度）
- `communication_bandwidth`：通信带宽（GB/s）
- `compute_throughput`：计算吞吐量（TFLOPS）

**代码结构**：

```python
@dataclass
class HardwareMetrics:
    timestamp: float = field(default_factory=time.time)
    gpu_memory_used: int = 0
    gpu_memory_peak: int = 0
    gpu_memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    communication_bandwidth: float = 0.0
    compute_throughput: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        # 转换为字典，单位已转换（MB、GB/s、TFLOPS）
```

### 5.3 RealTimeHardwareMonitor

**RealTimeHardwareMonitor**：实时硬件监控器，在训练过程中持续监控 GPU 内存、计算能力、通信带宽等硬件状态。

**核心功能**：

- 实时收集 GPU 内存使用、峰值、百分比
- 实时监控 GPU 利用率和温度
- 定期测量计算吞吐量（TFLOPS）
- 定期测试通信带宽（GB/s）
- 支持历史记录和平均值/峰值统计

**代码结构**：

```python
class RealTimeHardwareMonitor:
    def __init__(self, local_rank=0, max_history_size=1000,
                 bandwidth_test_interval=60.0, compute_check_interval=30.0):
        # 初始化监控器参数
        # 创建空的历史记录列表
    
    def collect_metrics(self) -> HardwareMetrics:
        # 收集当前硬件指标
        # 调用 _collect_gpu_metrics()
        # 调用 _collect_compute_metrics()
        # 调用 _check_bandwidth()
        # 添加到历史记录
    
    def _collect_gpu_metrics(self, metrics) -> HardwareMetrics:
        # 收集 GPU 内存、利用率、温度
        # 使用 torch.cuda.memory_allocated()
        # 使用 torch.cuda.utilization()
        # 使用 torch.cuda.temperature()
    
    def _collect_compute_metrics(self, metrics) -> HardwareMetrics:
        # 测量计算吞吐量
        # 执行矩阵乘法测试
        # 计算 TFLOPS
    
    def _check_bandwidth(self, metrics, current_time) -> HardwareMetrics:
        # 测试通信带宽
        # 执行 all_reduce 操作
        # 计算 GB/s
    
    def get_average_metrics(self, last_n=100) -> Optional[HardwareMetrics]:
        # 获取最近 N 个指标的平均值
    
    def get_peak_metrics(self) -> Optional[HardwareMetrics]:
        # 获取历史记录中的峰值指标
    
    def print_metrics_summary(self) -> None:
        # 打印硬件监控摘要
    
    def clear_history(self) -> None:
        # 清空历史记录
```

### 5.4 监控策略

**GPU 内存监控**：

- 每个训练 batch 都会收集
- 实时跟踪已使用内存、峰值内存、使用百分比
- 使用 `torch.cuda.memory_allocated()` 和 `torch.cuda.max_memory_allocated()`

**GPU 利用率和温度监控**：

- 每个训练 batch 都会收集
- 使用 `torch.cuda.utilization()` 和 `torch.cuda.temperature()`

**计算吞吐量监控**：

- 每 30 秒测量一次
- 执行矩阵乘法测试
- 计算 TFLOPS

**通信带宽监控**：

- 每 60 秒测试一次
- 执行 `dist.all_reduce()` 操作
- 计算 GB/s

### 5.5 历史记录和统计

**历史记录管理**：

- 默认最多保存 1000 个数据点
- 自动删除最旧的数据点
- 支持清空历史记录

**统计功能**：

- `get_average_metrics()`：计算最近 N 个指标的平均值
- `get_peak_metrics()`：获取历史记录中的峰值指标
- `print_metrics_summary()`：打印详细的监控摘要

### 5.6 与 ParaEngine 集成

**集成方式**：

- ParaEngine 在初始化时创建 RealTimeHardwareMonitor
- 在训练过程中每 100 个 batch 调用一次 `collect_metrics()`
- 每 1000 个 batch 打印一次硬件监控摘要

**代码示例**：

```python
class ParaEngine:
    def __init__(self, model, optimizer, auto_parallel=True):
        # ... 其他初始化代码 ...
        self.hardware_monitor = None
        self.monitor_interval = 100
    
    def train(self, dataloader, epochs=1):
        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                # ... 训练代码 ...
                
                # 实时监控硬件状态
                if self.global_step % self.monitor_interval == 0:
                    self._monitor_hardware()
    
    def _monitor_hardware(self) -> None:
        if self.hardware_monitor is None:
            self.hardware_monitor = create_hardware_monitor(
                local_rank=self.local_rank
            )
        
        metrics = self.hardware_monitor.collect_metrics()
        
        if self.global_step % (self.monitor_interval * 10) == 0:
            self.hardware_monitor.print_metrics_summary()
```

***

## 6. 量化训练设计

### 6.1 整体架构

```
QuantizationAwareTraining
    ├── QuantizationConfig
    ├── FakeQuantize
    │   ├── MinMaxObserver / MovingAverageObserver
    │   └── 伪量化操作
    └── 模型准备和转换
```

### 6.2 伪量化层

**FakeQuantize**：在训练过程中模拟量化-反量化过程。

**工作流程**：

1. 使用观察器收集统计信息（min/max）
2. 计算量化参数（scale, zero\_point）
3. 量化：x\_q = round((x - zero\_point) / scale)
4. 钳制：x\_q = clamp(x\_q, qmin, qmax)
5. 反量化：x\_dq = x\_q \* scale + zero\_point

**代码结构**：

```python
class FakeQuantize(nn.Module):
    def __init__(self, config):
        self.observer = MinMaxObserver(config)  # 或 MovingAverageObserver
        self.scale = torch.tensor(1.0)
        self.zero_point = torch.tensor(0.0)
    
    def forward(self, x):
        if self.observer_enabled:
            self.observer.update(x.detach())
        
        if self.fake_quant_enabled:
            scale, zero_point = self.observer.calculate_qparams()
            x_quant = torch.round(x / scale + zero_point)
            x_quant = torch.clamp(x_quant, qmin, qmax)
            x_dequant = (x_quant - zero_point) * scale
            return x_dequant
        return x
```

### 6.3 观察器

**MinMaxObserver**：记录观察到的最小值和最大值。

**MovingAverageObserver**：使用移动平均平滑观察到的数值范围。

### 6.4 QAT 管理

**QuantizationAwareTraining**：管理 QAT 的整个流程。

**流程**：

1. 准备模型（prepare）：融合模块、插入伪量化层
2. 训练：使用伪量化层模拟量化误差
3. 冻结观察器（freeze\_observer）：停止收集统计信息
4. 转换模型（convert）：导出量化模型

### 6.5 PTQ 设计

**PostTrainingQuantization**：训练后量化的管理类。

**整体架构**：

```
PostTrainingQuantization
    ├── QuantizationConfig
    ├── 模型准备（prepare）
    │   ├── 模块融合
    │   └── 观察器插入
    ├── 校准（calibrate）
    │   └── 使用校准数据收集统计信息
    ├── 转换（convert）
    │   └── 生成量化模型
    └── 模型保存/加载
```

**流程**：

1. 准备模型（prepare）：融合 Conv+BN+ReLU 模块，插入观察器
2. 校准（calibrate）：使用校准数据前向传播，收集激活统计信息
3. 转换（convert）：根据统计信息计算量化参数，生成量化模型
4. 保存/加载：支持量化模型的序列化和反序列化

**代码结构**：

```python
class PostTrainingQuantization:
    def __init__(self, model, config, example_inputs=None):
        self.model = model
        self.config = config
        self.example_inputs = example_inputs
        self.prepared_model = None
        self.quantized_model = None
        self.calibrated = False
    
    def prepare(self):
        # 复制模型
        # 融合模块（Conv+BN+ReLU）
        # 插入伪量化层
        # 复制权重
    
    def calibrate(self, calib_loader, progress_callback=None):
        # 前向传播校准数据
        # 收集激活值统计信息
        # 计算量化参数（scale, zero_point）
    
    def freeze_observer(self):
        # 停止收集统计信息
        # 固定量化参数
    
    def quantize_weights(self):
        # 量化并反量化权重
        # 模拟量化误差
    
    def convert(self):
        # 禁用伪量化
        # 得到最终量化模型
    
    def export(self, save_path):
        # 保存量化模型和参数
    
    def evaluate(self, test_loader, criterion):
        # 评估量化模型性能
    
    def print_quantization_info(self):
        # 打印量化信息
```

**PTQ 完整流程**：

```
1. 加载预训练模型（FP32）
   ↓
2. prepare(): 准备模型
   - 复制模型
   - 融合模块（Conv+BN+ReLU）
   - 插入伪量化层
   - 复制权重
   ↓
3. calibrate(): 校准
   - 前向传播校准数据
   - 收集激活值统计信息
   - 计算量化参数（scale, zero_point）
   ↓
4. freeze_observer(): 冻结观察器
   - 停止收集统计信息
   - 固定量化参数
   ↓
5. quantize_weights(): 量化权重
   - 量化并反量化权重
   - 模拟量化误差
   ↓
6. convert(): 转换模型
   - 禁用伪量化
   - 得到最终量化模型
   ↓
7. export(): 导出模型（可选）
   - 保存量化模型和参数
```

**性能指标**：

- **模型大小**: INT8 量化后减少 75%（FP32 → INT8）
- **推理速度**: 提升 2-4 倍（取决于硬件）
- **内存占用**: 减少 75%
- **带宽需求**: 减少 75%

**精度损失**（典型）:

- **INT8 PTQ**: 1-3% top-1 精度损失
- **INT4 PTQ**: 5-15% top-1 精度损失
- **INT8 QAT**: <1% top-1 精度损失

**最佳实践**：

- **校准数据选择**: 100-1000 个样本通常足够，应覆盖模型的输入分布
- **INT8**: 推荐用于大多数场景，精度损失小
- **INT4**: 仅在对内存/速度要求极高时使用
- **对称量化**: 适用于激活值分布对称的场景
- **非对称量化**: 适用于激活值分布不对称（如 ReLU 后）
- **逐通道**: 推荐启用，精度更高
- **模块融合**: Conv+BN+ReLU 融合可提升精度和速度

**使用方式**：

```python
# 方式 1: 标准流程
from parascale import PostTrainingQuantization, QuantizationConfig

config = QuantizationConfig(mode="ptq", bits=8, calib_batches=100)
ptq = PostTrainingQuantization(model, config)
ptq.prepare()
ptq.calibrate(calib_loader)
quantized_model = ptq.convert()

# 方式 2: 便捷函数
from parascale import ptq_quantize
quantized_model = ptq_quantize(model, config, calib_loader)
```

**代码统计**：

| 文件                            | 行数         | 类型     |
| ----------------------------- | ---------- | ------ |
| parascale/quantization/ptq.py | 662        | 核心实现   |
| tests/test\_ptq.py            | 468        | 单元测试   |
| examples/ptq\_example.py      | 465        | 使用示例   |
| **总计**                        | **\~1595** | <br /> |

### 6.6 量化模块架构分析

**核心类关系图**：

```
QuantizationConfig (配置)
        ↓
   BaseObserver (基类)
        ↓
┌───────┴────────┐
│                │
MinMaxObserver  MovingAverageObserver
│                │
└───────┬────────┘
        ↓
   FakeQuantize (伪量化层)
        ↓
┌───────┴────────┐
│                │
FakeQuantizedLinear  FakeQuantizedConv2d
        ↓
QuantizationAwareTraining (QAT 管理)
        ↓
PostTrainingQuantization (PTQ 管理)
```

**伪量化层工作流程**：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 1. 收集统计信息（训练时）
    if self.observer_enabled:
        self.observer.update(x.detach())
    
    # 2. 计算量化参数
    if self.training:
        scale, zero_point = self.observer.calculate_qparams()
        self.scale = scale
        self.zero_point = zero_point
    
    # 3. 执行伪量化（量化 - 反量化）
    return self._fake_quantize(x, self.scale, self.zero_point)

def _fake_quantize(self, x, scale, zero_point):
    # 量化：x_q = round((x - zero_point) / scale)
    x_quant = torch.round(x / scale + zero_point)
    # 钳制：x_q = clamp(x_q, qmin, qmax)
    x_quant = torch.clamp(x_quant, qmin, qmax)
    # 反量化：x_dq = x_q * scale + zero_point
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant
```

**观察器实现**：

- **MinMaxObserver**: 记录最小/最大值，逐通道或逐张量统计
- **MovingAverageObserver**: 使用移动平均平滑统计，适合动态范围

**量化参数计算**：

```python
def calculate_scale_zero_point(min_val, max_val, config):
    if config.scheme == "symmetric":
        max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
        scale = max_abs / qmax
        zero_point = torch.zeros_like(scale)
    else:
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
```

### 6.7 PTQ vs QAT 对比

| 特性    | PTQ  | QAT  |
| ----- | ---- | ---- |
| 训练需求  | 无需训练 | 需要训练 |
| 实现复杂度 | 简单   | 复杂   |
| 精度    | 可能较低 | 较高   |
| 适用场景  | 快速部署 | 精度敏感 |
| 校准数据  | 需要   | 不需要  |

***

## 7. 分布式训练设计

### 7.1 环境检测

**自动检测的环境**：

- torchrun：通过 RANK、WORLD\_SIZE 环境变量
- SLURM：通过 SLURM\_PROCID、SLURM\_NTASKS 环境变量
- OpenMPI：通过 OMPI\_COMM\_WORLD\_RANK 环境变量
- MPICH：通过 PMI\_RANK 环境变量

**代码结构**：

```python
def detect_torchrun_environment():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return rank, world_size, local_rank, master_addr, master_port
    return None

def detect_slurm_environment():
    if "SLURM_PROCID" in os.environ:
        return rank, world_size, local_rank, master_addr, master_port
    return None
```

### 7.2 初始化流程

```python
def initialize_distributed(backend=None, rank=None, world_size=None, ...):
    # 1. 检测环境（torchrun > SLURM > MPI）
    # 2. 设置环境变量（RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT）
    # 3. 初始化进程组（dist.init_process_group）
    # 4. 设置 CUDA 设备（torch.cuda.set_device）
```

### 7.3 多节点信息

**工具函数**：

- `get_rank()`：获取全局 rank
- `get_world_size()`：获取进程总数
- `get_local_rank()`：获取本地 rank（当前节点内的 GPU 编号）
- `get_node_rank()`：获取节点编号
- `get_num_nodes()`：获取节点总数

***

## 8. 配置系统设计

### 8.1 设计原则

- 使用 dataclass 简化配置定义
- 自动参数验证
- 支持配置更新和序列化

### 8.2 验证机制

```python
@dataclass
class ParaScaleConfig:
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        # 验证并行大小参数
        # 验证 ZeRO 阶段
        # 验证训练参数
```

### 8.3 序列化

```python
def to_dict(self) -> Dict[str, Any]:
    # 将配置转换为字典

@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> "ParaScaleConfig":
    # 从字典创建配置实例
```

***

## 9. 测试设计

### 9.1 测试分类

**单元测试**：

- 配置测试（test\_refactoring.py）
- 量化测试（test\_quantization.py）
- 多节点测试（test\_multi\_node.py）

**集成测试**：

- 并行策略测试（test\_all\_parallel.py）
- 引擎测试（test\_engine.py）

### 9.2 测试覆盖

| 模块     | 测试文件                      | 覆盖内容                                           |
| ------ | ------------------------- | ---------------------------------------------- |
| 配置     | test\_refactoring.py      | ParaScaleConfig, QuantizationConfig            |
| 并行策略   | test\_all\_parallel.py    | DataParallel, TensorParallel, PipelineParallel |
| 3D混合并行 | test\_hybrid\_parallel.py | HybridParallel, 2D/3D并行组合                      |
| 量化     | test\_quantization.py     | FakeQuantize, Observers, QAT                   |
| PTQ量化  | test\_ptq.py              | PostTrainingQuantization, ptq\_quantize        |
| 分布式    | test\_multi\_node.py      | initialize\_distributed, 环境检测                  |
| 引擎     | test\_engine.py           | ParaScaleEngine                                |
| 智能引擎   | test\_para\_engine.py     | ParaEngine, 自动策略选择                             |

***

## 9. 部署计划

### 9.1 单节点部署

```bash
# 安装
pip install -e .

# 运行基础并行示例
torchrun --nproc_per_node=4 examples/basic_parallel_examples.py --example 1

# 运行量化示例
torchrun --nproc_per_node=2 examples/quantization_examples.py --example 1
```

### 9.2 多节点部署

```bash
# Node 0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
    --master_addr=<ip> --master_port=29500 \
    examples/basic_parallel_examples.py --example 1

# Node 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
    --master_addr=<ip> --master_port=29500 \
    examples/basic_parallel_examples.py --example 1
```

### 9.3 SLURM 部署

```bash
# 提交作业
sbatch --nodes=2 --gpus-per-node=4 multi_node_job.sh
```

***

## 10. 总结

ParaScale 框架采用模块化设计，主要包含以下核心组件：

1. **ParaScaleEngine**：核心引擎，协调训练过程
2. **并行策略**：数据并行、模型并行、张量并行、流水线并行
3. **量化训练**：量化感知训练，支持 INT8/INT4 量化
4. **分布式工具**：自动检测和初始化多种分布式环境
5. **配置系统**：灵活的 dataclass 配置管理

设计特点：

- 易用性：简洁的 API，与 PyTorch 无缝集成
- 灵活性：支持多种并行策略的组合使用，包括3D混合并行
- 可扩展性：模块化设计，易于扩展
- 性能：高效的分布式通信
- 类型安全：完整的类型注解支持

