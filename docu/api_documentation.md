# ParaScale API 文档

**版本**: v0.1.0
**日期**: 2026-03-19  
**状态**: 已更新

本文档详细描述了 ParaScale 框架的 API 接口。

---

## 版本历史

| 版本 | 日期 | API变更 |
|------|------|---------|
| v0.1.0 | 2026-03-19 | 初始版本API |

**历史API变更参考**: 早期版本的API变更记录已归档保留。

---

## 目录

1. [配置模块](#1-配置模块)
2. [核心引擎](#2-核心引擎)
3. [并行策略](#3-并行策略)
4. [优化器](#4-优化器)
5. [量化训练](#5-量化训练)
6. [分布式工具](#6-分布式工具)
7. [工具函数](#7-工具函数)

---

## 1. 配置模块

### 1.1 QuantizationConfig

`QuantizationConfig` 是量化配置类，用于配置量化感知训练（QAT）和训练后量化（PTQ）的参数。

#### 构造函数

```python
QuantizationConfig(
    enabled: bool = False,
    mode: Literal["qat", "ptq"] = "qat",
    bits: int = 8,
    scheme: Literal["symmetric", "asymmetric"] = "symmetric",
    per_channel: bool = True,
    observer_type: Literal["minmax", "moving_average"] = "minmax",
    moving_average_ratio: float = 0.9,
    fuse_modules: bool = True,
    qat_epochs: int = 10,
    calib_batches: int = 100,
    backend: Literal["fbgemm", "qnnpack"] = "fbgemm",
    quantizable_layers: Optional[List[str]] = None
)
```

**参数：**
- `enabled`：是否启用量化，默认为 False
- `mode`：量化模式，支持 "qat"（量化感知训练）或 "ptq"（训练后量化），默认为 "qat"
- `bits`：量化位数，支持 8 或 4，默认为 8
- `scheme`：量化方案，"symmetric"（对称）或 "asymmetric"（非对称），默认为 "symmetric"
- `per_channel`：是否逐通道量化，默认为 True
- `observer_type`：观察器类型，"minmax" 或 "moving_average"，默认为 "minmax"
- `moving_average_ratio`：移动平均比例，仅用于 moving_average 观察器，默认为 0.9
- `fuse_modules`：是否融合模块（Conv+BN+ReLU），默认为 True
- `qat_epochs`：QAT 训练轮数，默认为 10
- `calib_batches`：PTQ 校准批次数量，默认为 100
- `backend`：量化后端，"fbgemm"（x86）或 "qnnpack"（ARM），默认为 "fbgemm"
- `quantizable_layers`：需要量化的层类型列表，默认为 ["Conv2d", "Linear", "ConvTranspose2d"]

#### 方法

##### get_qmin_qmax

```python
get_qmin_qmax() -> tuple
```

**功能：** 获取量化的最小值和最大值

**返回值：**
- 元组 (qmin, qmax)

##### to_dict

```python
to_dict() -> dict
```

**功能：** 将配置转换为字典

**返回值：**
- 配置字典

##### from_dict

```python
@classmethod
from_dict(config_dict: dict) -> QuantizationConfig
```

**功能：** 从字典创建配置

**参数：**
- `config_dict`：配置字典

**返回值：**
- QuantizationConfig 实例

**示例：**
```python
config_dict = {
    "enabled": True,
    "mode": "ptq",
    "bits": 8,
    "scheme": "symmetric"
}
config = QuantizationConfig.from_dict(config_dict)
```

### 1.2 ParaScaleConfig

`ParaScaleConfig` 是 ParaScale 框架的主配置类，使用 dataclass 实现。

#### 构造函数

```python
ParaScaleConfig(
    data_parallel_size: int = 1,
    model_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    tensor_parallel_mode: Literal["row", "column"] = "row",
    pipeline_parallel_size: int = 1,
    pipeline_parallel_chunks: int = 1,
    zero_optimization: bool = False,
    zero_stage: int = 0,
    zero_offload: bool = False,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-3,
    checkpoint_save_path: str = "./checkpoints",
    checkpoint_save_interval: int = 1000,
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
)
```

**参数：**
- `data_parallel_size`：数据并行大小，默认为 1
- `model_parallel_size`：模型并行大小，默认为 1
- `tensor_parallel_size`：张量并行大小，默认为 1
- `tensor_parallel_mode`：张量并行模式，"row" 或 "column"，默认为 "row"
- `pipeline_parallel_size`：流水线并行大小，默认为 1
- `pipeline_parallel_chunks`：流水线并行分块数，默认为 1
- `zero_optimization`：是否启用 ZeRO 优化器，默认为 False
- `zero_stage`：ZeRO 优化器阶段（0, 1, 2, 3），默认为 0
- `zero_offload`：是否启用 ZeRO offload，默认为 False
- `batch_size`：批次大小，默认为 32
- `gradient_accumulation_steps`：梯度累积步数，默认为 1
- `learning_rate`：学习率，默认为 1e-3
- `checkpoint_save_path`：检查点保存路径，默认为 "./checkpoints"
- `checkpoint_save_interval`：检查点保存间隔（步数），默认为 1000
- `quantization`：量化配置，默认为 QuantizationConfig()

#### 方法

##### update

```python
update(config_dict: Dict[str, Any]) -> "ParaScaleConfig"
```

**功能：** 从字典更新配置

**参数：**
- `config_dict`：包含配置项的字典

**返回值：**
- 更新后的配置实例

##### to_dict

```python
to_dict() -> Dict[str, Any]
```

**功能：** 将配置转换为字典

**返回值：**
- 包含所有配置项的字典

##### from_dict

```python
from_dict(config_dict: Dict[str, Any]) -> "ParaScaleConfig"
```

**功能：** 从字典创建配置实例（类方法）

**参数：**
- `config_dict`：包含配置项的字典

**返回值：**
- 新创建的配置实例

---

## 2. 核心引擎

### 2.1 Engine

`Engine` 是 ParaScale 框架的核心引擎类，负责协调各种并行策略和训练过程。

#### 构造函数

```python
Engine(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: Optional[ParaScaleConfig] = None,
    auto_init_distributed: bool = True
)
```

**参数：**
- `model`：PyTorch 模型实例
- `optimizer`：PyTorch 优化器实例
- `config`：ParaScaleConfig 配置实例，如果为 None 则使用默认配置
- `auto_init_distributed`：是否自动初始化分布式环境，默认为 True

**成员变量：**
- `model`：PyTorch 模型实例
- `optimizer`：优化器实例
- `config`：ParaScaleConfig 配置实例
- `rank`：当前进程的 rank
- `world_size`：世界大小（进程总数）
- `local_rank`：本地 rank（当前节点内的 GPU 编号）
- `node_rank`：节点编号
- `num_nodes`：节点总数
- `data_parallel`：数据并行实例
- `model_parallel`：模型并行实例
- `tensor_parallel`：张量并行实例
- `pipeline_parallel`：流水线并行实例
- `qat_handler`：量化感知训练处理器
- `is_quantized`：是否启用量化
- `global_step`：全局训练步数

#### 方法

##### train

```python
train(dataloader: Any, epochs: int = 1) -> None
```

**功能：** 训练模型

**参数：**
- `dataloader`：数据加载器
- `epochs`：训练轮数，默认为 1

##### evaluate

```python
evaluate(dataloader: Any) -> Tuple[float, float]
```

**功能：** 评估模型

**参数：**
- `dataloader`：数据加载器

**返回值：**
- 元组 (平均损失, 准确率百分比)

##### save_checkpoint

```python
save_checkpoint(save_path: Optional[str] = None) -> None
```

**功能：** 保存检查点

**参数：**
- `save_path`：检查点保存路径，如果为 None 则使用配置中的路径

##### load_checkpoint

```python
load_checkpoint(checkpoint_path: str) -> None
```

**功能：** 加载检查点

**参数：**
- `checkpoint_path`：检查点文件路径

##### freeze_quantization_observer

```python
freeze_quantization_observer() -> None
```

**功能：** 冻结量化观察器，停止收集统计信息

##### unfreeze_quantization_observer

```python
unfreeze_quantization_observer() -> None
```

**功能：** 解冻量化观察器，恢复收集统计信息

##### export_quantized_model

```python
export_quantized_model(save_path: str) -> None
```

**功能：** 导出量化模型

**参数：**
- `save_path`：保存路径

##### get_quantization_info

```python
get_quantization_info() -> dict
```

**功能：** 获取量化信息

**返回值：**
- 量化信息字典

---

## 3. 并行策略

### 3.1 BaseParallel

`BaseParallel` 是并行策略的抽象基类，定义了所有并行策略必须实现的接口。

#### 构造函数

```python
BaseParallel(model: nn.Module, rank: int, world_size: int)
```

**参数：**
- `model`：PyTorch 模型实例
- `rank`：当前进程的 rank
- `world_size`：世界大小

**成员变量：**
- `model`：PyTorch 模型实例
- `rank`：当前进程的 rank
- `world_size`：世界大小
- `device`：当前设备

#### 方法

##### forward

```python
forward(inputs: torch.Tensor) -> Optional[torch.Tensor]
```

**功能：** 前向传播（抽象方法，子类必须实现）

**参数：**
- `inputs`：输入数据张量

**返回值：**
- 模型输出张量

##### gather_gradients

```python
gather_gradients() -> None
```

**功能：** 收集梯度（默认空实现，子类可覆盖）

##### to_device

```python
to_device(tensor: torch.Tensor) -> torch.Tensor
```

**功能：** 将张量移动到当前设备

**参数：**
- `tensor`：要移动的张量

**返回值：**
- 移动到当前设备后的张量

### 3.2 DataParallel

`DataParallel` 实现数据并行策略，将数据分割到不同 GPU，每个 GPU 运行完整的模型副本。

#### 构造函数

```python
DataParallel(model: nn.Module, rank: int, world_size: int)
```

#### 方法

##### broadcast_model

```python
broadcast_model() -> None
```

**功能：** 广播模型参数，将 rank 0 的模型参数广播到所有其他进程

##### gather_gradients

```python
gather_gradients() -> None
```

**功能：** 收集并平均梯度，使用 all-reduce 操作

##### prepare_dataloader

```python
prepare_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader
```

**功能：** 准备分布式数据加载器

**参数：**
- `dataset`：数据集实例
- `batch_size`：批次大小
- `shuffle`：是否打乱数据，默认为 True
- `num_workers`：数据加载线程数，默认为 0

**返回值：**
- 配置好的分布式数据加载器

##### forward

```python
forward(inputs: torch.Tensor) -> torch.Tensor
```

**功能：** 前向传播

### 3.3 ModelParallel

`ModelParallel` 实现模型并行策略，将模型分割到不同设备。

#### 构造函数

```python
ModelParallel(model: nn.Module, rank: int, world_size: int)
```

#### 方法

##### _split_model

```python
_split_model() -> None
```

**功能：** 将模型分割到不同设备，支持 layers、encoder/decoder 和普通模型结构

##### forward

```python
forward(inputs: torch.Tensor) -> Optional[torch.Tensor]
```

**功能：** 前向传播，按顺序执行各个阶段，通过点对点通信传递中间结果

### 3.4 TensorParallel

`TensorParallel` 实现张量并行策略，将模型的张量（权重矩阵）分割到不同 GPU。

#### 构造函数

```python
TensorParallel(
    model: nn.Module,
    rank: int,
    world_size: int,
    mode: Literal["row", "column"] = "row"
)
```

**参数：**
- `mode`：张量并行模式，"row"（行并行）或 "column"（列并行），默认为 "row"

**成员变量：**
- `mode`：张量并行模式
- `parallel_layers`：已并行化的层名称列表

#### 方法

##### _parallelize_model

```python
_parallelize_model() -> None
```

**功能：** 并行化模型中的线性层

##### _parallelize_linear

```python
_parallelize_linear(linear_layer: nn.Linear) -> nn.Linear
```

**功能：** 并行化单个线性层

##### _parallelize_row

```python
_parallelize_row(linear_layer: nn.Linear) -> nn.Linear
```

**功能：** 行并行，按输出维度分割权重

##### _parallelize_column

```python
_parallelize_column(linear_layer: nn.Linear) -> nn.Linear
```

**功能：** 列并行，按输入维度分割权重

##### forward

```python
forward(inputs: torch.Tensor) -> torch.Tensor
```

**功能：** 前向传播，根据并行模式执行 all-gather 或 all-reduce

### 3.5 PipelineParallel

`PipelineParallel` 实现流水线并行策略，将模型按层分割到不同设备。

#### 构造函数

```python
PipelineParallel(
    model: nn.Module,
    rank: int,
    world_size: int,
    chunks: int = 1
)
```

**参数：**
- `chunks`：微批次数量，默认为 1

**成员变量：**
- `chunks`：微批次数量
- `stage_layers`：当前阶段的层
- `is_first`：是否为第一个阶段
- `is_last`：是否为最后一个阶段

#### 方法

##### _partition_model

```python
_partition_model() -> None
```

**功能：** 将模型分割到不同的流水线阶段

##### _extract_layers

```python
_extract_layers() -> List[nn.Module]
```

**功能：** 从模型中提取所有层，支持多种模型结构

**返回值：**
- 层列表

##### forward

```python
forward(input_data: torch.Tensor) -> Optional[torch.Tensor]
```

**功能：** 前向传播，支持单批次和微批次模式

##### _forward_single_batch

```python
_forward_single_batch(input_data: torch.Tensor) -> Optional[torch.Tensor]
```

**功能：** 单批次前向传播

##### _forward_micro_batches

```python
_forward_micro_batches(input_data: torch.Tensor) -> Optional[torch.Tensor]
```

**功能：** 微批次前向传播

##### _send_tensor

```python
_send_tensor(tensor: torch.Tensor, dst: int) -> None
```

**功能：** 发送张量到目标 rank（支持动态形状）

##### _recv_tensor

```python
_recv_tensor(src: int) -> torch.Tensor
```

**功能：** 从源 rank 接收张量（支持动态形状）

**返回值：**
- 接收到的张量

##### get_stage_model

```python
get_stage_model() -> nn.Module
```

**功能：** 获取当前阶段的模型

**返回值：**
- 当前阶段的模型（nn.Sequential）

### 3.6 HybridParallel

`HybridParallel` 实现3D混合并行策略，将数据并行、张量并行和流水线并行组合在一起。

#### 构造函数

```python
HybridParallel(
    model: nn.Module,
    rank: int,
    world_size: int,
    dp_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    tensor_parallel_mode: str = "row",
    pipeline_chunks: int = 1
)
```

**参数：**
- `model`：PyTorch 模型实例
- `rank`：当前进程的全局rank
- `world_size`：总进程数
- `dp_size`：数据并行大小，默认为 1
- `tp_size`：张量并行大小，默认为 1
- `pp_size`：流水线并行大小，默认为 1
- `tensor_parallel_mode`：张量并行模式，"row" 或 "column"，默认为 "row"
- `pipeline_chunks`：流水线微批次数量，默认为 1

**约束条件：**
- `dp_size × tp_size × pp_size` 必须等于 `world_size`

**成员变量：**
- `dp_size`：数据并行大小
- `tp_size`：张量并行大小
- `pp_size`：流水线并行大小
- `dp_rank`：数据并行组内的rank
- `tp_rank`：张量并行组内的rank
- `pp_rank`：流水线并行组内的rank（即stage id）
- `dp_group`：数据并行进程组
- `tp_group`：张量并行进程组
- `pp_group`：流水线并行进程组
- `is_first_stage`：是否为流水线第一个阶段
- `is_last_stage`：是否为流水线最后一个阶段
- `stage_layers`：当前流水线阶段的层
- `tensor_parallel_mode`：张量并行模式

#### 方法

##### forward

```python
forward(inputs: torch.Tensor) -> Optional[torch.Tensor]
```

**功能：** 3D混合并行前向传播

**执行流程：**
1. 如果是第一个流水线阶段，接收输入数据
2. 如果是张量并行模式为column，先切分输入
3. 执行当前阶段的层
4. 如果是张量并行模式为row，all-gather输出
5. 如果不是最后一个流水线阶段，发送给下一阶段

**参数：**
- `inputs`：输入数据张量

**返回值：**
- 模型输出张量（仅最后一个流水线阶段返回非None）

##### gather_gradients

```python
gather_gradients() -> None
```

**功能：** 收集梯度（用于数据并行）

**说明：** 在数据并行组内对梯度进行all-reduce并平均

##### broadcast_model

```python
broadcast_model() -> None
```

**功能：** 广播模型参数（用于数据并行）

**说明：** 将rank 0的模型参数广播到所有其他进程

##### get_stage_model

```python
get_stage_model() -> nn.Module
```

**功能：** 获取当前流水线阶段的模型

**返回值：**
- 当前阶段的模型（nn.Sequential）

##### get_parallel_info

```python
get_parallel_info() -> Dict[str, Any]
```

**功能：** 获取并行配置信息

**返回值：**
- 包含以下键的字典：
  - `global_rank`：全局rank
  - `world_size`：总进程数
  - `dp_size`、`tp_size`、`pp_size`：并行大小
  - `dp_rank`、`tp_rank`、`pp_rank`：各维度rank
  - `is_first_stage`、`is_last_stage`：阶段信息
  - `tensor_parallel_mode`：张量并行模式
  - `pipeline_chunks`：微批次数量

---

## 4. 优化器

### 4.1 ZeroOptimizer

`ZeroOptimizer` 是 ZeRO（Zero Redundancy Optimizer）优化器的实现。

#### 构造函数

```python
ZeroOptimizer(
    model: nn.Module,
    optimizer: optim.Optimizer,
    stage: int = 0,
    offload: bool = False
)
```

**参数：**
- `model`：PyTorch 模型实例
- `optimizer`：基础优化器
- `stage`：ZeRO 阶段（0, 1, 2, 3），默认为 0
- `offload`：是否启用 offload，默认为 False

#### 方法

##### step

```python
step() -> None
```

**功能：** 执行优化步骤

##### zero_grad

```python
zero_grad() -> None
```

**功能：** 清零梯度

---

## 5. 量化训练

### 5.1 QuantizationAwareTraining

`QuantizationAwareTraining` 是量化感知训练的管理类。

#### 构造函数

```python
QuantizationAwareTraining(model: nn.Module, config: QuantizationConfig)
```

**参数：**
- `model`：原始模型
- `config`：QuantizationConfig 实例

**成员变量：**
- `model`：原始模型
- `config`：量化配置
- `prepared_model`：准备好的模型

#### 方法

##### prepare

```python
prepare() -> nn.Module
```

**功能：** 准备模型进行 QAT

**返回值：**
- 准备好的模型

##### freeze_observer

```python
freeze_observer() -> None
```

**功能：** 冻结观察器

##### unfreeze_observer

```python
unfreeze_observer() -> None
```

**功能：** 解冻观察器

##### enable_fake_quant

```python
enable_fake_quant(enabled: bool = True) -> None
```

**功能：** 启用/禁用伪量化

##### convert

```python
convert() -> nn.Module
```

**功能：** 转换模型为真正的量化模型

**返回值：**
- 量化后的模型

##### get_quantization_params

```python
get_quantization_params() -> Dict[str, Any]
```

**功能：** 获取所有层的量化参数

**返回值：**
- 量化参数字典

### 5.2 FakeQuantize

`FakeQuantize` 是伪量化层，在训练过程中模拟量化-反量化过程。

#### 构造函数

```python
FakeQuantize(config: QuantizationConfig)
```

**成员变量：**
- `config`：量化配置
- `observer`：观察器实例
- `scale`：量化缩放因子
- `zero_point`：量化零点
- `fake_quant_enabled`：是否启用伪量化
- `observer_enabled`：是否启用观察器

#### 方法

##### forward

```python
forward(x: torch.Tensor) -> torch.Tensor
```

**功能：** 前向传播，执行伪量化操作

##### enable_fake_quant

```python
enable_fake_quant(enabled: bool = True) -> None
```

**功能：** 启用/禁用伪量化

##### enable_observer

```python
enable_observer(enabled: bool = True) -> None
```

**功能：** 启用/禁用观察器

##### calculate_qparams

```python
calculate_qparams() -> tuple
```

**功能：** 计算量化参数

### 5.3 观察器

#### MinMaxObserver

```python
MinMaxObserver(config: QuantizationConfig)
```

**功能：** 记录观察到的最小值和最大值

#### MovingAverageObserver

```python
MovingAverageObserver(config: QuantizationConfig)
```

**功能：** 使用移动平均来平滑观察到的数值范围

### 5.4 便捷函数

#### prepare_qat_model

```python
prepare_qat_model(model: nn.Module, config: QuantizationConfig) -> nn.Module
```

#### convert_qat_model

```python
convert_qat_model(model: nn.Module, config: QuantizationConfig) -> nn.Module
```

---

## 6. 分布式工具

### 6.1 initialize_distributed

```python
initialize_distributed(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
    master_addr: Optional[str] = None,
    master_port: Optional[int] = None,
    timeout: Optional[int] = None
) -> Tuple[int, int, int]
```

**功能：** 自动初始化分布式训练环境

**自动检测的环境：**
- torchrun（通过 RANK、WORLD_SIZE 环境变量）
- SLURM（通过 SLURM_PROCID、SLURM_NTASKS 环境变量）
- OpenMPI（通过 OMPI_COMM_WORLD_RANK 环境变量）
- MPICH（通过 PMI_RANK 环境变量）

**返回值：**
- 元组 (rank, world_size, local_rank)

### 6.2 cleanup_distributed

```python
cleanup_distributed() -> None
```

**功能：** 清理分布式环境，销毁进程组

### 6.3 get_distributed_info

```python
get_distributed_info() -> dict
```

**功能：** 获取分布式环境信息

**返回值：**
- 包含 initialized、backend、rank、world_size、local_rank、master_addr、master_port、environment 的字典

### 6.4 print_distributed_info

```python
print_distributed_info() -> None
```

**功能：** 打印分布式环境信息（只在 rank 0 上打印）

---

## 7. 工具函数

### 7.1 日志和打印

#### setup_logging

```python
setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None
```

**功能：** 配置日志系统

#### print_rank_0

```python
print_rank_0(msg: str, rank: int = 0) -> None
```

**功能：** 仅在指定 rank 上打印消息

### 7.2 分布式信息

#### get_rank

```python
get_rank() -> int
```

**功能：** 获取当前进程的 rank

**返回值：**
- 当前进程的 rank，如果分布式未初始化返回 0

#### get_world_size

```python
get_world_size() -> int
```

**功能：** 获取世界大小（进程总数）

**返回值：**
- 进程总数，如果分布式未初始化返回 1

#### get_local_rank

```python
get_local_rank() -> int
```

**功能：** 获取本地 rank（当前节点内的 GPU 编号）

**返回值：**
- 本地 rank

#### get_node_rank

```python
get_node_rank() -> int
```

**功能：** 获取当前节点的编号

**返回值：**
- 节点编号

#### get_num_nodes

```python
get_num_nodes() -> int
```

**功能：** 获取节点总数

**返回值：**
- 节点总数

#### is_main_process

```python
is_main_process() -> bool
```

**功能：** 判断当前进程是否为主进程

**返回值：**
- 如果当前进程是主进程（rank 0）返回 True

### 7.3 其他工具

#### ensure_directory

```python
ensure_directory(directory: str) -> None
```

**功能：** 确保目录存在，如果不存在则创建

#### barrier
```python
barrier() -> None
```
**功能：** 同步所有进程

---

## 7. 实时硬件监控

### 7.1 RealTimeHardwareMonitor

`RealTimeHardwareMonitor` 是实时硬件监控器，在训练过程中持续监控 GPU 内存、计算能力、通信带宽等硬件状态，为性能优化和资源管理提供实时数据支持。

#### 构造函数
```python
RealTimeHardwareMonitor(
    local_rank: int = 0,
    max_history_size: int = 1000,
    bandwidth_test_interval: float = 60.0,
    compute_check_interval: float = 30.0
) -> RealTimeHardwareMonitor
```

**参数：**
- `local_rank`：本地 rank，默认为 0
- `max_history_size`：最大历史记录数量，默认为 1000
- `bandwidth_test_interval`：带宽测试间隔（秒），默认为 60.0
- `compute_check_interval`：计算能力检查间隔（秒），默认为 30.0

**成员变量：**
- `local_rank`：本地 rank
- `metrics_history`：指标历史记录列表
- `max_history_size`：最大历史记录数量
- `last_bandwidth_check`：上次带宽检查时间
- `bandwidth_test_interval`：带宽测试间隔
- `last_compute_check`：上次计算能力检查时间
- `compute_check_interval`：计算能力检查间隔
- `start_time`：监控开始时间

#### collect_metrics
```python
collect_metrics() -> HardwareMetrics
```

**功能：** 收集当前硬件指标

**返回值：**
- 当前硬件指标（HardwareMetrics 对象）

#### get_average_metrics
```python
get_average_metrics(last_n: int = 100) -> Optional[HardwareMetrics]
```

**功能：** 获取最近 N 个指标的平均值

**参数：**
- `last_n`：要平均的指标数量，默认为 100

**返回值：**
- 平均硬件指标（HardwareMetrics 对象）

#### get_peak_metrics
```python
get_peak_metrics() -> Optional[HardwareMetrics]
```

**功能：** 获取历史记录中的峰值指标

**返回值：**
- 峰值硬件指标（HardwareMetrics 对象）

#### print_metrics_summary
```python
print_metrics_summary() -> None
```

**功能：** 打印硬件监控摘要

**输出：**
- 平均 GPU 内存使用、峰值、利用率、温度
- 平均通信带宽、计算吞吐量
- 峰值统计
- 监控时长和数据点数

#### clear_history
```python
clear_history() -> None
```

**功能：** 清空历史记录

### 7.2 HardwareMetrics

`HardwareMetrics` 是硬件指标数据类，存储单个时间点的硬件状态信息。

#### 构造函数
```python
HardwareMetrics(
    timestamp: float = 0.0,
    gpu_memory_used: int = 0,
    gpu_memory_peak: int = 0,
    gpu_memory_percent: float = 0.0,
    gpu_utilization: float = 0.0,
    gpu_temperature: float = 0.0,
    communication_bandwidth: float = 0.0,
    compute_throughput: float = 0.0
) -> HardwareMetrics
```

**参数：**
- `timestamp`：时间戳，默认为当前时间
- `gpu_memory_used`：已使用的 GPU 内存（字节）
- `gpu_memory_peak`：峰值 GPU 内存（字节）
- `gpu_memory_percent`：内存使用百分比
- `gpu_utilization`：GPU 利用率（0-100）
- `gpu_temperature`：GPU 温度（摄氏度）
- `communication_bandwidth`：通信带宽（GB/s）
- `compute_throughput`：计算吞吐量（TFLOPS）

#### to_dict
```python
to_dict() -> Dict[str, float]
```

**功能：** 转换为字典

**返回值：**
- 包含所有硬件指标的字典，单位已转换（MB、GB/s、TFLOPS）

### 7.3 create_hardware_monitor

`create_hardware_monitor` 是创建实时硬件监控器的便捷函数。

#### 函数签名
```python
create_hardware_monitor(
    local_rank: int = 0,
    max_history_size: int = 1000,
    bandwidth_test_interval: float = 60.0,
    compute_check_interval: float = 30.0
) -> RealTimeHardwareMonitor
```

**参数：**
- `local_rank`：本地 rank，默认为 0
- `max_history_size`：最大历史记录数量，默认为 1000
- `bandwidth_test_interval`：带宽测试间隔（秒），默认为 60.0
- `compute_check_interval`：计算能力检查间隔（秒），默认为 30.0

**返回值：**
- 实时硬件监控器实例

**使用示例：**
```python
from parascale.utils import create_hardware_monitor

# 创建监控器
monitor = create_hardware_monitor(local_rank=0)

# 收集当前指标
metrics = monitor.collect_metrics()

# 打印摘要
monitor.print_metrics_summary()

# 获取平均值
avg_metrics = monitor.get_average_metrics(last_n=100)

# 获取峰值
peak_metrics = monitor.get_peak_metrics()
```
