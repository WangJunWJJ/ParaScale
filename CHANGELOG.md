# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-21

### Breaking Changes

#### Parallel Strategy Refactoring

The parallel strategy module has been completely refactored to improve maintainability and performance. This release includes significant breaking changes.

**Removed Classes and Modules:**
- `TensorParallelV2` - Removed, use `TensorParallel` instead
- `HybridParallelV2` - Removed, use `HybridParallel` instead
- `tensor_parallel_v2.py` - Module removed
- `hybrid_parallel_v2.py` - Module removed
- `tensor_parallel_unified.py` - Module removed (merged into `tensor_parallel.py`)
- `hybrid_parallel_unified.py` - Module removed (merged into `hybrid_parallel.py`)

**Changed APIs:**

1. **TensorParallel Configuration**
   ```python
   # Before (v0.1.x)
   from parascale.parallel import TensorParallel
   tp = TensorParallel(model, rank=0, world_size=4, mode="column")
   
   # After (v0.2.0)
   from parascale.parallel import TensorParallel, TensorParallelConfig, ParallelStrategy
   config = TensorParallelConfig(tp_size=2, strategy=ParallelStrategy.TRANSFORMER)
   tp = TensorParallel(model, rank=0, world_size=4, config=config)
   ```

2. **HybridParallel Configuration**
   ```python
   # Before (v0.1.x)
   from parascale.parallel import HybridParallel
   hp = HybridParallel(model, rank=0, world_size=8, tensor_parallel_mode="column")
   
   # After (v0.2.0)
   from parascale.parallel import HybridParallel, HybridParallelConfig
   config = HybridParallelConfig(dp_size=2, tp_size=2, pp_size=2)
   hp = HybridParallel(model, rank=0, world_size=8, config=config)
   ```

### Added

- **New Parallel Layer Components:**
  - `ColumnParallelLinear` - Column-wise parallel linear layer
  - `RowParallelLinear` - Row-wise parallel linear layer
  - `VocabParallelEmbedding` - Vocabulary parallel embedding layer
  - `ParallelSelfAttention` - Optimized self-attention for Transformers
  - `ParallelMLP` - Optimized MLP for Transformers

- **Configuration Classes:**
  - `TensorParallelConfig` - Configuration for tensor parallelism
  - `HybridParallelConfig` - Configuration for 3D hybrid parallelism
  - `ParallelStrategy` - Enum for parallel strategies (SIMPLE, TRANSFORMER, AUTO)
  - `PipelineSchedule` - Enum for pipeline schedules (1F1B, FILL_DRAIN, INTERLEAVED)

- **New Utilities:**
  - `TensorParallelConverter` - Automatic model conversion to parallel version
  - `PipelineScheduler` - 1F1B pipeline scheduling implementation
  - `PipelineCommunicator` - Inter-stage communication for pipeline parallelism

### Changed

- **Unified Implementation:**
  - Merged v1 and v2 implementations into single, optimized versions
  - Reference implementation based on Megatron-LM and DeepSpeed best practices
  - Improved communication efficiency with custom autograd functions

- **Architecture Improvements:**
  - Strategy pattern for selecting parallel behavior
  - Cleaner separation of concerns between DP/TP/PP layers
  - Better error handling and validation

### Performance Improvements

- **Communication Optimization:**
  - Fused all-reduce operations in ColumnParallelLinear
  - Optimized scatter/gather for RowParallelLinear
  - Reduced communication overhead in 1F1B scheduling

- **Memory Efficiency:**
  - Better memory layout for parallel layers
  - Reduced activation memory footprint

### Migration Guide

To migrate from v0.1.x to v0.2.0:

1. **Update imports:**
   ```python
   # Remove these imports
   from parascale.parallel import TensorParallelV2, HybridParallelV2
   
   # Use these instead
   from parascale.parallel import TensorParallel, HybridParallel
   ```

2. **Update TensorParallel usage:**
   ```python
   # Old way
   tp = TensorParallel(model, rank, world_size, mode="column")
   
   # New way
   from parascale.parallel import TensorParallelConfig
   config = TensorParallelConfig(tp_size=2, strategy="simple")
   tp = TensorParallel(model, rank, world_size, config=config)
   ```

3. **Update HybridParallel usage:**
   ```python
   # Old way
   hp = HybridParallel(model, rank, world_size, tensor_parallel_mode="column")
   
   # New way
   from parascale.parallel import HybridParallelConfig
   config = HybridParallelConfig(dp_size=2, tp_size=2, pp_size=2)
   hp = HybridParallel(model, rank, world_size, config=config)
   ```

## [0.1.0] - 2026-03-08

### Added

- Initial release of ParaScale framework
- Basic parallel strategies:
  - `DataParallel` - Data parallelism implementation
  - `ModelParallel` - Model parallelism (layer-wise splitting)
  - `PipelineParallel` - Pipeline parallelism with micro-batching
  - `TensorParallel` - Basic tensor parallelism (v1 implementation)
  - `HybridParallel` - Basic 3D parallelism (v1 implementation)

- Communication optimizations:
  - `GradientCompressor` - Base class for gradient compression
  - `TopKCompressor` - Top-K gradient compression
  - `OneBitAdamCompressor` - 1-bit Adam compression
  - `CommunicationOverlap` - Communication-computation overlap
  - `RingAllReduce` - Ring all-reduce implementation

- Optimizers:
  - `ZeROOptimizer` - ZeRO Stage 0/1/2/3 optimizer
  - `FusedOptimizer` - Fused optimizer for mixed precision

- Utilities:
  - `BaseParallel` - Abstract base class for all parallel strategies
  - `ParallelConfig` - Configuration management
  - Checkpoint and recovery utilities

### Notes

- This is the initial release with basic functionality
- TensorParallel and HybridParallel implementations are preliminary
- Known issues with Transformer model compatibility in v1 implementations

[Unreleased]: https://github.com/yourusername/parascale/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yourusername/parascale/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/parascale/releases/tag/v0.1.0
