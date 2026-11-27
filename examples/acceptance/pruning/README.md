# 模型剪枝验收

对应INF-05；当前`pruning.quality`返回BLOCKED。

复用[017 Llama MLP结构剪枝](../../gpu/example_017_llama_pruning/README.md)的公共API。正式比较原始模型、剪枝模型、恢复训练模型，验证实际结构缩小、HF导出重载、独立质量、端到端速度和显存。

剪枝后创建新优化器，不恢复原结构检查点。MLP通道比例不等于整模型参数比例，参数减少不等于推理加速；量化与剪枝组合需另行验证。
