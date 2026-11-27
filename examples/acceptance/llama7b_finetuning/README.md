# LLaMA-2-7B微调

对应TRN-01/02/03；当前`llama7b_finetuning.quality`返回BLOCKED。

正式资产固定为有权使用的`meta-llama/Llama-2-7b-hf`，TinyLlama或SmolLM不能替代。必须先冻结训练/验证数据、可训练参数、预算、精度和质量阈值，再与合理优化的AdamW对照。

显存指标固定为优化器实际更新后的完整CUDA持久状态节省至少40%，包括尺度、补偿等状态且存储去重；不以整体显存、offload或增加分片数替代。需同时记录质量、恢复和端到端开销。

已有训练入口参考[因果LM训练](../../gpu/example_012_causal_lm_training/README.md)。当前目录只登记未闭合要求，不提前宣称该指标已实现。
