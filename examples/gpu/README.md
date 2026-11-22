# GPU Examples

GPU examples use the shared ParaScale runtime with CUDA/NCCL hardware hints.
They are examples of configuration and launch style, not separate framework code.

Examples provide launch instructions and JSON configuration; model and dataset
paths are explicit or bound through environment variables.

[Example 012](example_012_causal_lm_training/README.md) prepares immutable token
data and runs pretrained causal-LM training through the product CLI, with
environment-bound model/data/output paths and 1/2/4-process launch options.

[Example 013](example_013_causal_lm_inference/README.md) runs local Llama generation
and current product PTQ/QAT artifacts through `parascale infer`, with bounded
JSONL input/output and explicit eager or static-cache compilation policies.
