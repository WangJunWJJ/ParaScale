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

[Example 014](example_014_clip_strong_scaling/README.md) qualifies real CLIP
single-/four-GPU strong scaling at a fixed global batch, with a native-HF numerical
gate, direct PyTorch controls and optional bounded profiling.

[Example 015](example_015_clip_retrieval_quality/README.md) prepares pinned,
deduplicated image/text splits and validates real training, checkpoint reload and
held-out multi-positive retrieval through the product inference adapter.

[Example 016](example_016_diffusion_inference/README.md) runs local Stable
Diffusion text-to-image generation with bounded PNG/manifest output, explicit
sampling budgets and optional UNet compilation. Performance and image-quality
acceptance require pretrained-model evidence beyond the small component tests.
