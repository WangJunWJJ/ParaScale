# GPU Examples

GPU examples use the shared ParaScale runtime with CUDA/NCCL hardware hints.
They are examples of configuration and launch style, not separate framework code.

Every example provides `run.sh`; model and dataset paths remain in `config.json`.

[Example 012](example_012_causal_lm_training/README.md) prepares immutable token
data and runs pretrained causal-LM training through the product CLI, with
environment-bound model/data/output paths and 1/2/4-process launch options.
