# Ascend Examples

Ascend examples use the shared ParaScale runtime with NPU/HCCL hardware hints.
They are examples of configuration and launch style, not separate framework code.

Every example provides `run.sh`; distributed examples launch HCCL through
`torchrun` while using the same ParaScale CLI.
