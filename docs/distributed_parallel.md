# Distributed Parallel Training

ParaScale has two intentionally separate execution paths:

- The workload CLI supports data parallelism and backend-managed state sharding.
  Requests for TP, PP or generic model parallelism now fail explicitly there.
- The experimental Python API accepts an explicitly partitioned stage and a
  model-specific PyTorch tensor-parallel plan. It does not automatically split
  arbitrary CLIP, YOLO, VLM or Hugging Face models.

## Scope

The execution target is one CUDA/NCCL node with at most four GPUs. Install
`pip install -e '.[parallel]'` in an isolated environment. This path pins
PyTorch 2.9.1; older prerelease PyTorch 2.5 builds encountered upstream FSDP2
checkpoint and DDP/pipeline composition failures during development.

| Mode | DP | TP | PP | Data parallel execution |
| --- | ---: | ---: | ---: | --- |
| Single device | 1 | 1 | 1 | none |
| Data parallel | 2 or 4 | 1 | 1 | DDP |
| Tensor parallel | 1 | 2 | 1 | none |
| Pipeline parallel | 1 | 1 | 2 | none |
| DP + TP | 2 | 2 | 1 | FSDP2 sharded DP |
| DP + PP | 2 | 1 | 2 | DDP |
| TP + PP | 1 | 2 | 2 | none |

These are the admission/test matrix, not a blanket compatibility guarantee for
all models. Keep run results in the selected output directory. Full
DP2 x TP2 x PP2 requires eight ranks and is rejected, not simulated on four GPUs.
Ascend, multi-node execution, AMP, variable shapes, interleaved stages and
unvalidated topology combinations are outside this implementation's boundary.

## Python API

Initialize `torch.distributed` with NCCL under `torchrun`, set the CUDA device to
`LOCAL_RANK`, and construct only this rank's local stage on that device:

```python
from parascale.parallel import DistributedParallelConfig
from parascale.parallel.execution import DistributedTrainer

config = DistributedParallelConfig(tp_size=2, microbatches=2)
trainer = DistributedTrainer(
    local_stage, config, example_microbatch, mean_loss,
    tensor_plan=model_specific_tp_plan,
)
optimizer = torch.optim.SGD(trainer.parameters(), lr=0.01, foreach=False)
optimizer.zero_grad(set_to_none=True)
result = trainer.forward_backward(local_inputs, local_targets)
optimizer.step()
```

`local_stage`, `example_microbatch`, `mean_loss`, the TP plan and data are supplied
by the caller; this fragment is an integration contract, not a standalone model.
The runnable example below supplies all of them.

- Rank coordinates are `(DP, PP, TP)`, with TP varying fastest. Singleton mesh
  dimensions are omitted for upstream FSDP2 composition.
- Parameters and buffers are synchronized within TP and DP groups before
  partitioning. Each stage's DP=0/TP=0 rank supplies its initial state.
- Use non-fused, `foreach=False` optimizers for mixed Tensor/DTensor parameter
  sets. SGD with momentum is the validated optimizer in this matrix.
- TP/PP peers must process the same local samples; DP replicas process disjoint
  samples. Global batch = DP x microbatches x microbatch size, not world size x
  local batch.
- Inputs and stage parameters are FP32. The scalar loss must be mean-reduced
  over equal-size microbatches with equal denominators. Variable valid-token
  counts, ignored labels and sum-reduced losses require a separate adapter.
- GPipe handles inter-stage forward/backward communication. Loss scaling occurs
  exactly once; the result reports DP-averaged loss on every rank.
- Replicated DDP without TP/PP synchronizes gradients only on the final
  microbatch. Intermediate forward/backward passes use DDP's `no_sync` context.
  FSDP2 and pipeline schedules retain their own synchronization behavior.
- Output tensors are returned only on the last PP stage. Optimizer steps,
  dataloaders, schedulers and process-group teardown remain caller-owned.

## Checkpoints

All ranks call `save_checkpoint(new_directory, optimizer, step=...)` and
`load_checkpoint(directory, optimizer)`. Rank-local PyTorch state includes model,
optimizer and Torch CPU/CUDA RNG. SHA-256-checked files are published before the
final manifest; an existing committed directory cannot be overwritten.
All ranks must agree on the canonical checkpoint directory, topology and step
before saving, and on the directory and complete manifest before loading any
payload. Mixed snapshots are rejected before applying model or optimizer state.

Only trusted local checkpoint files may be loaded. Topology, microbatch count
and rank/device mapping must remain unchanged. This format is not portable
inference export, topology-changing resharding, dataloader state or automatic
fault recovery. Python validation failures are coordinated; a dead process or
failure inside native collectives still requires launcher-level teardown.

## Reproducible Validation

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 bash examples/gpu/example_009_distributed_parallel/run.sh tp2pp2 /tmp/parascale-tp2pp2
```

Use a new output directory per run. Modes: `single`, `dp2`, `dp4`, `tp2`, `pp2`,
`dp2tp2`, `dp2pp2`, `tp2pp2`. Set `MODEL=llama_decoder` for the Hugging Face
decoder-operator diagnostic (Transformers 4.51.3 on the remote reference image).

The validator compares the full-model reference against distributed logits or
hidden outputs, scalar loss, gradients and updated parameters. It also perturbs
weights and clears optimizer momentum after saving, then verifies continuation
after restore. This is not an independent job restart unless explicitly marked
as such in the result artifact.

Both residual-MLP and Llama decoder fixtures use random initialization and
synthetic hidden states. Llama covers attention and MLP TP, but does not include
token embeddings, an LM head, pretrained weights or language-model quality.
Reported timing excludes reference checks, optimizer steps and checkpoints;
memory includes the reference model. Do not interpret these diagnostic numbers
as production end-to-end throughput or framework speedup.

For the fixed-width A6000 fixtures, the [batch-throughput example](../examples/gpu/example_009_distributed_parallel/README.md#a6000-batch-throughput)
provides larger-batch settings and matched single/four-GPU comparison commands.
These settings do not change runtime defaults or establish a batch size for
arbitrary models. Keep measurements, repeat dispersion and source/runtime
provenance alongside the generated results, outside source-code commits.
