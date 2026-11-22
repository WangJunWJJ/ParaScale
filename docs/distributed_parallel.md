# Distributed Parallel Training

ParaScale distinguishes ordinary data-parallel backends from stage execution:

- Ordinary workload backends support data parallelism and backend-managed state
  sharding. Requests for TP, PP or generic model parallelism fail explicitly there.
- The explicit `native_parallel` CLI backend connects local Llama stages and an
  immutable token corpus to the same stage executor, with bounded update loops,
  complete checkpoint/resume and coordinated rank-local failures.
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
Ascend, multi-node execution, FP16/GradScaler, variable shapes, interleaved stages and
unvalidated topology combinations are outside this implementation's boundary.

## Unified Training CLI

Use `training_backend: native_parallel` with `training.workload: causal_lm`, a
local Llama checkpoint, and `data.type: token_corpus`. The
[TinyLlama example](../examples/gpu/example_012_native_parallel_training/README.md)
provides complete YAML files and `prepare-text`, `plan`, `train --dry-run`, and
torchrun commands. Even one-rank execution requires torchrun metadata.

`training.gradient_accumulation_steps` is the microbatch count. Global batch
counts DP replicas only. `training.max_steps` means additional optimizer updates
per invocation, including resume. FP32/BF16 computation, AdamW, fixed token
blocks, and the layouts below are admitted; QAT, distillation, activation
checkpointing, FP16 and unrelated backend options are explicitly rejected on
this path. They remain separate capabilities of their supported runtimes.

Use a dedicated trusted checkpoint root, one writer, no links or parent
traversal, and at least two retained snapshots. The CLI writes a final complete
checkpoint, and resumes committed data position, optimizer, scheduler and RNG
state. Changed model/data/optimizer/implementation identities fail closed.
Discovery does not authorize overwriting corrupt snapshots: any newer canonical
directory blocks continuation before model construction. Preserve and explicitly
relocate conflicting artifacts before retrying, or start a fresh run.

Reported throughput covers the update loop, including coordinated data access,
transfers, optimizer updates, checkpoint writes and retention scans. Model
loading and identity hashing are excluded. Do not compare it directly with
compute-only benchmark windows or interpret configuration admission as production
quality/performance certification.

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
  sets. The hidden-state fixture validates SGD with momentum; the token-loss
  fixture additionally compares AdamW parameters and optimizer state.
- TP/PP peers must process the same local samples; DP replicas process disjoint
  samples. Global batch = DP x microbatches x microbatch size, not world size x
  local batch.
- Stage parameters remain FP32. Stage inputs accept FP32 or first-stage int64
  token IDs; BF16 execution also admits BF16 inputs on later stages. Adjacent
  pipeline output/input shape and dtype must agree. The positional `loss_fn` remains a scalar mean over
  equal-size microbatches with equal denominators. Use `summed_loss` below for
  ignored labels and unequal valid-token counts; do not pass both objectives.
- Native GPipe or 1F1B handles inter-stage forward/backward communication. Loss scaling occurs
  exactly once; the result reports DP-averaged loss on every rank.
- Replicated DDP without TP/PP synchronizes gradients only on the final
  microbatch. Intermediate forward/backward passes use DDP's `no_sync` context.
  FSDP2 and pipeline schedules retain their own synchronization behavior.
- Output tensors are opt-in via `forward_backward(..., return_outputs=True)` and
  returned only on the last PP stage. The default avoids auxiliary retention and
  concatenation of microbatch logits in both execution paths. When requested,
  pipeline outputs retain native order/dtype; the schedule releases its output
  chunk references after the call, while the returned tensor remains caller-owned.
  Native GPipe still retains forward graphs until backward and allocates receive
  buffers per microbatch: this is not a constant-memory pipeline schedule.
  Optimizer steps, dataloaders, schedulers and process-group teardown remain
  caller-owned. Model/communication failures still require teardown, not reuse.

### Pipeline Schedule

`DistributedParallelConfig(pp_size=2, pipeline_schedule="1f1b")` opts into
PyTorch's native single-stage-per-rank 1F1B schedule. `"gpipe"` remains the default.
`trainer.config` is read-only. The selection is frozen, must agree across ranks before mesh creation, and is
part of checkpoint identity. 1F1B requires PP and at least as many microbatches
as stages; it is not a different tensor-parallel plan or virtual-stage schedule.

1F1B alternates backward and forward work after warmup, releasing backward graphs
earlier than all-forward/all-backward GPipe. Receive buffers still exist for each
microbatch, and requested output tensors still occupy memory: total allocation
is not constant. Compare identical per-microbatch shapes and measure all ranks'
peaks. Lower activation occupancy does not guarantee higher throughput.
Loss normalization, native DDP synchronization, BF16 forward scope and caller-owned
optimizer steps are unchanged. Cross-schedule bitwise equivalence is not promised;
validate each policy numerically and restore only with the same schedule.

### Execution Precision

`DistributedTrainer(..., precision="fp32")` is the default. Explicit
`precision="bf16"` enables CUDA BF16 autocast for forward/loss operations only,
on devices with native BF16 support. Parameters, gradients and Adam moments stay
FP32; FSDP2 retains its default parameter/reduction policy. This is not BF16
parameter storage, automatic communication compression, or support for AMP QAT/KD.

The policy is fixed at construction and must agree across ranks. The trainer
overrides an ambient autocast context during execution and restores it on exit;
backward runs outside autocast. Weight casts are not cached across calls, so an
outer autocast context cannot retain stale casts over optimizer updates.

Pipeline metadata inference uses the same forward precision. Before constructing
the schedule, every rank checks adjacent output/input shape and dtype. Supply
the actual receiving activation dtype, not simply the parameter dtype. For the
current Llama adapter, FP32 embedding/residual streams keep the inter-stage hidden
state FP32, while the final linear head produces BF16 logits. Generic linear
stages can instead need BF16 receiving examples. A mismatch fails collectively
before pipeline send/receive. These checks do not recover from arbitrary failures
inside model execution or NCCL; keep launcher timeouts.

BF16 and FP32 need separate numerical and held-out quality controls. A native
matched-TP comparison isolates the adapter/executor but does not prove equality
to an unsharded FP32 model or establish long-run convergence.

### Token-Weighted Loss

For a caller-supplied causal model/stage with replicated full-vocabulary logits:

```python
from parascale.parallel import CausalLMLoss

trainer = DistributedTrainer(
    local_stage, config, example_microbatch,
    summed_loss=CausalLMLoss(vocab_size),
    tensor_plan=model_specific_tp_plan,
)
```

`CausalLMLoss` shifts logits/labels for next-token prediction and excludes
`ignore_index=-100` labels. Targets have shape `[batch, sequence]`, dtype int64;
logits have shape `[batch, sequence, vocab_size]`. Ignored labels are not an
attention mask. The caller still supplies causal attention and any padding mask.

The trainer counts valid shifted labels across DP replicas, without counting
TP/PP copies again, and scales each summed microbatch loss by
`dp_size / global_valid_tokens` to compensate DP gradient averaging. Empty local
replicas/microbatches keep a backward graph; an entirely empty global objective
is rejected. `ParallelStepResult.normalization_count` exposes the denominator.
Custom `SummedLoss` implementations must return a differentiable scalar sum and
a nonnegative device-local int64 scalar from `normalization_count(targets)`.
That method must also validate all labels before model execution.

Constructor argument errors and topology/microbatch/loss-mode disagreements are
coordinated before mesh creation. Step preflight coordinates invalid batches and
rejects inconsistent global counts before forward/backward. Callers must still
provide identical data/objectives to TP/PP peers; equal counts do not prove equal
labels. Errors inside arbitrary user model/loss code or native communication
still require launcher timeouts and teardown.

This contract does not automatically partition pretrained models, synchronize
tied weights across stages or implement vocabulary-sharded cross entropy.
Checkpoint context stores and compares the declared loss configuration;
the caller must still reconstruct the objective before loading state.

### Llama Stages

`parascale.parallel.llama.LlamaStage` adapts an explicitly loaded
`transformers==4.51.3` `LlamaForCausalLM`. It reuses the original embedding,
decoder layers, norm and head, preserving parameter names. It does not load
weights, download assets, create optimizers or automatically choose a topology.

```python
from parascale.parallel.llama import LlamaStage

replica, pipeline_stage, tensor_rank = config.coordinates(dist.get_rank())
stage = LlamaStage(
    loaded_model, pipeline_stage, config.pp_size, tp_size=config.tp_size,
).to(device)
del loaded_model
trainer = DistributedTrainer(
    stage, config, example_microbatch,
    summed_loss=CausalLMLoss(vocab_size),
    tensor_plan=stage.tensor_parallel_plan() or None,
)
```

The source model's parameters must be materialized FP32. The source modules are
shared with the stage, not cloned; release the source after partitioning and use
an independent copy for reference comparisons. Each PP rank owns a contiguous
decoder range; only the first owns embedding and only the last owns norm/head.
Attention and MLP projections are TP-sharded; embedding, norms and head remain
replicated within TP. Q heads, KV heads and projection widths must divide TP.
Within attention, Q/K/V share one replicated DTensor input; gate/up share one
within the MLP. Their partial input gradients combine before redistribution,
instead of reducing each branch separately. Child outputs stay local tensors,
so RoPE, attention and residual operations retain the pinned HF behavior.
This changes floating-point reduction order and still requires numerical checks.

The contract is packed causal token blocks starting at position zero, zero
attention dropout, no KV cache and no padding/custom mask. Both eager
attention and SDPA use Hugging Face operators; SDPA retains native mask-free
causal dispatch. The stage's `validate_inputs` hook checks embedding indices
inside the trainer's coordinated batch preflight, separately from loss labels.

Single-stage weight ties are preserved. Cross-stage tied embedding/head is
rejected, so SmolLM2 is suitable for DP but not this PP path; its 9 Q / 3 KV heads
also exclude TP2. TinyLlama has untied weights and divisible heads. Do not change
ties or head counts to force an unsupported topology. Adapter construction and
model loading precede trainer preflight and still require launcher-level failure
handling. Checkpoints compare caller-declared Llama/RoPE identity but do not
introspect or reconstruct the live model. Reconstruct the exact pinned configuration
and loss before loading. Bounded fresh-process cases do not establish complete
pretrained production recovery qualification.

See the [full-model validation example](../examples/gpu/example_010_llama_parallel/README.md)
for real local weights/data and independent HF numerical controls.

### Prepared-Corpus Stage Builder

`parascale.workloads.build_causal_lm_stage_components` combines local pretrained
Llama weights with a corpus generated by `parascale prepare-text`. It accepts
the ordinary causal-LM `model` and `data` configuration sections:

```python
from parascale.workloads import build_causal_lm_stage_components

stage, cursor, objective = build_causal_lm_stage_components(
    workload_config, config, rank=dist.get_rank(), world_size=dist.get_world_size(),
)
```

The explicit launcher size must match the topology. The returned stage is on
CPU; unowned source layers are released before the caller moves it to CUDA.
Construct `DistributedTrainer` with `summed_loss=objective` and the stage's TP
plan, then construct the optimizer against its final local parameters. Do not
construct an optimizer before TP/DP wrapping. `parascale.optimizers.build_optimizer`
automatically uses non-fused `foreach=False` AdamW for mixed Tensor/DTensor
trainable parameters; ordinary and all-DTensor parameters keep PyTorch's native
selection. No parameters are converted or regrouped. Configured four-bit
optimizers reject trainable DTensor parameters before creating state. Manual
optimizers must also avoid fused/foreach updates over mixed parameters.
Llama stage examples use
int64 token IDs on the first stage and FP32 residual hidden states on later
stages, including when the executor uses BF16 autocast.
The executor calls an optional `validate_example_inputs` stage hook during its
coordinated constructor preflight, before mesh creation or shape-inference
forwards. Llama uses it to reject invalid token IDs and residual shape/dtype
across all ranks. Custom hooks must perform local validation only, without
forward execution, module mutation or collectives.

The cursor shards by DP coordinate, not global rank. TP/PP peers receive the
same ordered blocks when they use identical data settings and advance together;
different DP replicas receive disjoint blocks. Launcher environment variables
are neither read for this mapping nor modified. Each cursor batch is one
microbatch; the caller collects `config.microbatches` batches per update.
Global batch size is `data.batch_size * config.microbatches * config.dp_size`.

The Llama restrictions above still apply. Activation checkpointing is rejected.
The builder does not initialize collectives, coordinate construction failures,
execute training or enable TP/PP in the workload CLI. The caller owns process
group teardown, scheduler/cursor restoration and update/checkpoint boundaries;
a cursor snapshot alone is not a training recovery checkpoint.

## Checkpoints

All ranks call `save_checkpoint(new_directory, optimizer, step=...)` and
`load_checkpoint(directory, optimizer)`. Rank-local PyTorch state includes model,
optimizer and Torch CPU/CUDA RNG. SHA-256-checked files are published before the
final manifest; an existing committed directory cannot be overwritten.
All ranks must agree on the canonical checkpoint directory, topology and step
before saving, and on the directory and complete manifest before loading any
payload. Mixed snapshots are rejected before applying model or optimizer state.
The format also binds each rank's optimizer class and ordered parameter groups to
canonical stage parameter names. Type, order, group membership, duplicate slots
and external parameters are checked before state application; save-side ownership
checks precede model state capture. Shared parameters use their first registered
name, and optimizing only a subset is allowed. Saved hyperparameters are restored,
not compared with constructor defaults.

Schema v4 binds `pipeline_schedule` and requires an explicit `checkpoint_context` when constructing the trainer
if checkpoints will be used. Training without checkpoints may omit it. The context
contains `model`, `loss`, `data`, `tokenizer` and `precision` descriptors; the first
four must be nonblank strings or nonempty objects. Precision is `"fp32"` or
`"bf16"` and must match the trainer's execution policy; cross-precision resume
is rejected rather than silently changing the numerical computation.
Include effective model/RoPE/tie configuration, objective/reduction settings, asset
hashes, tokenizer configuration and packing/sampling policy, not just file paths.
Never include credentials or access tokens in this persisted, shared context.
Synthetic data can explicitly use `"tokenizer": "none"`.

```python
trainer = DistributedTrainer(
    stage, parallel_config, example_input, summed_loss=objective,
    tensor_plan=tensor_plan,
    checkpoint_context={
        "model": effective_model_config,
        "loss": objective_config,
        "data": dataset_manifest,
        "tokenizer": tokenizer_manifest,
        "precision": "fp32",
    },
)
```

The constructor freezes a detached declaration; `trainer.checkpoint_context`
returns a copy. All ranks declare the same global context, not their PP ownership
or local batch. Context JSON must have string object keys and finite values;
tuples, cycles and duplicate manifest keys are rejected. Object order is ignored,
array order is retained, and `true`, `1` and `1.0` are distinct identities.
Schema 1/2/3 checkpoints are rejected rather than inventing missing identity.
Load preflight checks context before deserializing any rank payload.

This verifies **matching declarations**, not arbitrary runtime behavior. Callers
must keep descriptors faithful to actual configuration/code/assets. Optimizer
class names do not fingerprint implementations or custom hooks. Context is not
sampler position, scheduler state or data-loader state. Schema4 does not restore
those; use the explicit schema5 runtime participant below for token training.
Automatic discovery of undeclared changes is not supported.

Only trusted local checkpoint files may be loaded. Topology, microbatch count, schedule
and rank/device mapping must remain unchanged. This format is not portable
inference export, topology-changing resharding or automatic
fault recovery. Python validation failures are coordinated; a dead process or
failure inside native collectives still requires launcher-level teardown.

### Complete Token Training State

Pass `runtime=StageTrainingCheckpoint(...)` to both save and load to include
the existing engine's committed token cursor, update/sample counters, scheduler
and Python/NumPy RNG in the same checksummed rank file. The executor continues to
own Torch CPU and its current CUDA device RNG. This supports one device per rank,
not arbitrary user generators or RNG streams on other devices.

```python
from parascale.optimizers.schedule import SchedulerSpec
from parascale.runtime.training.stage_checkpoint import StageTrainingCheckpoint

spec = SchedulerSpec("cosine", total_steps=1000, warmup_steps=20)
engine.scheduler = spec.build(optimizer)
engine.scheduler_metadata = spec.to_metadata()
runtime = StageTrainingCheckpoint(
    engine, cursor, optimizer,
    microbatches=trainer.config.microbatches, scheduler_spec=spec,
)
iterator = engine.data_resume.prepare_iterator(cursor)
# After each successful full update: optimizer.step(), scheduler.step(),
# increment engine.state.global_step, then record_consumption with local samples.
trainer.save_checkpoint(new_directory, optimizer, step=engine.state.global_step,
                        runtime=runtime)
# In a fresh process, construct the same model/optimizer/scheduler/runtime first.
trainer.load_checkpoint(new_directory, optimizer, runtime=runtime)
iterator = engine.data_resume.prepare_iterator(cursor)
```

Use the existing `TrainState` and `DataResumeController` in the runtime owner.
Only `TokenBatchCursor`, fixed full microbatches, counters starting at zero and
`SchedulerSpec.build()` schedules (or no scheduler) are admitted. A scheduler
step must follow each optimizer update; skipped updates and GradScaler are not
supported. `consumed_samples` is **local to the DP replica**, not multiplied by
TP/PP. All participants must use the trainer's microbatch count and the workload
builder's DP-coordinate cursor. Saving before iterator setup is rejected because
there is no committed cursor snapshot yet.

Runtime presence/configuration, cursor identity/position, counters, scheduler
progress, optimizer LR and RNG are checked before model/optimizer application.
Pending prefetch is not committed. Restore invalidates existing iterators; obtain
a new iterator from `prepare_iterator`. It restores RNG without an extra scheduler
step and clears stale metric history. Schema4/5 cannot be loaded interchangeably.
Publication is atomic; **state application is not rollback-transactional**. An
application failure requires process teardown and a fresh restart.

The reusable fresh-process validator uses a pretrained Llama and a corpus made
by `prepare-text`. Run each command in a separate process, with the same visible
GPUs, arguments and runtime. Allow space for model plus optimizer shards.

```bash
torchrun --standalone --nproc_per_node=2 tests/benchmarks/scripts/validate_stage_recovery.py \
  --mode baseline --model-path /models/tinyllama --corpus-path /data/token-corpus \
  --checkpoint-dir /checkpoints/new-run --output-dir /results/new-run --tp 2
torchrun --standalone --nproc_per_node=2 tests/benchmarks/scripts/validate_stage_recovery.py \
  --mode resume --model-path /models/tinyllama --corpus-path /data/token-corpus \
  --checkpoint-dir /checkpoints/new-run --output-dir /results/new-run --tp 2
```

Use four processes with `--dp 2 --tp 2`, or `--tp 2 --pp 2` and optional
`--pipeline-schedule 1f1b`, for supported mixed layouts. This compares update3
after a saved update2 against uninterrupted training, including exact local
parameter/optimizer hashes, data blocks, LR and RNG. It does not establish
convergence, long-run reliability, public CLI integration or throughput speedup.

### Discovery And Retention

For schema5 snapshots, use a dedicated root with direct children named
`step-00000000`, `step-00000001`, and so on. The catalog is filesystem-only and
does not import Torch or deserialize rank payloads:

```python
from parascale.parallel.checkpoint_catalog import (
    inspect_stage_checkpoint, latest_stage_checkpoint, prune_stage_checkpoints,
)

metadata = inspect_stage_checkpoint(root / "step-00000100")
selected, skipped = latest_stage_checkpoint(root)
# Only after the save has succeeded on every rank, call pruning on rank zero;
# coordinate its local I/O errors before any rank starts another collective.
report = prune_stage_checkpoints(root / "step-00000100", keep_last=3)
```

Inspection checks the manifest structure, paths and every rank SHA256. It does
**not** prove payload semantics or recoverability. `latest_stage_checkpoint`
reports rejected newer candidates and selects the newest intact snapshot;
execution/context mismatch must still fail in the all-rank loader, rather than
silently falling back to an older model. Schema4 snapshots are not admitted.

`keep_last` must be an integer of at least two, including the anchor and at least
one older intact copy when available. Retention groups snapshots by exact JSON
topology, context and runtime identity. An anchor older than another manifest in
its family is rejected even when that newer payload is corrupt. Other families
and corrupt payloads are retained; corrupt payloads do not count toward the
protected copies. An unknown or malformed manifest blocks all pruning because
its references cannot be proven self-contained. Extra files and directories are
never recursively deleted. The report lists `removed`, `protected` and `skipped`.

Paths and ancestors must not be links/reparse points. Use one writer with no
concurrent filesystem mutations; this is not an adversarial filesystem sandbox.
Integrity inspection checks independent rank files with at most four concurrent
readers; a single payload stays inline. Every SHA256 check is retained, errors
are reported in manifest order, and all readers finish before inspection returns
or pruning can proceed. A failed read is not a timeout guarantee: the launcher
must still bound the worker job if the underlying storage stalls.
Pruning unlinks and synchronizes the manifest before removing declared rank
files. An interruption may leave an uncommitted directory, which is preserved
and reported on subsequent scans. Such orphans require deliberate cleanup.
Full checksum scans add checkpoint I/O. Diagnostic benchmark windows exclude
them; the `native_parallel` CLI includes them in end-to-end update-loop timing
and invokes retention after each complete all-rank save.

Add `--validate-catalog` to both fresh-process validator commands above, using
a new checkpoint root. The baseline saves real updates0/1/2, prunes update0,
and checks that updates1/2 remain; the resume process discovers update2 and
must reproduce update3 exactly. Generated files remain local test artifacts.

To compare integrity inspection implementations on the same real checkpoint:

```bash
python tests/benchmarks/scripts/benchmark_checkpoint_verification.py \
  --checkpoint /checkpoints/run/step-00000100 \
  --baseline-catalog /trusted/baseline/parascale/parallel/checkpoint_catalog.py \
  --repeats 3 --output /results/checkpoint-verification.json
```

The baseline is executable Python: use only a trusted source file whose shared
helpers match the candidate. Both implementations inspect the same immutable
files, with one warmup each and alternating AB/BA pairs. Run without competing
work under the same CPU affinity and quota. The JSON records source hashes,
individual times, variance and median verification speedup. This measures
warm-cache integrity inspection only, not training, save, retention, cold-cache
I/O or recovery. The output must be a new file outside the repository and
checkpoint directory; do not commit generated results.

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
