# ParaScale Examples

Examples are organized by hardware target while keeping the framework entrypoint
and runtime architecture unified.

Runtime workload example directories contain:

- `config.json`: a runnable configuration for `python -m parascale.cli`.
- `run.sh`: a thin Linux launcher that resolves the repository root and calls
  the unified ParaScale CLI.
- `README.md`: the command and hardware notes for that example.

From the repository root, run an example with:

```bash
bash examples/<hardware>/<example>/run.sh
```

The scripts resolve the repository root internally, so they can also be invoked
from another working directory by using an absolute script path.

Run outputs, checkpoints, datasets, model weights, and benchmark reports must stay
outside `examples/`.

## Compression Validation Examples

These examples contain `run.sh` and `README.md`, without a new `config.json`
schema. They reuse the existing validation programs under `tests/benchmarks/`.
Llama uses the Python compression APIs; YOLO validation invokes the standard
ParaScale training runtime. They are real execution paths, not dry runs.

| Example | Explicit inputs | Verification |
| --- | --- | --- |
| [005 Llama PTQ](gpu/example_005_llama_ptq/README.md) | `--tiny` or local pretrained Llama and separate text corpora | Calibration, conversion, export/reload, CE and timing |
| [006 Llama QAT](gpu/example_006_llama_qat/README.md) | `--tiny` or local pretrained Llama and separate text corpora | Genuine updates, observer freeze, conversion and reload |
| [007 Llama distillation](gpu/example_007_llama_distillation/README.md) | `--tiny` or aligned pretrained teacher/student and separate text corpora | Frozen teacher, student updates, CE baseline and export |
| [008 YOLO distillation](gpu/example_008_yolo_distillation/README.md) | Trusted local teacher/student, images/labels, optional evaluation YAML | Official detection loss, feature KD, checkpoint validation and real-data comparison |

Set `PYTHON` to select the interpreter, `DEVICE` to choose CPU/CUDA, and
`OUTPUT_DIR` to a fresh directory outside `examples/`. Relative paths resolve from
the repository root even when launching elsewhere. Pass `--help` for all options.
Llama defaults to CPU; YOLO defaults to CUDA. These are not Ascend-validated
compression examples. No models or datasets are downloaded implicitly by the
documented local-asset commands.

`--tiny` is always opt-in: random-model smoke is not pretrained-quality evidence.
PTQ/QAT use floating-point reference kernels, not optimized integer inference.
YOLO COCO128 verifies real-data execution but not unseen-data generalization.
Use the [unified report](../tests/benchmarks/reports/BENCHMARK_REPORT.md) for the
exact executed configurations and limitations.

## Explicit Distributed Parallelism

[009 Distributed parallel training](gpu/example_009_distributed_parallel/README.md)
exercises the experimental Python stage API for DP, TP, PP and four-GPU
two-dimensional combinations. Residual-MLP and Llama decoder fixtures test
numerical parity and checkpoints using synthetic inputs. This does not enable
TP/PP in the ordinary workload CLI or certify full eight-GPU 3D parallelism.
See [the API boundary](../docs/distributed_parallel.md).

[010 Complete Llama parallel validation](gpu/example_010_llama_parallel/README.md)
adds local pretrained weights and real text, including embedding, causal loss and
AdamW update checks. SmolLM2 is a DP case; untied TinyLlama admits PP/TP layouts.
These bounded correctness cases do not establish production throughput, model
quality or fresh-process recovery.
