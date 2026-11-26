# Llama MLP Structured Pruning

This example physically removes intermediate MLP channels. It is not a weight
mask, cache cleanup, or sparse-kernel claim. Attention and vocabulary are unchanged.
The initial baseline ranks each down-projection column by its L2 weight norm;
the corresponding gate/up rows and down columns are retained together.

This is a magnitude heuristic, not a reproduction of LLM-Pruner's gradient-based
method. Coupled-structure removal is motivated by
[LLM-Pruner](https://arxiv.org/abs/2305.11627). Quality recovery and acceleration
must be measured independently; removing parameters does not guarantee either.

## Local Model Example

Install a matching PyTorch and Transformers environment. The functional tests
cover Transformers 4.51.3 on A6000 and the local test environment. Use an ordinary
HF Llama directory, including tokenizer files. No implicit downloads or remote
model code are needed.

```python
import json
from pathlib import Path

from transformers import AutoTokenizer, LlamaForCausalLM
from parascale.pruning import prune_llama_mlp

source = "/data/models/llama"
output = Path("/data/models/llama-pruned")
if output.exists():
    raise FileExistsError(output)
model = LlamaForCausalLM.from_pretrained(source, local_files_only=True).eval()
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
pruned, report = prune_llama_mlp(model, keep_ratio=0.75, alignment=128)
output.mkdir(parents=True)
pruned.save_pretrained(output)
tokenizer.save_pretrained(output)
(output / "pruning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
restored = LlamaForCausalLM.from_pretrained(output, local_files_only=True).eval()
inputs = tokenizer("Explain distributed training:", return_tensors="pt")
print(tokenizer.decode(restored.generate(**inputs, max_new_tokens=32)[0]))
```

The API copies the model, so budget memory for both copies. Prune on CPU when
GPU capacity is limited. Retained width rounds DOWN to the requested alignment;
the report records the actual width and total-model parameter reduction. A
25% MLP channel removal is NOT a 25% total parameter reduction.

Only ordinary floating-point, single-device CPU/CUDA Llama models in eval mode
are accepted. Do not prune an already wrapped DDP/TP/PP, quantized, dispatched,
hooked or LoRA model. Create a new optimizer and wrap the resulting model for
training afterward; old optimizer states/checkpoints cannot be resumed into
the changed architecture. Quantization/pruning combinations need separate tests.

## Functional Validation

```bash
python -m pytest tests/test_llama_pruning.py -q
```

The offline tests use randomly initialized small Llama networks: physical tensor
shapes/parameter count, equivalent masked reference, unchanged input model,
HF export/reload, generation, backward/update, FP16/BF16, and unsupported inputs.
They do not establish pretrained model quality or throughput improvement.

Before deployment evaluate original/pruned/recovery-trained models on identical
held-out tokens and generation tasks. Record parameter/file size, actual peak
memory, cold startup, prefill/decode throughput and latency at fixed batch and
sequence lengths. Keep quality and speed gates explicit and preserve failures.
