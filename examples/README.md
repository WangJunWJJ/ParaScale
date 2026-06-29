# ParaScale Examples

Examples are organized by hardware target while keeping the framework entrypoint
and runtime architecture unified.

Each example directory contains:

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
