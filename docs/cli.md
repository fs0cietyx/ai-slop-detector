# CLI Reference

The `slopguard` CLI provides access to local inference and model training workflows.

## Global Options

- `--help`: Show the help message and exit.

## Commands

### `predict`
Analyze a string of text to determine if it is AI-generated.

```bash
slopguard predict "Your text payload goes here."
```

**Output**:
A visual inference report containing the classification label (`HUMAN-WRITTEN` or `AI-GENERATED`), model confidence percentage, compute time (latency), and the hardware context (e.g., `CPU`, `CUDA`, `MPS`).

### `train`
Initiates a fine-tuning job on the transformer model using datasets located in `data/training_dataset.csv`.

```bash
slopguard train
```

**Features**:
- Uses LoRA adapters to minimize compute footprint.
- Automatically handles test/train splits.
- Exports production-ready adapters to the `models/` directory upon completion.
