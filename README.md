# on-the-fly-models

Task-conditioned models that generate or adapt a small target model from a task's examples instead of training a separate model per task.

The current codebase is a config-driven 1D ARC experimentation repo with two active tracks: task-level hypermodel experiments and direct-supervised baselines. Shared registries and runtime utilities let the same training and evaluation entrypoints run both tracks from YAML configs.

## Overview

- [train.py](/home/fabio/Projects/on-the-fly-models/train.py) loads a YAML config, instantiates the selected datamodule and model, trains with PyTorch Lightning, saves the resolved config, and writes final metrics to `results.txt`.
- [validate.py](/home/fabio/Projects/on-the-fly-models/validate.py) reloads a saved run and evaluates `best`, `last`, `auto`, or an explicit checkpoint path.
- Model and datamodule selection are registry-driven via `models/__init__.py` and `data_modules/__init__.py`, so most new experiments only require YAML changes.

## Quick start

```bash
uv sync --python 3.12 --managed-python
uv run python scripts/build_arc_1d.py --padded-multiclass
uv run python train.py --config configs/experiments/arc1d_multiclass/move_1p/hyper_model.yaml
uv run python validate.py --config outputs/<run_name>/config.yaml --checkpoint best --mode both
```

## Setup

```bash
uv sync --python 3.12 --managed-python
```

## Data

Build the base task-level ARC1D dataset:

```bash
uv run python scripts/build_arc_1d.py
```

This creates a task-level HuggingFace `DatasetDict` with complete ARC tasks kept intact across splits. Each task record contains:

- `task_category`
- `task_id`
- `sequence_length`
- `support_inputs`
- `support_outputs`
- `query_input`
- `query_output`

Build the simplified binary dataset used by the binary track:

```bash
uv run python scripts/build_arc_1d.py --simple
```

This writes `data/arc_1d_simple`.

Build the padded multiclass dataset used by the multiclass track:

```bash
uv run python scripts/build_arc_1d.py --padded-multiclass
```

This writes `data/arc_1d_padded_multiclass`.

## Training

Training is config-driven. Each experiment YAML selects a registered model/data pair and overrides only the fields that differ from its inherited base configs.

Hypermodel example:

```bash
uv run python train.py --config configs/experiments/arc1d_binary/overfit/hyper_model.yaml
```

Multiclass hypermodel example:

```bash
uv run python train.py --config configs/experiments/arc1d_multiclass/move_1p/hyper_model.yaml
```

Direct-supervised example:

```bash
uv run python train.py --config configs/experiments/arc1d_capacity_binary/1d_move_1p/rnn.yaml
```

Supported model families are:

- `binary_hyper_model` / `hyper_model` — both resolve to `HyperModelLightning`; hypernetwork and target template are selected via `hyper_model.name` and `target_model.name` in the experiment config
- `direct_supervised` — resolves to `DirectSupervisedLightning`; trains a backbone (RNN, CNN, Transformer, MLP) directly on (input, output) pairs without a hypernetwork; see the capacity baselines under `configs/experiments/arc1d_capacity_*`

### Experiment 1: Target Model Capacity (direct supervised)

Run one binary direct-supervised baseline config for `1d_move_1p`:

```bash
uv run python train.py --config configs/experiments/arc1d_capacity_binary/1d_move_1p/rnn.yaml
uv run python train.py --config configs/experiments/arc1d_capacity_binary/1d_move_1p/cnn.yaml
uv run python train.py --config configs/experiments/arc1d_capacity_binary/1d_move_1p/transformer.yaml
uv run python train.py --config configs/experiments/arc1d_capacity_binary/1d_move_1p/mlp.yaml
```

Capacity sweeps can be launched across multiple single-GPU workers and multiple
seeds with:

```bash
uv run python scripts/run_arc1d_capacity.py --config-dir configs/experiments/arc1d_capacity_multiclass --gpus 0,1,2,3 --seeds 42,43,44
```

Variable-length multiclass sweep (18 tasks × CNN/RNN/Transformer, 3 seeds, 8 GPUs):

```bash
uv run python scripts/run_arc1d_capacity.py --config-dir configs/experiments/arc1d_capacity_variable_multiclass --gpus 0,1,2,3,4,5,6,7 --seeds 0,1,2
```

Scaling experiment — small/medium/large capacity tiers (run each size independently):

| Size | Params (CNN/RNN/TF) | Steps | WandB project |
|------|---------------------|-------|---------------|
| small | ~4K / 5.6K / 5.4K | 4 000 | `arc1d_capacity_small` |
| medium | ~9K / 9.2K / 10.3K | 4 000 | `arc1d_capacity_medium` |
| large | ~100K / 100K / 95K | 10 000 | `arc1d_capacity_large` |

```bash
uv run python scripts/run_arc1d_capacity.py --config-dir configs/experiments/arc1d_capacity_small --gpus 0,1,2,3,4,5,6,7 --seeds 0,1,2
uv run python scripts/run_arc1d_capacity.py --config-dir configs/experiments/arc1d_capacity_medium --gpus 0,1,2,3,4,5,6,7 --seeds 0,1,2
uv run python scripts/run_arc1d_capacity.py --config-dir configs/experiments/arc1d_capacity_large --gpus 0,1,2,3,4,5,6,7 --seeds 0,1,2
```

The capacity plotter aggregates seeded runs under one output root:

- `solved_per_model.png` uses the best seed per task/model cell
- `heatmap_task_model.png` uses mean validation exact match with `mean±std`
  shown inside each cell

## Validation

After a training run, validate or test a saved run with the resolved config in its output directory:

```bash
uv run python validate.py --config outputs/<run_name>/config.yaml --checkpoint best --mode both
```

Key flags from [validate.py](/home/fabio/Projects/on-the-fly-models/validate.py):

- `--checkpoint auto|best|last|<path>` selects which checkpoint to evaluate
- `--mode validate|test|both` controls which evaluation pass runs
- `--log-hard-examples --num-hard-examples N` exports the hardest validation failures
- `--log-to-wandb` opens a separate evaluation run for metrics and artifacts

## Models

`HyperModelLightning` is the active model class. It encodes the full task context (3 support examples + query input) using a configurable hypernetwork (`hyper_model.name`), pools the token representations into a single task vector, projects it through a hyper-head to a flat parameter vector, and applies those generated weights to a target model template (`target_model.name`) for all support and query examples.

For the binary track, the wrapper can now encode those task tokens in two ways:

- `scalar`: the existing 5-scalar feature vector
- `shared_embeddings`: summed learned embeddings for `value`, `position`,
  `example_id`, `role`, and `is_query`, reused by both the hypernetwork and the
  target model's input-side tokens

Available encoder architectures for both hypernetwork and target roles: `cnn`, `rnn`, `transformer` (defined in `models/`).

Shared experiment defaults:

- `val_exact_match_accuracy` is the primary metric; elementwise accuracy is also logged
- each run writes the resolved config, model summary, checkpoints, and `results.txt`
- early stopping triggers at `val_exact_match_accuracy == 1.0` unless disabled

## Repository layout

```text
on-the-fly-models/
├── .agents/                        # Agent contract, style guide, task briefs
├── configs/
│   └── experiments/
│       ├── arc1d_binary/                  # Binary hypermodel experiments
│       ├── arc1d_multiclass/              # Multiclass hypermodel experiments
│       ├── arc1d_capacity_binary/         # Binary direct-supervised baselines
│       ├── arc1d_capacity_multiclass/     # Fixed-length multiclass baselines
│       ├── arc1d_capacity_variable_multiclass/
│       ├── arc1d_capacity_small/
│       ├── arc1d_capacity_medium/
│       └── arc1d_capacity_large/
├── data/
│   ├── arc_1d/                      # Task-level DatasetDict
│   ├── arc_1d_simple/               # Binary padded baseline dataset
│   └── arc_1d_padded_multiclass/    # Multiclass padded baseline dataset
├── data_modules/
│   ├── __init__.py                 # Datamodule registry
│   ├── arc1d_direct.py             # Flat supervised datamodule (Exp 1)
│   ├── arc1d_meta_simple.py        # Binary task-level datamodule
│   ├── arc1d_meta_multiclass.py    # Multiclass task-level datamodule
│   └── arc1d_meta_padded_multiclass.py
├── models/
│   ├── __init__.py                 # Model registry
│   ├── direct_supervised_lightning.py  # DirectSupervisedLightning (Exp 1)
│   ├── hypermodel.py               # Generic hypernetwork-target wrapper
│   ├── hypermodel_lightning.py     # HyperModelLightning training module
│   ├── task_token_embedder.py      # Shared task-token embedding components
│   ├── cnn.py                      # Generic 1D CNN encoder
│   ├── mlp.py                      # Global MLP backbone
│   ├── rnn.py                      # Generic RNN encoder
│   └── transformer.py              # Generic transformer encoder
├── scripts/
│   ├── build_arc_1d.py             # Build task-level and derived ARC1D datasets
│   ├── run_arc1d_capacity.py       # Batch launcher for capacity sweeps
│   └── analyze_weight_space_pca.py # Offline PCA of hyper-generated vs direct RNN weights
├── tests/
├── training/                       # Shared config, logging, and trainer utilities
├── visualisation/
├── metrics.py
├── train.py
├── validate.py
├── pyproject.toml
└── README.md
```

## Tests

```bash
uv run pytest
uv run ruff check .
```

## Agentic workflow

The `.agents/` folder is the repository's operating contract for coding agents. It defines instruction precedence, style, task processes, verification expectations, and prompting structure so repo work stays consistent across tools.
