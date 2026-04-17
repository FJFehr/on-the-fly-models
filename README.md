# on-the-fly-models

Task-conditioned modelling research repo focused on generating or adapting a small target model from a task's examples instead of training a separate model per task.

The long-term goal is a hypernetwork-style system that consumes support examples, predicts a task-specific target model, and solves the query example. The current implemented system is the baseline stage of that roadmap: config-driven target-model experiments on padded 1D ARC tasks.

## Repository layout

```text
on-the-fly-models/
├── .agents/                        # Agent contract, style guide, task briefs
├── configs/
│   ├── base/
│   │   └── arc1d_simple.yaml
│   └── experiments/
│       ├── arc1d_binary/                  # Binary hypermodel experiments
│       ├── arc1d_multiclass/              # Multiclass hypermodel experiments
│       ├── arc1d_capacity/                # Exp 1: direct target-model capacity baseline
│       └── _archived/                     # Legacy experiment configs (deleted models)
├── data_modules/
│   ├── __init__.py                 # Datamodule registry
│   ├── arc1d_simple.py
│   ├── arc1d_direct.py             # Flat supervised datamodule (Exp 1)
│   ├── arc1d_meta_simple.py
│   ├── arc1d_padded_multiclass.py
│   └── arc1d_meta_padded_multiclass.py
├── models/
│   ├── __init__.py                 # Model registry
│   ├── hypermodel.py               # Generic hypernetwork-target wrapper
│   ├── hypermodel_lightning.py     # HyperModelLightning training module
│   ├── direct_supervised_lightning.py  # DirectSupervisedLightning (Exp 1)
│   ├── cnn.py                      # Generic 1D CNN encoder
│   ├── mlp.py                      # Global MLP backbone
│   ├── rnn.py                      # Generic RNN encoder
│   └── transformer.py              # Generic transformer encoder
├── scripts/
│   ├── build_arc_1d.py
│   ├── analyze_weight_space_pca.py   # Offline PCA of hyper-generated vs direct RNN weights
│   ├── visualise_tasks.py
│   └── eda_arc_1d_boxplots.py
├── tests/
├── visualisation/
├── metrics.py
├── train.py
├── train_utils.py                 # Shared runtime, checkpoint, and visualisation helpers
├── pyproject.toml
└── README.md
```

## Current state

The training entrypoint is [train.py](/home/fabio/Projects/on-the-fly-models/train.py). It loads a YAML config, instantiates a registered datamodule and model, trains with PyTorch Lightning, saves the resolved config to the output directory, evaluates the best checkpoint, and writes a `results.txt` summary.

The active track is `arc1d_simple_meta_hypermodel`: task-level binary meta-learning using the simplified `HyperModelLightning` pipeline with one generated parameter vector per task.

The repo supports task-level dataloaders for the binary hypermodel meta-learning track.

## Setup

```bash
uv sync --python 3.12 --managed-python
```

## GPU setup (Linux, NVIDIA)

The training codepath already supports GPU execution through PyTorch Lightning.
The shared trainer passes `accelerator: auto` from the experiment config into
`pl.Trainer`, so Lightning will select CUDA when the environment is valid.

GPU support is therefore an environment question, not a missing training
feature. At the time of writing, the main failure mode has been a stale or
mixed Python environment: `torch` and `lightning` were being imported from an
incompatible stack (`torch 1.11.0+cu102`, no visible CUDA device, and a
`lightning` import failure). Do not reuse a global/system Python for training.

This repo does not currently ship a `build.sh`. The recommended setup is
environment-variable driven through `uv`.

### GPU readiness requirements

- bare-metal Linux host with an NVIDIA GPU and working driver stack
- Python `3.12.x` only
- `uv`-managed project environment only
- `torch` and `lightning` resolved from the same environment and mutually compatible
- training launched through `uv run ...`, not a system Python

The runtime knobs that already matter are:

- `accelerator` in experiment YAMLs; keep the existing `auto` default
- `devices` in experiment YAMLs; use `1` for a single GPU or `auto` to use all visible GPUs
- `num_workers` for dataloader parallelism; useful for throughput tuning but not required for first GPU bring-up
- `UV_TORCH_BACKEND` for choosing the CUDA wheel family

Typical GPU settings in experiment configs:

```yaml
accelerator: auto
devices: 1
```

Set `devices: auto` when you want Lightning to use all visible GPUs on the
host. Keep `accelerator: auto` unless you need to force a specific backend.

### Recommended backend: CUDA 12.8 (`cu128`)

Create a clean managed environment and select the CUDA backend before syncing:

```bash
export UV_TORCH_BACKEND=cu128
uv python install 3.12
uv sync --python 3.12 --managed-python
```

This repo expects Python `3.12.x` via `uv`. PyTorch is resolved transitively by
the dependency stack; the CUDA wheel family is selected through
`UV_TORCH_BACKEND`.

### Alternate backend: CUDA 12.1 (`cu121`)

If your host or driver/toolchain needs the older backend, recreate the same
environment with:

```bash
export UV_TORCH_BACKEND=cu121
uv python install 3.12
uv sync --python 3.12 --managed-python
```

Use one backend per environment. If you switch between `cu128` and `cu121`,
recreate or fully resync the managed environment rather than mixing packages
from different runs.

### Verify the environment before training

1. Confirm `uv` is using Python `3.12.x`:

```bash
uv run python -c "import sys; print(sys.version)"
```

2. Confirm the project environment can see CUDA through PyTorch:

```bash
uv run python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count()); print('cuda_device_0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

3. Confirm `lightning` imports cleanly from the same environment:

```bash
uv run python -c "import lightning as pl; print('lightning', pl.__version__)"
```

4. Run one training command from the managed environment and check that
Lightning selects CUDA rather than CPU:

```bash
uv run python train.py --config configs/experiments/arc1d_binary/mixes/move_1p3p_validate_2p/hyper_model.yaml
```

Expected pass condition:

- no Python import errors
- `torch.cuda.is_available()` prints `True`
- `torch.cuda.device_count()` is at least `1`
- the training startup logs show Lightning using a CUDA accelerator

### Troubleshooting

`lightning` import fails:
- likely cause: mixed environment or incompatible `torch`
- fix: recreate the `uv` environment under Python `3.12.x` and rerun the verification commands above

`torch.cuda.is_available()` is `False`:
- likely cause: wrong backend, missing NVIDIA driver visibility, or a CPU-only wheel resolution
- fix: re-check `UV_TORCH_BACKEND`, confirm the host GPU/driver is visible, and recreate the managed environment cleanly

Training still uses CPU:
- verify the experiment config still uses `accelerator: auto`
- verify the run is launched with `uv run ...` from the project environment, not from a stale system interpreter

### End-to-end GPU bring-up commands

For a clean Linux GPU setup, data build, and hypermodel overfit run:

```bash
export UV_TORCH_BACKEND=cu128
uv python install 3.12
uv sync --python 3.12 --managed-python

uv run python -c "import sys; print(sys.version)"
uv run python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count()); print('cuda_device_0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
uv run python -c "import lightning as pl; print('lightning', pl.__version__)"

uv run python scripts/build_arc_1d.py --simple

uv run python train.py --config configs/experiments/arc1d_binary/overfit/hyper_model.yaml
```

If you need the alternate CUDA backend, switch the first line to:

```bash
export UV_TORCH_BACKEND=cu121
```

The overfit config used above is:

- [hyper_model.yaml](/home/fabio/Projects/on-the-fly-models/configs/experiments/arc1d_binary/overfit/hyper_model.yaml)

It is a single-task overfit run over task category `1d_move_1p`, task id `8`,
with `train`, `val`, and `test` all pointed at the training split and
`overfit_single_batch: true`.

## Training

Training is config-driven. Each experiment YAML inherits from a shared base config and overrides only the fields that differ.

Example:

```bash
uv run python train.py --config configs/experiments/arc1d_padded_multiclass/move_1p/cnn.yaml
```

Supported model families are:

- `binary_hyper_model` / `hyper_model` — both resolve to `HyperModelLightning`; hypernetwork and target template are selected via `hyper_model.name` and `target_model.name` in the experiment config
- `direct_supervised` — resolves to `DirectSupervisedLightning`; trains a backbone (RNN, CNN, Transformer, MLP) directly on (input, output) pairs without a hypernetwork; see Experiment 1 configs in `configs/experiments/arc1d_capacity/`

### Experiment 1: Target Model Capacity (direct supervised)

Run a single architecture on all binary task categories:

```bash
uv run python train.py --config configs/experiments/arc1d_capacity/rnn.yaml
uv run python train.py --config configs/experiments/arc1d_capacity/cnn.yaml
uv run python train.py --config configs/experiments/arc1d_capacity/transformer.yaml
uv run python train.py --config configs/experiments/arc1d_capacity/mlp.yaml
```

Training pools support examples from all tasks per category; validation uses query examples from held-out tasks. Training stops at `val_all_examples_exact_match == 1.0`. The `base_multiclass.yaml` base config is available for the multiclass variant.

For a quick smoke test (single batch overfit):

```bash
uv run python train.py --config configs/experiments/arc1d_capacity/rnn.yaml overfit_single_batch=true max_steps=500
```

The current system is optimized for rapid baseline iteration:

- models are registered in `models/__init__.py`
- datamodules are registered in `data_modules/__init__.py`
- resolved configs and model summaries are saved per run
- training stops early once the configured primary validation metric reaches `1.0` unless disabled in config
- validation examples and hard failures can be logged for inspection

Capacity sweeps can be launched across multiple single-GPU workers and multiple
seeds with:

```bash
uv run python scripts/run_arc1d_capacity.py --config-dir configs/experiments/arc1d_capacity_multiclass --gpus 0,1,2,3 --seeds 42,43,44
```

Variable-length multiclass sweep (18 tasks × CNN/RNN/Transformer, 3 seeds, 8 GPUs):

```bash
uv run python scripts/run_arc1d_capacity.py --config-dir configs/experiments/arc1d_capacity_variable_multiclass --gpus 0,1,2,3,4,5,6,7 --seeds 0,1,2
```

The capacity plotter aggregates seeded runs under one output root:

- `solved_per_model.png` uses the best seed per task/model cell
- `heatmap_task_model.png` uses mean validation exact match with `mean±std`
  shown inside each cell

The `HyperModelLightning` path keeps one generated parameter vector per task. Its hypernetwork and target template are selected via `hyper_model.name` and `target_model.name` in the experiment config:

```bash
uv run python train.py --config configs/experiments/arc1d_simple_meta_hypermodel/move_1p/hyper_model.yaml
```

The binary hypermodel path also supports a wrapper-owned `task_encoding`
section:

- `task_encoding.name: scalar` keeps the current 5-scalar hypernetwork tokens
  plus `[value, normalized_position]` target inputs
- `task_encoding.name: shared_embeddings` switches both paths to a shared
  learned token embedder over `value`, `position`, `example_id`, `role`, and
  `is_query`

The hypermodel wrapper also owns a `hyper_head` section:

- `hyper_head.bottleneck_dim` controls the latent bottleneck before target
  parameter projection
- `hyper_head.pooling: attention` keeps the legacy single learned-query pool
- `hyper_head.pooling: hierarchical` uses hierarchical learned pooling over the
  6 serialized support segments, then 3 support examples, before the unchanged
  bottleneck and projection path

## Data

### Build task-level ARC1D

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

### Build the legacy simple baseline dataset

```bash
uv run python scripts/build_arc_1d.py --simple
```

This creates the binary padded baseline dataset at `data/arc_1d_simple`.

### Build the padded multiclass baseline dataset

```bash
uv run python scripts/build_arc_1d.py --padded-multiclass
```

This creates the active comparison-track dataset at `data/arc_1d_padded_multiclass`.

## Models

`HyperModelLightning` is the active model class. It encodes the full task context (3 support examples + query input) using a configurable hypernetwork (`hyper_model.name`), mean-pools the token representations into a single task vector, projects it through a hyper-head to a flat parameter vector, and applies those generated weights to a frozen target model template (`target_model.name`) for all support and query examples.

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

### Shared-embedding A/B on `move_1p_denoising_1c`

The binary mix experiment now has an explicit scalar-vs-embedding comparison
pair:

```bash
uv run python train.py --config configs/experiments/arc1d_binary/mixes/move_1p_denoising_1c/hyper_model_scalar.yaml
uv run python train.py --config configs/experiments/arc1d_binary/mixes/move_1p_denoising_1c/hyper_model_shared_embeddings.yaml
```

Compare the resulting runs on:

- `val_query_exact_match`
- `test_query_exact_match`
- `val_query_accuracy`
- `test_query_accuracy`
- total and trainable parameter counts in `model.txt` or W&B

The same A/B pattern can also be run on the larger
`move_1p_fill_hollow_denoising_1c` mix:

```bash
uv run python train.py --config configs/experiments/arc1d_binary/mixes/move_1p_fill_hollow_denoising_1c/hyper_model_scalar.yaml
uv run python train.py --config configs/experiments/arc1d_binary/mixes/move_1p_fill_hollow_denoising_1c/hyper_model_shared_embeddings.yaml
```

The shared-embedding variant there uses `embedding_dim: 16`.

## Deleted architectures — reference for future multiclass port

The following model families were removed from `models/hypermodels/` and `models/target_models/` in favour of the cleaner `HyperModelLightning` pipeline. This section records the key design decisions needed to extend the new pipeline to multiclass.

### Multiclass hypernetwork — what to re-implement

The deleted `hyper_rnn` and `hyper_cnn` used a richer task encoder than the current path:

- **TaskTransformerEncoder**: 4 separate learned embedding tables for `value`, `position`, `example_index`, and `role` (input vs. output side). All four are **summed** into a single token representation before the transformer encoder. This is more expressive than the new path's 5-scalar feature vector and is required to meaningfully encode multiclass token identities in the task context.
- **Loss / prediction**: `CrossEntropyLoss` over logits of shape `(batch, num_classes, seq_len)`. Predictions via `argmax`. Replaces the binary `BCEWithLogitsLoss` / sigmoid-threshold path.

### Future planned track: task-conditioned target training (no hypernetwork)

A planned experiment where target models are trained directly on task examples (support + query, using the task-level datamodule) without any hypernetwork. The target model itself is fit on task context end-to-end. This is a third contract distinct from both pair-based baselines and hypernetwork meta-learning, and should live in its own Lightning module and model-registry key when implemented.

### binary_hyper_rnn vs HyperModelLightning — diff

The deleted `BinaryHyperRNNMetaModelLightning` differed from the current `HyperModelLightning` as follows:

| Aspect | `binary_hyper_rnn` (deleted) | `HyperModelLightning` (current) |
|--------|------------------------------|----------------------------------|
| RNN execution | Hand-coded stateless forward: manually loops over timesteps, applies generated `weight_ih` / `weight_hh` per layer without `torch.nn.RNN` | `torch.func.functional_call` against a frozen `torch.nn.RNN` module |
| Parameter spec system | Custom `_build_param_specs()` derives weight shapes from config (input_size, hidden_size, num_layers, bidirectional) | Generic: uses `dict(target_model.named_parameters())` and reshapes the flat vector to match those shapes |
| Feature vector | 5 scalars: value, normalised position, example_id (0–3), role (0=input / 1=output), is_query (0/1) | Same |
| Task serialisation order | support1-in, support1-out, support2-in, support2-out, support3-in, support3-out, query-in | Same |
| Loss masking | `loss_on_support` / `loss_on_query` config flags | Always loss on both support and query |
| Bidirectionality | Manually implemented (concat forward + backward hidden states per timestep) | Delegated to `torch.nn.RNN` |

## Current scope and limitations

- active path is binary `arc1d_simple_meta_hypermodel`; multiclass and variable-length ARC1D are future stages
- `HyperModelLightning` always applies loss on both support and query; selective masking is a future addition
- full 2D ARC is a future stage

## Tests

```bash
uv run pytest
```

## Weight-space PCA analysis

Use the offline PCA script to compare hyper-generated target RNN weights
against a direct trained `1d_move_1p` baseline in one shared canonical weight
space:

```bash
uv run python scripts/analyze_weight_space_pca.py \
  --hyper-run outputs/arc1d_simple_meta_hypermodel_experiments/<hyper_run> \
  --direct-run outputs/arc1d_capacity_binary_baseline_experiments/<direct_run> \
  --split val \
  --checkpoint best \
  --output-dir outputs/weight_space_pca/<analysis_name>
```

This v1 analysis is intentionally narrow:

- binary prediction only
- `1d_move_1p` only
- bidirectional 1-layer RNN only
- PCA only, no t-SNE

The script writes `summary.json`, `hyper_vectors.csv`, `metadata.csv`, and
`pca.png`.

Coverage is intentionally focused on:

- dataset construction and datamodule contracts
- `HyperModelLightning` shape, gradient, and smoke-path coverage
- exact-match sequence metric behavior

## Agentic workflow

The `.agents/` folder is the repository's operating contract for coding agents. It defines instruction precedence, style, task processes, verification expectations, and prompting structure so repo work stays consistent across tools.
