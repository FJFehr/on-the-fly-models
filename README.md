# on-the-fly-models

Task-conditioned modelling research repo focused on generating or adapting a small target model from a task's examples instead of training a separate model per task.

The long-term goal is a hypernetwork-style system that consumes support examples, predicts a task-specific target model, and solves the query example. The current implemented system is the baseline stage of that roadmap: config-driven target-model experiments on padded 1D ARC tasks.

## Repository layout

```text
on-the-fly-models/
├── .agents/                        # Agent contract, style guide, task briefs
├── configs/
│   ├── base/
│   │   ├── arc1d_simple.yaml
│   │   └── arc1d_padded_multiclass.yaml
│   └── experiments/
│       ├── arc1d_simple/           # Legacy binary padded ARC1D experiments
│       └── arc1d_padded_multiclass/
├── data_modules/
│   ├── __init__.py                 # Datamodule registry
│   ├── arc1d_simple.py
│   ├── arc1d_padded_multiclass.py
│   └── arc1d_meta_padded_multiclass.py
├── models/
│   ├── __init__.py                 # Model registry
│   ├── hypermodels/                # Task-conditioned hypernetwork model families
│   └── target_models/
│       ├── base.py                 # Shared Lightning training/eval logic
│       ├── mlp.py                  # Positional MLP baseline
│       ├── cnn.py                  # Positional 1D CNN baseline
│       ├── cnn_core.py             # Shared pure CNN core for baseline/hypermodels
│       ├── rnn.py                  # Positional RNN baseline
│       └── transformer.py          # Positional transformer baseline
├── scripts/
│   ├── build_arc_1d.py
│   ├── visualise_tasks.py
│   └── eda_arc_1d_boxplots.py
├── tests/
├── visualisation/
├── metrics.py
├── train.py
├── pyproject.toml
└── README.md
```

## Current state

The training entrypoint is [train.py](/home/fabio/Projects/on-the-fly-models/train.py). It loads a YAML config, instantiates a registered datamodule and model, trains with PyTorch Lightning, saves the resolved config to the output directory, evaluates the best checkpoint, and writes a `results.txt` summary.

The active comparison track is `arc1d_padded_multiclass`:

- padded to length 33
- original ARC token values `0..9` preserved
- multiclass prediction with per-position logits

The legacy `arc1d_simple` path remains available as the binary padded baseline.

The repo now supports both training contracts:

- pair-based dataloaders for the baseline target-model experiments
- task-level dataloaders for the `hyper_rnn` and `hyper_cnn` meta-learning experiments

## Setup

```bash
uv sync --python 3.12 --managed-python
```

## Training

Training is config-driven. Each experiment YAML inherits from a shared base config and overrides only the fields that differ.

Example:

```bash
uv run python train.py --config configs/experiments/arc1d_padded_multiclass/move_1p/cnn.yaml
```

Supported model families are:

- `mlp`
- `cnn`
- `rnn`
- `transformer`
- `hyper_cnn`
- `hyper_rnn`

Model defaults:

- positional features are built into all four model families
- multiclass embeddings are built into all multiclass models
- transformer causality is controlled with `causal_attention: false|true`

See the Baseline Assumptions section below for how these baselines are wired and why the repo treats those choices as the working comparison setup.

The current system is optimized for rapid baseline iteration:

- models are registered in `models/__init__.py`
- datamodules are registered in `data_modules/__init__.py`
- resolved configs and model summaries are saved per run
- training stops early once the configured primary validation metric reaches `1.0` unless disabled in config
- validation examples and hard failures can be logged for inspection

The first task-conditioned experiment lives under its own meta-learning track and can be launched with:

```bash
uv run python train.py --config configs/experiments/arc1d_padded_multiclass_meta/move_1p/hyper_rnn.yaml
```

The hyper-CNN variant uses the same task-level datamodule and can be launched with:

```bash
uv run python train.py --config configs/experiments/arc1d_padded_multiclass_meta/move_1p/hyper_cnn.yaml
```

For a single-task overfit sanity check:

```bash
uv run python train.py --config configs/experiments/arc1d_padded_multiclass_meta/overfit/hyper_rnn.yaml
```

```bash
uv run python train.py --config configs/experiments/arc1d_padded_multiclass_meta/overfit/hyper_cnn.yaml
```

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

The implemented target-model baselines are:

- `mlp`: sequence-to-sequence MLP with explicit positional features
- `cnn`: 1D CNN with an explicit x-position channel
- `rnn`: RNN with explicit positional features
- `transformer`: transformer over per-position value-plus-position features
- `hyper_cnn`: transformer task encoder that predicts a task-specific 1D CNN
- `hyper_rnn`: transformer task encoder that reads whole support/query context and predicts a task-specific bidirectional RNN

For multiclass tasks, all models use learned token embeddings by default before the architecture-specific positional features are applied.

### Baseline Assumptions

The current baselines are built around a small number of shared assumptions. These are partly implementation defaults and partly deliberate comparison choices for the current stage of the project.

| Assumption | Current behavior | Why this assumption is used |
|------|---------|---------|
| Fixed-length padded ARC1D | Baselines use `input_dim = output_dim = 33` in the shared configs. | This keeps the comparison controlled before variable-length hypernetwork work. |
| Pair-based supervision | The datamodules expand tasks into support/query `(input, target)` pairs. | This isolates target-model baseline performance before adding task-conditioning machinery. |
| Active track | `arc1d_padded_multiclass` is the main comparison track; `arc1d_simple` remains the binary legacy baseline. | This keeps the repo focused on the non-binary setting while preserving the earlier baseline path. |
| Multiclass token space | The multiclass track uses `prediction_task: multiclass` with `num_classes: 10` and preserves ARC tokens `0..9`. | This keeps token identity intact instead of collapsing colors into binary presence/absence. |
| Train/eval split shape | Training uses support pairs; validation and test use query pairs from the configured sources. | This is closer to the ARC-style distinction between examples seen during fitting and held-out task queries. |
| Success metric | `val_exact_match_accuracy` is the shared primary metric, with elementwise accuracy also logged. | Exact match better reflects whether a task output is actually solved. |
| Shared optimizer defaults | Base configs use `Adam`, `learning_rate: 0.001`, `weight_decay: 0.01`, `batch_size: 32`, and `seed: 42`. | This gives a stable common starting point across baseline families. |
| Run artifacts | Each run writes the resolved config, model summary, checkpoints, and `results.txt`. | This keeps each experiment self-documenting and easier to reproduce. |
| Positional information | All four baseline families inject absolute position information directly into the model input path. | ARC tasks depend on absolute location, not only local pattern matching. |
| Multiclass embeddings | Multiclass models embed token IDs before architecture-specific processing. | Learned embeddings preserve token identity without imposing a fake ordinal meaning on colors. |

### Base Models

- `mlp`
  - Multiclass inputs are embedded first, then flattened.
  - A normalized position ramp in `[0, 1]` is appended before the first linear layer.
  - The MLP supports configurable depth, `use_skip_connections`, `use_layernorm`, `activation`, `init_scheme`, `dropout`, and `ffn_expansion_factor`.
- `cnn`
  - Inputs are converted into channels; for multiclass runs, embeddings become the value channels.
  - One extra normalized x-position channel is concatenated before the convolution stack.
  - Same-padding convolutions preserve sequence length, so `input_dim == output_dim` is required.
  - Kernels must be odd so padding stays symmetric.
- `rnn`
  - Per-position value features are concatenated with a normalized position scalar and then fed to the RNN.
  - Current experiment configs commonly use `rnn_bidirectional: true`.
  - An optional input skip projection can add the input features back into the hidden-state space.
- `transformer`
  - Per-position value features are concatenated with a normalized position scalar and then linearly projected into the transformer width.
  - Multiclass runs use learned token embeddings before feature concatenation.
  - `causal_attention: false|true` controls encoder-style versus causal attention; current baseline examples are encoder-style (`false`).

### Recommended Baseline Defaults

The repo keeps a number of architectural knobs configurable, but the current guidance for new baseline comparisons is:

- treat positional information as part of the baseline definition for all model families
- treat multiclass embeddings as part of the baseline definition for multiclass runs
- treat skip connections as recommended for multilayer baselines, especially deeper MLP and CNN variants
- treat layer norm as a strong stabilizer to try next for deeper MLP variants
- treat transformer as an active comparison family, but not currently the strongest baseline

Skip connections are not hardcoded as a universal default in code. They remain experiment-level config choices, but the repo now treats them as recommended guidance for deeper multilayer baselines.

The hypernetwork paths now live under `models/hypermodels/`. They keep the task intact, encode the 3 support examples plus query input with explicit token, position, example-index, and input/output-role embeddings, and predict the full weights of a task-specific target model. The current target-model variants are a bidirectional RNN and a 1D CNN.

## Current scope and limitations

- the main active path is fixed-length padded ARC1D, not variable-length ARC1D
- task-conditioned hypernetwork training currently targets padded multiclass `1d_move_1p`, with overfit sanity-check configs for single-task debugging
- baseline comparison tracks remain pair-based outside that new meta-learning path
- `arc1d_simple` remains the binary legacy baseline
- hypernetwork training and full 2D ARC are future stages

## Tests

```bash
uv run pytest
```

Coverage is intentionally focused on:

- dataset construction and datamodule contracts
- task-conditioned hypernetwork shape, gradient, and smoke-path coverage
- target-model output-shape invariants
- shared validation logging behavior
- exact-match sequence metric behavior

## Agentic workflow

The `.agents/` folder is the repository's operating contract for coding agents. It defines instruction precedence, style, task processes, verification expectations, and prompting structure so repo work stays consistent across tools.
