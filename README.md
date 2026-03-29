# on-the-fly-models

Task-conditioned modelling research repo focused on generating or adapting a small target model from a task's examples instead of training a separate model per task.

The long-term goal is a hypernetwork-style system that consumes support examples, predicts a task-specific target model, and solves the query example. The current implemented system is the baseline stage of that roadmap: config-driven target-model experiments on padded 1D-ARC tasks.

The work is intentionally staged:

1. Binary padded ARC1D baselines
2. Multiclass padded ARC1D baselines
3. Variable-length ARC1D with hypernetworks
4. Full ARC (2D)

## Repository layout

```text
on-the-fly-models/
├── .agents/                        # Agent contract, style guide, task briefs
├── configs/
│   ├── base/
│   │   ├── arc1d_simple.yaml       # Shared base config for binary ARC1D baseline
│   │   └── arc1d_padded_multiclass.yaml
│   └── experiments/
│       ├── arc1d_simple/           # Binary ARC1D experiments by category/model
│       └── arc1d_padded_multiclass/
├── data_modules/
│   ├── __init__.py                 # Datamodule registry
│   └── arc1d_simple.py             # Pair-based datamodule for padded binary ARC1D
│   └── arc1d_padded_multiclass.py  # Pair-based datamodule for padded multiclass ARC1D
├── models/
│   ├── __init__.py                 # Model registry
│   ├── hyper_model.py              # Hyper-model placeholder / prototype territory
│   └── target_models/
│       ├── base.py                 # Shared Lightning training/eval logic
│       ├── mlp.py                  # Flat MLP baseline
│       ├── cnn.py                  # 1D CNN baseline
│       ├── positional_mlp.py       # MLP with explicit x-position features
│       ├── positional_cnn.py       # CNN with explicit x-position channel
│       └── positional_rnn.py       # Positional RNN baseline
├── scripts/
│   ├── build_arc_1d.py             # Build task-level ARC1D datasets
│   ├── visualise_tasks.py          # Render ARC tasks
│   └── eda_arc_1d_boxplots.py      # Length-distribution plots
├── tests/                          # Focused pytest coverage for data/models/metrics
├── visualisation/                  # Shared ARC plotting helpers
├── metrics.py                      # Sequence metrics used in training
├── train.py                        # Single config-driven training entrypoint
├── pyproject.toml                  # Dependencies and tool config
└── README.md
```

## Current state

The current training entrypoint is [train.py](/home/fabio/Projects/on-the-fly-models/train.py#L127). It loads a YAML config, instantiates a registered datamodule and model, trains with PyTorch Lightning, saves the resolved config to the output directory, and evaluates the best checkpoint.

The current baseline data path is built in [scripts/build_arc_1d.py](/home/fabio/Projects/on-the-fly-models/scripts/build_arc_1d.py#L264). The simplified ARC1D variant is:

- padded to length 33
- restricted to a simple subset of ARC1D categories
- binarised from `0/non-zero` to `0/1`

The current baseline dataloader in [data_modules/arc1d_simple.py](/home/fabio/Projects/on-the-fly-models/data_modules/arc1d_simple.py#L135) is pair-based, not task-conditioned. It expands task records into support/query `(input, target)` pairs so baseline target models can be trained and evaluated without yet introducing hypernetwork conditioning.

The current training stack in [models/target_models/base.py](/home/fabio/Projects/on-the-fly-models/models/target_models/base.py#L157) is binary end to end:

- `BCEWithLogitsLoss`
- sigmoid thresholding for predictions
- binary elementwise and exact-match metrics

The active config foundations are:

- [configs/base/arc1d_simple.yaml](/home/fabio/Projects/on-the-fly-models/configs/base/arc1d_simple.yaml#L1) for the legacy binary baseline
- [configs/base/arc1d_padded_multiclass.yaml](/home/fabio/Projects/on-the-fly-models/configs/base/arc1d_padded_multiclass.yaml#L1) for the new multiclass padded track

## Setup

```bash
uv sync --python 3.12 --managed-python
```

## Training

Training is config-driven. Each experiment is defined by a YAML config, with a shared base config plus small overrides for model/task-specific runs.

Example:

```bash
uv run python train.py --config configs/experiments/arc1d_simple/move_1p/cnn.yaml
```

Current experiment configs cover:

- `move_1p`
- `move_2p`
- `move_3p`
- `fill`
- `hollow`
- `denoising_1c`
- `pcopy_1c`
- `overfit`

The first multiclass padded experiments currently cover:

- `move_1p`
  - `positional_cnn`
  - `positional_rnn`

Current baseline model families include:

- `mlp`
- `cnn`
- `positional_mlp`
- `positional_cnn`
- `positional_rnn`

The current system is optimized for rapid baseline iteration:

- models are registered in `models/__init__.py`
- datamodules are registered in `data_modules/__init__.py`
- resolved configs and model summaries are saved per run
- validation examples and hard failures can be logged for inspection

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

### Build the current simplified ARC1D baseline dataset

```bash
uv run python scripts/build_arc_1d.py --simple
```

This is the current baseline dataset path used by the repo today: padded, binary, and limited to a simpler subset of ARC1D categories.

### Build the padded multiclass ARC1D baseline dataset

```bash
uv run python scripts/build_arc_1d.py --padded-multiclass
```

This creates the new comparison-track dataset at `data/arc_1d_padded_multiclass`:

- padded to length 33
- original ARC token values `0..9` preserved
- `1d_padded_fill` excluded

### Hold out an entire category

```bash
uv run python scripts/build_arc_1d.py --holdout-category 1d_mirror
```

## Models

The implemented models today are target-model baselines:

- `mlp`: flat sequence-to-sequence MLP
- `cnn`: translation-equivariant 1D CNN
- `positional_mlp`: MLP with explicit normalized x-position features
- `positional_cnn`: CNN with an explicit x-position channel
- `positional_rnn`: RNN baseline with explicit positional features

The hyper-model is not implemented yet. [models/hyper_model.py](/home/fabio/Projects/on-the-fly-models/models/hyper_model.py#L1) is currently placeholder/prototype territory and should be read as roadmap, not shipped functionality.

## Current scope and limitations

- The current documented and tested path is binary padded ARC1D, not multiclass ARC1D.
- The first multiclass padded path is now available as a separate config and dataset family.
- The current datamodule is pair-based baseline training, not full task-conditioned hypernetwork training.
- The current experiments are fixed-length through padding to 33.
- Hypernetwork training and full 2D ARC are planned next stages, not current capabilities.

## Roadmap

### Milestone 1: Add multiclass padded ARC1D alongside simple ARC1D

- Preserve the existing binary `arc1d_simple` dataset and configs unchanged.
- Add a new multiclass padded ARC1D track as a separate dataset/config family under `arc1d_padded_multiclass`.
- Keep padding, but remove binarisation for the new track.
- Exclude the padded-fill category from the new multiclass padded set.
- Update models and training to predict per-position multiclass outputs.

Default assumption for this stage:
- binary simple ARC1D remains preserved and runnable as the reproducible legacy baseline
- multiclass padded ARC1D becomes the new comparison track, not an in-place replacement

### Milestone 2: Benchmark the best multiclass padded baselines

- Run the strongest current positional RNN and strongest current CNN on multiclass padded ARC1D.
- Use this as the final fixed-length baseline comparison before hypernetwork work.
- Use the results to choose the target-model family for the hypernetwork phase.

Default assumption for this stage:
- "best" refers to the best-performing current baseline families surfaced by the existing ARC1D experiments, not a fresh architecture sweep

### Milestone 3: Prototype hypernetworks for variable-length ARC1D

- Move from flat pair supervision to full task conditioning.
- Feed all support input/output examples plus the query input into the hyper-model.
- Use a lightweight attention-based task encoder.
- Predict a task-specific target model's weights with a hypernetwork.
- Run the predicted target model on the query.
- Train on task structure and evaluate on the held-out query output.

Default assumptions for this stage:
- variable length is introduced here, not in the multiclass padded baseline stage
- one task produces one target-model instance
- support set plus query input are the conditioning inputs
- query output is the evaluation target

### Milestone 4: Add full ARC (2D)

- Once the hypernetwork approach is validated on ARC1D, extend the same ideas to full 2D ARC.
- Treat this as a new milestone after the first 1D hypernetwork prototypes, not as part of the initial prototype wave.
- Use full ARC as the harder spatial and generalization benchmark after the ARC1D pipeline is working.

Default assumption for this stage:
- full ARC comes after, not during, the first hypernetwork prototype phase

## Tests

```bash
uv run pytest
```

Current status: `20 passed`.

Coverage is intentionally focused on the current baseline scope:

- ARC1D dataset construction and split integrity
- simplified datamodule contracts
- target-model output-shape invariants
- shared validation logging behavior
- exact-match sequence metric behavior

## Agentic workflow

The `.agents/` folder is the repository's operating contract for coding agents. It defines instruction precedence, style, task processes, verification expectations, and prompting structure so README edits, features, refactors, and debugging work stay consistent across tools.

### What's inside

| File | Purpose |
|------|---------|
| `AGENTS.md` | Root contract: priorities, working rules, verification, version-control policy. |
| `STYLE.md` | Default coding and documentation style. |
| `PROMPT_TEMPLATE.md` | Reusable task prompt template for agent dispatch. |
| `tasks/feature.md` | Process guide for feature work. |
| `tasks/debug.md` | Process guide for debugging work. |
| `tasks/refactor.md` | Process guide for refactor work. |

Instruction precedence inside `.agents/` is:

`AGENTS.md` > `STYLE.md` > `PROMPT_TEMPLATE.md` > task brief > `README.md`
