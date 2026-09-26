# on-the-fly-models

Task-conditioned models that generate or adapt a small target model from a task's examples instead of training a separate model per task.

**→ For the paper's experiments, start at [`experiments/README.md`](experiments/README.md).**
It names the 6 experiments, their status, and the exact commands to reproduce
each figure. The rest of this README covers the shared codebase (data prep,
training/validation entry points, model classes, repository layout). The
original 1D-ARC capacity/binary/multiclass track that predates the paper
work has been removed (its configs and scripts, along with the exploratory
`arc1d_*` runs that predated the current 6 experiments, are still recoverable
from git history before this cleanup).

## Reproducing from a clean environment

The exact sequence for a full clean-slate rerun of the paper experiments under
`experiments/`: fresh environment, datasets rebuilt from scratch, everything else
downstream of that. Full per-experiment commands and findings live in
[`experiments/README.md`](experiments/README.md); this is the process, once.

```bash
uv sync --python 3.12 --managed-python

uv run python -m scripts.build_arc_1d
uv run python -m scripts.augment_arc_1d \
    --per-pair --n-color-permutations 199 --shifts 1 2 -1 -2 --no-mirror \
    --dev-test-n-permutations 19 \
    --output-dir data/arc_1d_looped_augmented
uv run python -m scripts.build_arc1d_compositional
```

This builds `data/arc_1d` (the raw benchmark), `data/arc_1d_looped_augmented`
(721,000 train rows, the dataset every current experiment shares) and
`data/arc_1d_compositional_holdout` (experiment 4's zero-shot eval set).
Optionally confirm there's no train/eval overlap:
`uv run pytest -m slow tests/test_data_leakage.py`.

**Known gotcha**: on some NFS-mounted home directories, `uv sync` can silently
strip the executable bit from `wandb`'s bundled binary, which fails every
training job instantly with `PermissionError: ... wandb/bin/wandb-core`. Fix
with `chmod +x .venv/lib/python3.12/site-packages/wandb/bin/wandb-core
.venv/lib/python3.12/site-packages/wandb/bin/gpu_stats` if training jobs fail
immediately after a fresh `uv sync`.

### Running on the cluster

Training runs on GPU nodes (`torrnodeN.priv`), reachable with a single
`ssh torrnodeN.priv` (a `ProxyJump` through the lab's login host handles the hop
transparently via `~/.ssh/config`). Standing convention: only launch jobs on
`torrnode8`, `torrnode9`, `torrnode11`-`torrnode15` (not `torrnode10`, not
`torrnode1`-`torrnode7`), per `experiments/02_hypernetwork_multitask/README.md`
(where the reasoning is explained).

```bash
ssh torrnode15.priv
git clone git@github.com:FJFehr/on-the-fly-models.git && cd on-the-fly-models
uv sync --python 3.12 --managed-python
# build the datasets as above, then launch a sweep across all 8 GPUs:
CFG_DIR=experiments/01_multitask_capacity SEEDS_OVERRIDE="1 2 3 4 5" \
    GPUS="0,1,2,3,4,5,6,7" bash scripts/run_config.sh
```

`scripts/run_config.sh` skips any `(config, seed)` pair that already has a
`results.txt`, so it's always safe to rerun. Launch it under `nohup ... &` (or
`tmux`) so it survives the SSH session ending. `CFG_DIR` recurses, so pointing
it at an experiment's whole directory sweeps every config folder underneath
it too (a sub-study living alongside the main configs, say). Narrow `CFG_DIR`
to a specific subfolder for a scoped launch, and pass `PROJECT=<name>`
explicitly if you want its results grouped under a project other than that
subfolder's own basename.

Once training finishes, fetch the results back (checkpoints excluded) and
regenerate the figures:

```bash
REMOTE_HOST=torrnode15.priv bash scripts/fetch_experiments.sh 01_multitask_capacity
uv run python experiments/01_multitask_capacity/plot_all.py --outputs-dir outputs
```

`experiments/01_multitask_capacity/plot_all.py` is that experiment's single
entry point: it rescans `outputs/`, refreshes every CSV under
`outputs/results/01_multitask_capacity/`, and renders every figure (the
capacity-cliff plot plus a per-task breakdown for every size present in the
data) in one pass. Other experiments' own READMEs list their exact
plot script(s); this one-script-does-everything convention isn't wired up
everywhere yet.

The current codebase is a config-driven 1D ARC experimentation repo with two active tracks: task-level hypermodel experiments and direct-supervised baselines. Shared registries and runtime utilities let the same training and evaluation entrypoints run both tracks from YAML configs.

## Overview

- [train.py](/home/fabio/Projects/on-the-fly-models/train.py) loads a YAML config, instantiates the selected datamodule and model, trains with PyTorch Lightning, saves the resolved config, and writes final metrics to `results.txt`.
- [validate.py](/home/fabio/Projects/on-the-fly-models/validate.py) reloads a saved run and evaluates `best`, `last`, `auto`, or an explicit checkpoint path.
- Model and datamodule selection are registry-driven via `models/__init__.py` and `data_modules/__init__.py`, so most new experiments only require YAML changes.

## Quick start

For a full clean-slate rerun of the paper experiments, see "Reproducing from
a clean environment" above. In general:

```bash
uv sync --python 3.12 --managed-python
uv run python train.py --config <path/to/experiment_config>.yaml
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

Build the augmented variable-length dataset used by the augmented capacity experiments:

```bash
uv run python scripts/augment_arc_1d.py
```

This writes `data/arc_1d_augmented`. Only the train split is augmented; dev and test are
passed through unchanged so results are directly comparable to the baseline.
Default settings produce up to 242 variants per task (22 colour variants × 11 shift positions:
0, ±1, ±2, ±3, ±4, ±5), giving ~9,680 tasks per category in train. For `1d_mirror` tasks,
colour 9 (the semantic pivot) is kept fixed and excluded from permutation targets.

Build the move-task augmented dataset used by the disentanglement experiments:

```bash
uv run python scripts/augment_arc_1d.py --per-pair --n-color-permutations 199 --shifts 1 2 -1 -2 --no-mirror --task-categories 1d_move_1p 1d_move_2p 1d_move_3p --output-dir data/arc_1d_move_augmented
```

This writes `data/arc_1d_move_augmented` with only the three move task categories.
Per-pair colour augmentation gives each support pair and the query independent injective
colour mappings, producing 200 colour variants × 5 shift positions = 1000 variants per task
(120,000 train examples total). Dev and test are filtered to the same categories and passed
through unchanged.

## Data Augmentation

`scripts/augment_arc_1d.py` augments the train split of `data/arc_1d` with two
transforms applied in order: colour permutation → shift. All transforms
apply consistently to every sequence in a task (support inputs, support outputs,
query input, and query output) so the rule relationship between input and output
is preserved.

### 1 — Colour permutation

Non-zero colours in a task are remapped via a random injective mapping to `{1…9}`.
Background (0) is always preserved. The same mapping is applied to every sequence,
so the transformation is semantically valid for all multiclass tasks.

For `1d_mirror` tasks, colour 9 is the semantic pivot and is excluded from both the
permutable set and the available target set.

**Example** — original task colours `{1, 3}` remapped to `{5, 2}`:

```
Before:  support_input  = [0, 1, 0, 3, 0]
         support_output = [0, 3, 0, 1, 0]

After:   support_input  = [0, 5, 0, 2, 0]
         support_output = [0, 2, 0, 5, 0]
```

### 2 — Shift (translation)

All sequences are extended by prepending or appending zeros. Content is never
discarded. The `sequence_length` field increases by `abs(shift)`.

- Positive shift: prepend zeros → content moves right.
- Negative shift: append zeros → content moves left.

**Example** — shift +2 on a task with sequence length 5:

```
Before (length 5):  [0, 1, 3, 0, 0]
After  (length 7):  [0, 0, 0, 1, 3, 0, 0]
```

Sequences can grow beyond the 33-token fixed-length limit used in the padded
track; the variable-length collator (`Arc1dDirectPaddingCollator`) handles any
length dynamically at batch time.

### Using the augmented dataset

The schema of `data/arc_1d_augmented` is identical to `data/arc_1d`, so no new
data module is needed. Point the `data_dir` field in any YAML config at the new
directory:

```yaml
data_dir: data/arc_1d_augmented
```

Both `Arc1dDirectDataModule` (capacity track) and `Arc1dMetaDataModule`
(hypernetwork / meta-learning track) are compatible.

## Training

Training is config-driven. Each experiment YAML selects a registered model/data pair and overrides only the fields that differ from its inherited base configs. See each experiment's own README under `experiments/` for its exact training commands.

Supported model families are:

- `binary_hyper_model` / `hyper_model` — both resolve to `HyperModelLightning`; hypernetwork and target template are selected via `hyper_model.name` and `target_model.name` in the experiment config
- `direct_supervised` — resolves to `DirectSupervisedLightning`; trains a backbone (RNN, CNN, Transformer, MLP) directly on (input, output) pairs without a hypernetwork
- `looped_supervised` — resolves to `LoopedSupervisedLightning`; recursion-supervised training over `N_supervision` loop steps

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

For the paper's 6 experiments, start at [`experiments/README.md`](experiments/README.md)
instead — it's the current, actively-maintained map of that tree. This
section is the whole-repo picture:

```text
on-the-fly-models/
├── train.py                        # entry point: config -> model/datamodule -> fit
├── validate.py                     # entry point: reload a saved run -> validate/test
├── README.md
├── pyproject.toml
│
├── models/                         # what gets trained: architectures + Lightning wrappers
│   ├── __init__.py                     # MODEL_REGISTRY
│   ├── metrics.py                      # accuracy / exact_match_accuracy
│   ├── direct_supervised_lightning.py  # DirectSupervisedLightning
│   ├── looped_supervised_lightning.py  # LoopedSupervisedLightning (N_supervision recursion)
│   ├── hypermodel.py                   # generic hypernetwork <-> target-model wrapper
│   ├── hypermodel_lightning.py         # HyperModelLightning training module
│   ├── task_token_embedder.py          # shared task-token embedding, both training tracks
│   ├── canon_layer.py, canon_transformer.py, rope.py, rope_looped_transformer.py
│   │                                    # the RoPE+Canon target/hypernetwork architecture family
│   ├── transformer.py, cnn.py, rnn.py, mlp.py, activations.py, looped_transformer.py,
│   │   recursive_transformer.py        # backbone building blocks / alternative encoders
│
├── data_modules/                   # what gets trained on: datasets
│   ├── __init__.py                     # DATA_REGISTRY
│   ├── arc1d_direct.py                 # flat (input, output) supervised datamodule
│   ├── arc1d_meta_multiclass.py        # task-level (support+query) datamodule for the hypernetwork
│   ├── arc1d_compositional.py          # synthetic chained-skill holdout generator
│   └── task_filtering.py               # shared, caching-aware task-category/id filter
│
├── training/                       # how a run executes: mechanics, not definitions
│   ├── __init__.py
│   ├── config.py                       # YAML + _base_ inheritance, dotlist overrides
│   ├── trainer.py                      # Lightning trainer/callbacks, checkpoint I/O
│   └── logging.py                      # model summaries, W&B, results.txt, run artifacts
│
├── visualisation/
│   ├── __init__.py                     # public rendering API, re-exported from core/
│   ├── core/                           # shared rendering: used by training AND paper figures
│   │   ├── arc.py, style.py, embedding_clusters.py
│   └── paper/                          # paper-specific figure generators only
│       ├── arc_paper.py, plot_tasks.py, plot_paper_tasks.py, plot_embedding_clusters.py
│
├── scripts/                        # data prep + standalone analysis, run directly
│   ├── build_arc_1d.py                 # ingest the raw 1D-ARC benchmark
│   ├── augment_arc_1d.py               # colour/shift/mirror augmentation
│   ├── build_arc1d_compositional.py    # build the compositional holdout set
│   ├── eval_compositional_holdout.py   # zero-shot-evaluate a trained run against it
│   ├── run_config.sh                   # generic train.py launcher (skip-on-done, GPU round-robin)
│   ├── visualise_augmentation.py, measure_compute_efficiency.py, fetch_experiments.sh
│
├── experiments/                    # the paper's 6 experiments -- see experiments/README.md
├── docs/arc1d_story/                # research-narrative writeups
├── tests/                          # pytest suite (testpaths); `slow`-marked tests need built data
└── .agents/                        # operating contract for coding agents working in this repo
```

## Tests

```bash
uv run pytest
uv run ruff check .
```

## Agentic workflow

The `.agents/` folder is the repository's operating contract for coding agents. It defines instruction precedence, style, task processes, verification expectations, and prompting structure so repo work stays consistent across tools.
