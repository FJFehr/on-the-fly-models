# on-the-fly-models

A hypernetwork that generates the weights of a small target model from a task's examples, on
the 1D-ARC benchmark, compared with training one model per task and one shared model for all
tasks. This repository reproduces every experiment in the paper from scratch.

## Setup

```bash
uv sync --python 3.12 --managed-python
```

This creates `.venv/`, which every run script uses (`.venv/bin/python`). Runs log to Weights
& Biases; set `WANDB_MODE=offline` to log locally only. On some NFS home directories `uv sync`
drops the executable bit from wandb's bundled binaries, so every training job fails instantly
with a `PermissionError`; fix it with
`chmod +x .venv/lib/python3.12/site-packages/wandb/bin/{wandb-core,gpu_stats}`.

## Build the data

```bash
uv run python -m scripts.build_arc_1d
uv run python -m scripts.augment_arc_1d \
    --per-pair --n-color-permutations 199 --shifts 1 2 -1 -2 --no-mirror \
    --dev-test-n-permutations 19 \
    --output-dir data/arc_1d_looped_augmented
uv run python -m scripts.build_arc1d_compositional
```

| Dataset | Contents | Used by |
|---|---|---|
| `data/arc_1d` | the raw benchmark: 18 categories, 901 tasks (721 train, 90 dev, 90 test) | the augmentation step |
| `data/arc_1d_looped_augmented` | 721,000 train, 1,800 dev and 1,800 test tasks | experiments 1 to 6 |
| `data/arc_1d_compositional_holdout` | 10 synthetic composite categories, 400 tasks | experiment 4 |

The build is deterministic: 1D-ARC is fetched at a pinned commit and all randomness is
seeded, so a rebuild is identical row for row. `uv run pytest -m slow tests/test_data_leakage.py`
checks that no training example appears in the dev, test or holdout splits.

Each task holds 3 support pairs and 1 query pair (`support_inputs`, `support_outputs`,
`query_input`, `query_output`, plus `task_category`, `task_id`, `sequence_length`). The
augmentation remaps non-zero colours with a random injective mapping (colour 9 stays fixed for
`1d_mirror`, where it is the pivot) and shifts every sequence by padding with zeros, the same
transform for all sequences of a task, so the rule is preserved. Splits are made before
augmenting.

## Reproduce the paper

Each experiment has one command to train (or evaluate) and one to make its figures and CSVs.
Experiments 3 and 4 only evaluate experiment 2's models, so run 2 first.

| Experiment | Question | Jobs | A40 GPU-hours |
|---|---|---:|---:|
| [1. Multi-task capacity](experiments/01_multitask_capacity/README.md) | Can one small model learn all 14 tasks at once? | 960 | 64 |
| [2. Hypernetwork](experiments/02_hypernetwork_multitask/README.md) | Does generating per-task weights close that gap? | 10 | 4 |
| [3. Reusability](experiments/03_reusability_generate_once_execute_many/README.md) | Do weights generated from one instance solve others? | eval | minutes |
| [4. Compositional generalisation](experiments/04_compositional_generalization/README.md) | Zero-shot on chained combinations of known rules | eval | minutes |
| [5. Leave-one-out](experiments/05_leave_one_out_task_generalization/README.md) | Zero-shot on a whole unseen task category | 140 | 57 |
| [6. Data efficiency](experiments/06_data_efficiency_ablation/README.md) | Does sharing across tasks need less data? | 870 | 340 |

```bash
GPUS=0,1,2,3,4,5,6,7 bash experiments/01_multitask_capacity/run.sh
uv run python experiments/01_multitask_capacity/plot_all.py --outputs-dir outputs

GPUS=0,1,2,3,4 bash experiments/02_hypernetwork_multitask/run.sh
uv run python experiments/02_hypernetwork_multitask/plot_all.py

bash experiments/03_reusability_generate_once_execute_many/run.sh
uv run python experiments/03_reusability_generate_once_execute_many/plot_generalization_loo.py --outputs-dir outputs

bash experiments/04_compositional_generalization/run.sh
uv run python experiments/04_compositional_generalization/plot_compositional.py --outputs-dir outputs
uv run python experiments/04_compositional_generalization/report_holdout_breakdown.py

GPUS=0,1,2,3,4,5,6,7 bash experiments/05_leave_one_out_task_generalization/run.sh
uv run python experiments/05_leave_one_out_task_generalization/plot_leave_one_out.py --outputs-dir outputs

GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/hypernetwork/run.sh
GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/joint/run.sh
GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/individual/run.sh
uv run python experiments/06_data_efficiency_ablation/aggregate_results.py
uv run python experiments/06_data_efficiency_ablation/plot_data_efficiency.py --outputs-dir outputs
```

Every training `run.sh` wraps [`scripts/run_config.sh`](scripts/run_config.sh) and accepts:

| Variable | Meaning |
|---|---|
| `GPUS` | comma-separated GPU ids; one job per GPU, each GPU takes the next job when free (unset: one job at a time) |
| `SHARD` | `k/N`: run only every N-th job starting at the k-th, to split a sweep across nodes |
| `SEEDS_OVERRIDE` | seeds to run (default `1 2 3 4 5`) |
| `PROJECT` | output folder and W&B project (defaults to the experiment's name) |
| `PYTHON` | interpreter (default `.venv/bin/python`) |

Finished jobs are skipped, so rerunning the same command resumes a sweep. Everything is
written under `outputs/` (not tracked by git):

- `outputs/<experiment>/<run>/`: `results.txt` (final validation and test metrics),
  `config.yaml` (the resolved config), `model.txt`, `train.log`, and the trained model:
  `best_model.ckpt` and `last.ckpt`, plus `final_model.ckpt` when the experiment evaluates its
  final weights.
- `outputs/results/<experiment>/*.csv`: numbers aggregated across runs by the plot scripts.
- `outputs/figures/<experiment>/*.{png,pdf}`: the figures.

The per-experiment READMEs give the setup, the figures produced, the results and the
reproducibility check for each experiment. Figures of the task categories themselves:
`uv run python -m visualisation.paper.plot_paper_tasks --first-per-category` (add
`--compositional` for the composite ones).

## Running on a GPU cluster

Launch sweeps under `nohup` (or `tmux`) so they survive the SSH session, and split a sweep
across nodes with `SHARD`, for example on two nodes:

```bash
SHARD=0/2 GPUS=0,1,2,3 nohup bash experiments/05_leave_one_out_task_generalization/run.sh > shard0.log 2>&1 &   # node A
SHARD=1/2 GPUS=0,1,2,3 nohup bash experiments/05_leave_one_out_task_generalization/run.sh > shard1.log 2>&1 &   # node B
```

`scripts/fetch_experiments.sh` copies an experiment's outputs, models included, from a node
(`REMOTE_HOST=torrnode15.priv bash scripts/fetch_experiments.sh 02_hypernetwork_multitask`;
`EXCLUDE_CHECKPOINTS=1` skips the models).

## How a run works

`train.py --config <file>.yaml [key=value ...]` loads the config (one level of `_base_`
inheritance plus command-line overrides), builds the Lightning module named by `model` and the
data module named by `data`, trains for `max_steps`, evaluates, and writes the run folder.
`validate.py --config outputs/<experiment>/<run>/config.yaml --checkpoint best|last|<path>`
re-evaluates a saved run.

The model part of a config:

```yaml
model: hypernetwork            # or: direct
evaluate_on: best              # best checkpoint by primary_metric, or final weights
task_encoding:
  embedding_dim: 4
  use_sinusoidal_pe: false     # position comes from RoPE in attention
target:                        # the generated model (direct models use `backbone:` instead)
  hidden_dim: 4
  num_layers: 3
  num_heads: 1
  dropout: 0.1
  canon_set: ABCD              # Canon layer positions; '' for none
  canon_kernel: 5
encoder:                       # the hypernetwork's Transformer
  hidden_dim: 4
  num_layers: 1
  num_heads: 1
  output_dim: 4
hyper_head:
  bottleneck_dim: 8
  num_tasks: 18                # one-hot task identity; null for none
  freeze_task_indicator: true
optimizer: Muon                # or AdamW, RAdam
```

## Code layout

```text
train.py, validate.py      entry points
models/                    network architectures (plain PyTorch modules)
  transformer.py             the RoPE + Canon Transformer used everywhere
  canon.py, rope.py          Canon layer, rotary position embedding
  embedding.py               token embeddings and the task-category index
  hypernetwork.py            encoder, pooling, weight decoder, and running the generated target
lightning_modules/         training: loss, metrics, optimiser, logging
  direct.py                  trains one Transformer directly (individual and joint models)
  hypernetwork.py            trains the hypernetwork
  common.py                  shared optimiser (Muon, AdamW, RAdam), schedule, per-category metrics
data_modules/              ARC-1D datasets: direct pairs, task episodes, compositional generator
training/                  run mechanics: config loading, trainer and callbacks, run artefacts
experiments/               one folder per paper experiment: configs, run.sh, plot scripts, README
scripts/                   data build, shared launcher, evaluation scripts, fetching outputs
visualisation/             figure rendering (core/ shared, paper/ paper-only figures)
tests/                     pytest suite, see tests/README.md
.agents/                   conventions for coding agents working in this repository
```

## Tests

```bash
uv run pytest            # about a minute; trains tiny models, so use a machine with CPU to spare
uv run ruff check .
```

See [`tests/README.md`](tests/README.md). In particular `tests/test_structure_snapshot.py`
checks that every experiment config still builds exactly the model it built before the
`models/` refactor, so refactoring cannot silently change an experiment.
