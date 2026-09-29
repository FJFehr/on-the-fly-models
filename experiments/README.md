# Paper experiments

One folder per experiment, in the order of the paper's argument. Each folder's README has
the question, setup, figures, results and cost. Installing and building the data are in the
[top-level README](../README.md).

1. [**Multi-task capacity**](01_multitask_capacity/README.md): one small model cannot hold all
   14 tasks at once, although it can solve each alone; a task-identity embedding helps, and
   Canon layers and Muon matter most at small sizes.
2. [**Hypernetwork**](02_hypernetwork_multitask/README.md): generating per-task weights from
   the support examples recovers the individual-model level with task identity, at a matched
   parameter budget.
3. [**Reusability**](03_reusability_generate_once_execute_many/README.md): weights generated
   from one instance solve other instances of the same task almost as well (except `mirror`).
4. [**Compositional generalisation**](04_compositional_generalization/README.md): without task
   identity the hypernetwork generalises better to unseen combinations of known rules.
5. [**Leave-one-out**](05_leave_one_out_task_generalization/README.md): the same holds for a
   whole unseen task category (13 of 14).
6. [**Data efficiency**](06_data_efficiency_ablation/README.md): individual, joint and
   hypernetwork models at shrinking amounts of training data.
8. [**Optimiser tuning**](08_optimiser_tuning/README.md): a Muon grid and a separately tuned
   plain-AdamW grid on the task-ID-free hypernetwork, to pick the recipe for later runs and
   compare the two optimisers fairly.

| | Task identity arms | Other axes |
|---|---|---|
| 1 | `td` (learned embedding) / `notd` | size (dim 4 to 14), Canon, optimiser |
| 2 to 5 | `frozen_td` (fixed random one-hot projection) / `notd` | |
| 6 | hypernetwork `frozen_td`/`notd`, joint `td`/`notd`, individual none | training data amount |
| 8 | hypernetwork `notd` | Muon and AdamW learning rates, weight decay |

## Every command

```bash
GPUS=0,1,2,3,4,5,6,7 bash experiments/01_multitask_capacity/run.sh                  # 960 jobs
uv run python experiments/01_multitask_capacity/plot_all.py --outputs-dir outputs

GPUS=0,1,2,3,4 bash experiments/02_hypernetwork_multitask/run.sh                    # 10 jobs
uv run python experiments/02_hypernetwork_multitask/plot_all.py

bash experiments/03_reusability_generate_once_execute_many/run.sh                   # needs 2
uv run python experiments/03_reusability_generate_once_execute_many/plot_generalization_loo.py --outputs-dir outputs

bash experiments/04_compositional_generalization/run.sh                             # needs 2
uv run python experiments/04_compositional_generalization/plot_compositional.py --outputs-dir outputs
uv run python experiments/04_compositional_generalization/report_holdout_breakdown.py

GPUS=0,1,2,3,4,5,6,7 bash experiments/05_leave_one_out_task_generalization/run.sh   # 140 jobs
uv run python experiments/05_leave_one_out_task_generalization/plot_leave_one_out.py --outputs-dir outputs

GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/hypernetwork/run.sh       # 120 jobs
GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/joint/run.sh              # 120 jobs
GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/individual/run.sh         # 630 jobs
uv run python experiments/06_data_efficiency_ablation/aggregate_results.py
uv run python experiments/06_data_efficiency_ablation/plot_data_efficiency.py --outputs-dir outputs

GPUS=0,1,2,3 JOBS_PER_GPU=2 bash experiments/08_optimiser_tuning/run.sh             # 225 jobs
uv run python experiments/08_optimiser_tuning/plot_all.py
```

Figures of the task types themselves:
`uv run python -m visualisation.paper.plot_paper_tasks --first-per-category` (add
`--compositional` for experiment 4's composite ones).

## Run options

Every training `run.sh` wraps [`scripts/run_config.sh`](../scripts/run_config.sh), and
finished jobs are skipped, so rerunning a command resumes a sweep.

| Variable | Meaning |
|---|---|
| `GPUS` | GPUs to use: a list such as `0,1,3`, or `all` (unset: your default GPU) |
| `JOBS_PER_GPU` | jobs to run at once on each GPU (default 1); a job needs about 350 MB of GPU memory, 1 CPU core and 3 to 5 GB of RAM, so this is usually limited by CPU and RAM |
| `SHARD` | `k/N`: run only every N-th job starting at the k-th, to split a sweep across machines |
| `SEEDS_OVERRIDE` | seeds to run (default `1 2 3 4 5`) |
| `PROJECT` | output folder and W&B project (defaults to the experiment's name) |
| `PYTHON` | interpreter (default `.venv/bin/python`) |

Examples:

```bash
bash experiments/05_leave_one_out_task_generalization/run.sh                       # one GPU, one job at a time
JOBS_PER_GPU=4 bash experiments/05_leave_one_out_task_generalization/run.sh        # one GPU, 4 jobs at a time
GPUS=all JOBS_PER_GPU=2 bash experiments/05_leave_one_out_task_generalization/run.sh
```

To split a sweep across machines, give each its own shard, and launch under `nohup` (or
`tmux`) so it survives the terminal closing:

```bash
SHARD=0/2 GPUS=all nohup bash experiments/05_leave_one_out_task_generalization/run.sh > shard0.log 2>&1 &   # machine A
SHARD=1/2 GPUS=all nohup bash experiments/05_leave_one_out_task_generalization/run.sh > shard1.log 2>&1 &   # machine B
```

`scripts/fetch_experiments.sh` copies an experiment's outputs, models included, back from
another machine
(`REMOTE=user@host:/path/to/on-the-fly-models bash scripts/fetch_experiments.sh 05_leave_one_out_task_generalization`;
`EXCLUDE_CHECKPOINTS=1` skips the models).

If every training job fails instantly with a `PermissionError` from wandb, `uv sync` dropped
the executable bit on an NFS home directory:
`chmod +x .venv/lib/python3.12/site-packages/wandb/bin/{wandb-core,gpu_stats}`.

## Outputs

Everything goes under `outputs/`, which git ignores:

- `outputs/<experiment>/<run>/`: `results.txt` (final validation and test metrics),
  `config.yaml` (the resolved config), `model.txt`, `train.log`, `compute.json` (GPU model
  and wall-clock minutes, totalled by `python scripts/compute_cost.py`), and the trained model
  (`best_model.ckpt`, `last.ckpt`, and `final_model.ckpt` when the experiment evaluates its
  final weights).
- `outputs/results/<experiment>/*.csv`: numbers aggregated across runs by the plot scripts.
- `outputs/figures/<experiment>/*.{png,pdf}`: the figures.

## Data

| Dataset | Contents | Used by |
|---|---|---|
| `data/arc_1d` | the raw benchmark: 18 task types, 901 tasks (721 train, 90 dev, 90 test) | the augmentation step |
| `data/arc_1d_looped_augmented` | 721,000 train, 1,800 dev and 1,800 test tasks | experiments 1 to 6 |
| `data/arc_1d_compositional_holdout` | 10 synthetic composite task types, 400 tasks | experiment 4 |

1D-ARC is fetched at a pinned commit and all randomness is seeded, so a rebuild is identical
row for row. Augmentation remaps non-zero colours (colour 9 stays fixed for `1d_mirror`,
where it is the pivot) and shifts every sequence by padding with zeros, the same transform
for all sequences of a task, so the rule is preserved; splits are made before augmenting.
`uv run pytest -m slow tests/test_data_leakage.py` checks that no training example appears in
the dev, test or holdout splits.

## Configs

`python train.py --config <file>.yaml [key=value ...]` trains one config: one level of
`_base_` inheritance, then command-line overrides. The model part of a config:

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

`validate.py --config outputs/<experiment>/<run>/config.yaml --checkpoint best|last|<path>`
re-evaluates a saved run.

## Folder layout

```
0N_experiment_name/
  README.md       question, setup, run and plot commands, results, cost
  configs/        the configs train.py reads (base.yaml plus one file per cell)
  gen_*.py        writes generated configs (rerunning reproduces them exactly)
  run.sh          runs every config
  plot_*.py       aggregates outputs/ into outputs/results/ and outputs/figures/
```

`tests/test_structure_snapshot.py` checks that every config here still builds exactly the
model it built before the `models/` refactor, so refactoring cannot silently change an
experiment.
