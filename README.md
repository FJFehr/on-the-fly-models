# on-the-fly-models

**Can a network write the weights of a small model for a task it is shown, instead of us
training one model per task?**

The tasks come from [1D-ARC](https://github.com/khalil-research/1D-ARC): short
one-dimensional puzzles such as "move the block one step right", "fill the gap" or "mirror
the pattern" (14 task types are used). Each task comes with three input-output examples and
a query input to solve.

A **hypernetwork** reads the three examples and generates all the weights of a tiny
Transformer (about 1,200 parameters), which then solves the query. We compare it with
training one small model per task type and with one shared model trained on every task, and
ask when generating weights helps.

## Reproduce

```bash
# 1. Install (creates .venv)
uv sync --python 3.12 --managed-python

# 2. Build the data (deterministic, a few minutes on CPU)
uv run python -m scripts.build_arc_1d
uv run python -m scripts.augment_arc_1d --per-pair --n-color-permutations 199 \
    --shifts 1 2 -1 -2 --no-mirror --dev-test-n-permutations 19 \
    --output-dir data/arc_1d_looped_augmented
uv run python -m scripts.build_arc1d_compositional

# 3. Run an experiment on GPUs 0-3, then make its figures (here experiment 1)
GPUS=0,1,2,3 bash experiments/01_multitask_capacity/run.sh
uv run python experiments/01_multitask_capacity/plot_all.py --outputs-dir outputs
```

Every experiment works the same way: one `run.sh`, then its plot script. Results, trained
models and figures are written to `outputs/`. Runs log to Weights & Biases; set
`WANDB_MODE=offline` to keep them local.

## Experiments

| | Question | Finding | A40 GPU-hours |
|---|---|---|---:|
| [1](experiments/01_multitask_capacity/README.md) | Can one small model learn all 14 tasks at once? | No: it solves each alone but not all together | 64 |
| [2](experiments/02_hypernetwork_multitask/README.md) | Does generating weights close that gap? | Yes, with the task's identity: 93% vs 92.6% for per-task models | 4 |
| [3](experiments/03_reusability_generate_once_execute_many/README.md) | Do weights generated from one example set work for other instances? | Yes, losing about 2 points | minutes |
| [4](experiments/04_compositional_generalization/README.md) | Unseen combinations of known rules? | Better without the task's identity (75.5% vs 67.7%) | minutes |
| [5](experiments/05_leave_one_out_task_generalization/README.md) | A task type never seen in training? | Better without the task's identity, 13 of 14 types | 57 |
| [6](experiments/06_data_efficiency_ablation/README.md) | How much training data is needed? | Sharing across tasks helps at moderate data sizes | 340 |

Run experiment 2 before 3 and 4, which evaluate its models. Each experiment's README has the
full setup, commands, figures and results; [`experiments/README.md`](experiments/README.md)
covers running on several GPUs or nodes, the output layout and the config format.

## Code

```text
models/              the networks: Transformer, hypernetwork, embeddings
lightning_modules/   training: direct models and the hypernetwork
data_modules/        the ARC-1D datasets
experiments/         one folder per experiment: configs, run.sh, plots, README
scripts/             data build, the shared launcher, evaluation scripts
train.py             trains one config: python train.py --config <file>.yaml
tests/               uv run pytest (see tests/README.md)
```
