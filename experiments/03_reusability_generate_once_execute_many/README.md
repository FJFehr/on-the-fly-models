# Experiment 3: reusability ("generate once, execute many")

## Question

Do the weights generated from one instance of a task also solve other instances of the same
task category? If so, one generation can be reused across instances, and the hypernetwork's
cost per task is paid once.

## Setup

Evaluation only, no training. Uses experiment 2's 10 dim-4 checkpoints (both arms, 5 seeds,
`best_model.ckpt`). For each task category, every instance in the validation and test splits
(200 per category) takes a turn as the reference: its generated weights are scored on every
other instance of that category ("cross-instance", leave-one-out) and compared with each
instance solving its own query ("own instance"). Implemented by
`scripts/measure_compute_efficiency.py --stage generalization`.

## Run

Needs experiment 2's checkpoints in `outputs/02_hypernetwork_multitask/`.

```bash
bash experiments/03_reusability_generate_once_execute_many/run.sh
```

10 evaluations (2 arms x 5 seeds), a few minutes in total. Uses a GPU if one is visible.
`CKPT_DIR` points it at another run of experiment 2.

## Plots

```bash
uv run python experiments/03_reusability_generate_once_execute_many/plot_generalization_loo.py --outputs-dir outputs
```

Writes `outputs/results/03_reusability_generate_once_execute_many/results_generalization_loo{,_raw}.csv`
and `outputs/figures/03_reusability_generate_once_execute_many/generalization_loo.{png,pdf}`.

## Outputs

`outputs/03_reusability_generate_once_execute_many/hyper_multitask_dim4_{notd,frozentd}_seed{1..5}/`:
`generalization_table.csv` (per category) and `generalization.json`.

## Result

Exact match, macro-average over the 14 categories (each category averaged over 5 seeds):

| Arm | Own instance | Cross-instance | Difference |
|---|---:|---:|---:|
| frozen_td | 94.0% | 91.8% | -2.2 points |
| notd | 60.5% | 58.4% | -2.1 points |

Reuse costs almost nothing: in 12 of 14 categories the drop is under 5 points (mostly under
1) for both arms. `mirror` is the clear exception (frozen_td 84.5% to 56.4%, notd 51.7% to
38.3%), with a smaller drop for `fill` in notd (-6.5 points). So reusability holds almost
everywhere, with or without task identity; the exception is specific to one category.

**Reproducibility.** Re-evaluated with the refactored code on the rerun checkpoints of
experiment 2: all 1,260 table entries match, the largest difference being 0.0001 (GPU
rounding).

## Cost

Evaluation only: minutes on one GPU.
