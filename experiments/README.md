# Paper experiments

One folder per experiment, in the order of the paper's argument. Setup, data build and the
commands to reproduce everything are in the [top-level README](../README.md); each folder's
README has the question, setup, figures, results and cost.

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

## Arms across experiments

| | Task identity | Other axes |
|---|---|---|
| 1 | `td` (learned embedding) / `notd` | size (dim 4 to 14), Canon, optimiser |
| 2 to 5 | `frozen_td` (fixed random one-hot projection) / `notd` | |
| 6 | hypernetwork `frozen_td`/`notd`, joint `td`/`notd`, individual none | training data amount |

## Folder layout

```
0N_experiment_name/
  README.md       question, setup, run and plot commands, results, cost
  configs/        the configs train.py reads (base.yaml plus one file per cell)
  gen_*.py        writes generated configs (rerunning reproduces them exactly)
  run.sh          runs every config (see the top-level README for its options)
  plot_*.py       aggregates outputs/ into outputs/results/ and outputs/figures/
```
