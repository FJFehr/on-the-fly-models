# Experiment 6: data efficiency

## Question

Does sharing across tasks let a model learn each task from less data? Three arms are trained
at progressively smaller amounts of training data:

| Arm | Model | Parameters (total) | Parameters used per prediction | Sharing across tasks | Task identity |
|---|---|---:|---:|---|---|
| [`hypernetwork/`](hypernetwork/README.md) | experiment 2's dim-4 hypernetwork | 11.4K | 1,204 (the generated dim-4 target) | generated weights | `frozen_td` / `notd` |
| [`joint/`](joint/README.md) | one direct dim-14 model for all 14 categories | 9.7K | 9.7K (all of them) | one shared model | `td` / `notd` |
| [`individual/`](individual/README.md) | one direct dim-4 model per category | 1.4K per category | 1.4K | none | none |

The joint model is sized to the hypernetwork's total parameter budget, and `individual` is
the no-sharing baseline at the hypernetwork's target size. The two shared models do
fundamentally different things, though: the hypernetwork decouples the task from the model
that executes it, so every prediction runs through a generated 1,204-parameter target,
while the joint model runs every prediction through all of its 9.7K parameters.

## Data levels

Only the training data is reduced; validation and test sets are the same at every level.
Selection is deterministic (`data_seed` 42, independent of the training seed) and nested, so
a smaller level is a subset of a larger one.

| Level | Training data per category |
|---|---|
| `t1`, `t3`, `t5`, `t10`, `t20` | 1, 3, 5, 10 or 20 base tasks, original examples only (`base_tasks_per_category`) |
| `v1`, `v2`, `v3`, `v4`, `v5`, `v20` | all 40 base tasks, with 1 (original only), 2, 3, 4, 5 or 20 variants each (`variants_per_base_task`) |
| `full` | no reduction (about 40,000 augmented examples) |

`individual` runs `t*`, `v1`, `v2`, `v3` and `full`. Training compute is fixed across levels
within each arm (8,000 steps), and every arm evaluates its final weights
(`evaluate_on: final`), 5 seeds per cell. Each arm's `gen_configs.py` writes its configs.

## Run

```bash
GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/hypernetwork/run.sh   # 120 jobs
GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/joint/run.sh          # 120 jobs
GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/individual/run.sh     # 630 jobs
```

## Plots

```bash
uv run python experiments/06_data_efficiency_ablation/aggregate_results.py
uv run python experiments/06_data_efficiency_ablation/plot_data_efficiency.py --outputs-dir outputs
```

`aggregate_results.py` writes per-arm CSVs to `outputs/results/06_data_efficiency_ablation/`.
`plot_data_efficiency.py` writes `data_efficiency_cliff.{png,pdf}` (individual vs
hypernetwork) to `outputs/figures/06_data_efficiency_ablation/`; `--with-joint` adds a second
figure with the joint arm, and `--drop-mid-augmentations` leaves out `v2` to `v4`.

## Outputs

`outputs/06_data_efficiency_ablation_{hypernetwork,joint,individual}/<run>/`: `results.txt`
and the models (`final_model.ckpt` is the one evaluated).

## Result

Test exact match, mean over 5 seeds (macro over categories for `individual`):

| Level | Individual | Hypernetwork, frozen_td | Hypernetwork, notd | Joint, td | Joint, notd |
|---|---:|---:|---:|---:|---:|
| t1 | 10.0% | 7.1% | 2.9% | 17.7% | 10.3% |
| t3 | 33.7% | 25.7% | 10.0% | 57.4% | 19.1% |
| t5 | 50.0% | 42.6% | 20.3% | 71.1% | 27.4% |
| t10 | 65.7% | 68.3% | 31.4% | 84.3% | 42.6% |
| t20 | 81.1% | 84.9% | 47.7% | 93.4% | 47.7% |
| v1 | 85.4% | 89.4% | 65.1% | 98.0% | 57.1% |
| v2 | 91.7% | 94.3% | 64.3% | 98.0% | 59.7% |
| v3 | 92.3% | 94.6% | 63.4% | 98.9% | 62.9% |
| v20 | | 95.4% | 61.4% | 98.9% | 64.9% |
| full | 94.6% | 92.9% | 63.1% | 98.9% | 66.9% |

With task identity, the hypernetwork is ahead of individual models of its target's size
from `t10` to `v3` (for example `v1`: 89.4% vs 85.4%), where sharing across tasks makes up
for the missing data; with very little data (`t1` to `t5`) and at full data the individual
models are ahead. The joint model is ahead of both at every level, as expected from the
parameters each uses per prediction: the joint model executes with all of its 9.7K
parameters, while the hypernetwork has to distil everything a task needs into the weights of
a 1,204-parameter target (8 times fewer, the size of an individual model). This is the cost
of decoupling the task from the executing model, not a failure of the hypernetwork. Without
task identity, neither shared model reaches the individual models' level at any data
amount.

**Reproducibility.** Not yet re-run on the refactored code. These numbers come from the
original runs, which were evaluated on an older build of the validation and test splits
with about 5 tasks per category (the current build, used by experiments 1 to 5, has about
100), so they will move somewhat when re-run.

## Cost

870 jobs, about 340 A40 GPU-hours in the original runs (hypernetwork `t*` cells about 2 hours
each, `v*`/`full` about 30 minutes; individual `v*`/`full` about 30 minutes, `t*` about 6
minutes; joint 10 to 20 minutes).
