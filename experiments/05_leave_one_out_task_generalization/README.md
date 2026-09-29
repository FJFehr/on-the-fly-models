# Experiment 5: leave-one-out task generalisation

## Question

A hypernetwork is trained on 13 of the 14 base task categories and tested zero-shot on the
14th, which it never saw in any form. Every category is held out in turn. Does it generalise
better without a task-identity anchor (`notd`) than with one (`frozen_td`)? Unlike experiment
4, the held-out category is a whole base rule, not a combination of known ones.

## Setup

- **Model and training**: experiment 2's dim-4 recipe unchanged (10,156 trainable
  hypernetwork parameters, same optimiser, 8,000 steps). Evaluation uses the final weights
  (`evaluate_on: final`).
- **Held-out category**: each config's `task_categories` lists the 13 training categories,
  while `val_task_categories` always lists all 14. The held-out category's zero-shot score is
  therefore in the run's own `results.txt` (`val_query_*_by_task_<category>`), with no
  separate evaluation step.
- **Arms**: `notd` (`hyper_head.num_tasks: null`) and `frozentd` (`num_tasks: 18`,
  `freeze_task_indicator: true`; the frozen projection keeps a column for every category, so
  the architecture does not change with the held-out category).
- 14 held-out categories x 2 arms x 5 seeds = 140 configs, written by `gen_configs.py` with
  the seed built into each config.

## Run

```bash
GPUS=0,1,2,3,4,5,6,7 bash experiments/05_leave_one_out_task_generalization/run.sh
```

## Plots

```bash
uv run python experiments/05_leave_one_out_task_generalization/plot_leave_one_out.py --outputs-dir outputs
```

Writes CSVs to `outputs/results/05_leave_one_out_task_generalization/` and
`per_category_holdout.{png,pdf}` to `outputs/figures/05_leave_one_out_task_generalization/`.

## Outputs

`outputs/05_leave_one_out_task_generalization/<category>_{notd,frozentd}_seed{1..5}/`:
`results.txt` and the models (`final_model.ckpt` is the one evaluated).

## Result

Validation set, macro-average over categories, mean over 5 seeds:

| Arm | Held-out token accuracy | Held-out exact match | In-distribution token accuracy | In-distribution exact match |
|---|---:|---:|---:|---:|
| frozen_td | 67.5% | 0.4% | **99.7%** | **94.5%** |
| notd | **81.6%** | 10.2% | 96.6% | 62.9% |

`notd` has the higher held-out token accuracy in 13 of 14 categories; `frozen_td` dominates
in-distribution. This is the same trade-off as experiments 2 and 4, now at full
category-level coverage.

| Held-out category | notd | frozen_td | Difference (points) |
|---|---:|---:|---:|
| `1d_denoising_mc` | 81.2% | 48.4% | +32.9 |
| `1d_scale_dp` | 87.0% | 59.5% | +27.5 |
| `1d_pcopy_mc` | 84.2% | 58.2% | +25.9 |
| `1d_move_3p` | 89.5% | 65.9% | +23.6 |
| `1d_move_2p_dp` | 94.9% | 72.2% | +22.7 |
| `1d_pcopy_1c` | 99.7% | 77.8% | +21.9 |
| `1d_move_1p` | 92.1% | 80.3% | +11.7 |
| `1d_flip` | 82.8% | 72.3% | +10.6 |
| `1d_move_2p` | 92.2% | 82.4% | +9.8 |
| `1d_move_dp` | 86.1% | 77.6% | +8.4 |
| `1d_fill` | 69.6% | 63.0% | +6.6 |
| `1d_hollow` | 72.2% | 66.2% | +6.1 |
| `1d_mirror` | 63.0% | 56.9% | +6.0 |
| `1d_denoising_1c` | 48.5% | 63.7% | -15.1 |

`1d_denoising_1c` is the one category where `frozen_td` is ahead; it has not been
investigated further.

**Reproducibility.** These numbers are from the rerun on the refactored code with the data
rebuilt from scratch. An earlier run evaluated on a smaller validation set (about 5 tasks per
category instead of about 100) found the same 13 of 14 split (held-out token accuracy 81.9%
vs 68.5%).

## Cost

Wall-clock minutes per run on one NVIDIA A40 (one run per GPU), including data setup and evaluation. Measured on the refactor rerun: 140 runs of about 23 minutes, 53 A40 GPU-hours.
