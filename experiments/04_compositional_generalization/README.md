# Experiment 4: compositional generalisation

## Question

A hypernetwork trained on the 14 base task categories is tested zero-shot on 10 new
categories that chain two of those rules (for example denoise a block, then shift it). Does
it generalise better without a task-identity anchor (`notd`) than with one (`frozen_td`)?
The hypothesis: without an explicit anchor the model must build its task representation from
the support examples alone, which should blend two rules more naturally.

## Setup

Evaluation only, no training: both arms are experiment 2's dim-4 checkpoints
(`best_model.ckpt`, 5 seeds each), so the in-distribution numbers are experiment 2's.

- **Held-out set**: `data/arc_1d_compositional_holdout`, built by
  `scripts/build_arc1d_compositional.py` from `data_modules/arc1d_compositional.py`: 10
  composite categories, 40 instances each. Each composite applies one base rule and then a
  second (for example `shift3(denoise_1c(x))`), drawn from `denoise_1c`, `denoise_mc`,
  `shift`, `move_dynamic`, `mirror`, `fill`, `hollow` and `copy`. Pairs were excluded when
  they cancel out (`hollow` + `fill` returns the input), when their conventions do not hand
  off cleanly (`denoise_1c` + `fill`), or when they did not work out on visual review
  (`move_dynamic` + `mirror`); the `recolor` categories are not used.
- **notd** is evaluated directly (`scripts/eval_compositional_holdout.py`).
- **frozen_td** needs a task-indicator column for each new category, so
  `pad_frozentd_checkpoint.py` widens its projection from 18 to 28 columns before evaluation.
  This is exact, not an approximation: with `freeze_task_indicator` the projection is random
  and never trained, so 10 extra columns drawn from the same initialisation are
  indistinguishable from having built the model with 28. The padded model is built from
  `configs/frozen_td.yaml`.

## Run

Needs experiment 2's checkpoints in `outputs/02_hypernetwork_multitask/`.

```bash
bash experiments/04_compositional_generalization/run.sh
```

10 evaluations (2 arms x 5 seeds) on the CPU, a few minutes. `CKPT_DIR` points it at another
run of experiment 2.

## Plots

```bash
uv run python experiments/04_compositional_generalization/plot_compositional.py --outputs-dir outputs
uv run python experiments/04_compositional_generalization/report_holdout_breakdown.py
```

Writes `results_indist.csv`, `results_holdout.csv` and `results_holdout_paper_table.csv` (the
per-composition paper table) to `outputs/results/04_compositional_generalization/`, and
`per_task_indist` and `per_task_holdout` figures to
`outputs/figures/04_compositional_generalization/`.

## Outputs

`outputs/04_compositional_generalization/{notd,frozentd}_seed{1..5}/results.txt`, and the
padded frozen_td models in `outputs/04_compositional_generalization/padded_checkpoints/`.

## Result

Macro-average over categories, mean over 5 seeds:

| Arm | In-distribution exact match (14) | Held-out token accuracy (10) | Held-out exact match |
|---|---:|---:|---:|
| frozen_td | **93.0%** | 67.7% | 0.05% |
| notd | 58.3% | **75.5%** | 1.00% |

The hypothesis holds: `notd` has higher held-out token accuracy in 8 of 10 composite
categories (largest gap `denoisemc_mirror`, 85.6% vs 47.7%; `frozen_td` ahead only on
`denoisemc_copy` and `denoisemc_denoise1c`), while `frozen_td` dominates in-distribution.
Exact match on the held-out set stays near zero for both arms.

**Reproducibility.** Re-evaluated with the refactored code on the rerun checkpoints of
experiment 2: all 10 results files are identical.

## Cost

Evaluation only, on the CPU: about 0.7 minutes per checkpoint, under 10 minutes for all 10.
