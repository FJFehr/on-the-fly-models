# Experiment 3: reusability ("Generate Once, Execute Many")

## Question

Does a single generated weight set -- from one support-set instance of a task -- solve
*other* instances of the same task category, or does the hypernetwork need to regenerate
weights per instance? If one generation reuses across instances, the hypernetwork's per-task
cost is amortised: generate once, execute many times.

## The finding

**With a task-identity signal, cross-instance reuse costs almost nothing on average --
own-instance 94.0% mean exact match drops to 91.8% cross-instance (-2.2pp). Without one, it's
the same story (60.5% -> 58.4%, -2.1pp): the drop from reusing weights across instances is
small either way. What differs is the absolute level, matching every other result in this
repo -- `frozen_td` starts and stays far higher than `notd`. One category, `mirror`, is a
sharp exception for both arms: -28.1pp (`frozen_td`) and -13.4pp (`notd`), a real,
category-specific failure to generalise across instances, not noise.**

| Condition | Same-instance (own) | Cross-instance (leave-one-out) | diff |
|---|---:|---:|---:|
| `frozen_td` | 94.0% ± 13.1pp | 91.8% ± 16.4pp | -2.2pp |
| `notd` | 60.5% ± 35.5pp | 58.4% ± 35.8pp | -2.1pp |

(macro-average across the 14 task categories, mean ± population s.d. of those 14 per-category
means -- each per-category mean is itself already averaged across 5 seeds, not a single run.)

![Own vs. cross-instance generalisation](../../outputs/figures/03_reusability_generate_once_execute_many/generalization_loo.png)

*(Not committed -- see "Figures and results" in `experiments/README.md`. Regenerate with
`plot_generalization_loo.py --outputs-dir outputs` once `run.sh` has produced the raw data.)*

**12 of 14 categories show essentially zero degradation** (within ±5pp, most within ±1pp) for
both conditions -- for the large majority of task categories, one instance's generated weights
solve every other instance in that category just as well as its own. `mirror` is the one real
outlier for both arms (`frozen_td`: 84.5% own -> 56.4% loo; `notd`: 51.7% -> 38.3%), and to a
lesser extent `fill` for `notd` (77.0% -> 70.5%, -6.5pp). Every other category's own/loo gap is
under 5pp. **This is the headline result: reusability holds almost everywhere, task-identity
signal or not -- the exception is category-specific, not a general property of removing the
identity signal.**

This is a genuinely different axis from experiment 2's own move-family finding (task-ID
signal vs. not, at the *same* instance) -- here the question is whether a *fixed* generation
transfers to *other* instances of the same category, and the answer is mostly yes for both
conditions, `mirror` aside.

## Method

**Models**: experiment 2's own dim=4 matched-scale hypernetwork checkpoints
(`outputs/02_hypernetwork_multitask/hyper_multitask_dim4_{notd,frozentd}_seed{1..5}/`,
10,156 trainable params each, real checkpoints -- see that experiment's README's "Dim=4
rerun"). All 5 seeds, both arms -- no cherry-picked "best" seed (the original design called
for picking one; using all 5 is strictly more robust and was already sitting there once the
rerun landed).

**Measurement**: `scripts/measure_compute_efficiency.py --stage generalization` (Phase E,
pre-existing, shared infra also documented in that script's own module docstring). For each
of the 14 task categories, every instance in the eval pool takes a turn as the reference --
its generated weights (from its own support set) scored against every *other* instance's own
support set + query in that category ("cross-instance" / leave-one-out), compared against each
instance solving only its own query ("same-instance"). Not a single fixed reference: for a
category with *n* instances, every instance in turn is the reference, giving *n* leave-one-out
accuracies per category, so the result reflects whether it matters *which* instance you
generate from, not just whether the first one happens to work.

**Data**: `--generalization-split val,test` -- both splits of `data/arc_1d_looped_augmented`
combined (100 rows/category each, confirmed by loading the dataset directly -- 200/category
total), all from data this repo already builds from scratch. Deliberately not the earlier
preliminary pass's separate `devtest_shifted` dataset (richer, but hand-built with no
reproducible build command in this repo).

**Compute**: no training, no GPU, no cluster. ~30s/cell on CPU, 10 cells (2 arms x 5 seeds),
a few minutes total -- a genuinely different reproduction story from experiments 1/2/4, which
all need cluster training.

## Reproducing

One script to run, one to plot -- same pattern as every other experiment:

```bash
bash experiments/03_reusability_generate_once_execute_many/run.sh
uv run python experiments/03_reusability_generate_once_execute_many/plot_generalization_loo.py --outputs-dir outputs
```

`run.sh` loops the 10 known (condition, seed) cells directly against
`outputs/02_hypernetwork_multitask/`'s own checkpoint dirs (each checkpoint's own saved
`config.yaml` is the "config" -- no `configs/` folder of its own needed here, since nothing
trains). Skips any cell whose `generalization_table.csv` already exists, so it's always safe
to rerun. If a checkpoint is missing locally, fetch it first:

```bash
REMOTE_HOST=<node> INCLUDE_CHECKPOINTS=1 bash scripts/fetch_experiments.sh 02_hypernetwork_multitask
```

`plot_generalization_loo.py --outputs-dir outputs` rescans the 10 raw
`generalization_table.csv` files, writes `results_generalization_loo_raw.csv` (140 rows, one
per condition/seed/category -- provenance) and `results_generalization_loo.csv` (28 rows, one
per condition/category, averaged across the 5 seeds -- what the figure reads), prints the
macro-average summary above, and renders `generalization_loo.png`/`.pdf`. Without
`--outputs-dir`, it replots from the committed-shape CSV as-is.

## Status

Done. 5 seeds, both arms, no cherry-picking -- the full design this experiment was blocked on
until experiment 2's dim=4 checkpoints existed at the right (matched) scale.

## Open follow-ups

- `mirror`'s sharp own/loo drop (both arms) isn't diagnosed further here -- is it a specific
  instance-to-instance sensitivity in how `mirror` tasks vary, or something about the
  category's support-set structure? Worth a closer look before drawing a general conclusion.
- Only evaluated at dim=4 (matched-scale) -- dim=6 isn't resized to match yet (see experiment
  2's own "Open follow-ups"), so this hasn't been checked at that size.
- Only same-category reuse tested (a category's generated weights only ever evaluated against
  that category's own other instances) -- no cross-category matrix.
