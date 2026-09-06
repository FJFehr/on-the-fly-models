# Experiment 3: reusability ("Generate Once, Execute Many")

**Question**: does a single generated weight set — from one support-set
instance of a task — solve *other* instances of the same task category, or
does the hypernetwork need to regenerate weights per instance? If one
generation reuses across instances, the hypernetwork's per-task cost is
amortised: generate once, execute many times.

## What's here so far

A first leave-one-out (LOO) pass, from `scripts/measure_compute_efficiency.py`'s
Phase E: for each of the 14 base task categories, every instance in a
100-instance eval pool (`data/arc_1d_looped_augmented_devtest_shifted`) took
a turn as the reference — its generated weights scored against the other 99
("cross-instance") — compared against each instance solving only its own
query ("own"). Run on the dim=4 `02_hypernetwork_multitask` checkpoints:
`notd` (seed2, best of 3 freshly-checkpointed retrains) vs. `frozen_td`
(seed1).

**Finding**: `frozen_td` barely degrades from own-instance to cross-instance
(98.7% → 94.6% macro-average exact match); `notd` degrades more and is far
noisier (75.8% → 73.0%, population std 0.38–0.41 across categories vs.
`frozen_td`'s 0.05–0.12). A task-identity signal doesn't just help
in-distribution — it's also what lets one generation reuse across instances
without collapsing.

![Own vs. cross-instance generalisation](../../outputs/figures/03_reusability_generate_once_execute_many/generalization_loo.png)

*(Not committed — see "Figures and results" in `experiments/README.md`.
Regenerate with `python plot_generalization_loo.py` from this folder, once
`outputs/results/03_reusability_generate_once_execute_many/results_generalization_loo.csv`
exists.)*

## Status: preliminary — not yet the full design

This is one data point (dim=4, 2 checkpoints), not the formal experiment.
The full design, per the paper plan: take the **best seed** from
`02_hypernetwork_multitask`'s upcoming 10K-scale rerun, then run this same
own-vs-cross-instance LOO comparison properly (multi-seed, at the paper's
standard sizes) for both `notd` and `frozen_td`. **Blocked on that rerun's
best-seed pick.**

## TODO

- Wait on `02_hypernetwork_multitask`'s 10K rerun and best-seed selection.
- Formalise the LOO evaluation as a standalone script (currently embedded in
  `scripts/measure_compute_efficiency.py`'s Phase E) with its own configs/
  and run script, matching the other experiments' shape.
- Re-run at the paper's standard dims (4 and 6, see `experiments/README.md`'s
  architecture conventions) with proper seed counts, not this one-off pair.
