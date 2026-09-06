# arc1d_lowdata

Phase 1 of a 3-phase data-efficiency study. Phases 2-3 are direction only, not designed yet.

## Goal

Fabio's hypothesis: the hypernetwork should need less training data than a model with no
cross-task transfer, because it shares a single hypernetwork backbone across all 15 tasks
rather than learning each task in isolation. This experiment trains the fixed architecture from
`arc1d_hypermodel_looped_rope_canon_muon` (Zhu backbone, `frozen_td`, Muon at its already-
validated stable `muon_lr=0.005`) at progressively less training data per task, to see how low
it can go before performance collapses.

## Fixed architecture (carried over from the muon experiment's conclusions)

- Zhu block backbone (`rope_canon_zhu_transformer`, RMSNorm + QK-norm + SwiGLU).
- `hyper_head.num_tasks: 18, freeze_task_indicator: true` (`frozen_td`) — near-ceiling ~99%
  exact match vs ~68% for `notd` in the muon experiment; `notd` isn't worth carrying into a
  new axis.
- `optimizer: Muon, muon_lr: 0.005, muon_momentum: 0.95` — Keller Jordan's published default
  (0.02) diverged to NaN at global_step ~915/4000 in this exact architecture (see
  `arc1d_hypermodel_looped_rope_canon_muon/README.md`); not worth re-risking across a new
  18-job sweep, so this experiment uses the already-validated stable value directly rather
  than re-testing 0.02.
- `max_steps: 2000, warmup_steps: 200`, `learning_rate: 0.001` (AdamW-aux group) — held fixed
  across every data level so training compute stays matched; only the amount of *data* varies.
- 15 in-distribution task categories, `task_categories == val_task_categories` (no held-out
  axis).

## Data-reduction mechanism

`Arc1dMetaMulticlassDataModule` (`data_modules/arc1d_meta_multiclass.py`) exposes
`variants_per_base_task`/`data_seed`, applied **only** to `train_dataset` construction in
`setup()` — `val_dataset`/`test_dataset` are built exactly as before, so every data level is
evaluated on the **same fixed 100 examples/category** validation and test split
(`data/arc_1d_looped_augmented`'s separate `dev`/`test` splits, independent of `train`).

Sampling is **stratified per underlying base task**, not a raw random count over the whole
per-category pool. An earlier version of this experiment used pool-uniform sampling
(`max_examples_per_task`, sampling N rows uniformly at random across ~40,000 rows/category —
40 base tasks × up to 1000 augmented colour/shift variants each); that silently conflated data
*volume* with task *diversity*, since low `N` ended up covering far fewer distinct base tasks
(e.g. `N=1000` covered all 40/40 base tasks, but `N=10` covered only 10/40). The corrected
design fixes this: every level covers **every** base task, only the augmentation depth per task
changes.

Augmented rows carry `task_id = original_task_id * 10000 + aug_index`
(`scripts/augment_arc_1d.py`'s `augment_task`), and `aug_index == 0` is always the
untransformed, original example for that base task. For each `(category, base task)` group, the
selection always includes that original first, then a fixed `data_seed`-ordered sequence of the
remaining augmented variants — so level `K`'s selection is always level `K-1`'s plus exactly
one more variant per base task (**nested/cumulative** across levels), and level 1 is exactly the
original, zero-augmentation example for every base task.

`data_seed` (fixed at 42, independent of the training `seed`) makes the subsample deterministic
and **identical across all 3 training seeds at a given data level** — so the only thing varying
between seed 1/2/3 runs at the same data level is training initialization, not which examples
were seen. (Confirmed by tests in `tests/test_arc1d_meta_multiclass.py`, notably
`test_levels_are_nested_across_increasing_variants_per_base_task` and
`test_variants_per_base_task_independent_of_training_seed`.)

Most categories have exactly 40 base tasks, but a few have slightly more (e.g. `1d_scale_dp` has
41), so the "≈ total/category" column below is approximate, not exact, across categories.

## Data levels

| Level | `variants_per_base_task` | ≈ total/category | Note |
|---|---:|---:|---|
| `cell_v1` | 1 | 40 | original only, zero augmentation |
| `cell_v2` | 2 | 80 | +1 augmented variant |
| `cell_v3` | 3 | 120 | +2 |
| `cell_v4` | 4 | 160 | +3 |
| `cell_v5` | 5 | 200 | +4 |
| `cell_v20` | 20 | 800 | larger jump, further up the curve |

6 levels × 3 seeds = **18 jobs**.

## Running

```bash
python experiments/06_data_efficiency_ablation/hypernetwork/gen_configs.py
bash experiments/06_data_efficiency_ablation/hypernetwork/run.sh
```

Split across nodes via `SEEDS_OVERRIDE`/`CELL_GLOB` (see script header for examples).

## Reading results

For each data level, compare `val_query_exact_match`/`test_query_exact_match` (mean ± std
across 3 seeds) against the muon experiment's `frozentd_muon` full-data result (~99%, at the
implicit full-data level this experiment doesn't re-run — the full ~40,000/category pool). The
headline question: at what `variants_per_base_task` does performance start dropping off, and
does zero augmentation (`cell_v1`, the raw original examples only) still let the hypernetwork
solve every task via cross-task transfer?

## Deferred (not part of this experiment — direction only)

- **Phase 2**: a capacity/data-need baseline *without* the hypernetwork, on a single task — how
  much data does a plain model need to solve the task, repeating the same data-reduction axis.
- **Phase 3**: a shared backbone (no hypernetwork) + frozen task-identity embedding, trained
  across multiple tasks — this architecture doesn't exist yet (`DirectSupervisedLightning`/
  `LoopedSupervisedLightning` have no task-identity conditioning today) and would need new code.
  The delta between Phase 1 and Phase 3 is the intended measure of "knowledge sharing via the
  hypernetwork."
