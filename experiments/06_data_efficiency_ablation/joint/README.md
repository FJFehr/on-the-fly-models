# 06_data_efficiency_ablation_joint

Third arm of the data-efficiency study, alongside `06_data_efficiency_ablation_hypernetwork` (hypernetwork) and
`06_data_efficiency_ablation_individual` (fully-isolated per-task individual). Added 2026-09-10.

## Goal

`06_data_efficiency_ablation_individual`'s per-task individual models can't share anything across
categories by construction, so a gap between it and the hypernetwork can't tell you
*why* the hypernetwork needs less data — cross-task weight generation, or just having a
shared backbone at all. This arm isolates that: one backbone, trained **jointly** across
all 14 categories, **with vs. without a task-identity embedding**, but generating no
per-task weights at all — plain `direct_supervised` training, exactly experiment 1's
own "Joint" mechanism (`01_multitask_capacity/configs/td_dim4.yaml` /
`notd_dim4.yaml`), just swept across data levels instead of only at full data.

`joint_td` vs. the hypernetwork's `frozen_td`, and `joint_notd` vs. the hypernetwork's
`notd`, are the two sharpest pairs in this experiment: same sharing, same task-identity
signal (or lack of it) on both sides, differing only in *generating* per-task weights
from that signal vs. *conditioning* a shared model on it. `joint_notd` is also expected
to reproduce experiment 1's own capacity-cliff finding on its own (a real ceiling well
below individual-training, even at full data), now measured across data levels instead
of only at the top.

## Fixed architecture — sized to the hypernetwork's parameter *budget*, not its target

Unlike `06_data_efficiency_ablation_hypernetwork`/`06_data_efficiency_ablation_individual` (both at experiment 2's dim=4 target,
a few hundred params on its own), this arm is deliberately **not** dim=4. The
hypernetwork's own total parameter count (~11.4K `frozen_td` / ~11.3K `notd`) is what a
no-hypernetwork baseline needs to match to isolate weight *generation* from mere extra
*capacity* — sizing this arm at dim=4 would make it 10-20x smaller than the hypernetwork
it's compared against, confounding "no weight generation" with "far less capacity."

- `rope_canon_looped_transformer`, `hidden_dim: 14, num_heads: 1, inner_dim: 14,
  inner_num_heads: 1, n_loops: 1, dropout: 0.1` — experiment 1's own already-validated
  dim=14 "~10K-param scaffold" (`01_multitask_capacity/configs/td_dim14.yaml` /
  `notd_dim14.yaml`), 9,508 (`notd`) / 9,688 (`td`) params, reused verbatim rather than
  picking a new, unvalidated size.
- `model: direct_supervised`, `task_encoding.use_task_embedding: true` (`td`) /
  `false` (`notd`) — experiment 1's own mechanism, reused verbatim (no new code).
  `task_encoding.embedding_dim: 10`, matching experiment 1's own dim=14 convention exactly
  (so the 9,508/9,688 param counts hold precisely, unlike the other two arms' own
  `embedding_dim: 4`).
- `optimizer: Muon, muon_lr: 0.005, muon_momentum: 0.95` — matches every other arm.
- `max_steps: 8000, warmup_steps: 200, learning_rate: 0.0005`, no gradient clipping — matches
  experiment 1's own `td_dim14`/`notd_dim14` recipe exactly (2026-09-14 unification; was
  `max_steps: 2000, learning_rate: 0.001, gradient_clip_val: 10.0`, matching
  `06_data_efficiency_ablation_hypernetwork`/`06_data_efficiency_ablation_individual`'s own then-budget instead of this arm's actual
  reference — caught alongside the same discrepancy in `06_data_efficiency_ablation_individual`'s own
  README). Compute is still held fixed across every data level within this arm — just
  recalibrated to experiment 1's own validated 8000-step recipe instead of an untested
  smaller one.
- All 14 in-distribution task categories trained jointly in one model per (condition,
  level, seed) — not per-category like `06_data_efficiency_ablation_individual`.

## Data-reduction mechanism

Identical stratified/nested `variants_per_base_task` sampling to the other two arms
(`data_modules/arc1d_direct.py`'s `_stratified_variants_per_base_task`, same
`data_seed=42`, same `data/arc_1d_looped_augmented` dataset) — grouped per (category,
base task), so it extends naturally to this arm's multi-category `task_categories` list.
At a given level, all three arms train on the identical underlying task-variant rows.

**Levels**: `variants_per_base_task` in {1, 2, 3, 4, 5, 20, full} — the full range plus a
"full" cell (no reduction, the entire train split, added 2026-09-10), matching
`06_data_efficiency_ablation_hypernetwork`, not `06_data_efficiency_ablation_individual`'s restricted {1, 2, 3, full}. A joint run is
one model across all 14 categories, the same cost profile as a hypernetwork run, so
there's no reason to restrict it the way the *per-category* individual baseline is
(210→280 jobs already, without needing 4 more levels x 14 categories on top). "full" runs
at `max_steps: 8000` (since the 2026-09-14 unification, matching experiment 1's own
`td_dim14`/`notd_dim14` budget exactly), so it should now land close to experiment 1's own
full-data dim=14 numbers as a sanity check.

7 levels × 2 conditions × 5 seeds = **70 jobs**.

**2026-09-15: added a second, independent data-reduction axis** —
`base_tasks_per_category` in {1, 3, 5, 10, 20}, always at `variants_per_base_task=1` (zero
additional augmentation), tagged `cell_{cond}_t{N}.yaml`. Reduces task *diversity* rather
than augmentation *depth*, reaching below the `v1` floor's ~40 base-tasks/category — the
sharpest test of the cross-task-sharing hypothesis: at `t1`, this arm still pools 14
categories' worth of single-base-task task-instances, while `06_data_efficiency_ablation_individual`'s
per-category models each get only that one base task. Selection uses its own
`data_seed`-seeded RNG stream, independent of the `v*` axis's own (see
`data_modules/arc1d_direct.py`'s `_stratified_base_tasks_per_category`). 5 more levels × 2
conditions × 5 seeds = **50 more jobs**, total **120**.

## Known asymmetry (same as `06_data_efficiency_ablation_individual`)

`Arc1dDirectDataModule` unpacks each selected row into its 3 support pairs only (the
row's own query is held out for the separate dev/test splits), vs. `06_data_efficiency_ablation_hypernetwork`'s 4
(3 support + the row's own query, supervised as part of the hypernetwork's training
loss). Same inherent datamodule difference documented in `06_data_efficiency_ablation_individual`'s
README — this arm shares it, not something new.

## Running

```bash
python experiments/06_data_efficiency_ablation/joint/gen_configs.py
bash experiments/06_data_efficiency_ablation/joint/run.sh                  # sequential, 1 GPU
GPUS="0,1,2" bash experiments/06_data_efficiency_ablation/joint/run.sh      # 3-way parallel
```

Split across nodes via `SEEDS_OVERRIDE`/`CELL_GLOB` (see script header for examples).

## Reading results

For each level and condition, compare `val_query_exact_match`/`test_query_exact_match`
(mean ± std across 5 seeds) against the other two arms at the same level, same condition:
`06_data_efficiency_ablation_hypernetwork`'s matching `frozen_td`/`notd` cell (hypernetwork) and
`06_data_efficiency_ablation_individual`'s per-category numbers averaged to a macro mean (only defined at
levels 1, 2, 3, full — its restricted range, missing 4/5/20 — and with no task-id axis of
its own to match against, one task per model there, so `notd` isn't a meaningful category
for it). `joint_td`
tracking close to the hypernetwork's `frozen_td` at every level would suggest the
hypernetwork's specific weight-generation mechanism isn't what's buying the data
efficiency — plain task-id conditioning of a shared model would be doing the real work.
A large, persistent gap (joint well below the hypernetwork, especially at the lowest
levels) supports the cross-task-transfer-via-weight-generation hypothesis specifically,
not just "sharing a backbone helps." Checking the same comparison at `notd` too (joint vs.
hypernetwork, both without a task-identity signal) tests whether that conclusion holds
independent of the task-identity signal, not just alongside it.
