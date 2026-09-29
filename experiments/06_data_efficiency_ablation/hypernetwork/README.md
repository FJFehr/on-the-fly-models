# 06_data_efficiency_ablation_hypernetwork

Phase 1 of a 3-phase data-efficiency study. Phases 2-3 are direction only, not designed yet.

## Goal

Fabio's hypothesis: the hypernetwork should need less training data than a model with no
cross-task transfer, because it shares a single hypernetwork backbone across all 14 tasks
rather than learning each task in isolation. This experiment trains experiment 2's own
dim=4 matched-scale recipe (`frozen_td` and `notd`, Muon at its already-validated stable
`muon_lr=0.005`) at progressively less training data per task, to see how low it can go
before performance collapses.

**2026-09-10: resized from the original scaffolding's inherited
`arc1d_hypermodel_looped_rope_canon_muon` recipe (Zhu backbone, dim=16 target, ~1.58M-param
hypernetwork) to experiment 2's dim=4 matched-scale recipe (~11.4K params) — that legacy recipe
predated (and was never reconciled with) the 2026-09-09 dim=4 unification that made matched-scale
"the standard" for 02-05; running data-efficiency at a ~139x-larger, incomparable scale wasn't
worth the extra compute. Also dropped `1d_recolor_cmp` (14 categories now, matching 01/02/05),
moved 3→5 seeds per the main README's proposed standard, added `notd` as a second condition
(originally `frozen_td`-only, reasoned as "not worth carrying into this axis" from experiment 2's
full-data finding — but `notd` is exactly what pairs against `../joint/`'s own `notd` arm at
every data level, needed for a like-for-like comparison, not just at full data), and added a
"full data" level (own-budget anchor, see "Data levels" below). Job count: 18→70.**

## Fixed architecture (experiment 2's own dim=4 matched-scale recipe)

- Plain encoder (`rope_canon_transformer`, no Zhu tricks), matched-scale to the dim=4 target —
  identical to `02_hypernetwork_multitask/configs/dim4_frozentd.yaml` /
  `dim4_notd.yaml`. ~11.4K (`frozen_td`) / ~11.3K (`notd`) total hypernetwork params.
- `hyper_head.num_tasks: 18, freeze_task_indicator: true` (`frozen_td`) vs. `num_tasks: null,
  freeze_task_indicator: false` (`notd`) — near-ceiling ~99% exact match vs ~68% for `notd` in
  experiment 2's own dim=4 rerun at full data; both are carried into this axis to see whether
  that full-data gap holds, narrows, or widens as training data shrinks, and to pair against
  `../joint/`'s own `td`/`notd` split at every level.
- `optimizer: Muon, muon_lr: 0.005, muon_momentum: 0.95` — the same setting experiment 2's own
  dim=4 matched-scale recipe already validated at this exact architecture (not a separate,
  different-architecture ablation, unlike the original scaffolding's justification).
- `max_steps: 8000, warmup_steps: 800, N_supervision: 1`, `learning_rate: 0.001` (AdamW-aux
  group), `devices: 1` — matches experiment 2's own dim=4 recipe exactly (2026-09-14
  unification; was `max_steps: 2000, warmup_steps: 200, N_supervision: 2, devices: auto` —
  `N_supervision x max_steps` is the real optimizer-step budget, per `training/trainer.py`'s
  `Trainer(max_steps=max_steps*N_supervision)` wiring, so that was half experiment 2's own
  "Round 2 convention" budget, not the 4x-fewer it looked like from `max_steps` alone;
  `devices: auto`'s 8-way DDP also silently gave `batch_size=512` an effective global batch of
  4096, 8x experiment 2's own single-GPU 512, since nothing in this codebase rescales
  `batch_size` for DDP). Held fixed across every data level so training compute stays matched;
  only the amount of *data* varies.
- 14 in-distribution task categories (the paper's standard set, matching 01/02/05 — not the
  original scaffolding's 15, which added `1d_recolor_cmp`), `task_categories ==
  val_task_categories` (no held-out axis).

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

| Level | `variants_per_base_task` | `base_tasks_per_category` | ≈ total/category | Note |
|---|---:|---:|---:|---|
| `cell_{cond}_t1` | 1 | 1 | 1 | added 2026-09-15, extreme floor: a single base task |
| `cell_{cond}_t3` | 1 | 3 | 3 | |
| `cell_{cond}_t5` | 1 | 5 | 5 | |
| `cell_{cond}_t10` | 1 | 10 | 10 | |
| `cell_{cond}_t20` | 1 | 20 | 20 | |
| `cell_{cond}_v1` | 1 | `null` (all ~40) | 40 | original only, zero augmentation |
| `cell_{cond}_v2` | 2 | `null` | 80 | +1 augmented variant |
| `cell_{cond}_v3` | 3 | `null` | 120 | +2 |
| `cell_{cond}_v4` | 4 | `null` | 160 | +3 |
| `cell_{cond}_v5` | 5 | `null` | 200 | +4 |
| `cell_{cond}_v20` | 20 | `null` | 800 | larger jump, further up the curve |
| `cell_{cond}_full` | `null` (no reduction) | `null` | ≈40,000 | added 2026-09-10, own-budget full-data anchor |

`{cond}` is `frozentd` or `notd`. `full` runs at this experiment's own fixed `max_steps:
8000, N_supervision: 1` budget — since the 2026-09-14 unification this is the exact same
recipe experiment 2's own full-data numbers use, so `full` is both a within-experiment
anchor *and* expected to reproduce those numbers (see "Reading results" below).

The `t{N}` levels (added 2026-09-15) are a second, independent data-reduction axis: they
fix `variants_per_base_task=1` (zero additional augmentation) and instead reduce how many
*distinct base tasks* per category are used, reaching below the `v1` floor's ~40
base-tasks/category. This is the sharpest test of the paper's cross-task-sharing
hypothesis — at `t1`, `../individual/`'s per-category model trains on a single base
task's original example, while this arm and `../joint/` still pool 14 categories' worth
(14x that) of task-instances. Selection is deterministic given `data_seed` (a separate RNG
stream from the `v*` axis's own, so the two axes don't entangle — see
`data_modules/arc1d_meta_multiclass.py`'s `_stratified_base_tasks_per_category` and its
own test suite in `tests/test_arc1d_meta_multiclass.py`).

7 v/full levels + 5 t levels (12 total) × 2 conditions × 5 seeds = **120 jobs**.

## Running

```bash
python experiments/06_data_efficiency_ablation/hypernetwork/gen_configs.py
bash experiments/06_data_efficiency_ablation/hypernetwork/run.sh
```

Split across nodes via `SEEDS_OVERRIDE`/`CELL_GLOB` (see script header for examples).

## Reading results

For each data level and condition, compare `val_query_exact_match`/`test_query_exact_match`
(mean ± std across 5 seeds) against this experiment's **own** `cell_{cond}_full` cell (the
same fixed `max_steps: 8000, N_supervision: 1` budget as every reduced level, since the
2026-09-14 unification, exactly matching experiment 2's own recipe) — `full` should now
land close to experiment 2's own full-data numbers (`frozen_td` ~93.0%, `notd` ~57.4% test
exact match) as a sanity check, since it's the same recipe at the same effective data
volume; a real gap there would flag a remaining bug, not an expected budget difference.
The headline questions: at what `variants_per_base_task` does performance start dropping
off relative to this experiment's own `full` cell; does zero augmentation (`cell_{cond}_v1`,
the raw original examples only) still let the hypernetwork solve every task via cross-task
transfer; and does `frozen_td`'s advantage over `notd` hold, narrow, or widen as data shrinks?

At the `t{N}` levels specifically: compare against `../individual/`'s own `t{N}` numbers
(macro-averaged across categories) at the same `N`. This is the sharpest version of the
question — does this arm's cross-task weight generation pull ahead of `individual`'s
fully-isolated per-category models as `N` shrinks toward 1, where `individual` has almost
no signal to learn a category's rule from at all?

## Companion arms

- **`../individual/`** ("Phase 2"): a capacity/data-need baseline *without* the hypernetwork,
  one model per task category, no sharing at all — how much data does a plain, fully-isolated
  model need to solve each task, repeating the same data-reduction axis.
- **`../joint/`** ("Phase 3", added 2026-09-10): a shared backbone (no hypernetwork), trained
  jointly across all 14 categories, with vs. without a task-identity embedding —
  `direct_supervised` + `task_encoding.use_task_embedding`, reusing experiment 1's own
  mechanism verbatim (this was originally flagged here as "doesn't exist yet, would need new
  code," which was wrong — the mechanism already existed in experiment 1, just not previously
  reused in this experiment). `frozen_td` vs. `td`, and `notd` vs. `notd`, are the two intended
  pairs: same task-identity signal (or lack of it) on both sides, differing only in weight
  *generation* vs. plain conditioning — isolating what generation specifically buys, at both
  ends of the task-identity axis, not just with a signal present.
