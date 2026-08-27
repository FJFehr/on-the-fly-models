# 01 - Data

## Scope

This note covers the data recipe actually used by the ARC-1D hypernetwork story: the
`data/arc_1d_looped_augmented` dataset and its direct siblings (`data/arc_1d_augmented`,
`data/arc_1d_all_tasks_augmented`), plus the synthetic compositional-holdout set built on top
of it. The older binary/non-multiclass ARC-1D track (`data/arc_1d_simple`,
`data/arc_1d_padded_multiclass`, the fixed-length binary datamodules) is explicitly **out of
scope** here. Every number below was read from source code, a config YAML, or the actual
on-disk `DatasetDict` (not recalled from memory).

## 1. Canonical augmentation recipe - `data/arc_1d_looped_augmented`

Built by `scripts/augment_arc_1d.py` from the raw variable-length `data/arc_1d` (itself
ingested from the external 1D-ARC benchmark by `scripts/build_arc_1d.py`). The command that
matches the dataset actually on disk (verified against its row counts and per-instance colour
content, see below) is:

```bash
python scripts/augment_arc_1d.py \
  --per-pair --n-color-permutations 199 --shifts 1 2 -1 -2 --no-mirror \
  --dev-test-n-permutations 19 \
  --output-dir data/arc_1d_looped_augmented
```

(all 18 task categories, no `--task-categories` filter, no `--canonical-recolor-colors`).

Pipeline, applied per task in this order (`scripts/augment_arc_1d.py:augment_task`):

1. **Colour** - `--per-pair`: each support pair *and* the query get an independent injective
   colour remapping (colours may repeat across pairs), rather than one global mapping for the
   whole task instance. `--n-color-permutations 199` gives 199 additional colour variants plus
   the untouched original = **200 unique colour variants per base task**. A handful of tasks
   (`1d_move_dp`, `1d_move_2p_dp`, `1d_scale_dp`, `1d_recolor_oe`, `1d_recolor_cmp`,
   `1d_recolor_cnt` - `GLOBAL_ONLY_TASKS`) fall back to global colour augmentation regardless
   of `--per-pair`, because their rule depends on a colour staying identical across every
   support pair and the query. `1d_mirror`'s pivot colour 9 is never remapped and never a
   remap target (`FIXED_RULE_COLOURS`).
2. **Shift** - `--shifts 1 2 -1 -2`: the unshifted position plus 4 shifts = **5 shift
   positions**. Positive shift prepends zeros (content moves right), negative shift appends
   zeros (content moves left); `sequence_length` grows by `abs(shift)`, no content is
   discarded.
3. **Mirror** - `--no-mirror`: disabled (mirroring is redundant with `1d_flip`/`1d_mirror`
   already being trained tasks, and would otherwise double every variant count).

**Result**: up to 200 colour variants × 5 shift positions × 1 (no mirror) = **up to 1000
variants per base task**. Confirmed on disk: `data/arc_1d_looped_augmented`'s `train` split
has 721,000 rows across 18 categories - 40,000 rows/category (1000 × 40 base tasks) for every
category except `1d_scale_dp`, which has 41,000 (41 base tasks in the raw benchmark for that
category).

Any augmented train row whose full `(support_inputs, support_outputs, query_input,
query_output)` content exactly matches a dev/test row is dropped
(`filter_held_out_contamination`), so no train/eval leakage survives augmentation.

Dev/test are **not** shifted or mirrored, but the on-disk dataset was built with
`--dev-test-n-permutations 19`: the 5 original held-out benchmark tasks per category get 19
extra colour-permuted variants each (20 × 5 = **100 examples/category** in both `dev` and
`test`, 1,800 rows each), which reduces eval metric variance relative to the raw 5-example
splits without changing which underlying tasks are held out.

## 2. Older/adjacent recipes (out of scope for new work)

Two other augmented datasets exist from earlier stages of the project. Both are documented
here only for contrast - neither should be used for new hypernetwork work.

| | `data/arc_1d_augmented` | `data/arc_1d_all_tasks_augmented` |
|---|---|---|
| Colour mode | global (one injective mapping per task instance) | per-pair (with the same `GLOBAL_ONLY_TASKS` fallback) |
| Colour variants | 21 + original = 22 | 199 + original = 200 |
| Shifts | 11 positions: `0, ±1, ±2, ±3, ±4, ±5` | 5 positions: `0, ±1, ±2` |
| Mirror | off | off |
| Variants/base task | 242 (22 × 11) | 1000 (200 × 5) |
| Train rows (on disk) | 174,482 - 9,680/category (9,922 for `1d_scale_dp`) | 721,000 - 40,000/category (41,000 for `1d_scale_dp`) |
| Dev/test | unaugmented, passed through (90 rows total = 5/category × 18) | unaugmented, passed through (90 rows total) |
| Used by | the original augmented-capacity experiments (`README.md`'s Data Augmentation section) | `configs/experiments/arc1d_hypermodel_disentanglement` only |

`data/arc_1d_augmented` is the project's original recipe (`README.md:68-141`): default script
settings, global colour mode, a wider shift range. It predates the per-pair augmentation
mechanism entirely.

`data/arc_1d_all_tasks_augmented` is historical and specific to the disentanglement
experiment. Per its own README
(`configs/experiments/arc1d_hypermodel_disentanglement/README.md:19-38`): "Each task in the
train split receives 1000 augmented variants via three transformations applied in order:
colour permutation → shift → (no mirror)", built with per-pair colour augmentation (200
variants) crossed with 5 shift positions, over all 18 task categories including
`1d_padded_fill`. Structurally close to the canonical recipe above (same per-pair mechanism,
same 200×5 variant count) but built and frozen before the working task-category set and the
`--dev-test-n-permutations` convention were established - hence "different per-task-varying
recipe" and not the dataset any current hypernetwork experiment points at.

## 3. Task categories

The full 18-category index, from `TASK_CATEGORY_INDEX` in `models/hypermodel_lightning.py:37-56`:

| # | Category | # | Category |
|---:|---|---:|---|
| 0 | `1d_move_1p` | 9 | `1d_denoising_1c` |
| 1 | `1d_move_2p` | 10 | `1d_denoising_mc` |
| 2 | `1d_move_3p` | 11 | `1d_pcopy_1c` |
| 3 | `1d_move_dp` | 12 | `1d_pcopy_mc` |
| 4 | `1d_move_2p_dp` | 13 | `1d_recolor_oe` |
| 5 | `1d_fill` | 14 | `1d_recolor_cnt` |
| 6 | `1d_hollow` | 15 | `1d_recolor_cmp` |
| 7 | `1d_flip` | 16 | `1d_scale_dp` |
| 8 | `1d_mirror` | 17 | `1d_padded_fill` |

### Exclusions from the working set

- **`1d_padded_fill`** - excluded from the hypernetwork multi-task work from the start of
  `arc1d_hypermodel_looped`'s phase-1 sweep (17 configs, "all `arc1d_recursion_ablation` task
  categories except `1d_padded_fill`"). Per that experiment's README
  (`configs/experiments/arc1d_hypermodel_looped/README.md`): "`1d_padded_fill` is excluded —
  it was never part of the per-task `arc1d_hypermodel_augmented` configs this phase mirrors
  (only the multi-task `mixes/`/`_td` configs include it), so it simply never enters the
  picture here." Not a data-quality finding, just scope inherited from an earlier
  single-task-config set.
- **`1d_recolor_cnt` and `1d_recolor_oe`** - dropped later, at the
  `arc1d_hypermodel_looped_rope_canon_capacity_baseline` stage (bringing the set from 17 down
  to 15). That README states: "`1d_recolor_cnt` and `1d_recolor_oe`: stuck at a literal
  `0.000` in every single run regardless of layers or clip. No signal at all, a qualitatively
  different (likely structural) failure, not a capacity question"
  (`configs/experiments/arc1d_hypermodel_looped_rope_canon_capacity_baseline/README.md`). This
  is a purely empirical finding about model behaviour, not a support/query data bug: an
  earlier, explicit data-integrity check
  (`configs/experiments/arc1d_hypermodel_looped_recolor/README.md`) had derived the true rule
  for all three recolor categories from the 3 support pairs alone (`recolor_oe`: run-length
  parity; `recolor_cnt`: exact run length; `recolor_cmp`: max-length tie-break) and verified it
  against the query's actual output across 200 real task instances per category (600 total, on
  `data/arc_1d_looped_augmented`) - "200/200 OK for all three - the support pairs always fully
  and correctly determine the query." So the data is well-posed; the two categories are
  excluded because no model configuration tried has ever solved them, not because the task is
  underspecified. `1d_recolor_cmp` (the third recolor task, sharing the same 200/200
  verification) is *not* excluded - it does not show the same zero-signal failure and remains
  in the working set.

### Working 15-task set

```
1d_move_1p, 1d_move_2p, 1d_move_3p, 1d_move_dp, 1d_move_2p_dp,
1d_fill, 1d_hollow, 1d_flip, 1d_mirror,
1d_denoising_1c, 1d_denoising_mc,
1d_pcopy_1c, 1d_pcopy_mc,
1d_recolor_cmp,
1d_scale_dp
```

(matches `task_categories`/`val_task_categories` in e.g.
`configs/experiments/arc1d_lowdata/base.yaml`.)

## 4. Splits

### Standard track: augmented train, held-out val/test

Confirmed in `data_modules/arc1d_meta_multiclass.py` (`Arc1dMetaMulticlassDataModule.setup`)
and `data_modules/arc1d_direct.py` (`Arc1dDirectDataModule.setup`): both build `train_dataset`
from the `train` split (optionally subsampled, see §5) and build `val_dataset`/`test_dataset`
from the `dev`/`test` splits with no augmentation-related subsampling applied.
`SPLIT_ALIASES = {"val": "dev"}` lets configs refer to `val_split: val` and have it resolve to
the on-disk `dev` split. For `data/arc_1d_looped_augmented` specifically, `dev`/`test` are the
original held-out 1D-ARC benchmark tasks per category, colour-permuted for variance reduction
only (§1) - never shifted, never mirrored, and never containing an example whose full content
also appears in train.

### Compositional-holdout split - `data/arc_1d_compositional_holdout`

Built by `scripts/build_arc1d_compositional.py`, generating tasks with
`data_modules/arc1d_compositional.py`. ARC-1D has no native notion of chaining two rules
together (the benchmark JSON is static), so this is a **synthetic** dataset: each of 10
composite categories applies a "stage 1" rule (build a fresh random object, applying whatever
corruption/markers one known single-task rule needs) followed by a "stage 2" rule (a second,
different known single-task transform on that same object) to produce the final
input/output pair - genuinely chaining two single-step rules rather than sampling either rule
in isolation.

The 10 categories, from `COMPOSITE_TASK_SPECS` in `data_modules/arc1d_compositional.py:387-405`
(stage 1 rule → stage 2 rule):

| Composite category | Stage 1 rule | Stage 2 rule |
|---|---|---|
| `1d_comp_denoise1c_shift3` | denoise (single colour) | shift |
| `1d_comp_fill_mirror` | fill | mirror |
| `1d_comp_fill_shift3` | fill | shift |
| `1d_comp_fill_movedynamic` | fill | move-to-pointer |
| `1d_comp_hollow_shift3` | hollow | shift |
| `1d_comp_denoisemc_copy` | denoise (multi-colour) | copy-to-marker |
| `1d_comp_denoisemc_denoise1c` | denoise (multi-colour) | denoise (single colour) |
| `1d_comp_movedynamic_hollow` | move-to-pointer | hollow |
| `1d_comp_shift3_copy` | shift | copy-to-marker |
| `1d_comp_denoisemc_mirror` | denoise (multi-colour) | mirror |

These are **fully disjoint** from the 15-task training set: they carry entirely distinct
`task_category` names (`1d_comp_*`, never present anywhere in `data/arc_1d_looped_augmented`),
are generated procedurally rather than drawn from the 1D-ARC benchmark, and are used only as a
`holdout_test` split - never mixed into any `train`/`dev`/`test` split used for training. The
generator explicitly calibrates several conventions (marker/pivot side, mirror pivot colour 9,
copy template centring) to match the real training data's own conventions "so a
generalisation failure can't be blamed on ... a geometry the model never saw even in
single-task training" (`data_modules/arc1d_compositional.py:1-40`,
`configs/experiments/arc1d_hypermodel_compositional_generalization/README.md`). Recolor
categories are excluded from consideration entirely for this split, per the experiment's own
scope decision.

Holdout size: `N_PER_CATEGORY = 40` in `scripts/build_arc1d_compositional.py`, 10 categories ×
40 = **400 instances**, saved as a single `holdout_test` split in
`data/arc_1d_compositional_holdout`.

### Leave-one-category-out split

Mechanism (`configs/experiments/arc1d_hypermodel_looped_rope_canon_generalization/README.md`):
no separate dataset is built for this - `task_categories` (train) and `val_task_categories`
(eval) are simply set to different lists in the same config, which
`Arc1dMetaMulticlassDataModule` already supports natively with no code changes. Each leaf
config sets `task_categories` to the 14-task list (the 15-task working set minus the one held
out) and `val_task_categories` to the full 15-task list, so the same run reports both
in-distribution exact-match on the 14 trained categories and zero-shot exact-match on the
held-out 15th category. "Held out" is therefore an index-reservation concept, not a data-split
concept: the held-out category's rows exist in `data/arc_1d_looped_augmented` throughout, they
are simply never selected into the training batch by the `task_categories` filter. It also
interacts with task-identity conditioning: under the learned one-hot task descriptor
(`hyper_head.num_tasks`), a held-out category's embedding column never receives a gradient, so
the experiment tests both a `td` (learned, un-updated for the held-out slot) and a `frozentd`
(`hyper_head.freeze_task_indicator: true`, every column including the held-out one frozen at
random init) variant, alongside `notd` (no task descriptor at all).

The 5 categories held out (one at a time), confirmed against the README's own table:

```
1d_move_2p, 1d_denoising_mc, 1d_flip, 1d_pcopy_mc, 1d_hollow
```

Each was chosen because it has a structurally similar sibling left in the 14-task training
set (e.g. `1d_move_2p` held out alongside `1d_move_1p`/`1d_move_3p`/`1d_move_dp`/
`1d_move_2p_dp` remaining), so a generalisation failure can't be attributed to "no related
signal existed anywhere in training".

## 5. Low-data subsampling - `variants_per_base_task`

Both `Arc1dMetaMulticlassDataModule` (`data_modules/arc1d_meta_multiclass.py`,
`_stratified_variants_per_base_task`, used by the hypernetwork low-data track
`configs/experiments/arc1d_lowdata/`) and `Arc1dDirectDataModule`
(`data_modules/arc1d_direct.py`, `_stratified_variants_per_base_task`, ported verbatim, used
by the no-hypernetwork per-task control `configs/experiments/arc1d_lowdata_baseline/`) expose
the same subsampling mechanism, applied **only** to the train split - `dev`/`test` are always
built at full size regardless of the level, so every data level is evaluated on the same fixed
eval set.

Mechanism (identical in both modules):

- Augmented rows carry `task_id = original_task_id * 10000 + aug_index`
  (`scripts/augment_arc_1d.py`'s `augment_task`), so `aug_index == 0` is always the true,
  unaugmented original example for that base task.
- Rows are grouped by `(task_category, task_id // 10000)` - one group per base task.
- Within each group, the original (`aug_index == 0`) is always kept first, then `K - 1`
  further augmented variants are taken from a `data_seed`-shuffled ordering of the rest.
- Because each level's selection is the previous level's selection plus exactly one more
  variant per base task, **level `K`'s training set is always a strict superset of level
  `K - 1`'s** (nested/cumulative), and level 1 is exactly the original, zero-augmentation
  example for every base task.
- `data_seed` (independent of the training `seed`) makes the subsample deterministic and
  identical across training seeds at a fixed level - only initialization varies between seed
  1/2/3 runs at the same level, not which rows were seen.

Sampling is **stratified per base task**, not a uniform random draw over the category's whole
augmented pool - the `arc1d_lowdata` README notes an earlier pool-uniform version "silently
conflated data volume with task diversity, since low `N` ended up covering far fewer distinct
base tasks... the corrected design fixes this: every level covers every base task, only the
augmentation depth per task changes."

### Levels actually used

`configs/experiments/arc1d_lowdata/base.yaml` sets `variants_per_base_task: null` as a
placeholder overridden per leaf config; the level values are documented in that experiment's
README:

| Level | `variants_per_base_task` | ≈ raw examples/category | Note |
|---|---:|---:|---|
| `cell_v1` | 1 | 40 | original only, zero augmentation |
| `cell_v2` | 2 | 80 | +1 augmented variant |
| `cell_v3` | 3 | 120 | +2 |
| `cell_v4` | 4 | 160 | +3 |
| `cell_v5` | 5 | 200 | +4 |
| `cell_v20` | 20 | 800 | larger jump, further up the curve |

`configs/experiments/arc1d_lowdata_baseline/base.yaml` (also `variants_per_base_task: null`,
overridden per leaf) uses a narrower range per its README - `variants_per_base_task ∈ {1, 2,
3}` (≈40/80/120 examples/category), "matching `arc1d_lowdata`'s `cell_v1`/`cell_v2`/
`cell_v3`"; levels 4/5/20 are explicitly out of scope for that experiment. Both experiments
fix `data_seed: 42` and `data_dir: data/arc_1d_looped_augmented`, so at a given level both
train on the identical underlying rows - only the model (hypernetwork vs. one plain model per
task category) differs.

