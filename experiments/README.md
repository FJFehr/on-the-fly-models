# Paper experiments

This tree holds the experiments behind the paper, reorganised (2026-09) out
of the much larger `legacy/configs/experiments/` history of exploratory
runs. If you're trying to reproduce a specific figure or number in the
paper, start here.

## The story, in order

1. **[`01_multitask_capacity/`](01_multitask_capacity/)** — can one small
   model learn all 14 ARC-1D tasks jointly (one shared weight set, no
   per-task model)? Produces the capacity-cliff plot and the per-task
   breakdown at dim=6, plus Canon and optimizer (Muon vs. AdamW)
   ablations. **Status: done** (5 seeds, dims 4/6/10/14, Individual/Joint
   w/ td/Joint w/o td all self-contained here, both ablations across all
   four dims).
2. **[`02_hypernetwork_multitask/`](02_hypernetwork_multitask/)** — the
   hypernetwork: does generating per-task weights close the gap #1
   couldn't, at the same parameter budget? Produces the cluster plots
   (task-representation disentanglement, with vs. without a task-identity
   signal) and the dim=4 per-task breakdown. **Status: done.** dim=4 has
   been resized in place to the "matched-scale" recipe (~11.4K total
   params, every pipeline dimension sized to match the dim=4 target itself
   -- was previously a ~200K fixed-width recipe), rerun clean-slate at 5
   seeds, both arms, `save_checkpoints: true` -- real, formal, multi-seed
   checkpoints at the true matched scale, not the old fixed-width one (see
   its own README's "Dim=4 rerun" for the numbers). dim=6 still at the
   original 3 seeds, larger fixed-width recipe, no checkpoints -- not
   resized this round. TODO: dim=6's own resize + rerun to match.
3. **[`03_reusability_generate_once_execute_many/`](03_reusability_generate_once_execute_many/)**
   — reusability: does one generated weight set, from one support-set instance, solve *other*
   instances of the same task type at inference time ("generate once, execute many"), with and
   without td? **Status: done**, 5 seeds, both arms, no cherry-picked seed — against #2's own
   dim=4 matched-scale checkpoints. Reuse holds almost everywhere (own→cross degradation
   under 5pp for 12/14 categories, both arms); `mirror` is a sharp, category-specific
   exception for both.
4. **[`04_compositional_generalization/`](04_compositional_generalization/)**
   — zero-shot generalisation to *chained-skill* tasks the model never saw
   combined during training (e.g. denoise-then-shift). With vs. without td.
   **Status: done**, 5 seeds, both arms — no longer trains its own models at
   all; both arms reuse #2's own dim=4 checkpoints directly (in-distribution
   numbers are literally #2's own results, `frozen_td` via a
   mathematically-justified padding of its frozen, never-trained
   task-indicator projection, see its README). `notd` beats `frozen_td` on
   zero-shot token accuracy in 8/10 held-out categories, confirming the
   original hypothesis.
5. **[`05_leave_one_out_task_generalization/`](05_leave_one_out_task_generalization/)**
   — train on 13 of the 14 base task categories, 5 seeds, and test zero-shot
   on the held-out 14th category. With vs. without td. Distinct from #4:
   this holds out a whole *base* category, not a chained composition of
   known ones. **Status: done** (5 seeds, both arms, all 14 base categories
   held out in turn — 140 jobs at experiment 2's matched-scale dim=4
   recipe). Extends `legacy/configs/experiments/arc1d_v2_generalization`'s
   3-seed, 5-category-subset precedent to full 14-category coverage. `notd`
   beats `frozen_td` on held-out zero-shot token accuracy in 13/14
   categories (macro 81.9% vs. 68.5%), while `frozen_td` dominates
   in-distribution — the same explicit-anchor tradeoff experiments 2 and 4
   already found, now confirmed at full category-level coverage.
6. **[`06_data_efficiency_ablation/`](06_data_efficiency_ablation/)** —
   hypernetwork (`hypernetwork/`, dim=4 matched-scale, `frozen_td`/`notd`) vs. joint direct
   training with no weight generation (`joint/`, experiment 1's ~10K-param dim=14 scaffold —
   sized to the hypernetwork's own param *budget*, not its dim=4 target — `td`/`notd`) vs.
   fully-isolated per-task models (`individual/`, dim=4, no task-id axis) across shrinking
   training-data levels: `variants_per_base_task` in {1,2,3,4,5,20,full} at a fixed ~40
   base-tasks/category (`individual/` restricted to {1,2,3,full}), plus (2026-09-15)
   `base_tasks_per_category` in {1,3,5,10,20} at a fixed `variants_per_base_task=1` — a second,
   independent axis reducing task *diversity* rather than augmentation *depth*, reaching below
   the 40-row/category floor to test the paper's cross-task-sharing hypothesis at its sharpest
   (at `t1`, `individual` trains on a single base task's 3 support pairs for an entire
   category, while `hypernetwork`/`joint` still pool 14x that across categories). `full` is
   each arm's own no-reduction cell at its own fixed compute budget, a within-experiment
   full-data anchor. **Status: `v*`/`full` family (420 jobs) run and unified 2026-09-14** —
   each arm's training regime (not just architecture) now matches its own reference recipe
   exactly (01 for `individual`/`joint`, 02 for `hypernetwork`) after a first pass surfaced
   several silent mismatches (wrong Lightning module, halved optimizer-step budget, an 8x
   DDP-vs-single-GPU effective-batch-size gap, and more — see `individual/README.md`'s and
   `hypernetwork/README.md`'s own "2026-09-14 unification" notes); confirmed via `individual`'s
   `full` cell landing at 94.6%, matching 01's own 92.6% dim=4 ceiling. The `t*` family (450
   more jobs, 870 total) is additive on top and does not touch or invalidate the `v*`/`full`
   results.

Every experiment's own README has the full method, findings, and an exact
"Running" command.

## Data preparation

Everything above shares one dataset, built once, from scratch, in three
commands (none needs a GPU — pure CPU data prep):

All three run as modules (`build_arc1d_compositional.py` imports
`data_modules`, which needs the repo root on `sys.path` — `-m` gives it
that; the other two don't strictly need it but are shown the same way for
one consistent invocation style):

```bash
# 1. Ingest the raw 1D-ARC benchmark -> data/arc_1d
python -m scripts.build_arc_1d

# 2. The canonical augmentation recipe -> data/arc_1d_looped_augmented
#    (per-pair colour augmentation, 200 colour variants x 5 shift positions,
#    no mirror -> up to 1000 variants/base task, 721,000 train rows across
#    18 categories; dev/test also lightly colour-augmented to reduce metric
#    variance -- see docs/arc1d_story/01_data.md for the full derivation)
python -m scripts.augment_arc_1d \
    --per-pair --n-color-permutations 199 --shifts 1 2 -1 -2 --no-mirror \
    --dev-test-n-permutations 19 \
    --output-dir data/arc_1d_looped_augmented

# 3. The compositional-generalisation holdout (experiment 4) -> data/arc_1d_compositional_holdout
python -m scripts.build_arc1d_compositional
```

Verified: all three ran end-to-end (small-scale, into a scratch directory)
while writing this — step 3 does fail with a plain `python
scripts/build_arc1d_compositional.py` (`ModuleNotFoundError: No module
named 'data_modules'`), which is exactly why all three are shown as `-m`
invocations here rather than mixing styles.

Every experiment's `data_dir` points at `data/arc_1d_looped_augmented`
(step 2); experiment 4 additionally reads `data/arc_1d_compositional_holdout`
(step 3) for its zero-shot eval. Once both exist, optionally confirm there's
no train/eval content overlap (see "Data integrity" below):

```bash
pytest -m slow tests/test_data_leakage.py
```

## Task examples

Every task category the paper discusses — the 14 base ARC-1D categories
plus the 10 compositional (chained-skill) ones — has a reference figure
generated by `visualisation/plot_paper_tasks.py`:

```bash
python -m visualisation.paper.plot_paper_tasks --first-per-category               # base 14
python -m visualisation.paper.plot_paper_tasks --first-per-category --compositional  # compositional 10
```

Lands in `outputs/figures/00_task_examples/` (see "Figures and results"
below — not committed, regenerated on demand from `data/arc_1d` /
`data/arc_1d_compositional_holdout`, both built by `scripts/build_arc_1d.py`
/ `scripts/build_arc1d_compositional.py`).

## Folder layout

Each experiment folder follows the same shape:

```
0N_experiment_name/
  README.md          # method, findings, exact running/reading-results commands
  configs/            # every training yaml — the only thing train.py reads
  gen_*.py            # (if present) generates configs/*.yaml — provenance, run once
  run*.sh             # launches configs/ against train.py, skips completed runs
  plot_*.py           # renders figures from outputs/results/, described below
```

`configs/` is the only thing `train.py`/`scripts/run_config.sh` ever read;
everything else in an experiment folder is code that produces or consumes
`outputs/`.

## Reproduction workflow

One training entry point, always: `train.py --config <path>` (every
`run*.sh` — the generic launcher or an experiment's own — is a thin wrapper
around exactly this call, nothing bespoke per experiment). `train.py`
already writes final val/test metrics to `results.txt` itself, so that's
usually the whole pipeline: train → aggregate → plot.

```
configs/*.yaml  →  scripts/run_config.sh (or the experiment's own run*.sh)
                →  train.py --config <leaf>.yaml   (per (config, seed) pair)
                →  outputs/<project>/<run>/results.txt
                →  experiment's plot_*.py --outputs-dir outputs
                →  outputs/results/<experiment>/*.csv    (aggregated numbers)
                →  experiment's plot_*.py (no args)
                →  outputs/figures/<experiment>/*.{png,pdf}
```

Two things sit outside that default path, both optional:
- `validate.py` — a generic, standalone re-evaluation tool (reload a saved
  run, rescore `best`/`last`/`auto`/an explicit checkpoint). None of the 6
  experiments' own pipelines call it — `results.txt` already has what they
  need — it's there for ad-hoc checkpoint inspection.
- `scripts/eval_compositional_holdout.py` — experiment 4's own extra step,
  scoring a checkpoint against the disjoint compositional-holdout set
  (`validate.py` has no notion of a second, different eval set). Called
  automatically by `04_compositional_generalization`'s own `run.sh` — which,
  unlike every other experiment's `run*.sh`, trains nothing at all: both its
  arms reuse experiment 2's own dim=4 checkpoints directly (see its README).

Generic launcher (most experiments):

```bash
CFG_DIR=experiments/01_multitask_capacity/configs SEEDS_OVERRIDE="1 2 3 4 5" \
    bash scripts/run_config.sh                              # sequential, 1 GPU
CFG_DIR=experiments/01_multitask_capacity/configs SEEDS_OVERRIDE="1 2 3 4 5" GPUS="0,1,2" \
    bash scripts/run_config.sh                              # 3-way parallel
```

Skips any `(config, seed)` pair that already has a `results.txt`, so it's
always safe to rerun to backfill missing seeds or resume a killed sweep.
`04_compositional_generalization` and `06_data_efficiency_ablation/*` have
their own `run*.sh` instead — `04`'s is eval-only (reuses #2's checkpoints,
no data-build/GPU claim needed at all), `06`'s need a data-build step and a
non-round-robin GPU claim policy the generic launcher doesn't cover — see
their own READMEs.

## Figures and results

`outputs/` (repo-root, gitignored) holds everything a script generates:

- `outputs/<project>/<run>/results.txt` — one training run's final metrics
  (the existing convention, unchanged).
- `outputs/results/<experiment>/*.csv` — an experiment's aggregated,
  cross-seed numbers, written by that experiment's own `plot_*.py
  --outputs-dir outputs` (rescans every `results.txt` under `outputs/`).
- `outputs/figures/<experiment>/*.{png,pdf}` — the rendered figures,
  written by the same `plot_*.py` scripts, reading only the CSV above (no
  live `outputs/<project>/` tree needed once the CSV exists).

None of this is committed — it's regenerated, not source. **The exception
is at a paper milestone**: once an experiment's numbers are locked in,
force-add that specific snapshot so the repo captures the exact reported
result:

```bash
git add -f outputs/results/01_multitask_capacity/results.csv
git add -f outputs/figures/01_multitask_capacity/*.png outputs/figures/01_multitask_capacity/*.pdf
```

`git add -f` on an explicit path overrides `.gitignore` for that file only
and doesn't affect anything else under `outputs/`. Do this only when a
result is genuinely final — a force-added file has to be force-updated
again by hand if the experiment is rerun.

The `embeddings_*.npz` dumps used by `02_hypernetwork_multitask`'s cluster
plots follow the same rule and live in
`outputs/results/02_hypernetwork_multitask/`, not committed. Every new
training run with `log_embedding_clusters: true` saves this file itself,
naturally, as an end-of-run artifact
(`outputs/<project>/<run>/embedding_clusters/embeddings.npz`, via
`training.logging.log_embedding_cluster_plots`) — no `save_checkpoints`, no
separate dump step. `legacy/scripts/dump_embedding_clusters.py` (reload a
checkpoint, rerun the forward pass, dump the .npz) only remains useful for
a run that predates this and has no `embeddings.npz` of its own.

## Data integrity

Before any large rerun, `tests/test_data_leakage.py` independently
re-verifies that no augmented training example exactly matches a
validation/test/compositional-holdout example — the same exact-content
fingerprint `scripts/augment_arc_1d.py`'s build-time filter already uses,
recomputed from the final on-disk splits rather than trusted to have run
correctly. Marked `slow` (excluded from the default `pytest tests/` run —
fingerprinting the full ~700K-row train split takes about a minute) and
`skipif`-guarded on the dataset actually existing on disk:

```bash
pytest -m slow tests/test_data_leakage.py
```

Last run against `data/arc_1d_looped_augmented` (the dataset every
experiment above shares): **PASSED** — 676,665 distinct train examples
(721,000 raw rows), zero overlap with dev (1,800), test (1,800), or the
compositional holdout (400). The augmentation pipeline splits raw task
*instances* into train/dev/test before augmenting (not the other way
round), so a colour/shift/mirror variant of a train task can only coincide
with held-out content by exact fingerprint collision — which the build
already filters and this script re-confirms. The compositional holdout is
a fully separate synthetic generator with disjoint category names
(`1d_comp_*`), so no overlap is possible there by construction, independent
of this check.

## Architecture & seed conventions (needs a decision before the next reruns)

Sizes and seed counts have grown organically and aren't yet uniform:

| Experiment | Sizes used | Seeds |
|---|---|---|
| 01 (multitask capacity) | dim 4, 6, 10, 14 (~10K) all done -- Joint, Individual, and both ablations (Canon, optimizer), all four dims | 5 throughout |
| 02 (hypernetwork) | dim 4 (matched-scale, ~11.4K), dim 6 (fixed-width, ~337K) | dim=4: 5, checkpoints saved; dim=6: 3, no checkpoints (original grid, not resized/rerun). 1 (smoke/sizing checks), 5 (LoRA rank sweep) |
| 03 (reusability) | dim 4 (matched-scale, reuses #2's own checkpoints) | 5, both arms |
| 04 (compositional) | dim 4 (matched-scale) | 5 |
| 05 (leave-one-out) | dim 4 (matched-scale, reuses #2's recipe) | 5, both arms, all 14 categories -- done |
| 06 (data efficiency) | `hypernetwork/`/`individual/`: dim 4 (matched-scale, reuses #2's recipe). `joint/`: dim 14 (~10K-param scaffold, reuses #1's recipe -- sized to the hypernetwork's own param budget, not dim=4) | 5 throughout |

**Proposed standard** (confirm before the next round of reruns): dims **4
and 6** as the paper's stable sizes (drop dim=10 from new work -- keep the
existing dim=10 result in #1 for capacity-cliff breadth, don't extend it
elsewhere), **5 seeds** everywhere, and (for the hypernetwork specifically)
the matched-scale sizing #2's dim=4 now uses rather than a fixed-width
encoder. #1 is fully backfilled, #2's dim=4 is now matched-scale/5-seed;
this now mainly means #2's dim=6 (resize + rerun to match dim=4) and #06,
once completed.

## Ablation axes

| Experiment | with/without td | with/without canon |
|---|---|---|
| 01 | yes | yes (dims 4/6/10/14) |
| 02 | yes (`notd` / `frozen_td`) | — |
| 03 | yes (`notd` / `frozen_td`) | — |
| 04 | yes (`notd` / `frozen_td`) | — |
| 05 | yes (`notd` / `frozen_td`) | — |
| 06 | `hypernetwork/` and `joint/` both have `frozen_td`(`td`)/`notd`, paired at every data level; `individual/` has no task-id axis at all (one task per model) — data amount is the primary axis under test throughout | — |

`td` = a task-identity signal added to the model (a learned per-task
embedding, or `hyper_head.freeze_task_indicator: true`'s frozen one-hot
projection for the hypernetwork). `notd` = no task signal at all. See each
experiment's README for the exact mechanism.

## What's in `legacy/`

`legacy/{data_modules,scripts,configs/experiments}/` holds everything that
predates or was superseded by the experiments above — ~40 exploratory
config families (`arc1d_capacity_*`, `arc1d_hypermodel_looped_*`,
`arc1d_uniform_ablation`, ...), their one-off `gen_*.py`/`run_*.sh`
scripts, and the two datamodules (`arc1d_meta_padded_multiclass`,
`arc1d_meta_simple`) only they used. Kept for provenance via `git mv`
(history intact), not wired into the active registries — see
`legacy/README.md`.

## Open TODOs

- #2's dim=6 resize + rerun to the matched-scale recipe (dim=4 is done: 5
  seeds, both arms, checkpoints -- see its README's "Dim=4 rerun"). Its raw
  dim=6 run dirs are also currently missing from local `outputs/` entirely
  (only dim=4's are present) -- not just unresized, unreconstructable locally
  without a refetch or rerun.
- #6 is planned (resized to dim=4 matched-scale/14 categories/5 seeds, `joint/`
  arm added) but not yet launched -- see its own three READMEs before running.
- #3's `mirror`-category own/loo drop isn't diagnosed further (see its own
  "Open follow-ups") -- worth a closer look before generalising.
- #5's `1d_denoising_1c` outlier (the one category where `frozen_td` beats `notd` on held-out
  token accuracy) isn't diagnosed further -- see its own "Open follow-ups".
