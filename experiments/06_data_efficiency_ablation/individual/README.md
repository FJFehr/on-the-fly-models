# arc1d_lowdata_baseline

Companion experiment to `arc1d_lowdata`, testing the same data-reduction axis
**without the hypernetwork**.

## Goal

`arc1d_lowdata` shows that its hypernetwork reaches near-full performance at
`variants_per_base_task=2` (~80 examples/category). The open question: does
that data efficiency come specifically from **cross-task transfer** — the
hypernetwork sharing one backbone across all 15 task categories — or would a
plain single-task model need just as little data on its own?

This experiment trains the **exact same target architecture directly**, one
model per task category, with no hypernetwork, no weight generation, and no
cross-task sharing at all. If the baseline needs much *more* data than the
hypernetwork to reach the same performance at a given category, that supports
the cross-task-transfer hypothesis. If it needs about the same, the
hypernetwork isn't buying data efficiency at these scales.

**2026-09-10: resized alongside `arc1d_lowdata`'s own dim=4 rescale** — was
`hidden_dim=16/n_loops=4` (the pre-unification legacy target this experiment's
own "EXACT SAME target architecture" framing always required matching), now
experiment 2's own dim=4 matched-scale target. Also dropped `1d_recolor_cmp`
(14 categories now) and moved 3→5 seeds (135→210 jobs), then added a `full`
level the same day alongside the other two arms' own `full` cells (210→280
jobs — see "Levels this round" below).

**2026-09-14: unified onto experiment 1's own dim=4 Individual recipe exactly**
(`01_multitask_capacity/configs/individual/*/dim4.yaml`), not just its
architecture — this arm's own "exact same target architecture" claim had
quietly stopped being true beyond the backbone: `model` was
`looped_supervised` (now `direct_supervised`, matching 01 — `LoopedSupervised`'s
`N_supervision: 2` doubled the real optimizer-step count per batch via
`training/trainer.py`'s `Trainer(max_steps=max_steps*N_supervision)`, not
something 01's own reference does at all), `max_steps` 2000→8000,
`learning_rate` 0.001→0.0005, `gradient_clip_val: 10.0`→removed (01 has none,
and switching to `direct_supervised`'s automatic optimization means
Lightning's own automatic clipping would have newly activated here if left
in), `task_encoding.embedding_dim` 4→10 (01 holds this fixed at 10 across its
whole dim sweep — confirmed in code that it sizes the embedding tables
feeding a `Linear(embedding_dim → hidden_dim)` projection into the backbone,
real input capacity independent of `hidden_dim`, not something that should
shrink with it). Caught because this arm's own `full`-level result (macro
86.3%) didn't match 01's established dim=4 ceiling (92.6%) — `1d_move_dp`
alone was 24% vs. 01's own ~65%. All 280 jobs from before this were
invalidated and rerun.

## Fixed architecture (identical to arc1d_lowdata's target_model)

- `rope_canon_looped_transformer` (`hidden_dim: 4, num_heads: 1, inner_dim: 4,
  inner_num_heads: 1, n_loops: 1, dropout: 0.1`, same Canon settings —
  experiment 2's own dim=4 matched-scale target), trained directly via
  `DirectSupervisedLightning` (`model: direct_supervised`, matching
  experiment 1's own Individual arm exactly) — no task-identity conditioning
  of any kind.
- `optimizer: Muon, muon_lr: 0.005, muon_momentum: 0.95` — matches
  `arc1d_lowdata` exactly, to remove optimizer choice as a confound between
  the two experiments, and the same value experiment 1's own dim=4 optimizer
  ablation already validated at this exact target architecture (the
  alternative there, AdamW, is not re-tested here). `learning_rate: 0.0005`
  and `task_encoding.embedding_dim: 10` match experiment 1's own dim=4
  Individual convention exactly (not `arc1d_lowdata`'s — the two references
  differ here, and this arm now follows its own).
- `max_steps: 8000, warmup_steps: 200` — matches experiment 1's own dim=4
  Individual compute budget exactly, not `arc1d_lowdata`'s own (reduced)
  budget.
- One model **per task category** (14 categories, the paper's standard set —
  same list as `arc1d_lowdata/base.yaml` and matching 01/02/05, not the
  original 15 or `arc1d_uniform_ablation`'s 17).

## Data-reduction mechanism

`Arc1dDirectDataModule` (`data_modules/arc1d_direct.py`) now exposes
`variants_per_base_task`/`data_seed`, using the identical stratified/nested
sampling method ported verbatim from `arc1d_meta_multiclass.py`'s
`_stratified_variants_per_base_task`: for each base task, the true original
(unaugmented) example is always included first, then `K-1` more augmented
variants in a fixed `data_seed`-ordered sequence, so level `K`'s selection is
always level `K-1`'s plus one more per base task. Applied **only** to the
train split (val/test stay at their full, fixed size).

Because both experiments use the same `data_dir: data/arc_1d_looped_augmented`
and the same `data_seed=42`, **at a given level both experiments train on the
identical underlying task-variant rows** — the data itself is matched, not
just the count.

**Levels this round**: `variants_per_base_task` ∈ {1, 2, 3, full} (≈40/80/120/≈40,000
examples/category), matching `arc1d_lowdata`'s `cell_{cond}_v1`/`v2`/`v3`/`full`. Levels
4/5/20 are out of scope for now. `full` (no reduction, added 2026-09-10) runs at
`max_steps: 8000` (since the 2026-09-14 unification, matching experiment 1's own dim=4
Individual budget exactly) — so `full` here genuinely reproduces experiment 1's own
full-data ceiling (94.6% here vs. 01's own 92.6%), the same as `../hypernetwork/`'s and
`../joint/`'s own `full` cells now do against their own references, since all three arms
went through this same unification together.

**2026-09-15: added a second, independent data-reduction axis** —
`base_tasks_per_category` in {1, 3, 5, 10, 20}, always at `variants_per_base_task=1` (zero
additional augmentation), tagged `{category}/t{N}.yaml`. Reduces task *diversity* rather
than augmentation *depth*, reaching below the `v1` floor's ~40 base-tasks/category. This
is the extreme end of the isolated-model floor: at `t1`, a single per-category model here
trains on exactly one base task's 3 support pairs — the sharpest test of whether
`../hypernetwork/`'s and `../joint/`'s cross-task sharing (which still pool 14x that
across categories at the same `N`) pulls ahead. Selection uses its own `data_seed`-seeded
RNG stream, independent of the `v*` axis's own (see `data_modules/arc1d_direct.py`'s
`_stratified_base_tasks_per_category`). 5 more levels × 14 categories × 5 seeds = **350
more jobs**, total **630**.

## Known asymmetry (documented, not "fixed")

`Arc1dDirectDataModule` unpacks each selected row into its **3 support**
`(input, output)` pairs only — the row's own query is held out entirely for
the separate dev/test splits. `arc1d_lowdata`'s hypernetwork, by contrast,
supervises reconstruction of **all 4** examples per row (the 3 support pairs
plus that row's own query, confirmed by tracing the training loss in
`models/hypermodel_lightning.py`). So at a given `variants_per_base_task`
level, this experiment trains on 3 raw pairs/row vs. 4 there.

This is an inherent difference between the two datamodules' pre-existing,
independently-used conventions, not something introduced for this
comparison — changing either would break compatibility with other
experiments that already rely on them. Note it when interpreting a close
result; it does not by itself explain a *large* gap in either direction.

Separately: the hypernetwork's 3 support examples are **not** shown to the
target model in-context (each of the 3 support + 1 query runs independently
through the same generated weights, not attended jointly as a few-shot
prompt) — so this is not a few-shot-ICL advantage for the hypernetwork side,
only the extra-supervision-target asymmetry above.

## Running

```bash
python experiments/06_data_efficiency_ablation/individual/gen_configs.py
bash experiments/06_data_efficiency_ablation/individual/run.sh
```

Split across nodes via `SEEDS_OVERRIDE`/`CATEGORY_GLOB` (see script header for
examples). 630 jobs total (14 categories × 9 levels [4 v/full + 5 t] × 5
seeds), each tiny (≤360 raw training pairs, or up to ≈40,000 rows at `full`,
or as few as 3 at `t1`) and single-GPU
(`devices: 1`) — no benefit to claiming multiple GPUs per job here, unlike
`arc1d_lowdata`.

## Reading results

For each category and level, compare `val_query_exact_match`/
`test_query_exact_match` (mean ± std across 5 seeds) against `arc1d_lowdata`'s
per-category numbers at the same `variants_per_base_task` level (via its
`val_query_exact_match_by_task_{category}` metrics). A much larger gap in
favour of the hypernetwork at low levels (1/2) that narrows by level 3 or
`full` supports the cross-task-transfer hypothesis; a small gap at every
level suggests the architecture itself, not the hypernetwork's weight
sharing, is doing most of the work.
