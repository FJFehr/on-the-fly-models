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

## Fixed architecture (identical to arc1d_lowdata's target_model)

- `rope_canon_looped_transformer` (`hidden_dim: 16, num_heads: 2, inner_dim: 16,
  inner_num_heads: 2, n_loops: 4, dropout: 0.1`, same Canon settings), trained
  directly via `LoopedSupervisedLightning` (`model: looped_supervised`) — no
  task-identity conditioning of any kind, matching the already-validated
  `arc1d_uniform_ablation/*/T5_dim16_looped.yaml` config for this exact
  architecture.
- `optimizer: Muon, muon_lr: 0.005, muon_momentum: 0.95` — matches
  `arc1d_lowdata` exactly, to remove optimizer choice as a confound between
  the two experiments. (The alternative, RAdam, was already proven for this
  architecture in `arc1d_uniform_ablation`/`arc1d_recursion_ablation`, but
  consistency with `arc1d_lowdata` was prioritised; this exact
  Muon+direct-training combination is new, not previously validated here.)
- `max_steps: 2000, warmup_steps: 200, N_supervision: 2` — matches
  `arc1d_lowdata`'s compute budget exactly.
- One model **per task category** (15 categories, same list as
  `arc1d_lowdata/base.yaml` — not `arc1d_uniform_ablation`'s 17; the 2 extra
  categories there are intentionally excluded here too).

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

**Levels this round**: `variants_per_base_task` ∈ {1, 2, 3} (≈40/80/120
examples/category), matching `arc1d_lowdata`'s `cell_v1`/`cell_v2`/`cell_v3`.
Levels 4/5/20 are out of scope for now.

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
python scripts/gen_lowdata_baseline_configs.py
bash scripts/run_lowdata_baseline.sh
```

Split across nodes via `SEEDS_OVERRIDE`/`CATEGORY_GLOB` (see script header for
examples). 135 jobs total (15 categories × 3 levels × 3 seeds), each tiny
(≤360 raw training pairs) and single-GPU (`devices: 1`) — no benefit to
claiming multiple GPUs per job here, unlike `arc1d_lowdata`.

## Reading results

For each category and level, compare `val_query_exact_match`/
`test_query_exact_match` (mean ± std across 3 seeds) against `arc1d_lowdata`'s
per-category numbers at the same `variants_per_base_task` level (via its
`val_query_exact_match_by_task_{category}` metrics). A much larger gap in
favour of the hypernetwork at low levels (1/2) that narrows by level 3
supports the cross-task-transfer hypothesis; a small gap at every level
suggests the architecture itself, not the hypernetwork's weight sharing, is
doing most of the work.
