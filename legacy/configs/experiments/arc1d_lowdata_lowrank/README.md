# arc1d_lowdata_lowrank

Follow-on from `arc1d_lowdata`. That experiment found `variants_per_base_task=2` (only 2
augmented examples per base task, on top of the true original) already reaches full
performance at the default `lora_adapter_rank=8`.

## Goal

Does that same low-data conclusion hold at lower LoRA adapter rank, or does rank become
the bottleneck once training data is this scarce? And is the LoRA-style low-rank
parameterization itself even necessary at low data, or does full-rank (direct, unrestricted
weight generation) do just as well? This experiment crosses two axes:

- `hyper_head.lora_adapter_rank` in `{4, 2, 1}`, plus a `full` arm
  (`hyper_head.lora_adapter: false` — the hypernetwork predicts every target weight
  directly, full rank, no frozen random backbone at all; see `models/hypermodel.py`)
- `variants_per_base_task` in `{1, 2, 3}`

`rank=8` is **not** re-run here — use `arc1d_lowdata`'s `cell_v1`/`cell_v2`/`cell_v3`
results (same 3 seeds, same data levels) as the rank=8 reference column.

## Fixed architecture (carried over from arc1d_lowdata/arc1d_hypermodel_looped_rope_canon_muon)

- Zhu block backbone (`rope_canon_zhu_transformer`, RMSNorm + QK-norm + SwiGLU).
- `hyper_head.num_tasks: 18, freeze_task_indicator: true` (`frozen_td`).
- `optimizer: Muon, muon_lr: 0.005, muon_momentum: 0.95` — the already-validated stable
  value (see `arc1d_hypermodel_looped_rope_canon_muon/README.md`).
- `max_steps: 2000, warmup_steps: 200`, `learning_rate: 0.001` (AdamW-aux group) — held
  fixed across every cell so training compute stays matched.
- 15 in-distribution task categories, `task_categories == val_task_categories`.

## Data-reduction mechanism

Identical to `arc1d_lowdata` — see that experiment's README for the full explanation of
`Arc1dMetaMulticlassDataModule.variants_per_base_task`/`data_seed`
(`data_modules/arc1d_meta_multiclass.py`'s `_stratified_variants_per_base_task`):
stratified per base task, nested/cumulative across levels, `data_seed` fixed at 42 so all
3 training seeds at a given cell see identical training data.

## Grid

| | `variants_per_base_task=1` | `=2` | `=3` |
|---|---:|---:|---:|
| `lora_adapter_rank=4` | `cell_r4_v1` | `cell_r4_v2` | `cell_r4_v3` |
| `lora_adapter_rank=2` | `cell_r2_v1` | `cell_r2_v2` | `cell_r2_v3` |
| `lora_adapter_rank=1` | `cell_r1_v1` | `cell_r1_v2` | `cell_r1_v3` |
| `full` (no LoRA) | `cell_full_v1` | `cell_full_v2` | `cell_full_v3` |

12 cells × 3 seeds = **36 jobs**.

## Running

```bash
python scripts/gen_lowdata_lowrank_configs.py
bash scripts/run_lowdata_lowrank.sh
```

Split across nodes via `SEEDS_OVERRIDE`/`CELL_GLOB` (see script header for examples).

## Reading results

For each `(arm, variants_per_base_task)` cell, compare `val_query_exact_match`/
`test_query_exact_match` (mean ± std across 3 seeds) against:
1. `arc1d_lowdata`'s `rank=8` result at the same `variants_per_base_task` (the reference
   column this experiment doesn't re-run).
2. The other cells in this grid's own rank=8-free 4×3.

Two headline questions:
- Does the "2 augmentations is enough" conclusion hold as rank drops, or is there a rank
  floor below which more data is needed to compensate — i.e. does the data-efficiency win
  from cross-task transfer depend on having enough adapter capacity to actually express it?
- Does the `full` (no-LoRA) arm do better, worse, or the same as the rank-8 reference at
  each data level? Worse would suggest the low-rank constraint itself is acting as a useful
  regularizer at low data (full-rank has far more free parameters to overfit ~40-120
  examples/category with); better or equal would suggest the constraint isn't buying
  anything and the data-efficiency result is about the hypernetwork/cross-task transfer
  mechanism generally, not specifically about low-rank adaptation.
