# Experiment 2: multi-task capacity

**Question**: can one small model learn all 14 ARC-1D tasks *jointly* (a
single shared weight set, no per-task model), with and without a per-task
identity signal? Full background: `docs/arc1d_story/06_phase1_findings.md`.

## The plot

![Capacity cliff](capacity_cliff.png)

X-axis is total parameter count (the same RoPE+Canon architecture at three
widths, `hidden_dim` 4/6/10, `embedding_dim` fixed at 10 throughout for
comparability), log scale. Y-axis is test-split exact match accuracy. Solid
lines are joint training (one shared model across all 14 categories); the
dotted line is individual training (one model per task, no sharing). Colour
separates the two joint arms: light purple has no task signal, dark purple
has a per-task identity embedding. Shaded bands are &plusmn;1 s.d. across
seeds.

All raw per-seed results live in `results.csv` next to the plot script, so
the plot can be rebuilt anywhere without needing a live `outputs/` directory
(which is gitignored and normally only exists on whichever cluster node ran
the jobs):

```bash
# rebuild the plot from the committed CSV (no outputs/ needed)
uv run python configs/experiments/arc1d_v2_multitask/plot_capacity_cliff.py

# refresh results.csv from a live outputs/ dir (run where it exists), then plot
uv run python configs/experiments/arc1d_v2_multitask/plot_capacity_cliff.py \
    --outputs-dir outputs
```

## The finding

Individual training saturates almost immediately -- 94.6% at 1,398 params,
98.9% at 2,444, 98.6% at 5,400. Joint training does not, even with task
identity:

| params | Individual | Joint, no task ID | Joint + task-ID embedding |
|---:|---:|---:|---:|
| 1,398 (dim=4) | 94.6% (n=5) | 12.0% (n=5) | 19.7% (n=5, unstable: 4.3-48.6%) |
| 2,444 (dim=6) | 98.9% (n=5) | 40.9% (n=5) | 74.0% (n=5) |
| 5,400 (dim=10) | 98.6% (n=3) | 59.4% (n=5) | 93.1% (n=5) |

The sharpest single-size demonstration is at **2,444 params**: individual
training is already fully saturated there (98.9%, matching the 5,400-param
ceiling), but joint training with task identity is still 25 points behind
(74.0%). Task identity alone cannot close that gap at this size -- there is
a real model-capacity regime that solves every task *alone* but cannot hold
them *together*, even when told which task it's looking at.

This motivates the hypernetwork: a mechanism that generates per-task
weights, rather than sharing one fixed set across tasks.

## Which tasks actually fail

![Per-task breakdown at dim=6](per_task_dim6.png)

The aggregate 74.0% (joint + task ID, dim=6) hides a sharp split: most
tasks are fully recovered, but a few collapse completely. `Flip` is the
starkest case -- 100% individually, **4%** in both joint arms, task ID or
not. `Mirror` (88% -> 20%) and `Move Dynamic` (88% -> 28%) show task ID
recovering *some* signal but nowhere near the individual ceiling. Most
other tasks (`Denoise`, `Pattern Copy`, `Move 1 Pixel`, ...) are at or near
100% in every condition -- the joint-training penalty is concentrated in a
handful of tasks, not spread evenly across all 14.

Only the validation split has a per-task breakdown (`on_validation_epoch_end`
accumulates `val_query_exact_match_by_task_<category>`; there's no test-time
equivalent), so this chart uses validation exact match throughout, for all
three conditions -- internally consistent, but not directly comparable
split-wise to the capacity-cliff plot above (which uses test). Currently
covers dim=6 only; dim=4 and dim=10 follow with `--dim 4` / `--dim 10` once
run.

```bash
uv run python configs/experiments/arc1d_v2_multitask/plot_per_task.py --dim 6
uv run python configs/experiments/arc1d_v2_multitask/plot_per_task.py \
    --outputs-dir outputs --dim 6   # refresh results_per_task.csv first
```

## Method

- **Individual**: one model per task category, no cross-task sharing (the
  Phase 1 recipe). dim=6/4 from `configs/experiments/arc1d_v2_minimal_size/`;
  dim=10 from `configs/experiments/arc1d_v2_backbone_capacity/` (RC1, the
  original Phase 1 run -- 3 seeds instead of 5).
- **Joint, no task ID** (`notd*.yaml` here): one model trained across all 14
  categories at once, no task signal.
- **Joint + task ID** (`td*.yaml` here): same joint setup, plus a per-task
  `nn.Embedding(18, 10)` looked up via `TASK_CATEGORY_INDEX` and added into
  the token embeddings before the backbone
  (`models/direct_supervised_lightning.py`,
  `task_encoding.use_task_embedding: true`), mirroring the hypernetwork's
  `task_indicator_proj` (`models/hypermodel.py`).
- All test-split exact match (`train_split=train`, `val_split=dev` for
  monitoring only -- no checkpoint selection since `save_checkpoints: false`
  everywhere in this project -- `test_split=test` reported once at the end).
- RoPE + Canon backbone, Muon optimizer (`muon_lr=0.005`,
  `muon_momentum=0.95`), flat (`n_loops=1`), `N_sup=1`, `max_steps=8000`.
- 5 seeds per condition/size (dim=10 individual: 3, from the original run).

## Configs in this folder

| File | Condition | Size |
|---|---|---|
| `notd.yaml` / `td.yaml` | Joint, no ID / with ID | dim=10 |
| `notd_dim6.yaml` / `td_dim6.yaml` | Joint, no ID / with ID | dim=6 |
| `notd_dim4.yaml` / `td_dim4.yaml` | Joint, no ID / with ID | dim=4 |
| `overfit/td_overfit.yaml` | sanity check for the task-embedding code path | dim=10, 3 categories |
