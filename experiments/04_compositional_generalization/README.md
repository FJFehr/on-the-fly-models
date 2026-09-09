# Experiment 4: compositional generalisation

## Question

Train a hypernetwork on 14 in-distribution ARC-1D task categories, then test zero-shot on 10
synthetic categories that chain two of those rules together (e.g. denoise a block, *then*
shift it) -- a combination it was never trained on, built from two skills it was.

**Hypothesis**: `notd` (no task-identity embedding at all) should generalise to these
compositions *better* than `frozen_td` (a frozen, untrained one-hot task-identity embedding).
Without an explicit per-task anchor, the model has to organise its own internal task
representation from the support examples alone -- and that self-organised representation
should blend two rules together more naturally when the query actually needs both. An
explicit anchor, even an untrained/frozen one, instead pulls the model toward "this looks most
like known task X," which helps when the anchor happens to be right and works against
blending when it isn't.

## The finding

**Confirmed: `notd` beats `frozen_td` on zero-shot token accuracy in 8 of 10 held-out
composite categories (macro mean 75.5% vs. 67.7%), while `frozen_td` dominates in-distribution
(93.0% vs. 58.3% exact match) -- the same explicit-anchor tradeoff the hypothesis predicted.
Exact match on the held-out set stays near zero for both arms.**

| | In-distribution (14 categories) | Zero-shot holdout, seq_accuracy (10 categories) | Zero-shot holdout, exact_match |
|---|---:|---:|---:|
| `frozen_td` | **93.0%** | 67.7% | 0.05% |
| `notd` | 58.3% | **75.5%** | 1.00% |

(macro-average across categories, mean across 5 seeds each -- see
`outputs/figures/04_compositional_generalization/per_task_indist.png` /
`per_task_holdout.png`.)

Per held-out category (seq_accuracy, mean across 5 seeds):

| category | `notd` | `frozen_td` | diff |
|---|---:|---:|---:|
| `denoisemc_mirror` | 85.6% | 47.7% | **+37.9pp** |
| `shift3_copy` | 70.3% | 57.2% | +13.1pp |
| `fill_mirror` | 73.1% | 61.3% | +11.7pp |
| `fill_movedynamic` | 71.6% | 62.7% | +9.0pp |
| `hollow_shift3` | 85.2% | 78.2% | +7.0pp |
| `fill_shift3` | 73.9% | 68.8% | +5.1pp |
| `movedynamic_hollow` | 82.4% | 79.1% | +3.2pp |
| `denoise1c_shift3` | 77.0% | 75.5% | +1.5pp |
| `denoisemc_copy` | 67.3% | 69.6% | -2.3pp |
| `denoisemc_denoise1c` | 68.3% | 76.7% | -8.3pp |

`notd` wins 8/10; `denoisemc_mirror` shows both the largest gap and the same direction as the
original (larger-scale) version of this experiment's own finding, a consistent signal across
scales, not a one-off.

## Method

**Both arms reuse experiment 2's own dim=4 matched-scale hypernetwork checkpoints directly --
this experiment no longer trains anything.** Experiment 2's `dim4_notd.yaml`/
`dim4_frozentd.yaml` already train on the *exact* 14-category recipe this experiment needs
(confirmed by diffing the configs: identical `task_categories`/`val_task_categories`,
identical `data_dir`, identical architecture) -- so in-distribution numbers above are
literally experiment 2's own `results.txt`, and the only new thing this experiment adds is
the zero-shot compositional-holdout eval (`scripts/eval_compositional_holdout.py`) on top of
those checkpoints, 5 seeds each:

- **`notd`**: no dependency on `num_tasks` at all (`hyper_head.num_tasks: null` means no
  task-indicator projection exists in the model at all) -- experiment 2's checkpoints are
  structurally identical to what this eval needs. Evaluated directly, no changes.
- **`frozen_td`**: the eval script reserves indices 18-27 (10 composite categories) inside
  `hypermodel.task_indicator_proj.weight`, a plain `nn.Linear(num_tasks, hyper_output_dim,
  bias=False)`, requiring `num_tasks=28` at model-build time. Experiment 2's own `frozen_td`
  was trained with `num_tasks=18` (no reason for experiment 2 itself to reserve those 10
  slots) -- wrong shape to load directly.

  **Resolved by padding, not retraining**: `pad_frozentd_checkpoint.py` extends that
  checkpoint's projection matrix from 18 to 28 columns with 10 freshly-initialised values
  before the eval runs. This is mathematically sound, not a shortcut: `freeze_task_indicator:
  true` means that whole projection is random-init and **never gradient-updated**
  (`models/hypermodel.py:288-289` calls `requires_grad_(False)` right after construction) --
  both the real 18 columns and the 10 padded ones are equally "untrained random values from
  `nn.Linear`'s default init," never touched by an optimizer step either way. Padding with 10
  more is statistically identical to having trained with `num_tasks=28` from the start.
  Verified end-to-end: the padded-checkpoint numbers above land in the same ballpark as this
  experiment's own original, separately-trained `num_tasks=28` `frozen_td` run (see "Superseded
  original run" below).

**Data**: `data/arc_1d_compositional_holdout` (10 composite categories, 40 instances each,
built by `data_modules/arc1d_compositional.py` -- see
`legacy/configs/experiments/arc1d_hypermodel_compositional_generalization/README.md` for the
generator, rule semantics, and combo-selection rationale, unmodified since).

**Architecture** (documented in `configs/base.yaml`/`notd.yaml`/`frozen_td.yaml`, kept
buildable for a genuine from-scratch retrain if ever needed, e.g. a future dim=6 version, even
though the active pipeline doesn't train from them): identical to experiment 2's dim=4
matched-scale recipe (`task_encoding.embedding_dim=4`, target/hyper model `hidden_dim=4`,
`hyper_head.bottleneck_dim=8` -- 10,156 trainable params), `num_tasks` the only difference.

## Reproducing

One script to run (eval-only, no GPU/cluster needed once experiment 2's checkpoints are
local), one to plot:

```bash
bash experiments/04_compositional_generalization/run.sh
uv run python experiments/04_compositional_generalization/plot_compositional.py --outputs-dir outputs
```

`run.sh` evaluates `notd` directly against experiment 2's checkpoints, and pads +
evaluates `frozen_td`'s (`pad_frozentd_checkpoint.py`, skip-on-done, writes the padded
checkpoint under `outputs/04_compositional_generalization/padded_checkpoints/` so it's only
computed once). Both write to `outputs/compute_efficiency/04_compositional_generalization/
{notd,frozentd}_seed{1..5}/results.txt`. If experiment 2's checkpoints aren't local yet:

```bash
REMOTE_HOST=<node> INCLUDE_CHECKPOINTS=1 bash scripts/fetch_experiments.sh 02_hypernetwork_multitask
```

`plot_compositional.py --outputs-dir outputs` rescans both experiment 2's own tree
(in-distribution) and this experiment's own eval output (zero-shot), writes
`results_indist.csv`/`results_holdout.csv` under `outputs/results/
04_compositional_generalization/`, prints the macro-average summary above, and renders both
figures. Without `--outputs-dir`, replots from the CSVs as committed.

`report_holdout_breakdown.py` (run after `plot_compositional.py`, reads its
`results_holdout.csv`) writes the finer-grained per-composition breakdown -- token accuracy
and exact match for both arms, per held-out category, plus an Overall row matching the paper's
own aggregation convention (macro-averaged token accuracy; exact match pooled over all 2,000
held-out instances, mathematically the same computation here since every category/seed cell
has equal n=40) -- to `results_holdout_paper_table.csv`, the source for the per-composition
paper table (`tab:compositional_breakdown`).

`configs/notd.yaml`/`configs/frozen_td.yaml` (and `gen_seeds.py`'s generated `*_seed{N}.yaml`
leaves) remain accurate, buildable training recipes via `run_seeds.sh`/`run_train_eval.sh` --
documentation of exactly what would reproduce experiment 2's checkpoints from scratch, not the
active path. `frozen_td.yaml` specifically is still directly used, as the architecture
template the padded-checkpoint eval builds its model from.

## Superseded original run (kept for historical comparison, not part of the pipeline)

This experiment was originally run on 2026-09-01 with its own independently-trained
checkpoints (both arms, `num_tasks=28` for `frozen_td` from the start, no padding needed) --
5 seeds, both arms, on `torrnode7`, real checkpoints. That data still exists, undisturbed, at
`outputs/arc1d_v2_compositional_generalization_14task/` (old naming, predates this session's
cleanup) -- not migrated or deleted, just no longer authoritative. Its own numbers (macro
in-distribution `frozen_td` 91.4%/`notd` 62.0% test exact match; zero-shot seq_accuracy
`frozen_td` 60.0%/`notd` 74.1%) land close to the current ones above, as expected -- same
architecture, a genuinely different training-run instance rather than a literal
re-evaluation of the same weights, not a discrepancy.

One known gap in that original run, now closed as a side effect of the current pipeline:
its `embedding_clusters/` dumps are missing `embeddings.npz` (it predates that dump feature,
added to `training/logging.py` on 2026-09-06, five days later) -- the current, checkpoint-reuse
pipeline runs through today's code, so its own cluster dumps *do* include it automatically.

## Open follow-ups

- A dim=6 version of this experiment -- no spec exists; experiment 2's own dim=6 isn't
  resized to matched-scale yet either (see its README).
- `denoisemc_mirror`'s outsized `notd` advantage (+37.9pp) isn't diagnosed further -- is it
  something specific about how denoise and mirror rules combine, or a broader pattern that
  happens to show up most clearly there?
- Only same-architecture reuse tested here -- no attempt to see whether the padding approach
  (or the compositional-generalisation finding itself) holds at a different hypernetwork size.
