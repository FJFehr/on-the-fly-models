# Experiment 5: leave-one-out task generalisation

## Question

Train a hypernetwork on 13 of the 14 base ARC-1D task categories, then test zero-shot on the
held-out 14th category -- with vs. without task-identity (`notd`/`frozen_td`). Distinct from
experiment 4: that experiment holds out a *chained composition* of known base skills (e.g.
denoise-then-shift); this one holds out a whole base category itself, never seen in any form
during training. Every one of the 14 base categories is held out in turn, so this is full
category-level leave-one-out coverage, not a curated subset.

**Hypothesis** (carried over from the identical question already run once at this exact
architecture, see "Precedent" below): `notd` should beat `frozen_td` on held-out token accuracy,
and do so more *reliably* across seeds -- a frozen, untrained one-hot anchor gives the model
something to (mis)lean on when the held-out category has no column of its own, while `notd`
has no anchor to lean on wrong in the first place.

## Method

**Architecture**: identical to experiments 2-4's matched-scale dim=4 recipe (10,156 trainable
hypernetwork params, built around the dim=4 target model -- 1,398 params). See
`configs/base.yaml` for the full spec: `rope_canon_looped_transformer` target (`hidden_dim=4`,
flat, `n_loops=1`), `rope_canon_transformer` encoder (`hidden_dim=4, num_heads=1, num_layers=1,
output_dim=4`), `hyper_head.bottleneck_dim=8`, dense generation, `task_encoding.embedding_dim=4`,
Muon + AdamW-aux, `N_supervision=1, max_steps=8000, warmup_steps=800`.

**Mechanism**: each leaf config's `task_categories` is the 13 categories that remain after
dropping one; `val_task_categories` always stays the full 14. `results.txt`'s automatic
per-category validation breakdown (`val_query_exact_match_by_task_<category>`,
`val_query_accuracy_by_task_<category>`) then reports the held-out category's zero-shot score
alongside the 13 in-distribution ones, straight from that same training run -- **no separate
eval script or data-build step**, unlike experiment 4's compositional holdout: every base
category already lives in `data/arc_1d_looped_augmented`'s dev/test splits with a stable global
index (`TASK_CATEGORY_INDEX`, `models/hypermodel_lightning.py`).

`hyper_head.num_tasks` stays **18** for `frozen_td` regardless of which category is held out --
it's sized off that fixed global registry, not off how many categories any given leaf trains on,
so nothing about the model architecture changes leaf to leaf, only which 13 (of 14) rows of
`data/arc_1d_looped_augmented` get sampled during training.

`save_checkpoints: false` -- nothing downstream needs a checkpoint, the held-out score is read
directly from `results.txt`.

## Precedent

`legacy/configs/experiments/arc1d_v2_generalization` already ran this exact question at this
exact architecture: 3 seeds, a curated 5-category held-out subset (`1d_move_2p`,
`1d_denoising_mc`, `1d_flip`, `1d_pcopy_mc`, `1d_hollow`). Result: exact match a uniform 0
across all 30 cells (5 categories x 2 arms x 3 seeds); token accuracy told a decisive story --
`notd` beat `frozen_td` on **every** held-out category (5/5), 0.813 vs. 0.691 mean, and did so
far more consistently across seeds (sd 0.023-0.043 vs. `frozen_td`'s 0.044-0.398, most starkly
on `1d_denoising_mc` where `frozen_td` swung from 0.109 to 0.897 across 3 seeds). This experiment
extends that exact recipe to all 14 base categories at 5 seeds -- full coverage, formalised
rigour, same architecture, directly comparable numbers on the 5 categories that overlap.

## Arms

14 held-out categories x 2 variants x 5 seeds = **140 jobs**:

| Variant | `hyper_head.num_tasks` | `hyper_head.freeze_task_indicator` |
|---|---:|---:|
| `notd` | `null` | `false` |
| `frozentd` | `18` | `true` |

Plain (learned) `td` is not built, matching every other experiment in this repo's convention
("historically bimodal/unstable").

## Running

```bash
uv run python experiments/05_leave_one_out_task_generalization/gen_configs.py   # already generated, rerun only if the category/arm/seed grid changes

# Smoke test one leaf first -- see run.sh's header for why, and to get a real per-job timing
uv run python train.py --config experiments/05_leave_one_out_task_generalization/configs/hollow_notd_seed1.yaml

# Then the full 140-job sweep, round-robin across a torrnode's free GPUs
GPU_LIST="0 1 2 3 4 5 6 7" bash experiments/05_leave_one_out_task_generalization/run.sh

# Pull results.txt back (no checkpoints needed, save_checkpoints: false), then aggregate + plot
REMOTE_HOST=<node> bash scripts/fetch_experiments.sh 05_leave_one_out_task_generalization
uv run python experiments/05_leave_one_out_task_generalization/plot_leave_one_out.py --outputs-dir outputs
```

Resume-safe: skips any leaf whose `results.txt` already exists, so it's always safe to rerun
after an interruption or to backfill.

`plot_leave_one_out.py --outputs-dir outputs` rescans every leaf's `results.txt`, writes
`results_holdout.csv` (the held-out category's own score) and `results_indist.csv` (mean over
the other 13 categories, for context) under `outputs/results/
05_leave_one_out_task_generalization/`, prints the macro-average summary in "Findings" below,
and renders `per_category_holdout.png`. Without `--outputs-dir`, replots from the CSVs as
committed.

## Status

**Done.** All 140 jobs ran on `torrnode12` (8-way parallel, ~24 min/job, ~7h15m total wall
clock), 0 failures. Results pulled back and aggregated with `plot_leave_one_out.py`.

## Findings

**Confirmed, and more decisively than the legacy 5-category precedent: `notd` beats
`frozen_td` on held-out zero-shot token accuracy in 13 of 14 categories (macro mean 81.9% vs.
68.5%), while `frozen_td` dominates in-distribution (99.9% vs. 97.0% token accuracy; 97.0% vs.
64.7% exact match) -- the same explicit-anchor tradeoff experiments 2 and 4 already found, now
confirmed at full category-level coverage.**

| | In-distribution (13 categories/leaf, macro) | Held-out zero-shot (14 categories) |
|---|---:|---:|
| `frozen_td` token accuracy | **99.9%** | 68.5% |
| `notd` token accuracy | 97.0% | **81.9%** |
| `frozen_td` exact match | **97.0%** | 0.29% |
| `notd` exact match | 64.7% | 9.14% |

(macro-average across categories, mean across 5 seeds each -- see
`outputs/figures/05_leave_one_out_task_generalization/per_category_holdout.png`.)

Per held-out category (token accuracy, mean across 5 seeds):

| category | `notd` | `frozen_td` | diff |
|---|---:|---:|---:|
| `1d_denoising_mc` | 81.1% | 47.9% | **+33.2pp** |
| `1d_scale_dp` | 86.9% | 57.1% | +29.8pp |
| `1d_pcopy_mc` | 84.7% | 57.6% | +27.2pp |
| `1d_move_3p` | 91.8% | 68.4% | +23.4pp |
| `1d_pcopy_1c` | 99.5% | 77.3% | +22.2pp |
| `1d_move_2p_dp` | 95.5% | 74.9% | +20.6pp |
| `1d_mirror` | 68.1% | 59.0% | +9.1pp |
| `1d_move_dp` | 86.1% | 77.0% | +9.1pp |
| `1d_move_1p` | 92.0% | 84.0% | +8.0pp |
| `1d_move_2p` | 91.2% | 84.6% | +6.5pp |
| `1d_fill` | 69.2% | 62.8% | +6.4pp |
| `1d_flip` | 83.6% | 78.2% | +5.5pp |
| `1d_hollow` | 69.9% | 66.1% | +3.9pp |
| `1d_denoising_1c` | 46.9% | 64.6% | **-17.7pp** |

`notd` wins 13/14; `1d_denoising_1c` is the one exception, and the only category where
`frozen_td` wins outright -- worth a closer look (see "Open follow-ups"). The overall direction
and rough magnitude match the legacy 3-seed/5-category precedent closely: this run's macro mean
for `notd` (81.9%) lands almost exactly on legacy's own 5-category macro mean (81.3%), a good
cross-run consistency check despite the different (and larger) category coverage.

Exact match on the held-out set is mostly 0 but not perfectly uniform like the legacy 5-category
run -- `notd`'s macro exact-match mean is 9.14% (driven by a handful of categories, not all of
them), `frozen_td`'s is 0.29%, essentially the null result legacy found, just not *quite* as
clean at this larger category count.

## Open follow-ups

- `1d_denoising_1c` is the one category where `frozen_td` beats `notd` on held-out token
  accuracy -- not diagnosed further here. Worth checking whether it's something specific about
  that category (e.g. an unusually strong or weak sibling remaining in training) or a genuine
  seed-noise outlier (its `frozen_td` std is comparatively low, so probably not just noise).
- A dim=6 version -- no spec exists; experiment 2's own dim=6 isn't resized to matched-scale yet
  either (see its README).
- The non-uniform (if still small) `notd` exact-match rate on the held-out set, vs. legacy's
  perfectly uniform 0 -- which categories actually land bit-perfect zero-shot answers, and is
  there a pattern to which ones can?
