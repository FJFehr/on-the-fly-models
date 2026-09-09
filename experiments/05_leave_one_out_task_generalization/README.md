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
```

Resume-safe: skips any leaf whose `results.txt` already exists, so it's always safe to rerun
after an interruption or to backfill.

Reading results: pull `val_query_exact_match_by_task_<held_out_category>` and
`val_query_accuracy_by_task_<held_out_category>` out of each leaf's
`outputs/05_leave_one_out_task_generalization/<experiment_name>/results.txt`, across the 5 seeds
per (held-out category, arm) cell.

## Status

**Not yet run.** Configs generated (140 leaf configs, `gen_configs.py`), `run.sh` ready. Needs a
torrnode launch (see the repo-root README's cluster section for the allowed node range) --
smoke-test one leaf first per `run.sh`'s header before committing a node to the full sweep.

## Findings

TBD -- fill in after the 140-job sweep completes and results are pulled back. Expect a
plot/report script (following experiments 02/04's `plot_*.py --outputs-dir outputs` pattern) to
aggregate the per-category breakdown into `outputs/results/05_leave_one_out_task_generalization/
*.csv` before this section is written.

## Open follow-ups

- A dim=6 version -- no spec exists; experiment 2's own dim=6 isn't resized to matched-scale yet
  either (see its README).
- Whether the 9 categories beyond the legacy 5 (never run at this question before, at any scale)
  follow the same `notd`-favouring direction, or surface a category where `frozen_td` actually
  wins -- worth flagging in the Findings section rather than assuming the pattern is universal.
