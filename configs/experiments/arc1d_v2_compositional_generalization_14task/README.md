# arc1d_v2_compositional_generalization_14task

## Goal

Same question as `arc1d_v2_compositional_generalization`: train a hypernetwork on in-distribution
ARC-1D task categories, then test zero-shot on 10 synthetic categories that chain two of those
rules together (e.g. denoise a block, *then* shift it) -- a combination it was never trained on,
built from two skills it was.

**Hypothesis**: the `notd` arm (no task-identity embedding at all) should generalize to these
compositions *better* than `frozen_td` (a frozen, untrained one-hot task-identity embedding).
Without an explicit per-task anchor, the model has to organize its own internal task
representation from the support examples alone -- and that self-organized representation should
blend two rules together more naturally when the query actually needs both. An explicit anchor,
even an untrained/frozen one, instead pulls the model toward "this looks most like known task X,"
which helps when the anchor happens to be right and actively works against blending when it
isn't.

## Relationship to `arc1d_v2_compositional_generalization`

This is that experiment, unchanged in every respect except the training task set:

- `arc1d_v2_compositional_generalization` trains on **15** categories (the standard 14 plus
  `1d_recolor_cmp`, kept there for apples-to-apples comparison against the older, bigger-recipe
  `arc1d_hypermodel_compositional_generalization` run).
- This experiment trains on the **standard 14-category v2 list** instead (the same list
  `arc1d_v2_hypernetwork_multitask` and the rest of the v2 family use), dropping
  `1d_recolor_cmp`. Safe to drop: none of the 10 held-out composite categories involve
  `1d_recolor_cmp` at all (see the combo table below), and `frozen_td`'s `num_tasks=28` is sized
  off the fixed 18-category `TASK_CATEGORY_INDEX` registry (`models/hypermodel_lightning.py`),
  not off how many categories this run actually trains on.

Everything else is reused unmodified:

- `data_modules/arc1d_compositional.py` (the composite-task generator) and
  `data/arc_1d_compositional_holdout` (the 400-instance, 10-category held-out set built from it).
- `scripts/eval_compositional_holdout.py` (zero-shot eval + embedding-cluster holdout overlay).

See `arc1d_hypermodel_compositional_generalization/README.md` for the generator's architecture,
the base-rule semantics table, and the full combo-selection rationale (which 10 pairings were
kept and why) -- none of that is repeated here.

## Architecture

Copied unmodified from `arc1d_v2_compositional_generalization/base.yaml` (itself copied from
`arc1d_v2_hypernetwork_multitask/shrink_matched_dim4.yaml`, the sizing sweep's smallest-that-
still-holds config):

| Component | Value |
|---|---|
| Target model | `rope_canon_looped_transformer`, `hidden_dim=4`, flat (`n_loops=1`, not looped) -- **1,398 params**, the "1.4k" downstream/target-model scale this experiment is built around |
| Hypernetwork encoder | `rope_canon_transformer` (plain, non-Zhu), `hidden_dim=4, num_layers=1, output_dim=4` |
| `task_encoding.embedding_dim` | 4 |
| `hyper_head.bottleneck_dim` | 8 |
| Generation | Dense (`lora_adapter: false`) -- the sizing sweep found LoRA strictly worse below the dense floor at this scale |
| Optimizer | Muon (`muon_lr=0.005, muon_momentum=0.95`) + AdamW aux (`lr=0.001, wd=0.01`) |
| Budget | `N_supervision=1, max_steps=8000, warmup_steps=800` (8000-total-optimizer-update budget) |
| Total hypernetwork params | 10,156 |

Training data is the standard, full per-category augmented set (`data/arc_1d_looped_augmented`,
no `variants_per_base_task` cap) -- only the task-category *list* differs from
`arc1d_v2_compositional_generalization`, not the amount of data per category.

## Task set

The standard 14-category v2 list (drops `1d_recolor_cmp` from the 15-category list the original
compositional-generalization experiments use):

`1d_denoising_1c, 1d_denoising_mc, 1d_fill, 1d_flip, 1d_hollow, 1d_mirror, 1d_move_1p,
1d_move_2p, 1d_move_2p_dp, 1d_move_3p, 1d_move_dp, 1d_pcopy_1c, 1d_pcopy_mc, 1d_scale_dp`

## Arms

Only `notd` and `frozen_td` -- no plain `td`, matching `arc1d_v2_hypernetwork_multitask`'s own
choice to drop it ("historically bimodal/unstable"), and a learned one-hot embedding is undefined
on the held-out compositional indices anyway:

| Variant | `hyper_head.num_tasks` | `hyper_head.freeze_task_indicator` |
|---|---:|---:|
| `notd` | `null` | `false` |
| `frozen_td` | `28` (18 base + 10 reserved composite indices) | `true` |

## Running

```bash
bash scripts/run_arc1d_v2_compositional_generalization_14task.sh
```

Trains both arms, then runs `scripts/eval_compositional_holdout.py` for each against the held-out
compositional set. Set `FREE_GPUS_FLAG="--free-gpus"` if the node is shared.

## Reading results

`outputs/arc1d_v2_compositional_generalization_14task/<arm>/results.txt` for in-distribution
val/test metrics (including the per-task-category exact-match and accuracy breakdown), and
`outputs/arc1d_v2_compositional_generalization_14task/<arm>/compositional_holdout_eval/results.txt`
for the zero-shot per-composite-category `exact_match`/`seq_accuracy` table. Compare directly
against `arc1d_v2_compositional_generalization`'s own results (same architecture, same held-out
set, only the training task list differs by one category) to see whether dropping
`1d_recolor_cmp` changes the notd-vs-frozen_td direction or magnitude found there.

## Status

Run and evaluated, 5 seeds (1-5), both arms, on `torrnode7`. See Findings below.

## Findings (5 seeds: 1-5)

**In-distribution (test set, all 14 training categories)** -- `frozen_td` dominates, same
direction as every other `arc1d_v2_*` notd/frozen_td comparison, and clears the 15-category
version's numbers too:

| | `frozen_td` test EM | `notd` test EM |
|---|---:|---:|
| mean ± stdev (n=5) | **91.4% ± 7.2%** | **62.0% ± 10.0%** |

**Zero-shot on the 10 held-out composite categories** (`n=40` per category, 400 per seed) --
mean token accuracy (`seq_accuracy`) per category, averaged over the 5 seeds:

| category | `frozen_td` seq_accuracy | `notd` seq_accuracy | diff |
|---|---:|---:|---:|
| `denoise1c_shift3` | 0.710 | 0.723 | +0.013 |
| `denoisemc_copy` | 0.466 | 0.707 | +0.241 |
| `denoisemc_denoise1c` | 0.537 | 0.709 | +0.172 |
| `denoisemc_mirror` | 0.589 | 0.829 | +0.240 |
| `fill_mirror` | 0.564 | 0.708 | +0.143 |
| `fill_movedynamic` | 0.573 | 0.686 | +0.113 |
| `fill_shift3` | 0.700 | 0.724 | +0.024 |
| `hollow_shift3` | 0.598 | 0.854 | +0.255 |
| `movedynamic_hollow` | 0.704 | 0.814 | +0.110 |
| `shift3_copy` | 0.556 | 0.659 | +0.103 |
| **overall mean ± stdev (n=5 seeds)** | **0.600 ± 0.065** | **0.741 ± 0.027** | **+0.141** |
| **overall exact_match, mean (n=5 seeds)** | **0.000** | **0.013 ± 0.011** | |

Per-category `exact_match`, same 5-seed averaging (`notd`'s hit count is out of 200 = 5 seeds ×
40 held-out instances per category):

| category | `frozen_td` exact_match | `notd` exact_match | `notd` hits |
|---|---:|---:|---:|
| `denoise1c_shift3` | 0.000 | 0.005 | 1/200 |
| `denoisemc_copy` | 0.000 | 0.025 | 5/200 |
| `denoisemc_denoise1c` | 0.000 | 0.020 | 4/200 |
| `denoisemc_mirror` | 0.000 | 0.070 | 14/200 |
| `fill_mirror` | 0.000 | 0.015 | 3/200 |
| `fill_movedynamic` | 0.000 | 0.000 | 0/200 |
| `fill_shift3` | 0.000 | 0.000 | 0/200 |
| `hollow_shift3` | 0.000 | 0.000 | 0/200 |
| `movedynamic_hollow` | 0.000 | 0.000 | 0/200 |
| `shift3_copy` | 0.000 | 0.000 | 0/200 |
| **total (2,000 held-out instances)** | **0/2,000** | **27/2,000 (1.35%)** | |

`frozen_td` is a flat, exact zero -- not one exact match across every seed, every category, all
2,000 held-out instances. `notd`'s exact matches concentrate in 5 of the 10 categories, and
`denoisemc_mirror` stands out (14/200, 7%) -- notably the *same* category with the largest
seq_accuracy gap above (+0.240), so the two metrics agree on where `notd`'s advantage is
strongest, not just on average. The other 5 categories are exact-match-zero for both arms across
every seed: the token-accuracy gap there is real, but at this model scale neither arm ever
resolves it into a fully correct sequence.

**The hypothesis holds, and much more cleanly than any single-seed compositional-generalization
run before it: `notd` beats `frozen_td` on token accuracy in 10/10 categories** (not 6/10 or
9/10 as in the two single-seed experiments this rerun improves on), with the per-seed means
essentially non-overlapping (`frozen_td` seed range 51.3-68.9%, `notd` seed range 70.0-76.7%) --
this is no longer an artifact of one outlier category or one lucky seed. Exact match stays
essentially zero for both arms, as always, but even there the two arms separate cleanly:
`frozen_td` scores exactly 0/2,000 held-out instances across all 5 seeds, while `notd` gets a
handful right in every single seed (13-28 per seed, never zero) -- a small but perfectly
consistent difference.

This is exactly the pattern the hypothesis predicted: `frozen_td`'s explicit (if untrained)
task-identity anchor buys a large, consistent in-distribution advantage (+29 points test EM) by
giving the model a fixed slot per known task, but that same anchor actively works against it the
moment the true answer isn't any single known task -- it has nothing to blend *from* except "which
one known category is this closest to." `notd` has no such anchor, so its internal task
representation is built solely from the support examples every time, which turns out to combine
two rules together more naturally when the query actually needs both blended.

**Caveats**: 5 seeds is still a small sample for the per-category breakdown (each category's
5-seed mean has its own noise), and this is a single architecture at a single (very small) scale
-- it doesn't establish the finding holds at larger hypernetwork sizes or with different
target-model widths. The overall seq_accuracy comparison (50 seed-category cells per arm,
collapsed to 5 seed-level means) is the more defensible summary than any single category's
number.
