# arc1d_v2_compositional_generalization

## Goal

Same question as `arc1d_hypermodel_compositional_generalization`: train a hypernetwork on 15
in-distribution ARC-1D task categories, then test zero-shot on 10 synthetic categories that chain
two of those rules together (e.g. denoise a block, *then* shift it) -- a combination it was never
trained on, built from two skills it was.

This is a rerun of that question at the **"matched-scale" architecture** from
`arc1d_v2_hypernetwork_multitask`'s sizing sweep (10,156 params -- the smallest configuration in
that sweep that still matched individual-training's in-distribution ceiling, 95.7% mean exact
match), not the original experiment's much bigger `rope_canon_zhu_transformer`/hidden_dim=64,
8-layer recipe (~1.58M-ish params region before that sweep). Since the rest of the v2 story now
uses this minimal scale, this keeps the compositional-generalization question comparable to it.

## Reusing the original experiment's infrastructure

Nothing about the *task* changes, only the architecture -- so this experiment reuses, unmodified:

- `data_modules/arc1d_compositional.py` (the composite-task generator) and
  `data/arc_1d_compositional_holdout` (the 400-instance, 10-category held-out set built from it).
- `scripts/eval_compositional_holdout.py` (zero-shot eval + embedding-cluster holdout overlay).

See `arc1d_hypermodel_compositional_generalization/README.md` for the generator's architecture,
the base-rule semantics table, and the full combo-selection rationale (which 10 pairings were
kept and why) -- none of that is repeated here.

## Architecture

Copied from `arc1d_v2_hypernetwork_multitask/shrink_matched_dim4.yaml`, the sizing sweep's
smallest-that-still-holds config:

| Component | Value |
|---|---|
| Target model | `rope_canon_looped_transformer`, `hidden_dim=4`, flat (`n_loops=1`, not looped) |
| Hypernetwork encoder | `rope_canon_transformer` (plain, non-Zhu), `hidden_dim=4, num_layers=1, output_dim=4` |
| `task_encoding.embedding_dim` | 4 |
| `hyper_head.bottleneck_dim` | 8 |
| Generation | Dense (`lora_adapter: false`) -- the sizing sweep found LoRA strictly worse below the dense floor at this scale |
| Optimizer | Muon (`muon_lr=0.005, muon_momentum=0.95`) + AdamW aux (`lr=0.001, wd=0.01`) |
| Budget | `N_supervision=1, max_steps=8000, warmup_steps=800` (8000-total-optimizer-update budget) |

**Deviation from its v2 source**: `save_checkpoints: true` (the sizing-sweep configs this
architecture is copied from ran with `save_checkpoints: false`, since they never evaluated on a
held-out set -- this experiment needs the checkpoint for `scripts/eval_compositional_holdout.py`
afterward).

## Task set

Kept at the original compositional experiment's 15 categories (not v2's 14 -- v2 dropped
`1d_recolor_cmp` for an unrelated capacity-isolation reason), so this stays an apples-to-apples
task set against the already-run bigger-recipe version.

## Arms

Only `notd` and `frozen_td` -- no plain `td` this time, matching
`arc1d_v2_hypernetwork_multitask`'s own choice to drop it ("historically bimodal/unstable"), and
a learned one-hot embedding is undefined on the held-out compositional indices anyway (same reason
the original experiment excluded it from held-out eval, just taken one step further here by not
building it at all):

| Variant | `hyper_head.num_tasks` | `hyper_head.freeze_task_indicator` |
|---|---:|---:|
| `notd` | `null` | `false` |
| `frozen_td` | `28` (18 base + 10 reserved composite indices) | `true` |

## Running

```bash
bash scripts/run_arc1d_v2_compositional_generalization.sh
```

Trains both arms, then runs `scripts/eval_compositional_holdout.py` for each against the held-out
compositional set. Set `FREE_GPUS_FLAG="--free-gpus"` if the node is shared.

## Reading results

Same as the original experiment: `outputs/arc1d_v2_compositional_generalization/<arm>/results.txt`
for in-distribution val/test metrics (including the per-task-category exact-match *and* -- as of
the leave-one-out generalization experiment's metric addition -- accuracy breakdown), and
`outputs/arc1d_v2_compositional_generalization/<arm>/compositional_holdout_eval/results.txt` for
the zero-shot per-composite-category `exact_match`/`seq_accuracy` table. Compare directly against
`arc1d_hypermodel_compositional_generalization`'s own holdout results (same categories, same
holdout data, different architecture only) to see whether the matched-scale recipe changes the
exact-match-vs-token-accuracy pattern found there (near-zero exact match, 46-92% token accuracy
per category) or whether that's scale-independent.

## Status

Run and evaluated at seed 42 and, since, at 3 seeds (1, 2, 3). See Findings below -- the 3-seed
results supersede the seed-42-only pass.

## Findings (seed 42)

In-distribution (val/test, all 15 training categories) -- at this much smaller (10,156-param)
architecture, `frozen_td` is well below the original bigger-recipe experiment's near-ceiling
result (84.0% val / 82.7% test exact match here vs. 98.75% / 97.5% there), and `notd` is roughly
comparable (57.3% val / 49.3% test here vs. 48.8% / 46.3% there):

| | `notd` val EM | `notd` test EM | `frozen_td` val EM | `frozen_td` test EM |
|---|---:|---:|---:|---:|
| in-distribution | 57.3% | 49.3% | 84.0% | 82.7% |

So the matched-scale recipe trades away a meaningful chunk of in-distribution accuracy relative
to the original bigger recipe -- expected, since this architecture is ~150x smaller by parameter
count, and `arc1d_v2_hypernetwork_multitask`'s own sizing sweep was run on 14 (not 15) categories
without this experiment's extra `1d_recolor_cmp` task.

Zero-shot on the 10 held-out composite categories (`scripts/eval_compositional_holdout.py`,
`n=40` per category):

| category | `notd` exact_match | `notd` seq_accuracy | `frozen_td` exact_match | `frozen_td` seq_accuracy |
|---|---:|---:|---:|---:|
| `denoise1c_shift3` | 0.000 | 0.750 | 0.000 | 0.806 |
| `denoisemc_copy` | 0.050 | 0.688 | 0.000 | 0.708 |
| `denoisemc_denoise1c` | 0.000 | 0.675 | 0.000 | 0.646 |
| `denoisemc_mirror` | 0.250 | 0.885 | 0.000 | 0.697 |
| `fill_mirror` | 0.000 | 0.523 | 0.000 | 0.719 |
| `fill_movedynamic` | 0.000 | 0.620 | 0.000 | 0.611 |
| `fill_shift3` | 0.000 | 0.742 | 0.000 | 0.147 |
| `hollow_shift3` | 0.000 | 0.853 | 0.000 | 0.816 |
| `movedynamic_hollow` | 0.000 | 0.786 | 0.000 | 0.724 |
| `shift3_copy` | 0.000 | 0.620 | 0.000 | 0.626 |
| **overall** | **0.030** | – | **0.000** | – |

**The exact-match-vs-token-accuracy pattern from the original (bigger-recipe) experiment holds
at this much smaller scale too -- it isn't an artifact of that architecture.** Exact match is
essentially 0 for both arms on every composite category (one 5% and one 25% partial exception
for `notd`), while token accuracy sits at 52-89% per category (one outlier: `frozen_td` on
`fill_shift3` collapsed to 14.7%, worth a look at the qualitative renderings for that category
specifically).

**`notd` beats `frozen_td` on token accuracy in both experiments, despite `frozen_td` dominating
it in-distribution** -- and the direction is the *opposite* of the leave-one-out generalization
experiment's finding, not just a repeat of it:

| | mean `notd` seq_accuracy | mean `frozen_td` seq_accuracy | diff | categories where `notd` wins |
|---|---:|---:|---:|---:|
| original (big recipe) | 0.782 | 0.707 | +0.076 | 9/10 |
| this experiment (v2) | 0.714 | 0.650 | +0.064 | 6/10 |
| combined (20 category-runs) | | | | 15/20 |

A crude sign test on the pooled 20 category-level comparisons gives p ~= 0.04 -- treat that as
illustrative, not rigorous: each experiment is a single seed, so its 10 categories aren't
independent draws (they share one trained checkpoint), and the two experiments differ in
architecture, not just a repeated trial. The v2 gap is also partly carried by one outlier
(`frozen_td`'s `fill_shift3` collapse alone accounts for over half of it -- excluding that
category, v2's gap nearly vanishes: 0.711 vs. 0.706). The original experiment's gap isn't
outlier-driven the same way (9/10 categories, no single category dominating it), so it's the
sturdier half of this finding.

Why this might make sense: leave-one-out generalization found freezing the task-identity
projection *helps* (recovers or beats `notd`) when the true answer is a category the model
already knows individually, just never in this exact training run. Compositional generalization
is a different regime -- the true answer *isn't* any single known category. A task-identity
signal, even an untrained/frozen one, may give the model something to anchor to ("this looks
most like known task X"), which helps when the anchor is right and actively works against
blending two rules together when it isn't. `notd` has no such anchor, so there's less pulling it
away from combining what the support examples actually show.

**Caveats (seed-42 pass)**: single seed only, same as the original experiment -- no variance
estimate on either the in-distribution or holdout numbers, and see the sign-test caveat above
before treating the notd-vs-frozen_td gap as more than a suggestive pattern. **Superseded by the
3-seed rerun below**, which shows this single-seed pass actually *understated* the effect.

## Findings (3 seeds: 1, 2, 3)

Per-composition exact match and token accuracy, mean across the 3 seeds (`notd`/`frozen_td`,
`n=40` per seed per composition, 120 total per cell):

| Composition | `notd` EM | `notd` Acc | `frozen_td` EM | `frozen_td` Acc | Acc diff |
|---|---:|---:|---:|---:|---:|
| `denoise1c_shift3` | 0.000 | 0.721 | 0.000 | 0.582 | +0.139 |
| `denoisemc_copy` | 0.050 | 0.715 | 0.000 | 0.366 | **+0.349** |
| `denoisemc_denoise1c` | 0.075 | 0.772 | 0.000 | 0.598 | +0.174 |
| `denoisemc_mirror` | 0.133 | 0.833 | 0.000 | 0.325 | **+0.508** |
| `fill_mirror` | 0.000 | 0.730 | 0.000 | 0.549 | +0.181 |
| `fill_movedynamic` | 0.000 | 0.680 | 0.000 | 0.479 | +0.201 |
| `fill_shift3` | 0.000 | 0.753 | 0.000 | 0.528 | +0.225 |
| `hollow_shift3` | 0.000 | 0.855 | 0.000 | 0.751 | +0.104 |
| `movedynamic_hollow` | 0.000 | 0.815 | 0.000 | 0.700 | +0.115 |
| `shift3_copy` | 0.000 | 0.634 | 0.000 | 0.602 | +0.032 |
| **mean** | **0.026** | **0.751** | **0.000** | **0.548** | **+0.203** |

**With 3 seeds, `notd` beats `frozen_td` on token accuracy on all 10/10 compositions -- not the
noisy 6/10 the single seed suggested -- and the gap is 3x bigger (+0.203 vs. +0.064).** The
seed-42 pass wasn't wrong about direction, but it substantially undersold the effect: `frozen_td`
turns out to be markedly less stable across seeds than the single run implied, with several
compositions showing the same kind of wide seed-to-seed swings seen in `arc1d_v2_generalization`
(e.g. `denoisemc_mirror`: `frozen_td` accuracy is 0.440, 0.513, then collapses to 0.023 on seed 3
-- `notd` stays tight at 0.846/0.823/0.831 on the same composition). `notd` also picks up real,
repeatable partial exact-match credit on 3 categories now (`denoisemc_mirror`: 13.3% mean, hit in
all 3 seeds; `denoisemc_copy` and `denoisemc_denoise1c`: smaller but nonzero and multi-seed too)
-- not a one-off. `frozen_td` stays at a clean 0.0 exact match everywhere, every seed, every
composition.

This closes the loop on the question that motivated the rerun: is the compositional-generalization
`notd` advantage as robust as the category-generalization one (`arc1d_v2_generalization`, `notd`
wins 5/5 categories, +0.122)? **Yes -- more so, in fact.** Once both experiments are run at 3
seeds, `notd`'s token-accuracy advantage over `frozen_td` on out-of-training-configuration
examples is a clean sweep in both regimes, and the compositional gap (+0.203) is now the *larger*
of the two.

**Caveats (3-seed pass)**: 3 seeds is still fewer than the 5 used in some published compositional-
generalization tables; the qualitative pattern (clean sweep, large frozen_td variance) looks
settled, but exact effect sizes could still move somewhat with more seeds.
