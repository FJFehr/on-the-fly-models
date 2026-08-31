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

Run and evaluated (seed 42, both arms). See Findings below.

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
specifically). As in the original experiment, `frozen_td` does not clearly beat `notd` here
despite dominating it in-distribution -- if anything `notd` looks marginally better on average,
consistent with the original run's finding that the task-identity signal's in-distribution
advantage doesn't carry over to zero-shot composition.

**Caveats**: single seed (42) only, same as the original experiment -- no variance estimate on
either the in-distribution or holdout numbers.
