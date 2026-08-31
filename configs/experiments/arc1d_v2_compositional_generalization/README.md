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

Not yet run. Configs and launcher script verified structurally (config loading, a short smoke
run confirming forward/backward + checkpoint save + the held-out eval path all work end to end) --
no full-budget training has happened yet.
