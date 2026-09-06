# arc1d_hypermodel_looped_rope_canon_lr_sweep

## Goal

`arc1d_hypermodel_looped_rope_canon_optimizer` found AdamW/lr=0.0005 gave a real, clean
improvement over the RAdam/lr=0.001 baseline (mean exact-match 0.792 vs 0.713, tighter
variance, and every AdamW seed's val_loss beating every baseline seed's), though confounded
with the LR change at the time. Per Fabio's direction, AdamW is being kept as the optimizer
going forward (not re-litigated here). This experiment asks how low the LR should go, crossed
with whether the full combined Zhu block (`arm_zhu_all` from
`arc1d_hypermodel_looped_rope_canon_zhu_block`: RMSNorm + QK-norm + SwiGLU, mean 0.808 across
only 3 seeds, the best mean of anything tried so far, but with one seed below baseline and no
clean diagnostic explanation why) holds up or improves with more seeds and a proper LR sweep.

## Grid

Two architectures x four learning rates, AdamW throughout, 5 seeds each:

| Arm | architecture | learning_rate | seeds |
|---|---|---:|---|
| `arm_baseline_lr1e-3` | baseline (`rope_canon_transformer`) | 0.001 | 5 new |
| (topped up, no leaf here) | baseline | 0.0005 | reused from `arc1d_hypermodel_looped_rope_canon_optimizer/arm_adamw.yaml`: 3 existing + 2 topped up |
| `arm_baseline_lr1e-4` | baseline | 0.0001 | 5 new |
| `arm_baseline_lr5e-5` | baseline | 0.00005 | 5 new |
| `arm_zhuall_lr1e-3` | zhu_all (`rope_canon_zhu_transformer`, all 3 flags on) | 0.001 | 5 new |
| `arm_zhuall_lr5e-4` | zhu_all | 0.0005 | 5 new |
| `arm_zhuall_lr1e-4` | zhu_all | 0.0001 | 5 new |
| `arm_zhuall_lr5e-5` | zhu_all | 0.00005 | 5 new |

7 new leaf configs x 5 seeds = 35 jobs, plus 2 top-up jobs (seeds 4-5) run directly against
the existing `arc1d_hypermodel_looped_rope_canon_optimizer/arm_adamw.yaml` config (no new
leaf duplicated here for that cell). 37 new jobs total; all 8 cells end up at 5 seeds once
combined with the existing data.

Fixed across every cell: no task descriptor, `layers=8`, `max_steps=8000`/`warmup_steps=800`,
`gradient_clip_val=10`, `lora_adapter_rank=8`, attention pooling, 15-task list, target model
unchanged, matching every experiment since `capacity_baseline`.

## Running

```bash
# Full sweep: 7 new leaf configs, 5 seeds each, plus the 2 top-up seeds for the reused
# baseline_lr5e-4 cell. 37 jobs total.
bash scripts/run_hypermodel_looped_rope_canon_lr_sweep.sh
```

No overfit smoke configs this round, matching every experiment since the followup; running
directly on the cluster.

## Reading results

For each architecture, tabulate `val_query_exact_match` (mean ± std across 5 seeds) against
the 4 learning rates, looking for where performance peaks and whether it degrades going lower
(too small a step size to make progress in 8000 steps) or higher (closer to the original
RAdam-favoring instability regime). Compare baseline vs `arm_zhu_all` at each LR to see
whether the Zhu block's advantage (if real) holds across the LR range or is concentrated at
one setting. With 5 seeds now covering all 8 cells (up from 3 for the two reused/pre-existing
cells), also revisit whether `arm_zhu_all`'s previous below-baseline seed
(`looped_hyper_rope_canon_zhu_block_all_seed1`, 0.637) was noise or a real recurring failure
mode.
