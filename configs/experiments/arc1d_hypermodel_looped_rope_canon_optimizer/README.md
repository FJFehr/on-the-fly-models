# arc1d_hypermodel_looped_rope_canon_optimizer

## Goal

`arc1d_hypermodel_looped_rope_canon_capacity_baseline` found `layers=8`, no task descriptor,
8000-step training to be the strongest no-descriptor setting so far (0.738 mean exact-match).
This experiment tests whether a lower learning rate with AdamW closes more of the
descriptor-on/off gap than the settings already tried.

## Arm

| Arm | optimizer | learning_rate |
|---|---|---:|
| (reused baseline, not rerun) | RAdam | 0.001 |
| `arm_adamw` | AdamW | 0.0005 |

3 seeds. Everything else identical to the reused baseline
(`arc1d_hypermodel_looped_rope_canon_capacity_baseline/arm_layers8_notd_long.yaml`):
`layers=8`, no task descriptor, `max_steps=8000`/`warmup_steps=800`, `gradient_clip_val=10`,
`lora_adapter_rank=8`, attention pooling, RoPE+Canon encoder, 15-task list. Kept as a separate
experiment from `arc1d_hypermodel_looped_rope_canon_zhu_block` rather than crossed into one
factorial, so each stays a clean single-axis comparison against the known baseline.

## Running

```bash
bash scripts/run_hypermodel_looped_rope_canon_optimizer.sh
```

No overfit smoke configs, matching every experiment since the followup; running directly on
the cluster.

## Reading results

Compare `val_query_exact_match` (mean ± std across 3 seeds) against the reused baseline's
0.738 mean, per task and averaged. Particular attention to `1d_move_dp`.
