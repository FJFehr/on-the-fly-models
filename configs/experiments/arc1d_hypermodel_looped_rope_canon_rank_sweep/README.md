# arc1d_hypermodel_looped_rope_canon_rank_sweep

## Goal

Follows on from `arc1d_hypermodel_looped_rope_canon_followup`, whose results were:

- **Rank 8 vs. rank 1**: rank 8 unlocked every task except `1d_recolor_oe` and
  `1d_recolor_cnt`, clearly better than the rank 1 default.
- **N_supervision 4 vs. 2**: no measurable difference.
- **Hierarchical vs. attention pooling**: no measurable difference.

This experiment acts on those results. `N_supervision` is fixed at 2 (no benefit from 4) and
hierarchical pooling is dropped entirely (attention pooling only, one less piece of complexity
for no measured benefit). It then asks two new questions: where between rank 1 and rank 8 does
the task-unlocking actually happen (a rank 1/2/4/8 sweep), and does removing Canon from the
hypernetwork encoder make it worse (tested at rank 8 only, the winning rank, rather than at
every rank, to keep this experiment's size reasonable).

`gradient_clip_val` is lowered from 10.0 to 5.0, per Fabio's direction (the 10.0 value was
itself carried over from `batch_size=1024` loss-spike diagnostics and never independently
re-verified, so this is a fresh, unvalidated setting too, watch the loss curves for spikes).

## Arms

| Arm | `lora_adapter_rank` | `canon_set` | Question |
|---|:---:|:---:|---|
| `rank1` | 1 | ABCD | rank sweep |
| `rank2` | 2 | ABCD | rank sweep |
| `rank4` | 4 | ABCD | rank sweep |
| `rank8` | 8 | ABCD | rank sweep (this experiment's reference point for the canon comparison) |
| `rank8_no_canon` | 8 | `""` | does dropping Canon hurt at the winning rank? |

5 configs x 5 seeds = 25 jobs total.

Fixed across every arm: hypernetwork encoder is `rope_canon_transformer`,
`hidden_dim=64, num_heads=4, num_layers=4, output_dim=64` (the `num_layers=4` value comes
from the previous experiment's attention-pooling arm, kept as-is rather than reverted to 2,
since it was never shown to hurt), attention pooling, target model and 17-task list identical
to the previous two experiments, `batch_size=512, max_steps=4000, warmup_steps=400,
N_supervision=2, gradient_clip_val=5.0`.

`rank8_no_canon` drops `canon_set` from `"ABCD"` to `""`, which is not parameter-matched back
up (a straightforward architecture-presence question, not asked to be size-controlled):
measured directly, the encoder goes from 213,888 to 202,368 params, 11,520 fewer.

## Running

```bash
# Full sweep (5 configs, 5 seeds each, 25 jobs total).
bash scripts/run_hypermodel_looped_rope_canon_rank_sweep.sh
```

No overfit smoke configs this round, running directly on the cluster.

## Reading results

For the rank sweep (`rank1`/`rank2`/`rank4`/`rank8`), look at *per-task* exact-match rates
(mean ± std across 5 seeds), not just the average, to find where each task actually unlocks
between rank 1 and rank 8, and in particular whether `1d_recolor_oe`/`1d_recolor_cnt` ever
unlock at any intermediate rank or stay stuck through rank 8.

For `rank8_no_canon` vs. `rank8`: does Canon's presence in the encoder actually matter, or is
RoPE alone (plus the extra depth already present from the previous experiment) doing the
work? Keep the ~5% parameter difference (202,368 vs. 213,888 in the encoder alone) in mind
when interpreting a result either way.
