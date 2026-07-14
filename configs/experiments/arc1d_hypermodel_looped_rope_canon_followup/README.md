# arc1d_hypermodel_looped_rope_canon_followup

## Goal

`arc1d_hypermodel_looped_hypernet_rope_canon_ablation` varied the hypernetwork encoder's
architecture (current/RoPE/Canon/RoPE+Canon) and came back a bit inconclusive. This follow-up
fixes the encoder to RoPE+Canon (the combined arm) as the default and opens three new
single-axis questions on top of it: does hierarchical pooling beat plain attention pooling,
does raising LoRA rank from 1 to 8 unlock any task, and does N_supervision=4 beat
N_supervision=2. `batch_size` is dropped back to 512 (1024 was too large), `max_steps` is back
up to 4000, 3 seeds (down from the previous experiment's 5).

Target model, task list (17 categories), `hyper_head.{bottleneck_dim, num_tasks}`, and the
hypernetwork encoder's base shape all match
`arc1d_hypermodel_looped_hypernet_rope_canon_ablation` exactly, so results are directly
comparable to that project's `hypernet_rope_canon` arm (same encoder, same target, same task
list, same `max_steps`; the only difference from that arm is `batch_size` 512 vs. 1024 there
and 3 vs. 5 seeds).

## Structure: one shared default, three single-axis comparisons

Not a full factorial. One baseline arm (`arm_default`) is reused as the "before" side of all
three pairwise comparisons, rather than rerunning it three times:

| Arm | pooling | `lora_adapter_rank` | `N_supervision` | `hyper_model.num_layers` | Question |
|---|---|:---:|:---:|:---:|---|
| `arm_default` | hierarchical | 1 | 2 | 2 | baseline |
| `arm_pooling_attention` | attention | 1 | 2 | 4 | does hierarchical pooling help? |
| `arm_rank8` | hierarchical | 8 | 2 | 2 | does rank 8 unlock tasks rank 1 fails? |
| `arm_nsup4` | hierarchical | 1 | 4 | 2 | does more supervision granularity help? |

4 configs x 3 seeds = 12 jobs total.

## Pooling parameter-matching

`AttentionPooler` (`models/hypermodel.py`) has no depth/width knob, it's a fixed
`hidden_dim`-sized module (64 params at `hidden_dim=64`). `HierarchicalPooler` at this
experiment's implicit defaults (`segment_interaction_layers=1, example_interaction_layers=1,
interaction_num_heads=1`) has 98,752 params, a fixed overhead `attention` pooling can't match
on its own side. Instead, `arm_pooling_attention` bumps `hyper_model.params.num_layers` 2 → 4:
`RoPECanonTransformer(hidden_dim=64, num_heads=4, canon_set=ABCD)` adds 52,160 params per
layer (`num_layers=2` → 109,568; `num_layers=4` → 213,888, measured directly, not estimated).
`arm_default` totals 109,568 + 98,752 = 208,320 (encoder + pooler alone); `arm_pooling_attention`
totals 213,888 + 64 = 213,952, the closest an integer `num_layers` gets. At the full-model
level (verified by instantiating both configs end to end via `HyperModelLightning`), the gap
is smaller still since the shared target model and LoRA heads dilute it: `arm_default` totals
628,448 params, `arm_pooling_attention` totals 634,080, a 5,632-param, 0.9% difference. Not an
exact match, disclosed rather than engineered around further.

## N_supervision compute-matching

`N_supervision` inner iterations are full forward/backward/`opt.step()` passes on the *same*
batch (not resampled), confirmed by reading `HyperModelLightning.training_step`
(`models/hypermodel_lightning.py`). It's fully orthogonal to `target_model.params.n_loops`
(the target model's own recursive-application depth within one forward pass, fixed at 4 here).
Per this repo's compute-matching convention (`max_steps * N_supervision` held to a fixed
total-optimizer-updates budget, `warmup_steps` = 10% of `max_steps`), `arm_nsup4` uses
`max_steps=2000, warmup_steps=200` against `arm_default`'s `max_steps=4000, warmup_steps=400`
(both 8000 total optimizer updates), so the comparison isolates supervision granularity rather
than also confounding it with more total training.

## Carried-over caveat: `gradient_clip_val=10.0`

Carried forward from `arc1d_hypermodel_looped_hypernet_rope_canon_ablation/base.yaml`, added
there after loss-spike diagnostics traced periodic training-loss spikes to unclipped
hyper-projection/LoRA-adapter gradients at `batch_size=1024`. Not re-verified at this
experiment's `batch_size=512`; gradient statistics may differ at the smaller batch size
(typically noisier per-step gradients), so this value may be over- or under-conservative here.

## Running

```bash
# Full sweep (4 configs, 3 seeds each, 12 jobs total: 17 tasks, max_steps=4000 or 2000
# depending on arm, batch_size=512).
bash scripts/run_hypermodel_looped_rope_canon_followup.sh
```

No overfit smoke configs this round; running directly on the cluster.

## Reading results

Compare `val_query_exact_match`/`test_query_exact_match` (mean ± std across 3 seeds) for each
of the three pairwise comparisons against `arm_default`, per task and averaged:

- `arm_pooling_attention` vs. `arm_default`: does hierarchical pooling earn its ~2.7%-larger
  parameter budget and structural complexity over plain attention pooling?
- `arm_rank8` vs. `arm_default`: look at *per-task* exact-match rates specifically, not just
  the average, to see whether rank 8 flips any task from failing to passing under rank 1
  (rather than just moving the mean on tasks that already partially work).
- `arm_nsup4` vs. `arm_default`: does finer supervision granularity help at matched total
  compute?

Also directly comparable to `arc1d_hypermodel_looped_hypernet_rope_canon_ablation`'s
`hypernet_rope_canon` arm (`arm_default` here is the same encoder/target/task-list at
`batch_size=512` instead of `1024`, 3 seeds instead of 5).
