# arc1d_hypermodel_looped_rope_canon_mechanism_ablation

## Goal

`arc1d_hypermodel_looped_rope_canon_rank_sweep` confirmed RoPE+Canon and rank=8 are both
needed, but rank=8's result was lower than the earlier followup experiment's result. Two
things changed between those two experiments at once (`gradient_clip_val` 10 to 5, and
pooling hierarchical to attention), so which factor (or both) caused the regression is
currently confounded. This experiment re-isolates each factor with a single-axis flip from
the rank_sweep's `rank8.yaml` settings (not a full 2x2 factorial, per Fabio's direction), and
adds two new mechanism questions on top: does the LoRA adapter's random backbone matter
versus a zero backbone, and can the model still learn multiple tasks without the explicit
one-hot task-identity signal.

## Arms

The baseline (rank=8, RoPE+Canon `num_layers=4`, attention pooling, `gradient_clip_val=5`,
`N_supervision=2`) is exactly `arc1d_hypermodel_looped_rope_canon_rank_sweep/rank8.yaml`'s
settings, which **already has 5 completed seeds of data** and is not rerun here, only
referenced for comparison. Four new arms, each flipping exactly one axis, 3 seeds each:

| Arm | Change from baseline | Question |
|---|---|---|
| `arm_clip10` | `gradient_clip_val` 5 to 10 | did lowering clip cause the rank=8 regression? |
| `arm_hierarchical` | `hyper_head.pooling` attention to hierarchical | does pooling matter at rank=8 (the original test was at rank=1)? |
| `arm_zero_backbone` | `hyper_head.lora_adapter_zero_backbone: true` | does the random backbone earn its keep over a from-scratch low-rank generation? |
| `arm_no_task_indicator` | `hyper_head.num_tasks: null` | can the model learn multitask from support examples alone, without the oracle task label? |

4 arms x 3 seeds = 12 new jobs.

## New flag: `hyper_head.lora_adapter_zero_backbone`

`models/hypermodel.py`'s LoRA adapter backbone was previously always the target model's real
random init, read live via `get_parameter`. `lora_adapter_zero_backbone: true` makes it zero
instead, so the generated `B @ A` delta alone determines every weight.

**Found during implementation, not anticipated going in**: `lora_proj_b` is normally
zero-initialized (standard LoRA convention) so the delta starts at exactly zero *on top of a
nonzero backbone*. With the backbone also zero, that assumption breaks: `backbone=0` and
`delta=0` at step 0 means the entire target model starts as an all-zero network, the exact
collapse `low_rank_output` hit before its own variance-matching fix. Confirmed empirically:
gradient reached the generated weights but its norm was exactly `0.0`, never reaching the
hypernetwork, a dead trap, not a bug in the flag's wiring. Fixed per Fabio's direction: `B`
keeps its ordinary random init specifically when `lora_adapter_zero_backbone=True` (the
normal case is completely unaffected), so this arm can actually train. Verified: (a) `B` stays
zero-init when the flag is off (regression check), (b) `B` is *not* zero-init when the flag is
on, (c) the flag-based zero backbone produces byte-identical output to a manually-zeroed
target model with the same generated delta, (d) gradients reach the hypernetwork in both
cases.

## New arm: `arm_no_task_indicator`

No code change needed. `hyper_head.num_tasks` already gates `HyperModel.task_indicator_proj`
(a learned one-hot task-id embedding, additively injected into the pooled representation) via
existing `is not None` guards in both `models/hypermodel.py` and
`models/hypermodel_lightning.py`; setting it to `null` disables it cleanly. The 3-shot support
examples remain the model's only remaining task signal, built independently of `task_ids` in
`prepare_inputs`, so this isn't a zero-signal ablation, just removal of the oracle one-hot
label shortcut.

## Running

```bash
# 4 new arms, 3 seeds each, 12 jobs total.
bash scripts/run_hypermodel_looped_rope_canon_mechanism_ablation.sh
```

No overfit smoke configs this round, matching the last two experiments; running directly on
the cluster. The baseline's existing 5-seed results are at
`outputs/arc1d_hypermodel_looped_rope_canon_rank_sweep/looped_hyper_rope_canon_rank_sweep_r8_seed{1..5}/`.

## Reading results

Compare each new arm's `val_query_exact_match`/`test_query_exact_match` (mean ± std across 3
seeds) against the reused rank8 baseline's existing 5-seed results, per task and averaged.

- `arm_clip10` and `arm_hierarchical` together should reveal which factor (or both, or
  neither) caused the rank=8 regression relative to the followup experiment.
- `arm_zero_backbone`: does a properly-trainable (not dead-gradient) from-scratch low-rank
  generation match, beat, or lag the frozen-random-backbone default?
- `arm_no_task_indicator`: does removing the oracle one-hot task label hurt multitask
  learning, or do the 3-shot support examples already carry enough signal on their own?
