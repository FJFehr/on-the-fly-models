# arc1d_hypermodel_looped_rope_canon_muon

## Goal

The leave-one-task-out generalization experiment
(`arc1d_hypermodel_looped_rope_canon_generalization`) found the model doesn't generalize to a
truly unseen task category, and separately established `frozen_td` (a task-identity embedding
frozen at random init for every category) as an interesting middle ground between `td` (learned,
actively harmful for the one held-out category it helped least) and `notd` (no signal at all).

This experiment goes back to standard in-distribution training (task_categories ==
val_task_categories, full 15-task list, no held-out axis) and asks a different question: does
the Muon optimizer (Keller Jordan, https://github.com/KellerJordan/Muon) have any value here,
crossed with `notd` vs `frozen_td` (plain `td` dropped -- not needed without a held-out
category to expose its weakness).

Fabio's specific hypothesis: Muon may stabilise the `notd` runs, since `notd` combined with the
LoRA adapter parameterisation (`hyper_head.lora_adapter: true`, rank 8, active throughout this
backbone) may be a harder optimization landscape than AdamW's per-parameter adaptive scaling
handles well. Muon's orthogonalized-momentum update is a genuinely different update rule, so
this is a real "is this optimizer worth anything for this model" test, not just another LR
point.

## Backbone (fixed across every arm)

Zhu block (RMSNorm + QK-norm + SwiGLU, `rope_canon_zhu_transformer`, all three flags on),
lr=0.001 -- the current leader in `arc1d_hypermodel_looped_rope_canon_lr_sweep`'s zhu_all arm
(mean exact-match 0.8625 across 5 seeds). Not re-swept here.

## The Muon/AdamW parameter split

Muon only makes sense for genuine 2D "hidden weight matrix" parameters -- Keller Jordan's own
guidance is that embeddings, the final output layer, and any 1D gains/biases should stay on
standard AdamW. `models/hypermodel_lightning.py`'s `_build_muon_param_groups` implements this
as an allowlist: only `nn.Linear` weights are ever Muon-eligible (this alone excludes every
`nn.Embedding`, `RMSNorm`/`LayerNorm`, and Canon `nn.Conv1d` weight, and the attention pooler's
raw `pool_query` vector, none of which are `nn.Linear`), with three further explicit
exclusions kept on AdamW even though they are `nn.Linear`:

- `input_projection` -- the hypernetwork's input embedding layer.
- `output_head` -- the hypernetwork's final output layer.
- `task_indicator_proj` -- structurally `nn.Linear(num_tasks, hyper_output_dim, bias=False)`,
  but semantically a one-hot task-identity embedding table, not a matrix operator.

Everything else -- the internal attention/MLP Linear weights inside each transformer block,
**and** the hyper_head/LoRA projection Linears (`lora_proj_a`/`lora_proj_b`/`lora_proj_other`)
that generate the actual target-model weights -- is Muon-eligible. Including the LoRA
projection layers was a deliberate call (not the literal-strictest reading of "final output
layer"): it's most of where this model's trainable capacity actually sits, and it's exactly
the parameterisation Fabio's stabilisation hypothesis is about. Verified directly in
`tests/test_muon_param_groups.py`.

Muon and AdamW-aux hyperparameters: `muon_lr=0.02`, `muon_momentum=0.95` -- Keller Jordan's
published defaults, **not tuned for this repo**. A Muon LR sweep is a natural follow-up if
this first pass shows any signal. The AdamW-aux group inside Muon runs uses this experiment's
own `learning_rate`/`weight_decay` (0.001/0.01), matching the plain-AdamW arms, so the only
intended difference between a `_muon` and `_adamw` arm is the update rule applied to the
Muon-eligible matrix weights.

## Grid

2 task-identity variants x 2 optimizers, 3 seeds each = 12 jobs:

| Arm | `hyper_head.num_tasks` | `freeze_task_indicator` | `optimizer` |
|---|---:|---:|---|
| `arm_notd_adamw` | `null` | `false` | `AdamW` |
| `arm_notd_muon` | `null` | `false` | `Muon` |
| `arm_frozentd_adamw` | `18` | `true` | `AdamW` |
| `arm_frozentd_muon` | `18` | `true` | `Muon` |

## Running

```bash
# All 4 leaf configs, 3 seeds each. 12 jobs.
bash scripts/run_hypermodel_looped_rope_canon_muon.sh
```

Split across nodes via `SEEDS_OVERRIDE` and/or `CELL_GLOB` (see script header for examples).

## Reading results

1. For each task-identity variant, compare `AdamW` vs `Muon` on `val_query_exact_match` (mean
   +/- std across 3 seeds) and training-loss curves -- does Muon reach the same or better
   exact-match, does it get there faster, is variance across seeds tighter or wider. That's
   the direct answer to "is there value in this optimizer" for this model.
2. Fabio's stabilisation hypothesis, specifically: compare `arm_notd_adamw` vs
   `arm_notd_muon`'s training-loss/grad-norm trajectories, not just final exact match -- does
   `notd_muon` show visibly smoother/lower-variance loss curves or fewer spikes than
   `notd_adamw` across the 3 seeds? If Muon helps `notd` specifically more than it helps
   `frozentd` (where the frozen embedding is already providing some stability), that
   differential is the direct evidence for the hypothesis, not just an overall
   "Muon is better/worse" reading.
