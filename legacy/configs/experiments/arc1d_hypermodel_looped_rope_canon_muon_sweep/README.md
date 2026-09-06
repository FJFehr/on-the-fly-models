# arc1d_hypermodel_looped_rope_canon_muon_sweep

## Goal

`arc1d_hypermodel_looped_rope_canon_muon` established that Muon is worth investigating for this
model, but left `muon_lr=0.005` untuned beyond the one adjustment needed to stop a NaN
divergence (lowered 4x from Keller Jordan's published 0.02), and ran the AdamW aux group / weight
decay / batch size at whatever the AdamW arms happened to use.

Fabio's goal for this sweep: **speed up training, and get the notd (no task-identity signal)
runs to solve more task categories.** Two questions, tested together:

1. Is there a better `muon_lr` / aux `learning_rate` / `weight_decay` / `batch_size` combination
   than the untuned placeholder?
2. Does excluding the LoRA head's A/B factor projections (`lora_proj_a`/`lora_proj_b` in
   `models/hypermodel.py`, which generate the target model's low-rank adapter factors) from
   Muon help? Fabio's hypothesis: Muon's Newton-Schulz orthogonalization may be a poor fit for
   these particular projections specifically, as opposed to the block-internal attention/MLP
   weights Muon was designed for. `lora_proj_other` (generates the non-matrix target params,
   e.g. norms/Canon kernels) is never excluded by this flag — it has no low-rank structure to
   speak of, so it isn't "the low rank vectors" the hypothesis is about.

Only `notd` is covered here (no `frozen_td` arm) — the stabilisation hypothesis from the
previous experiment concerns `frozen_td` vs `notd` directly, which isn't what this sweep is
testing.

## The `muon_exclude_lora_heads` toggle

`models/hypermodel_lightning.py`'s `_build_muon_param_groups` now accepts a
`muon_exclude_lora_heads` flag (default `False`, preserving every existing config's behavior).
When `True`, `lora_proj_a`/`lora_proj_b` move from the Muon group to the AdamW aux group.
Implementing this required fixing a latent bug in the exclusion-matching logic: it previously
matched only the *last* dotted path component (`name.split(".")[-1]`), which works for direct
attributes like `output_head` but silently fails for `lora_proj_a`/`lora_proj_b` since they are
`nn.ModuleList`s — a member's `named_modules()` path ends in its list index (e.g.
`lora_proj_a.3`), not the attribute name. The check now matches against every path segment.
Covered by `tests/test_muon_param_groups.py::test_muon_param_split_excludes_lora_heads_when_flag_set`
and the accompanying default-unchanged sanity test.

## Grid

`4 (muon_lr) x 3 (learning_rate) x 3 (weight_decay) x 3 (batch_size) x 2
(muon_exclude_lora_heads) = 216 configs`, 1 seed each = 216 jobs:

| Axis | Values |
|---|---|
| `muon_lr` | `0.008, 0.01, 0.016, 0.02` |
| `learning_rate` (AdamW aux group) | `3e-4, 6e-4, 3e-3` |
| `weight_decay` (shared by both groups) | `0, 0.01, 0.1` |
| `batch_size` | `512` (current baseline), `1024`, `2048` (4x) |
| `muon_exclude_lora_heads` | `False` (current default), `True` |

Fixed across every cell: Zhu backbone, `rope_canon_looped_transformer` target, all 15
in-distribution task categories, `notd` (`hyper_head.num_tasks: null`,
`freeze_task_indicator: false`), `muon_momentum: 0.95`, `max_steps: 4000`/`warmup_steps: 400`,
`gradient_clip_val: 10.0`.

This is a large sweep (216 jobs) — confirm GPU/node allocation before launching the full run;
a smoke test on 1-2 cells first is cheap insurance, particularly for `batch_size: 2048` (confirm
it doesn't exceed the dataset size) and `muon_exclude_lora_heads: true` (confirm the new code
path trains without error).

## Running

```bash
# Generate the 216 leaf configs (only needs to be run once, or after changing the grid).
.venv/bin/python scripts/gen_hypermodel_muon_sweep_configs.py

# Then launch, GPU-parallel (not optional in practice at this size):
GPUS="0,1,2,3,4,5,6,7" bash scripts/run_hypermodel_looped_rope_canon_muon_sweep.sh
```

Split across nodes via `SEEDS_OVERRIDE` and/or `CELL_GLOB` (see script header for examples).

## Reading results

Primary: `val_query_exact_match` on the notd runs — does any cell solve more task categories
than `arc1d_hypermodel_looped_rope_canon_muon`'s `arm_notd_muon` (muon_lr=0.005,
learning_rate=0.001, weight_decay=0.01, batch_size=512, LoRA heads included)? Secondary:
training speed — steps/wall-clock to reach a given exact-match level, since the stated goal
includes speeding up training, not only solving more tasks eventually.

Specifically for the `muon_exclude_lora_heads` axis: compare matched cells (same
muon_lr/learning_rate/weight_decay/batch_size, only the flag differing) rather than the global
best of each — a global-best comparison would conflate the architecture question with the
hyperparameter question.
