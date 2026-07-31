# arc1d_hypermodel_looped_rope_canon_muon_moveablation

## Goal

Follow-up to the 216-cell `arc1d_hypermodel_looped_rope_canon_muon_sweep` (muon_lr x adam_lr x
weight_decay x batch_size x LoRA-exclusion, notd only), which was cancelled after 99 cells once
its findings pointed clearly enough in a direction to act on. Fabio's goal throughout this line of
work: speed up training and get notd (no task-identity signal) to solve more task categories.

## What the prior sweep found

- **Excluding the LoRA head's A/B factor projections (`lora_proj_a`/`lora_proj_b`) from Muon beat
  including them**: mean val_query_exact_match 0.654 vs 0.618 across the 99 completed cells, and 9
  of the top 10 individual cells used exclusion. Locked in here (`muon_exclude_lora_heads: true`,
  fixed, not swept).
- **batch_size=2048 pulled ahead** of 512/1024 (mean 0.658 vs ~0.623-0.629), 7 of the top 10 cells
  used it. This experiment tried pushing further to `batch_size=4096`, but that genuinely OOMs
  (44.21/44.40 GiB on a single GPU, reproducible across every other axis combination — the
  hypernetwork generates per-example target-model weights, so activation memory scales with
  batch_size much faster than a normal transformer's) — dropped from the grid entirely rather than
  chasing a memory fix. `batch_size=2048` is fixed here.
- **muon_lr itself showed no real trend** across 0.008/0.01/0.016 (means all ~0.63-0.64, within
  noise). The one standout result (0.973 at muon_lr=0.016, adam_lr=6e-4, wd=0.1, bsz=2048,
  lora=excl) is more likely explained by that adam_lr/wd/bsz/lora combination than by muon_lr
  specifically — it's being reseeded separately (2 more seeds) rather than folded into this grid.
- **muon_lr=0.02 was never reached** by the sweep. It's notable: 0.02 is the exact value that
  caused the original NaN divergence in the very first Muon experiment
  (`arc1d_hypermodel_looped_rope_canon_muon`), before LoRA-exclusion existed as an option. Fixed
  here as a deliberate, now-better-protected retest (LoRA-exclusion may remove exactly the
  instability source), not an assumption it's safe — see Verification below.
- Two cells failed in the prior sweep (`adam_lr=3e-3, weight_decay=0, batch_size=512, lora=incl`,
  at both muon_lr=0.01 and 0.016) — both combinations dropped here anyway (`adam_lr=3e-3` was also
  the worst mean performer; `weight_decay=0` is out of scope this round).

## The new hypothesis: move-task ablation

Two prior, separate experiments found real evidence that mixing `1d_move_1p`/`1d_move_2p`/
`1d_move_3p` **without** a task-identity signal (i.e. in `notd` mode, the only mode this entire
line of work uses) causes trouble:

- `arc1d_hypermodel_looped_rope_canon_generalization` held out `1d_move_2p` and found it a
  "near-miss" (0/5 exact match, but per-token accuracy up to 0.90 on some held-out examples) —
  the model places it "somewhere sensible" near its move-family siblings in latent task space
  without landing an exact match.
- `arc1d_hypermodel_disentanglement` found single-task `move_1p`/`move_2p`/`move_3p` each train to
  100% alone, but **mixed pairs fail completely** without a task descriptor — direct evidence of
  catastrophic interference specifically from mixing move variants in `notd`.

This experiment tests whether dropping `1d_move_1p`/`1d_move_2p` entirely from training (both
`task_categories` and `val_task_categories` — no zero-shot eval on them this round, that's a
separate question) removes enough of that interference to improve both in-distribution results
and the compositional-generalization eval below, on the remaining 13 categories.

## Grid

Fixed across every cell: Zhu backbone (`rope_canon_zhu_transformer`), `rope_canon_looped_transformer`
target, `lora_adapter_rank=8`, `max_steps=4000`/`warmup_steps=400`, `gradient_clip_val=10.0`,
`optimizer=Muon`, `muon_lr=0.02`, `muon_momentum=0.95`, `muon_exclude_lora_heads=true`,
`batch_size=2048`, notd (`hyper_head.num_tasks: null`).

Crossed, `2 x 2 x 2 = 8` configs, 3 seeds each = 24 jobs:

| Axis | Values |
|---|---|
| `weight_decay` | `0.01`, `0.1` |
| `learning_rate` (AdamW aux group) | `3e-4`, `6e-4` |
| move-task inclusion | `with` (standard 15 categories), `without` (13, drops `1d_move_1p`/`1d_move_2p`) |

### batch_size / data-volume caveat

The training pool is only 40-41 genuinely distinct raw tasks per category (~601,000
augmented-but-correlated rows total across the 15-category list). `max_steps` is fixed at 4000
regardless of `batch_size` (`training/trainer.py`), so **larger batches mean more recycling of
this same fixed pool, not more raw data seen**: bsz=512 (prior sweep's baseline) recycled the pool
~3.4x, bsz=2048 (fixed here) ~13.6x. `bsz=4096` was tried (kept as a literal
batch-size-at-fixed-max_steps test, Fabio's call — testing whether bigger batches help
optimization at this data scale, not data scaling) but hit a hard memory wall instead, see above.
(A genuinely more data-scarce regime, for a different question, already exists:
`arc1d_lowdata_lowrank`'s `variants_per_base_task` sweep — out of scope here.)

## Compositional-generalization eval (every run, not just in-distribution numbers)

Every training run is followed by `scripts/eval_compositional_holdout.py` (unmodified, reused from
`arc1d_hypermodel_compositional_generalization`) against its own best checkpoint — a zero-shot
eval on 10 composite task categories (chained two-rule compositions, e.g. "denoise then shift")
that were never trained on. This answers Fabio's question of whether good in-distribution settings
from this tuning effort also help compositional generalization, as part of this experiment's own
results rather than a separate follow-up.

Confirmed compatible with the `without`-moves (13-category) arms: the composite holdout categories
are built from a `move_dynamic` base rule matching `move_dp`/`move_2p_dp` semantics, never
`1d_move_1p`/`1d_move_2p`, and the model's task-category index registry is fixed and independent
of which subset of categories a given run actually trained on. `notd` also skips the eval script's
`num_tasks >= 28` check entirely (only relevant to `td`/`frozen_td`), so no `hyper_head` changes
are needed.

Every run also keeps `log_embedding_clusters: true` on (base.yaml) — both the training run and the
compositional eval produce a PCA/t-SNE/UMAP cluster map (the eval's version overlays the
compositional-holdout embeddings on top of the in-distribution validation clusters) plus linear
probe accuracy, "always on" per Fabio, not just for the in-distribution numbers.

## Running

```bash
# Generate the 8 leaf configs (only needs to be run once, or after changing the grid).
.venv/bin/python scripts/gen_hypermodel_muon_moveablation_configs.py

# Then launch, split across two nodes (full 8-GPU parallelism each) by move-task inclusion:
# node A (e.g. torrnode12):
CELL_GLOB="arm_bsz2048_*_with.yaml" GPUS="0,1,2,3,4,5,6,7" \
  bash scripts/run_hypermodel_looped_rope_canon_muon_moveablation.sh
# node B (e.g. torrnode8):
CELL_GLOB="arm_bsz2048_*_without.yaml" GPUS="0,1,2,3,4,5,6,7" \
  bash scripts/run_hypermodel_looped_rope_canon_muon_moveablation.sh
```

Each invocation trains its half of the grid (GPU-parallel, 4 cells x 3 seeds = 12 jobs), then runs
the compositional-holdout eval on each completed run in turn (sequential — eval is cheap, 400
holdout examples on a tiny model, and runs on CPU by default per
`eval_compositional_holdout.py`'s own convention).

## Reading results

1. **In-distribution**: `val_query_exact_match` per cell (`outputs/<project>/<exp_name>/results.txt`)
   — does any `with`/`without`-moves pair, at matched weight_decay/adam_lr, show the
   `without` arm winning? That's the direct test of the move-task-interference hypothesis. Compare
   the best cells here against the prior sweep's `notd_muon` baseline (mean ≈ 0.483, muon_lr=0.005
   untuned) and `notd_adamw` baseline (mean ≈ 0.675).
2. **Compositional generalization**:
   `outputs/<project>/<exp_name>/compositional_holdout_eval/results.txt` (fixed-width table, not
   `key: value` — parse by column) — does the `without`-moves arm, or any particular
   weight_decay/adam_lr combination, show better zero-shot exact-match or
   per-token seq-accuracy on the 10 composite categories than the already-run (untuned)
   `arc1d_hypermodel_compositional_generalization/notd` baseline (overall exact-match 0.013, but
   seq-accuracy 0.63-0.92 per category — the "near-miss" pattern)?
3. **Embedding clusters + linear probe**: `embedding_clusters/pooled_task_latent_with_compositional_holdout.png`
   and the linear-probe summary in both the training run's own output and the compositional eval's
   output directory — does removing move tasks visibly tighten or separate the task-category
   clusters, and does the compositional-holdout overlay land closer to sensible neighbors?
