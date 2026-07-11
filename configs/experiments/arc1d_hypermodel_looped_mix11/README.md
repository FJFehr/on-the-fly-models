# arc1d_hypermodel_looped_mix11

## Goal

First multi-task hypernetwork experiment: one hypernetwork trained across 11 task
categories simultaneously (not per-task, like `arc1d_hypermodel_looped`'s phase-1 sweep),
with a one-hot task descriptor so the model can disambiguate which task it's solving.

## Settings

- **Task set**: the same "simple" 11-task subset already established in
  `arc1d_hypermodel_augmented_td/rnn_simple_td.yaml`: `1d_denoising_1c, 1d_denoising_mc,
  1d_fill, 1d_hollow, 1d_mirror, 1d_move_1p, 1d_move_2p_dp, 1d_move_dp, 1d_pcopy_1c,
  1d_pcopy_mc, 1d_scale_dp`. `1d_padded_fill` stays excluded.
- **Backbone**: `rope_canon_looped_transformer`, `n_loops=8`, block+loop skip on,
  `N_supervision=4` — the winning capacity settings found by the flip+recolor sweep
  (`arc1d_hypermodel_looped_recolor`): `N_supervision=4` beat 2, `n_loops=8` beat 4 (16 was
  too much), skip connections were fine to keep.
- **Task descriptor**: `hyper_head.num_tasks: 18` (enabled from the start). The repo's own
  `arc1d_hypermodel_disentanglement` experiments already showed structurally similar task
  mixes fail outright without one — `1d_move_1p+1d_move_3p` and `1d_move_1p+2p+3p` both
  failed completely on the old RNN-target hypernetwork, while a structurally dissimilar pair
  succeeded with no descriptor at all. This 11-task subset includes three move variants
  together (`1d_move_1p`, `1d_move_2p_dp`, `1d_move_dp`), so the descriptor is included
  up front rather than as a fallback after an expected failure.

## Running

```bash
bash scripts/run_hypermodel_looped_mix11.sh        # up to 3 GPUs, seeds 1-3 in parallel
```

Single hand-written config (`mix11_td.yaml`), not a generated family — this is one
multi-task experiment, not a per-task sweep.

## Reading results

Compare each task's `val_query_exact_match` here (mean ± std across the 3 seeds) against
that task's phase-1 individual-training result in `arc1d_hypermodel_looped`. This is the
"accuracy and comparability" check: does training all 11 tasks together, with the
descriptor, hold each task's accuracy close to its individually-trained baseline, or does
any task degrade despite the descriptor being present.
