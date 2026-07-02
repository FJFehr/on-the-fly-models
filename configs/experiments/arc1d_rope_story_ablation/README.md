# arc1d_rope_story_ablation

Three new conditions that fill in the design journey between the plain transformer
(`arc1d_recursion_ablation`) and the full Canon+skip model. Used to build the 8-row
"story" heatmap in `scripts/plot_story_ablation.py`.

## Motivation

The story plot needs "no Canon" baselines for RoPE and wide-middle conditions. The
existing `arc1d_rope_sandwich_ablation` already uses Canon ABCD, so we cannot reuse
it for the intermediate story steps. These three conditions fill the gap.

## Architecture progression (S3 → S4 → S5)

| Step | hidden_dim | inner_dim | n_loops | Canon | Change shown |
|------|-----------|-----------|---------|-------|--------------|
| S3 | 16 | 16 | 1 | — | + RoPE (flat 3-layer equivalent) |
| S4 | 16 | 16 | 4 | — | + Looped middle (weight-shared, 6 effective blocks) |
| S5 | 8  | 32 | 4 | — | + Wide middle (outer↓ inner↑, concentrates capacity) |

All use: RoPE (`use_sinusoidal_pe=false`), `canon_set=""` (no Canon), N_sup=2,
lr=0.0005, max_steps=4000, no block/loop skip.

## Full 8-step story (combining multiple experiments)

| Step | Source | Description |
|------|--------|-------------|
| S1 | arc1d_recursion_ablation_large_8k cond A | Vanilla transformer, sin PE, N_sup=1 |
| S2 | arc1d_recursion_ablation_large_8k cond C | + N_sup=2 loop training |
| S3 | **this experiment** | + RoPE (flat 3L) |
| S4 | **this experiment** | + Looped middle |
| S5 | **this experiment** | + Wide middle |
| S6 | arc1d_rope_wide_middle_ablation cond A | + Canon ABCD |
| S7 | arc1d_rope_skip_ablation cond B | + Block skip |
| S8 | arc1d_rope_loop_skip_ablation cond F | + Block + per-iter loop h0 |

## Jobs

3 conditions × 17 tasks × 5 seeds = **255 jobs**  
(1d_padded_fill excluded from this experiment)

```bash
bash scripts/run_ablation_arc1d_rope_story.sh        # 8 GPUs
bash scripts/run_ablation_arc1d_rope_story.sh 4
```

## Fetching and Plotting

```bash
bash scripts/fetch_experiments.sh arc1d_rope_story_ablation
python scripts/plot_story_ablation.py
```
