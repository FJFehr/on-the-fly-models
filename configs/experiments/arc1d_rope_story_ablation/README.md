# arc1d_rope_story_ablation

Two conditions (SC1, SC2) that fill in the design journey between the plain transformer
(`arc1d_recursion_ablation`) and the full Canon+skip model, plus three legacy reference
conditions (S3, S4, S5) from an earlier ordering that are no longer part of the plotted
narrative. Used to build the 8-row "story" heatmap in `scripts/plot_story_ablation.py`.

## Motivation

The original story put Canon ABCD at step 6, after RoPE, looped middle, and wide middle had
already been introduced. That made the heatmap dip after step 2 (dim collapses 512→16 and the
architecture switches to RoPE-sandwich all at once) and not recover until Canon appeared. Moving
Canon to step 3 — right after the N_sup step, before RoPE — gives a monotonically increasing
story instead. Canon had never previously existed without RoPE already bundled in, so two new
conditions were needed to make this possible.

## Architecture progression (SC1 → SC2)

| Step | hidden_dim | inner_dim | n_loops | Canon | Change shown |
|------|-----------|-----------|---------|-------|--------------|
| SC1 | 512 | — | 1 (num_layers=4) | ABCD | + Canon ABCD onto the plain N_sup=4 transformer |
| SC2 | 16 | 16 | 1 | ABCD | + RoPE, flat (Canon carries over from SC1) |

SC1 mirrors `arc1d_recursion_ablation_large_8k` cond C exactly (`N_supervision=4`,
`learning_rate=0.00025`, `max_steps=2000`, `hidden_dim=512`, `num_layers=4`, `num_heads=8`) but
uses `canon_transformer` instead of `transformer` as the backbone (a drop-in superset interface,
see `models/canon_transformer.py`), with `canon_set=ABCD`, `canon_kernel=5`. SC2 is identical to
the legacy `S3` condition (flat RoPE, dim=16, n_loops=1) except `canon_set` is `ABCD` instead of
`""`, so Canon persists once introduced.

## Full 8-step story (combining multiple experiments)

| Step | Source | Description |
|------|--------|--------------|
| S1 | arc1d_recursion_ablation_large_8k cond A | Vanilla transformer, sin PE, N_sup=1 |
| S2 | arc1d_recursion_ablation_large_8k cond C | + N_sup=4 loop training |
| S3 | **this experiment (cond SC1)** | + Canon ABCD |
| S4 | **this experiment (cond SC2)** | + RoPE (flat) |
| S5 | arc1d_rope_sandwich_ablation cond E | + Looped middle (dim=16) |
| S6 | arc1d_rope_wide_middle_ablation cond A | + Wide middle |
| S7 | arc1d_rope_skip_ablation cond B | + Block skip |
| S8 | arc1d_rope_loop_skip_ablation cond F | + Block + per-iter loop h0 |

## Legacy conditions (S3, S4, S5)

The original three no-Canon RoPE baselines remain in this project, untouched:

  - S3: Flat 3L RoPE transformer (n_loops=1, dim=16, no Canon)
  - S4: Looped middle (n_loops=4, dim=16, no Canon)
  - S5: Wide middle (outer=8, inner=32, n_loops=4, no Canon)

They are kept on disk as full-reproducibility reference data (255/255 already run at 5 seeds) but
are excluded from `scripts/plot_story_ablation.py`'s step mapping, so they no longer appear in the
rendered heatmap.

## Jobs

5 conditions × 17 tasks × 5 seeds = **425 jobs max** (1d_padded_fill excluded). Most of the S3/S4/S5
jobs (255) are already complete; the new SC1/SC2 conditions account for the remaining ~170.

```bash
bash scripts/run_ablation_arc1d_rope_story.sh        # 8 GPUs, skips already-completed runs
bash scripts/run_ablation_arc1d_rope_story.sh 4
```

## Fetching and Plotting

```bash
bash scripts/fetch_experiments.sh arc1d_rope_story_ablation
python scripts/plot_story_ablation.py
```

Note: the plot script also reads from `outputs/arc1d_rope_sandwich_ablation` (source of S5, cond
E) and the other four reused projects listed in `scripts/plot_story_ablation.py`'s `main()`.
