# arc1d_rope_unet_skip_ablation

Tests proper U-Net style skip connections to clarify the role of each skip mechanism.

## Motivation

Previous experiments used two skip mechanisms:
- **Block skip**: `x = block(x_in) + x_in` inside each block — skips Canon+attn+MLP per iteration
- **Loop skip**: `h = middle(h) + h_loop_0` at every iteration — injects the same pre-loop
  anchor h_0 repeatedly, which is architecturally unusual

The loop skip gave the best flip result (61%) when combined with block skip, but the
mechanism is hard to interpret. This experiment tests cleaner U-Net style alternatives:

- **Inner bypass**: save state before loops, add back ONCE after all loops complete
- **Outer bypass**: save state before pre_layer, add back ONCE after post_layer

Together these form a proper U-Net: inner bypass skips the looping block, outer bypass
skips all three blocks (pre + loops + post).

## Architecture base (all conditions)

| Property | Value |
|---|---|
| Outer hidden dim | 8 |
| Inner hidden dim | 32 |
| Outer / Inner heads | 1 / 4 |
| n_loops | 4 |
| Positional encoding | RoPE |
| Canon set | ABCD, kernel=5 |
| N_sup | 2 |
| Learning rate | 0.0005 |
| Max steps | 4000 |
| Parameters | ~16,672 |

## Conditions

Conditions A, B, F are **reused** from previous experiments.

| Cond | block_skip | inner_bypass | outer_bypass | Notes |
|------|-----------|--------------|--------------|-------|
| A    | No  | No  | No  | Reference (existing) |
| B    | Yes | No  | No  | Block skip only (existing) |
| F    | Yes | —   | —   | Block + per-iter loop skip (existing, comparison) |
| N    | No  | Yes | No  | Inner bypass only |
| O    | No  | No  | Yes | Outer bypass only |
| P    | No  | Yes | Yes | U-Net: inner + outer |
| Q    | Yes | Yes | Yes | Block + U-Net: all skips |

## Jobs

4 new conditions × 18 tasks × 3 seeds = **216 jobs**

```bash
bash scripts/run_ablation_arc1d_rope_unet_skip.sh        # 8 GPUs
bash scripts/run_ablation_arc1d_rope_unet_skip.sh 4
```

## Fetching and Plotting

```bash
bash scripts/fetch_experiments.sh arc1d_rope_unet_skip_ablation
python scripts/plot_rope_unet_skip_ablation.py
```
