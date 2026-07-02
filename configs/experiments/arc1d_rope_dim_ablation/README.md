# arc1d_rope_dim_ablation

Joint sweep of **inner_dim** and **skip configuration** to find the smallest architecture
that solves all 18 ARC-1D tasks. Also includes outer `hidden_dim=16` conditions to test
whether the output-side representation is a bottleneck.

## Motivation

After the loop-skip ablation, the best architecture (outer=8, inner=32, block+loop skip,
4 loops) reaches 91% overall but stalls at:
- **flip** 61% — routing problem; Canon attention blurring in the looped middle layer
- **move_dp** 89% — close but not 100%
- **mirror** 92% — close but not 100%
- **padded_fill** 12–20% — likely a length-generalisation issue

Two questions motivate this sweep:
1. **Capacity**: is inner_dim=32 (16.7k params) enough, or do we need inner_dim=64 (55.6k)?
   And conversely, can inner_dim=16 (6.4k) + skips already match the best result?
2. **Outer dim**: is the output bottleneck (hidden_dim=8 → 10 classes) limiting performance?
   outer=16 tests whether a larger outer representation helps.

### Note on `non_background_loss_weight`

Set to 1.0 in the base config → **no-op** (uniform weighting over all valid tokens).
Would only affect training if set > 1.0 to upweight non-background predictions.

## Architecture base (all conditions)

| Property | Value |
|---|---|
| Outer num_heads | 1 (head_dim = hidden_dim, avoids head_dim=2 with num_heads=4) |
| Inner num_heads | 2 / 4 / 8 (scaled with inner_dim to keep head_dim=8) |
| n_loops | 4 |
| Canon set | ABCD, kernel=5, non-causal |
| Positional encoding | RoPE (use_sinusoidal_pe=False) |
| N_supervision | 2 |
| Learning rate | 0.0005 |
| Max steps | 4000 (= 8000 optimizer updates) |

## Conditions

Conditions A and F are **reused** from previous experiments (plot script loads them).

| Cond | outer hidden | inner dim | block_skip | loop_skip | Params  |
|------|-------------|-----------|------------|-----------|---------|
| A    | 8           | 32        | No         | No        | 16,672  |
| F    | 8           | 32        | Yes        | Yes       | 16,672  |
| H    | 8           | 16        | No         | No        | 6,448   |
| I    | 8           | 16        | Yes        | Yes       | 6,448   |
| J    | 8           | 64        | No         | No        | 55,552  |
| K    | 8           | 64        | Yes        | Yes       | 55,552  |
| L    | 16          | 32        | No         | No        | 22,624  |
| M    | 16          | 32        | Yes        | Yes       | 22,624  |

## Jobs

6 new conditions × 18 tasks × 3 seeds = **324 jobs**

```bash
bash scripts/run_ablation_arc1d_rope_dim.sh        # 8 GPUs (default)
bash scripts/run_ablation_arc1d_rope_dim.sh 4      # 4 GPUs
```

## Fetching and Plotting

```bash
bash scripts/fetch_experiments.sh arc1d_rope_dim_ablation
python scripts/plot_rope_dim_ablation.py
```
