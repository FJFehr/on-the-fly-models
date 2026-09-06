# arc1d_capacity_looped

## Goal

Test whether **deep supervision with iterative prediction refinement** (inspired by Tiny Recursive Models / TRMs) increases the effective capacity of a small transformer, enabling it to solve tasks it normally fails at.

The key question: does training with N_supervision=4 backward passes per example — where each pass feeds the previous prediction back as additional context — help a medium-sized transformer solve harder tasks?

## A/B Design

Both variants use an identical transformer backbone (hidden_dim=20, 2 layers, 4 heads, dropout=0.1 — "medium" capacity).

| Config | Model | Training | Inference |
|--------|-------|----------|-----------|
| `transformer.yaml` | standard transformer | 1 forward pass, 1 backward | single-shot prediction |
| `looped_transformer.yaml` | transformer + y_prev feedback | N_supervision=4 forward+backward passes | N_supervision=4 refinement steps |

**Looped variant mechanism:** at each supervision step, `argmax(logits)` from the previous step is embedded via a learned `y_prev_embedder` and summed into the input embedding. The model sees the same input but also its own previous guess, allowing it to correct errors iteratively. The first step always starts with y_prev = zeros (background class).

This is the outer deep-supervision loop from TRM ("Tiny Recursive Models"). The inner latent recursion and z-state are not included in this experiment.

## Task Difficulty Tiers

| Task | Category | Expected difficulty |
|------|----------|-------------------|
| `1d_denoising_1c` | Easy | Standard transformer solves this well |
| `1d_scale_dp` | Medium | Standard transformer struggles sometimes |
| `1d_fill` | Hard | Standard transformer largely fails |

The hypothesis: the looped variant should show the biggest gains on harder tasks, where single-shot prediction is insufficient.

## Data

Augmented via `scripts/augment_arc_1d.py --per-pair`, stored in `data/arc_1d_looped_augmented`.

**Generation command:**
```bash
python scripts/augment_arc_1d.py \
  --per-pair \
  --n-color-permutations 199 \
  --shifts 1 2 -1 -2 \
  --no-mirror \
  --task-categories 1d_denoising_1c 1d_scale_dp 1d_fill \
  --output-dir data/arc_1d_looped_augmented
```

This produces ~1000 variants per base task (200 colour permutations × 5 shift positions) for a total of 121,000 training examples:
- `1d_denoising_1c`: 40,000
- `1d_fill`: 40,000
- `1d_scale_dp`: 41,000

Val/test splits are the original unaugmented held-out tasks (5 per category).

## Running Experiments

```bash
# Looped transformer (A)
python train.py configs/experiments/arc1d_capacity_looped/1d_fill/looped_transformer.yaml

# Standard baseline (B)
python train.py configs/experiments/arc1d_capacity_looped/1d_fill/transformer.yaml
```

Substitute `1d_fill` with `1d_scale_dp` or `1d_denoising_1c` for the other difficulty tiers.

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `N_supervision` | 4 | Backward passes per batch item |
| `hidden_dim` | 20 | Same as capacity_medium baseline |
| `num_layers` | 2 | Same as capacity_medium baseline |
| `num_heads` | 4 | Same as capacity_medium baseline |
| `max_steps` | 4000 | Same wall-clock budget for both variants |
| `batch_size` | 32 | |
| `learning_rate` | 0.001 | RAdam + CosineAnnealingLR |

Note: the looped model does 4× more optimizer steps per Lightning step, but the LR scheduler ticks once per Lightning step so the learning rate schedule is equivalent to the baseline in terms of batches seen.
