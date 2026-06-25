# arc1d_recursion_ablation

## Research Question

For ARC-style reasoning tasks, where does the performance gain from recursion come from?

```
Performance gain ← { architectural recursion | training recursion | both }
```

while holding **parameter count** and **total optimizer steps** fixed.

## 4-Way Design

All conditions use the same backbone size: `hidden_dim=256`, `num_layers=2`, `num_heads=4`.

| Cond | File prefix | Model | Backbone | Architecture | Training | Optimizer steps |
|------|-------------|-------|----------|--------------|----------|-----------------|
| A | `A_transformer` | `direct_supervised` | `transformer` | 1 forward pass | 1 update/batch | 8 000 |
| B | `B_recursive_transformer` | `direct_supervised` | `recursive_transformer` | 4 loops, shared weights | 1 update/batch | 8 000 |
| C | `C_looped_transformer` | `looped_supervised` | `transformer` | 1 forward pass | 4 updates/batch | 2 000 × 4 = 8 000 |
| D | `D_looped_recursive_transformer` | `looped_supervised` | `recursive_transformer` | 4 loops, shared weights | 4 updates/batch | 2 000 × 4 = 8 000 |

### Architectural recursion (B, D)
The `recursive_transformer` backbone applies a 2-layer shared block **4 times** in a single forward pass (Universal Transformer style). Total computational depth = 8 layers. Parameter count = same 2-layer transformer — the weights are reused across loops, not duplicated.

### Training recursion (C, D)
`looped_supervised` runs `N_supervision=4` forward+backward passes on each batch before moving to the next one. Parameters are updated 4 times on the same examples. `max_steps=2000` so the total optimizer step budget matches A and B.

## Compute Matching

- A and B: `max_steps=4000`, 1 optimizer step per Lightning step → **4 000 total updates**
- C and D: `max_steps=1000`, `N_supervision=4` → **1 000 × 4 = 4 000 total updates**

Note: FLOPs are not fully matched. B and D pay 4× more compute per forward pass (4 loops vs 1). This is an intentional design choice — matching FLOPs would require halving the model size for B/D, which confounds parameters with depth.

## Expected Outcomes

| Model | Expected result |
|-------|----------------|
| A (plain) | Baseline |
| C (training recursion) | Small improvement on hard tasks |
| B (architectural recursion) | Larger improvement |
| D (combined) | Best |

The most interesting result would be **B > C**: architectural recursion outperforms training recursion, suggesting that iterative hidden-state computation is more important than repeatedly updating parameters. This is consistent with the hypotheses behind Universal Transformers, TRM, and HRM.

## Inference-Time Loop Sweep (Condition B and D)

After training B or D, you can evaluate the model with different numbers of loops at inference:

```python
model.backbone.n_loops = k  # set k = 1, 2, 3, 4
```

A model that learned an iterative algorithm should show progressive improvement:
```
n_loops=1: 40%
n_loops=2: 55%
n_loops=3: 65%
n_loops=4: 70%
```
Flat accuracy across loop counts suggests the recursion didn't produce meaningful refinement.

## Task Difficulty Tiers

| Task | Difficulty | Notes |
|------|-----------|-------|
| `1d_denoising_1c` | Easy | Standard transformer solves this |
| `1d_scale_dp` | Medium | Transformer struggles sometimes |
| `1d_fill` | Hard | Transformer largely fails |
| `1d_recolor_cmp` | ? | Recolor by comparison — hypothesis: looping helps |
| `1d_recolor_cnt` | ? | Recolor by count — hypothesis: looping helps |
| `1d_recolor_oe` | ? | Recolor odd/even — hypothesis: looping helps |

Expected: recursion benefits should grow with task difficulty.

## Data

`data/arc_1d_looped_augmented/` — 121k train examples across 3 task categories (per-pair colour augmentation, ~1 000 variants per base task). Val/test use the original held-out tasks (5 per category).

To regenerate:
```bash
python scripts/augment_arc_1d.py \
  --per-pair \
  --n-color-permutations 199 \
  --shifts 1 2 -1 -2 \
  --no-mirror \
  --task-categories 1d_denoising_1c 1d_scale_dp 1d_fill \
    1d_recolor_cmp 1d_recolor_cnt 1d_recolor_oe \
  --output-dir data/arc_1d_looped_augmented
```

## Running Experiments

```bash
# All 4 conditions on 1d_fill (hardest task)
python train.py --config configs/experiments/arc1d_recursion_ablation/1d_fill/A_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/1d_fill/B_recursive_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/1d_fill/C_looped_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/1d_fill/D_looped_recursive_transformer.yaml

# Overfit sanity check (all tasks, 1 batch, no dropout)
python train.py --config configs/experiments/arc1d_recursion_ablation/overfit/A_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/overfit/B_recursive_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/overfit/C_looped_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/overfit/D_looped_recursive_transformer.yaml
```
