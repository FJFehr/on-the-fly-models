# arc1d_recursion_ablation

## Research Question

For ARC-style reasoning tasks, where does the performance gain from recursion come from?

```
Performance gain ← { architectural recursion | training recursion | both }
```

while holding **total optimizer steps** fixed.

## 4-Way Design

| Cond | File prefix | Backbone | `num_layers` | `n_loops` | `N_supervision` | LR | Optimizer steps |
|------|-------------|----------|:---:|:---:|:---:|---|:---:|
| A | `A_transformer` | `transformer` | 2 | 1 | 1 | 0.001 | 4 000 |
| B | `B_recursive_transformer` | `recursive_transformer` | 1 | 2 | 1 | 0.001 | 4 000 |
| C | `C_looped_transformer` | `transformer` | 2 | 1 | 4 | **0.00025** | 4 000 |
| D | `D_looped_recursive_transformer` | `recursive_transformer` | 1 | 2 | 4 | **0.00025** | 4 000 |

All conditions: `hidden_dim=256`, `num_heads=4`, `batch_size=256`, `warmup_steps=200`.

### Architectural recursion (B, D)
`recursive_transformer` applies 1 shared encoder layer **2 times** in the forward pass.
Effective depth = 2 (same as A/C), parameter count ≈ half of A/C (one layer instead of two).
Tests whether weight sharing and iterative hidden-state refinement helps.

### Training recursion (C, D)
`looped_supervised` runs `N_supervision=4` forward+backward passes on the same batch before
moving to the next one — 4 optimizer steps per Lightning step.
`max_steps=1000` so total optimizer steps match A and B (1 000 × 4 = 4 000).

### Learning rate scaling for C and D
C and D use `learning_rate=0.00025` (`0.001 / N_supervision`).

**Why:** 4 gradient steps on the same batch is equivalent to multiplying the per-example
update magnitude by 4. Scaling the LR down keeps effective updates comparable to A and B.

**Why this matters for the C vs D comparison:** both must use the same LR so that any
performance difference between them is attributable solely to the recursive architecture in
D, not to a confounded LR advantage.

## Compute Matching

- A and B: `max_steps=4000`, 1 optimizer step/Lightning step → **4 000 total updates**
- C and D: `max_steps=1000`, `N_supervision=4` → **1 000 × 4 = 4 000 total updates**

Note: B and D pay 2× more compute per forward pass (2 loops vs 1). FLOPs are not matched
— matching them would require halving the model size for B/D, which confounds parameters
with depth.

## Task Difficulty Tiers

| Task | Difficulty | Notes |
|------|-----------|-------|
| `1d_denoising_1c` | Easy | Baseline transformer solves this |
| `1d_scale_dp` | Medium | Transformer struggles |
| `1d_fill` | Hard | Transformer largely fails |
| `1d_recolor_cmp` | ? | Recolor by comparison — hypothesis: looping helps |
| `1d_recolor_cnt` | ? | Recolor by count — hypothesis: looping helps |
| `1d_recolor_oe` | ? | Recolor odd/even — hypothesis: looping helps |

## Data

`data/arc_1d_looped_augmented/` — ~241k train examples across 6 task categories
(per-pair colour augmentation, ~40k per task). Dev and test each have **100 examples per
task** (5 original held-out tasks × 20 colour permutations), reducing metric variance from
±45% to ±10% compared to the raw 5-example splits.

To regenerate (all 18 tasks):
```bash
uv run python scripts/augment_arc_1d.py \
  --per-pair \
  --n-color-permutations 199 \
  --shifts 1 2 -1 -2 \
  --no-mirror \
  --dev-test-n-permutations 19 \
  --output-dir data/arc_1d_looped_augmented
```

Omitting `--task-categories` augments all tasks in the base dataset.

## Running Experiments

```bash
# All 4 conditions on a given task
TASK=1d_fill
python train.py --config configs/experiments/arc1d_recursion_ablation/${TASK}/A_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/${TASK}/B_recursive_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/${TASK}/C_looped_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/${TASK}/D_looped_recursive_transformer.yaml

# Overfit sanity check
python train.py --config configs/experiments/arc1d_recursion_ablation/overfit/A_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/overfit/B_recursive_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/overfit/C_looped_transformer.yaml
python train.py --config configs/experiments/arc1d_recursion_ablation/overfit/D_looped_recursive_transformer.yaml
```
