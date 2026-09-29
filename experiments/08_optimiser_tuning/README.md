# Experiment 8: optimiser tuning (Muon grid and a tuned AdamW control)

## Question

Experiments 2 to 7 train the hypernetwork with one optimiser recipe that was never tuned:
Muon learning rate 0.005, Adam-group learning rate 0.001, weight decay 0.01, cosine decay to
an absolute floor of 0.0001. This experiment asks:

1. Which Muon settings work best for the hypernetwork? The winning recipe is used for the
   task-identity hypernetworks from here on.
2. Does Muon still beat AdamW once **both** are tuned? Experiment 1 found Muon ahead, but its
   AdamW learning rate was not tuned. Tuning only Muon here would make that gap larger by
   construction, so plain AdamW gets its own sweep and the comparison is best tuned against
   best tuned.

Both grids use the 14-task hypernetwork **without task identity** (experiment 2's dim-4
`notd`). Tuning on `notd` and then using the recipe with task identity is fair to both
optimisers, since both are tuned on the same model. A transfer check on `frozen_td` (arm C
below) is deferred.

## Hypotheses

Written before any run. Each is judged against the s.d. across the 3 seeds.

| # | Comparison | Expected | Why | Measured by | Refuted if |
|---|---|---|---|---|---|
| H1 | Muon grid: where the optimum lies | Muon learning rate 0.01 to 0.02 is best, inside the grid; experiment 2's recipe is within 1 s.d. of the best | Muon's orthogonalised update has a fixed size per step, scaled to each matrix's shape, so its good learning-rate range is narrow and close to the library default (0.02). Experiment 2's 0.005 is conservative but not far off | mean val_loss per cell, one heat map per weight decay | the best cell is on a grid edge, or experiment 2's recipe is worse than the best by more than 1 s.d. |
| H2 | Muon learning rate vs Adam-group learning rate | the Muon learning rate matters much more | 97% of the trainable parameters are in the Muon group, including the weight decoder (9,632 of 10,156, see below). The Adam group holds only 300: embeddings, Canon convolutions, norms and biases | spread of val_loss along each axis (marginals) | the Adam-group learning rate moves val_loss as much as the Muon learning rate does |
| H3 | weight decay | 0 and 0.01 within 1 s.d. of each other; 0.1 worse | about 10K parameters and 8,000 steps leave little overfitting to regularise, and strong decay shrinks the small matrices | weight-decay marginal at the best learning-rate pair | 0.1 is best, or weight decay changes val_loss more than the learning rates do |
| H4 | tuned Muon vs tuned AdamW | Muon still wins, but by less than the untuned gap in experiment 1 | part of experiment 1's gap came from tuning; Muon's advantage for small matrices should remain | best Muon cell vs best AdamW cell: val_loss, then val and test query exact match | the tuned AdamW is within 1 s.d. of Muon, or better |
| H5 | transfer to task identity (arm C, deferred) | the top cells keep their order on `frozen_td` | the task projection is frozen, so `frozen_td` trains the same parameters in the same optimiser groups as `notd` | order of the top 3 cells of each arm on `frozen_td` vs on `notd` | the winner changes by more than 1 s.d. |

**Sanity check.** The Muon cell `muon_m0.005_a1e-3_wd0.01` is experiment 2's recipe except for
the learning-rate floor (0 here, 0.0001 there). Its gap to experiment 2's dim-4 `notd` shows
the effect of the floor alone.

**Before any Muon vs AdamW verdict**, each arm's best cell must be inside its grid. If either
is on an edge, stage 2 extends that grid first.

## Setup

Experiment 2's dim-4 `notd` hypernetwork, 8,000 steps, batch 512, 800 warmup steps then cosine
decay, best checkpoint by val_loss evaluated. [`configs/base.yaml`](configs/base.yaml) differs
from experiment 2 in three places:

- `eta_min: 0`: every learning rate decays to zero. With experiment 2's absolute floor of
  0.0001, an AdamW learning rate of 0.0001 would not decay at all, which would mix the
  schedule into the learning-rate comparison.
- `adam_betas: [0.9, 0.95]` for plain AdamW, the betas Muon's own Adam group uses (PyTorch's
  default is (0.9, 0.999)). Muon runs ignore it.
- no embedding-cluster plots.

| Arm | Optimiser | Grid | Configs | Jobs (3 seeds) |
|---|---|---|---:|---:|
| A | Muon with its Adam group: momentum 0.95, Nesterov, 5 Newton-Schulz steps, Adam betas (0.9, 0.95) (all `muon` library defaults) | Muon learning rate {0.002, 0.005, 0.01, 0.02, 0.04} x Adam-group learning rate {1e-4, 3e-4, 5e-4, 1e-3} x weight decay {0, 0.01, 0.1} | 60 (`configs/muon/`) | 180 |
| B | plain AdamW, betas (0.9, 0.95) | learning rate {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} x weight decay {0, 0.01, 0.1} | 15 (`configs/adamw/`) | 45 |
| C (deferred) | the top 3 cells of A and of B | on `frozen_td` | 6 | |

Weight decay is one value shared by both of Muon's groups. Seeds are 1, 2 and 3. Selection
is by mean val_loss over seeds; test metrics are reported only for the chosen cells.

A has three settings to tune and B two, so A has more cells. The aim is not an equal number
of cells but an optimum inside each grid, so that both optimisers are tuned properly.

### Parameter count

Same model in every run: 10,156 trainable and 11,360 total parameters (the total includes the
frozen target, which is only a shape template).

| Muon's groups | Trainable parameters | What |
|---|---:|---|
| Muon | 9,856 | every hidden `nn.Linear` weight; 9,632 of them are the weight decoder (8 to 1,204) |
| Adam | 300 | embeddings, Canon convolutions, norms, biases |

Plain AdamW trains all 10,156 with one learning rate.

## Run

```bash
uv run python experiments/08_optimiser_tuning/gen_configs.py    # rewrites the 75 configs
GPUS=0,1,2,3 JOBS_PER_GPU=2 bash experiments/08_optimiser_tuning/run.sh   # 225 jobs
ARM=adamw bash experiments/08_optimiser_tuning/run.sh           # one arm only
```

## Findings

Pending: verdicts for H1 to H4 once the runs finish.
