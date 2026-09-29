# Experiment 1: multi-task capacity

## Question

Can one small model learn all 14 ARC-1D task categories jointly (one shared set of weights),
with and without a task-identity signal, and how does that compare with one model per task?

## Setup

The same RoPE + Canon Transformer (3 blocks, 1 head) at four widths, `hidden_dim` 4, 6, 10
and 14 (1.4K to 9.7K parameters), `task_encoding.embedding_dim` fixed at 10. Muon
(`muon_lr` 0.005) with an AdamW group for embeddings, norms and the output layer
(`learning_rate` 0.0005), 8,000 steps, 200-step warmup then cosine decay. 5 seeds everywhere.
Evaluation uses the final weights (`evaluate_on: final`).

| Condition | Configs | What differs |
|---|---|---|
| Individual | `configs/individual/<category>/dim*.yaml` (56) | one model per task category |
| Joint, no task ID (notd) | `configs/notd*.yaml` (4) | one model for all 14 categories |
| Joint, task ID (td) | `configs/td*.yaml` (4) | as notd, plus a learned per-category embedding added to every token |
| Canon ablation | `configs/nocanon/` (64) | every config above with `canon_set: ''` |
| Optimiser ablation | `configs/adamw/` (64) | every config above with plain AdamW (`learning_rate` 0.001) |

`gen_individual_configs.py`, `gen_nocanon_configs.py` and `gen_adamw_configs.py` write the
generated configs; rerunning them reproduces the committed files exactly.

## Run

```bash
GPUS=0,1,2,3,4,5,6,7 bash experiments/01_multitask_capacity/run.sh
```

192 configs x 5 seeds = 960 jobs. To run one part only, call `scripts/run_config.sh` with
`CFG_DIR` narrowed to a subfolder and `PROJECT=01_multitask_capacity`, for example
`CFG_DIR=experiments/01_multitask_capacity/configs/nocanon`.

## Plots

```bash
uv run python experiments/01_multitask_capacity/plot_all.py --outputs-dir outputs
```

Writes CSVs to `outputs/results/01_multitask_capacity/` and figures to
`outputs/figures/01_multitask_capacity/`: `capacity_cliff`, `per_task_dim{4,6,10,14}`,
`capacity_cliff_canon_ablation` and `capacity_cliff_optimizer_ablation`. The per-task plots
use validation exact match (only validation has a per-category breakdown); all others use
test exact match.

## Outputs

`outputs/01_multitask_capacity/<run>/`: `results.txt`, `config.yaml`, `model.txt`,
`train.log`, and the models (`final_model.ckpt` is the one evaluated; `best_model.ckpt` and
`last.ckpt` are also kept).

## Result

Test exact match, mean over 5 seeds:

| Parameters | Individual | Joint, notd | Joint, td |
|---:|---:|---:|---:|
| 1,398 (dim 4) | 92.6% | 13.4% | 21.2% (unstable, 5.5% to 44.6%) |
| 2,444 (dim 6) | 98.3% | 40.7% | 73.2% |
| 5,400 (dim 10) | 98.7% | 57.5% | 93.2% |
| 9,508 (dim 14) | 98.6% | 65.6% | 97.5% |

Individual models saturate almost immediately; joint models do not. At 2,444 parameters a
model can solve every task alone (98.3%) but not all of them together, even with task
identity (73.2%). The joint-training penalty is concentrated in a few categories (`flip` is
100% individually and 4% jointly at dim 6), not spread evenly. This motivates generating
per-task weights with a hypernetwork (experiment 2).

**Canon ablation.** Canon helps in every condition and size. For individual models the boost
shrinks with size (+55.1 points at dim 4 down to +2.8 at dim 14); for joint models it peaks
at dim 6 (notd +35.0, td +49.5), because without Canon the dim-4 joint models barely train
(0.9% and 1.2%).

**Optimiser ablation.** Muon beats AdamW in every cell except individual dim 14 (98.6% vs
99.2%), most at small sizes and in joint training (joint notd dim 4: 13.4% vs 4.4%). The
AdamW learning rate was not tuned for this task, so part of the gap may be tuning.

**Reproducibility.** Rerun on the refactored code: all 37 cells are within 2 standard errors
of the numbers above (mean over all cells 0.5907 before, 0.5908 after) and 675 of 965 runs
give an identical test exact match.

## Cost

Wall-clock minutes per run on one NVIDIA A40 (one run per GPU), including data setup and evaluation. Measured on the refactor rerun.

| Runs | Count | Minutes per run | A40 GPU-hours |
|---|---:|---:|---:|
| individual (main / no-Canon / AdamW) | 280 each | 3.6 / 3.6 / 2.5 | 16.8 / 16.8 / 11.7 |
| joint (main / no-Canon / AdamW) | 40 each | 9.8 / 9.6 / 8.7 | 6.5 / 6.4 / 5.8 |
| **Total** | **960** | | **64** |
