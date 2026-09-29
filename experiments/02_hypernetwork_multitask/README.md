# Experiment 2: a joint hypernetwork

## Question

Experiment 1 showed that one shared set of weights cannot hold all 14 task categories at
small sizes. Can a hypernetwork that generates the target model's weights from each task's
support examples close that gap, at a matched parameter budget, with and without a
task-identity signal?

## Setup

- **Target**: the same Transformer as experiment 1 at `hidden_dim` 4 (3 blocks, 1 head),
  whose weights are generated, never trained directly.
- **Hypernetwork** (dim 4, "matched scale": every part sized to the dim-4 target):
  `task_encoding.embedding_dim` 4, encoder Transformer `hidden_dim` 4, 1 layer, 1 head,
  `output_dim` 4, attention pooling, projection MLP with `bottleneck_dim` 8. 11,360 (notd) /
  11,432 (frozen_td) parameters in total, 10,156 trainable.
- **Arms**: `notd` (the support examples are the only task signal) and `frozen_td` (a one-hot
  task identity, projected by a fixed random matrix, is added to the task representation).
- **Training**: Muon (`muon_lr` 0.005) with AdamW (`learning_rate` 0.001), gradient clipping
  10, batch 512, 8,000 steps with 800 warmup, bf16. 5 seeds per arm. Evaluation uses the
  best checkpoint by validation loss (`evaluate_on: best`), which experiments 3 and 4 reuse.

`configs/dim6_*.yaml` hold an earlier, larger dim-6 variant (encoder `hidden_dim` 32, about
337K parameters, 3 seeds); it is not part of `run.sh`.

## Run

```bash
GPUS=0,1,2,3,4 bash experiments/02_hypernetwork_multitask/run.sh
```

2 configs x 5 seeds = 10 jobs.

## Plots

```bash
uv run python experiments/02_hypernetwork_multitask/plot_all.py
```

Writes `outputs/results/02_hypernetwork_multitask/` (per-task CSVs, `results.csv` with
exact match and linear-probe accuracy per seed) and `outputs/figures/02_hypernetwork_multitask/`
(per-task bar charts, and each seed's PCA / t-SNE / UMAP maps of the pooled task
representation). `--skip-clusters` skips the slower cluster maps.

## Outputs

`outputs/02_hypernetwork_multitask/hyper_multitask_dim4_{notd,frozentd}_seed{1..5}/`:
`results.txt`, `best_model.ckpt` (evaluated, and used by experiments 3 and 4), `last.ckpt`,
`embedding_clusters/` (pooled task representations and linear-probe results).

## Result

Test exact match, dim 4:

| Arm | Seeds 1 to 5 | Mean | Linear probe |
|---|---|---:|---:|
| frozen_td | 0.936 / 0.922 / 0.949 / 0.955 / 0.889 | **93.0%** | 92.9% to 100% |
| notd | 0.541 / 0.531 / 0.551 / 0.671 / 0.575 | 57.4% | 54.5% to 68.1% |

Against experiment 1 at the same target size (dim 4): individual models 92.6%, joint direct
training 21.2% (td) and 13.4% (notd). With task identity the hypernetwork recovers the
individual-model level from one shared network. Without it, the hypernetwork still far
outperforms direct joint training; most of the remaining gap is the move family (`move_1p`,
`move_3p`, `move_dp`) being confused with each other. At this small scale `frozen_td` also
softens on `move_dp` (55%), `mirror` (74%) and `flip` (80%); every other category is at or
near 100%.

The linear probe (predicting the category from the pooled task representation) tells the same
story: task representations are almost perfectly separable with a task-identity signal and
much less so without.

**Reproducibility.** Rerun on the refactored code: all 10 checkpoints are bit-identical to the
original ones (every weight equal, same best epoch), so every metric matches exactly.

## Cost

Wall-clock minutes per run on one NVIDIA A40 (one run per GPU), including data setup and evaluation. Measured on the refactor rerun: 10 runs of about 24 minutes, 4.0 A40 GPU-hours.
