#!/usr/bin/env bash
# One command to reproduce experiment 2's dim=4 rerun: notd + frozen_td,
# 5 seeds each (10 jobs), checkpoints saved (base.yaml's save_checkpoints:
# true), so every seed's own weights are reusable afterward (experiment 3)
# and every seed's own embedding cluster is plottable directly (see
# plot_embedding_clusters.py --seed).
#
# Scoped to dim4 via CELL_GLOB -- dim6's configs live in this same folder
# but are the original 3-seed run, not part of this rerun (see README's
# "Dim=4 rerun" section for why).
#
# Usage:
#   bash experiments/02_hypernetwork_multitask/run.sh                        # sequential, 1 GPU
#   GPUS="0,1,2,3,4,5,6,7" bash experiments/02_hypernetwork_multitask/run.sh  # 8-way parallel
#
# Wraps scripts/run_config.sh with this experiment's own CFG_DIR/CELL_GLOB
# baked in. Skips any (config, seed) pair that already has a results.txt,
# so it's always safe to rerun.
#
# Then plot directly from what this wrote to outputs/ -- see
# plot_per_task_combined.py --outputs-dir outputs.

set -uo pipefail
cd "$(dirname "$0")/../.."

CFG_DIR=experiments/02_hypernetwork_multitask \
CELL_GLOB="dim4*.yaml" \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3 4 5}" \
GPUS="${GPUS:-}" \
bash scripts/run_config.sh
