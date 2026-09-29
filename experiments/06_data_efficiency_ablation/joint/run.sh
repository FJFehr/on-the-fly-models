#!/usr/bin/env bash
# Experiment 6, joint arm: one direct model trained on all 14 categories (td and notd), sized
# to the hypernetwork's parameter budget, at every data level in configs/cell_*.yaml, 5 seeds
# each. Runs land in outputs/06_data_efficiency_ablation_joint/. See README.md.
#
# Usage (GPUS, SHARD, PROJECT, SEEDS_OVERRIDE and PYTHON work as in scripts/run_config.sh):
#   GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/joint/run.sh

set -uo pipefail
cd "$(dirname "$0")/../../.."

CFG_DIR=experiments/06_data_efficiency_ablation/joint/configs \
CELL_GLOB="${CELL_GLOB:-cell_*.yaml}" \
PROJECT="${PROJECT:-06_data_efficiency_ablation_joint}" \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3 4 5}" \
bash scripts/run_config.sh
