#!/usr/bin/env bash
# Experiment 6, hypernetwork arm: experiment 2's dim=4 hypernetwork (frozen_td and notd)
# trained on reduced training data, at every level in configs/cell_*.yaml, 5 seeds each.
# Runs land in outputs/06_data_efficiency_ablation_hypernetwork/. See README.md.
#
# Usage (GPUS, SHARD, PROJECT, SEEDS_OVERRIDE and PYTHON work as in scripts/run_config.sh):
#   GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/hypernetwork/run.sh

set -uo pipefail
cd "$(dirname "$0")/../../.."

CFG_DIR=experiments/06_data_efficiency_ablation/hypernetwork/configs \
CELL_GLOB="${CELL_GLOB:-cell_*.yaml}" \
PROJECT="${PROJECT:-06_data_efficiency_ablation_hypernetwork}" \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3 4 5}" \
bash scripts/run_config.sh
