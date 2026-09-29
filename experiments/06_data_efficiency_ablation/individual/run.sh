#!/usr/bin/env bash
# Experiment 6, individual arm: one small direct model per task category (dim=4, the
# hypernetwork's target size), at every data level in configs/<category>/*.yaml, 5 seeds each.
# Runs land in outputs/06_data_efficiency_ablation_individual/. See README.md.
#
# Usage (GPUS, SHARD, PROJECT, SEEDS_OVERRIDE and PYTHON work as in scripts/run_config.sh):
#   GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/individual/run.sh
#   CATEGORY=1d_fill bash experiments/06_data_efficiency_ablation/individual/run.sh

set -uo pipefail
cd "$(dirname "$0")/../../.."

CFG_DIR="experiments/06_data_efficiency_ablation/individual/configs/${CATEGORY:-}" \
PROJECT="${PROJECT:-06_data_efficiency_ablation_individual}" \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3 4 5}" \
bash scripts/run_config.sh
