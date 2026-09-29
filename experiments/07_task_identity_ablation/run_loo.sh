#!/usr/bin/env bash
# Experiment 7, leave-one-out: each of the 14 base task categories held out in turn, x the 4
# task-identity arms, x 5 seeds = 280 jobs, on experiment 5's recipe. The held-out category is scored with a
# zero task vector. Seeds and run names are written into each config (gen_loo_configs.py), so
# each config runs once as-is (SEEDS_OVERRIDE=config).
# Runs land in outputs/07_task_identity_ablation/loo_<category>_<arm>_seed<N>/. See README.md.
#
# Usage (GPUS, JOBS_PER_GPU, SHARD, PROJECT and PYTHON work as in scripts/run_config.sh):
#   GPUS=all bash experiments/07_task_identity_ablation/run_loo.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

CFG_DIR=experiments/07_task_identity_ablation/configs/loo \
CELL_GLOB="*_seed*.yaml" \
PROJECT="${PROJECT:-07_task_identity_ablation}" \
SEEDS_OVERRIDE=config \
bash scripts/run_config.sh
