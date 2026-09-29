#!/usr/bin/env bash
# Experiment 5: each of the 14 base task categories held out of training in turn,
# x {notd, frozentd}, x 5 seeds = 140 jobs. Seeds and run names are written into each config
# (by gen_configs.py), so each config runs once as-is (SEEDS_OVERRIDE=config).
#
# No separate eval step: val_task_categories includes the held-out category, so its zero-shot
# score lands in each run's results.txt (val_query_*_by_task_<held_out_category>).
# Runs land in outputs/05_leave_one_out_task_generalization/. See README.md.
#
# Usage (GPUS, SHARD, PROJECT and PYTHON work as in scripts/run_config.sh):
#   GPUS=0,1,2,3 bash experiments/05_leave_one_out_task_generalization/run.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

CFG_DIR=experiments/05_leave_one_out_task_generalization/configs \
CELL_GLOB="*_seed*.yaml" \
PROJECT="${PROJECT:-05_leave_one_out_task_generalization}" \
SEEDS_OVERRIDE=config \
bash scripts/run_config.sh
