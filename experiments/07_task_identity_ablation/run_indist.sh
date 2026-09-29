#!/usr/bin/env bash
# Experiment 7, in-distribution: the 3 new task-identity arms (learnedtd_latent, frozentd_input,
# learnedtd_input) x 5 seeds = 15 jobs, on experiment 2's dim-4 recipe. notd and frozentd_latent
# are experiment 2's own runs (same recipe, same seeds), so they are not rerun here.
# Runs land in outputs/07_task_identity_ablation/indist_<arm>_seed<N>/. See README.md.
#
# Usage (GPUS, JOBS_PER_GPU, SHARD, PROJECT and PYTHON work as in scripts/run_config.sh):
#   GPUS=0,1,2 bash experiments/07_task_identity_ablation/run_indist.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

CFG_DIR=experiments/07_task_identity_ablation/configs/indist \
PROJECT="${PROJECT:-07_task_identity_ablation}" \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3 4 5}" \
bash scripts/run_config.sh
