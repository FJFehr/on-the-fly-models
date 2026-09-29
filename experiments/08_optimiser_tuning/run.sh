#!/usr/bin/env bash
# Experiment 8: the Muon grid (60 configs) and the plain-AdamW grid (15 configs), 3 seeds each,
# 225 jobs. Runs land in outputs/08_optimiser_tuning/<config name>_seed<N>/. See README.md.
#
# Usage (GPUS, JOBS_PER_GPU, SHARD, PROJECT and PYTHON work as in scripts/run_config.sh):
#   GPUS=0,1,2,3 JOBS_PER_GPU=2 bash experiments/08_optimiser_tuning/run.sh
#   ARM=adamw bash experiments/08_optimiser_tuning/run.sh     # one arm only (muon or adamw)

set -uo pipefail
cd "$(dirname "$0")/../.."

CFG_DIR="experiments/08_optimiser_tuning/configs${ARM:+/$ARM}" \
PROJECT="${PROJECT:-08_optimiser_tuning}" \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3}" \
bash scripts/run_config.sh
