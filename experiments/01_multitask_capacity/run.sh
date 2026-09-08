#!/usr/bin/env bash
# One command to reproduce the whole of experiment 1: the main Joint/
# Individual grid, and both ablations (Canon, optimizer) -- CFG_DIR recurses,
# so pointing it at this whole folder picks up configs/*.yaml,
# configs/individual/, configs/nocanon/, and configs/adamw/ together in one
# sweep. 5 seeds by default (193 leaf configs -> 965 jobs).
#
# Usage:
#   bash experiments/01_multitask_capacity/run.sh                        # sequential, 1 GPU
#   GPUS="0,1,2,3,4,5,6,7" bash experiments/01_multitask_capacity/run.sh  # 8-way parallel
#
# Wraps scripts/run_config.sh (the shared generic launcher) with this
# experiment's own CFG_DIR baked in -- no need to know that incantation to
# reproduce this experiment. Skips any (config, seed) pair that already has
# a results.txt, so it's always safe to rerun: backfill missing seeds,
# resume after an interruption, or run just the ablations you haven't done
# yet (already-complete arms are skipped automatically).
#
# Then plot directly from what this wrote to outputs/ -- see plot_all.py.

set -uo pipefail
cd "$(dirname "$0")/../.."

CFG_DIR=experiments/01_multitask_capacity \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3 4 5}" \
GPUS="${GPUS:-}" \
bash scripts/run_config.sh
