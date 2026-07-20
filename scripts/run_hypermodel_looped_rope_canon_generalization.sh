#!/usr/bin/env bash
# Run all 15 arc1d_hypermodel_looped_rope_canon_generalization arms (5 held-out task
# categories x td/notd/frozentd), 3 seeds each (45 jobs). Zhu backbone + AdamW/lr=1e-3 fixed
# across every arm -- the only axis under test is task-identity conditioning, evaluated on a
# task category held out of training entirely. See README.md for the grid and how to read
# results.txt's per-category breakdown.
#
# Each run uses ALL GPUs on the node by default (configs set devices: auto), including any
# already in use by other users' jobs -- CUDA compute is time-sliced/shared fine as long as
# there's free memory. Set FREE_GPUS_FLAG="--free-gpus" to instead restrict to GPUs with
# <500MB used, if a shared node's other jobs are memory-heavy enough that sharing risks OOM.
# Since each run claims every (free or shared) GPU, jobs run one after another, not in
# parallel. Skips already-completed runs (idempotent to rerun).
#
# To split this across multiple nodes without clashing, override CELL_GLOB and/or
# SEEDS_OVERRIDE to give each node a disjoint slice, e.g. split by seed:
#   node A: SEEDS_OVERRIDE="1 2" bash scripts/run_hypermodel_looped_rope_canon_generalization.sh
#   node B: SEEDS_OVERRIDE="3"   bash scripts/run_hypermodel_looped_rope_canon_generalization.sh
# or by held-out task (one glob per invocation; run the script multiple times with different
# CELL_GLOB values on different nodes to cover all 5 held-out tasks):
#   node A: CELL_GLOB="arm_move2p_*.yaml"      bash scripts/run_hypermodel_looped_rope_canon_generalization.sh
#   node B: CELL_GLOB="arm_denoisingmc_*.yaml" bash scripts/run_hypermodel_looped_rope_canon_generalization.sh
#
# Usage:
#   bash scripts/run_hypermodel_looped_rope_canon_generalization.sh

set -uo pipefail

PROJECT="arc1d_hypermodel_looped_rope_canon_generalization"
LOG_DIR="logs/arc1d_hypermodel_looped_rope_canon_generalization"
CFG_DIR="configs/experiments/arc1d_hypermodel_looped_rope_canon_generalization"
CELL_GLOB="${CELL_GLOB:-arm_*.yaml}"
FREE_GPUS_FLAG="${FREE_GPUS_FLAG:-}"
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3}"
mkdir -p "$LOG_DIR"

read -ra SEEDS <<< "$SEEDS_OVERRIDE"

JOBS=()
SKIPPED=0
for SEED in "${SEEDS[@]}"; do
    while IFS= read -r cfg; do
        logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
        exp_name="${logging_name}_seed${SEED}"
        results_file="outputs/${PROJECT}/${exp_name}/results.txt"

        if [ -f "$results_file" ]; then
            (( SKIPPED++ )) || true
            continue
        fi
        JOBS+=("${cfg}|${SEED}")
    done < <(find "$CFG_DIR" -maxdepth 1 -name "$CELL_GLOB" | sort)
done

N_JOBS=${#JOBS[@]}
echo "Running $N_JOBS jobs sequentially (each using all GPUs)"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/  |  CELL_GLOB: $CELL_GLOB  |  SEEDS: ${SEEDS[*]}"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run, everything already complete."
    exit 0
fi

for job in "${JOBS[@]}"; do
    IFS='|' read -r cfg seed <<< "$job"
    logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    exp_name="${logging_name}_seed${seed}"
    log="${LOG_DIR}/${exp_name}.log"

    echo "START  ${PROJECT} / ${exp_name}"
    if .venv/bin/python train.py --config "$cfg" $FREE_GPUS_FLAG \
        seed="$seed" \
        project_name="$PROJECT" \
        experiment_name="${exp_name}" \
        logging_name="${logging_name}" \
        > "$log" 2>&1; then
        echo "DONE   ${PROJECT} / ${exp_name}"
    else
        echo "FAILED ${PROJECT} / ${exp_name}  (see $log)"
    fi
done

echo ""
echo "All $N_JOBS jobs finished. Logs in $LOG_DIR/"
