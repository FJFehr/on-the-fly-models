#!/usr/bin/env bash
# Run all 9 arc1d_lowdata_lowrank cells (lora_adapter_rank in {1, 2, 4} x
# variants_per_base_task in {1, 2, 3} - see README.md), 3 seeds each (27 jobs total).
# Fixed architecture otherwise (Zhu backbone, frozen_td, Muon lr=0.005) - the two axes
# under test are LoRA adapter rank and training data amount, on all 15 tasks
# in-distribution. rank=8 is not re-run here; use arc1d_lowdata's cell_v1/v2/v3 results
# as the rank=8 reference column.
# Validation/test are fixed at 100 examples/category regardless of data level.
# See README.md for the hypothesis and how to read results.
#
# Each run uses ALL GPUs on the node by default (configs set devices: auto),
# including any already in use by other users' jobs -- CUDA compute is
# time-sliced/shared fine as long as there's free memory. Set
# FREE_GPUS_FLAG="--free-gpus" to instead restrict to GPUs with <500MB used.
# Since each run claims every (free or shared) GPU, jobs run one after
# another, not in parallel. Skips already-completed runs (idempotent to rerun).
#
# To split this across multiple nodes without clashing, override CELL_GLOB
# and/or SEEDS_OVERRIDE to give each node a disjoint slice, e.g. split by seed:
#   node A: SEEDS_OVERRIDE="1 2" bash scripts/run_lowdata_lowrank.sh
#   node B: SEEDS_OVERRIDE="3"   bash scripts/run_lowdata_lowrank.sh
# or by rank:
#   node A: CELL_GLOB="cell_r1_*.yaml" bash scripts/run_lowdata_lowrank.sh
#   node B: CELL_GLOB="cell_r2_*.yaml" bash scripts/run_lowdata_lowrank.sh
#
# Usage:
#   bash scripts/run_lowdata_lowrank.sh

set -uo pipefail

PROJECT="arc1d_lowdata_lowrank"
LOG_DIR="logs/arc1d_lowdata_lowrank"
CFG_DIR="configs/experiments/arc1d_lowdata_lowrank"
CELL_GLOB="${CELL_GLOB:-cell_*.yaml}"
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
