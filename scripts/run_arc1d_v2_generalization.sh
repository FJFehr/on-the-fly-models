#!/usr/bin/env bash
# Run all 10 arc1d_v2_generalization arms (5 held-out task categories x notd/frozentd), 3 seeds
# by default (30 jobs) -- the v2-scale (dim=4 target, 10,156-param matched-scale hypernetwork)
# rerun of arc1d_hypermodel_looped_rope_canon_generalization. Same architecture as
# arc1d_v2_compositional_generalization. See README.md for the grid and how to read
# results.txt's per-category breakdown.
#
# Each run claims 1 GPU (devices: 1 in base.yaml, not auto). Set FREE_GPUS_FLAG="--free-gpus" to
# restrict to a free GPU index on a shared node instead of whatever the accelerator resolves to:
#   FREE_GPUS_FLAG="--free-gpus" bash scripts/run_arc1d_v2_generalization.sh
#
# To split this across multiple nodes without clashing, override CELL_GLOB and/or SEEDS_OVERRIDE
# to give each node a disjoint slice, same convention as
# scripts/run_hypermodel_looped_rope_canon_generalization.sh:
#   node A: SEEDS_OVERRIDE="1 2" bash scripts/run_arc1d_v2_generalization.sh
#   node B: SEEDS_OVERRIDE="3"   bash scripts/run_arc1d_v2_generalization.sh
#
# Usage:
#   bash scripts/run_arc1d_v2_generalization.sh

set -uo pipefail

PROJECT="arc1d_v2_generalization"
LOG_DIR="logs/arc1d_v2_generalization"
CFG_DIR="configs/experiments/arc1d_v2_generalization"
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
echo "Running $N_JOBS jobs sequentially"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/  |  CELL_GLOB: $CELL_GLOB  |  SEEDS: ${SEEDS[*]}"
echo ""

for JOB in "${JOBS[@]}"; do
    cfg="${JOB%%|*}"
    seed="${JOB##*|}"
    logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    exp_name="${logging_name}_seed${seed}"
    log="${LOG_DIR}/${exp_name}.log"

    echo "START  ${PROJECT} / ${exp_name}"
    if .venv/bin/python train.py --config "$cfg" $FREE_GPUS_FLAG \
        seed="$seed" \
        project_name="$PROJECT" \
        experiment_name="$exp_name" \
        logging_name="$logging_name" \
        > "$log" 2>&1; then
        echo "DONE   ${PROJECT} / ${exp_name}"
    else
        echo "FAILED ${PROJECT} / ${exp_name}  (see $log)"
    fi
done

echo ""
echo "All ${N_JOBS} jobs finished. Logs in ${LOG_DIR}/"
