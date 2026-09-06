#!/usr/bin/env bash
# Run all 18 arc1d_hypermodel_looped_rope_canon_grid configs (num_layers x {2,4,8} x
# gradient_clip_val x {5,10,20} x task-descriptor x {present,absent}), 3 seeds each
# (54 jobs total). See scripts/gen_hypermodel_looped_rope_canon_grid_configs.py for how the
# 18 leaf configs are generated.
#
# Each run uses ALL GPUs on the node (configs set devices: auto, resolved by
# train.py's resolve_free_gpus to every free GPU). Since each run claims the whole node,
# jobs run one after another, not in parallel. Skips already-completed runs.
#
# To split this across multiple nodes without clashing (nodes don't share a filesystem, so
# the skip-logic below can't coordinate across machines on its own), override CELL_GLOB to
# give each node a disjoint slice of the 18 configs, e.g. an even 9/9 split on the
# task-descriptor axis:
#   node A: CELL_GLOB="cell_*_notd.yaml" bash scripts/run_hypermodel_looped_rope_canon_grid.sh
#   node B: CELL_GLOB="cell_*_td.yaml"   bash scripts/run_hypermodel_looped_rope_canon_grid.sh
#
# Usage:
#   bash scripts/run_hypermodel_looped_rope_canon_grid.sh

set -uo pipefail

PROJECT="arc1d_hypermodel_looped_rope_canon_grid"
LOG_DIR="logs/arc1d_hypermodel_looped_rope_canon_grid"
CFG_DIR="configs/experiments/arc1d_hypermodel_looped_rope_canon_grid"
CELL_GLOB="${CELL_GLOB:-cell_*.yaml}"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3)

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
echo "Project: $PROJECT  |  Logs: $LOG_DIR/  |  CELL_GLOB: $CELL_GLOB"
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
    if .venv/bin/python train.py --config "$cfg" \
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
