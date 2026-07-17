#!/usr/bin/env bash
# Run all 6 arc1d_hypermodel_looped_rope_canon_capacity_baseline configs (num_layers x {4,8}
# x task-descriptor x {on,off}, plus 2 longer-training arms for the no-descriptor case),
# 3 seeds each (18 jobs total). 15-task list, gradient_clip_val=10 fixed.
#
# Each run uses every GPU with <500MB used on the node (configs set devices: auto; train.py
# is invoked with --free-gpus so resolve_free_gpus restricts to actually-idle GPUs rather
# than grabbing all 8 regardless of other users' jobs, which is train.py's default without
# that flag). Since each run claims every free GPU, jobs run one after another, not in
# parallel. Skips already-completed runs.
#
# To split this across multiple nodes without clashing (nodes don't share a filesystem in
# general, though torrnode7/12/13 happen to share NFS home directories here, so the
# skip-logic below coordinates across those automatically once one node writes a
# results.txt), override CELL_GLOB to give each node a disjoint slice, e.g.:
#   node A: CELL_GLOB="arm_*_td.yaml"   bash scripts/run_hypermodel_looped_rope_canon_capacity_baseline.sh
#   node B: CELL_GLOB="arm_*_notd*.yaml" bash scripts/run_hypermodel_looped_rope_canon_capacity_baseline.sh
#
# Usage:
#   bash scripts/run_hypermodel_looped_rope_canon_capacity_baseline.sh

set -uo pipefail

PROJECT="arc1d_hypermodel_looped_rope_canon_capacity_baseline"
LOG_DIR="logs/arc1d_hypermodel_looped_rope_canon_capacity_baseline"
CFG_DIR="configs/experiments/arc1d_hypermodel_looped_rope_canon_capacity_baseline"
CELL_GLOB="${CELL_GLOB:-arm_*.yaml}"
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
    if .venv/bin/python train.py --config "$cfg" --free-gpus \
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
