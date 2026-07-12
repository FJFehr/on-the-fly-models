#!/usr/bin/env bash
# Run only the n2_loop4_noskip variant of the 11-task descriptor mix
# (arc1d_hypermodel_looped_mix11) across seeds 1-3.
#
# Each seed run uses ALL GPUs on the node (mix11_n2_loop4_noskip.yaml sets devices: auto,
# resolved by train.py's resolve_free_gpus to every free GPU). Since each run claims the
# whole node, seeds run one after another, not in parallel. Skips already-completed runs.
#
# Usage:
#   bash scripts/run_hypermodel_looped_mix11_n2_loop4_noskip.sh

set -uo pipefail

PROJECT="arc1d_hypermodel_looped_mix11"
LOG_DIR="logs/arc1d_hypermodel_looped_mix11"
CFG="configs/experiments/arc1d_hypermodel_looped_mix11/mix11_n2_loop4_noskip.yaml"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3)
logging_name=$(grep '^experiment_name:' "$CFG" | awk '{print $2}')

JOBS=()
SKIPPED=0
for SEED in "${SEEDS[@]}"; do
    exp_name="${logging_name}_seed${SEED}"
    results_file="outputs/${PROJECT}/${exp_name}/results.txt"

    if [ -f "$results_file" ]; then
        (( SKIPPED++ )) || true
        continue
    fi
    JOBS+=("${SEED}")
done

N_JOBS=${#JOBS[@]}
echo "Running $N_JOBS jobs sequentially (each using all GPUs)"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Config: $CFG  |  Logs: $LOG_DIR/"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run — all 3 seeds already complete."
    exit 0
fi

for seed in "${JOBS[@]}"; do
    exp_name="${logging_name}_seed${seed}"
    log="${LOG_DIR}/${exp_name}.log"

    echo "START  ${PROJECT} / ${exp_name}"
    if .venv/bin/python train.py --config "$CFG" \
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
