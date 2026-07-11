#!/usr/bin/env bash
# Run the 11-task descriptor mix (arc1d_hypermodel_looped_mix11) across seeds 1-3.
# One config, 3 seeds, each on its own GPU in parallel. Skips already-completed runs.
#
# Usage:
#   bash scripts/run_hypermodel_looped_mix11.sh       # up to 3 GPUs
#   bash scripts/run_hypermodel_looped_mix11.sh 3

set -uo pipefail

N_GPUS=${1:-3}
PROJECT="arc1d_hypermodel_looped_mix11"
LOG_DIR="logs/arc1d_hypermodel_looped_mix11"
CFG="configs/experiments/arc1d_hypermodel_looped_mix11/mix11_td.yaml"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3)

JOBS=()
SKIPPED=0
for SEED in "${SEEDS[@]}"; do
    logging_name=$(grep '^experiment_name:' "$CFG" | awk '{print $2}')
    exp_name="${logging_name}_seed${SEED}"
    results_file="outputs/${PROJECT}/${exp_name}/results.txt"

    if [ -f "$results_file" ]; then
        (( SKIPPED++ )) || true
        continue
    fi
    JOBS+=("${SEED}")
done

N_JOBS=${#JOBS[@]}
echo "Launching $N_JOBS jobs across $N_GPUS GPUs"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run — all 3 seeds already complete."
    exit 0
fi

for i in "${!JOBS[@]}"; do
    seed="${JOBS[$i]}"
    gpu=$(( i % N_GPUS ))
    logging_name=$(grep '^experiment_name:' "$CFG" | awk '{print $2}')
    exp_name="${logging_name}_seed${seed}"
    log="${LOG_DIR}/${exp_name}.log"

    (
        export CUDA_VISIBLE_DEVICES=$gpu
        echo "[GPU $gpu] START  ${PROJECT} / ${exp_name}"
        if .venv/bin/python train.py --config "$CFG" \
            seed="$seed" \
            project_name="$PROJECT" \
            experiment_name="${exp_name}" \
            logging_name="${logging_name}" \
            > "$log" 2>&1; then
            echo "[GPU $gpu] DONE   ${PROJECT} / ${exp_name}"
        else
            echo "[GPU $gpu] FAILED ${PROJECT} / ${exp_name}  (see $log)"
        fi
    ) &
done

wait
echo ""
echo "All $N_JOBS jobs finished. Logs in $LOG_DIR/"
