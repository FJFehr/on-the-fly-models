#!/usr/bin/env bash
# Run arc1d_hypermodel_looped's 17 individual-task configs (phase 1: no
# multi-task mixing, no task descriptor -- see configs/experiments/
# arc1d_hypermodel_looped/base_hypermodel_looped.yaml).
#
# 1d_padded_fill is excluded (see gen_hypermodel_looped_configs.py).
#
# Usage:
#   bash scripts/run_hypermodel_looped.sh       # 8 GPUs
#   bash scripts/run_hypermodel_looped.sh 4
#   bash scripts/run_hypermodel_looped.sh 1

set -uo pipefail

N_GPUS=${1:-8}
PROJECT="arc1d_hypermodel_looped"
LOG_DIR="logs/arc1d_hypermodel_looped"
CFG_DIR="configs/experiments/arc1d_hypermodel_looped"
mkdir -p "$LOG_DIR"

JOBS=()
SKIPPED=0
while IFS= read -r cfg; do
    logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    results_file="outputs/${PROJECT}/${logging_name}/results.txt"

    if [ -f "$results_file" ]; then
        (( SKIPPED++ )) || true
        continue
    fi
    JOBS+=("${cfg}")
done < <(find "$CFG_DIR" -mindepth 2 -maxdepth 2 -name "*.yaml" \
              ! -name "base_*" ! -path "*/overfit/*" | sort)

N_JOBS=${#JOBS[@]}
echo "Launching $N_JOBS jobs across $N_GPUS GPUs (~$((N_JOBS / N_GPUS)) jobs/GPU)"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run — all tasks already complete."
    exit 0
fi

for gpu in $(seq 0 $((N_GPUS - 1))); do
    sleep $((gpu * 5))
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        for i in $(seq "$gpu" "$N_GPUS" $((N_JOBS - 1))); do
            cfg="${JOBS[$i]}"
            logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
            log="${LOG_DIR}/${logging_name}.log"

            echo "[GPU $gpu] START  ${PROJECT} / ${logging_name}"
            if .venv/bin/python train.py --config "$cfg" \
                project_name="$PROJECT" \
                > "$log" 2>&1; then
                echo "[GPU $gpu] DONE   ${PROJECT} / ${logging_name}"
            else
                echo "[GPU $gpu] FAILED ${PROJECT} / ${logging_name}  (see $log)"
            fi
        done
    ) &
done

wait
echo ""
echo "All $N_JOBS jobs finished. Logs in $LOG_DIR/"
