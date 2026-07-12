#!/usr/bin/env bash
# Run the arc1d_hypermodel_looped_lora_adapter rank sweep (r=1,2,4,8) across seeds 1-3.
#
# Each seed run uses ALL GPUs on the node (configs set devices: auto, resolved by
# train.py's resolve_free_gpus to every free GPU). Since each run claims the whole node,
# jobs run one after another, not in parallel -- there's nothing left to parallelise
# against once a single run is already spanning every GPU. Skips already-completed runs.
#
# base.yaml is excluded from the glob: it's a shared _base_ target for the rank configs,
# not a standalone experiment variant (unlike arc1d_hypermodel_looped_mix11, this project's
# base.yaml lives directly in this directory, since the config loader only resolves one
# level of _base_ inheritance -- see base.yaml's header comment).
#
# Usage:
#   bash scripts/run_hypermodel_looped_lora_adapter.sh

set -uo pipefail

PROJECT="arc1d_hypermodel_looped_lora_adapter"
LOG_DIR="logs/arc1d_hypermodel_looped_lora_adapter"
CFG_DIR="configs/experiments/arc1d_hypermodel_looped_lora_adapter"
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
    done < <(find "$CFG_DIR" -maxdepth 1 -name "*.yaml" ! -name "base.yaml" | sort)
done

N_JOBS=${#JOBS[@]}
echo "Running $N_JOBS jobs sequentially (each using all GPUs)"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run — everything already complete."
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
