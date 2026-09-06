#!/usr/bin/env bash
# Run all 5 arc1d_hypermodel_looped_rope_canon_rank_sweep configs
# (rank1 / rank2 / rank4 / rank8 / rank8_no_canon), 5 seeds each (25 jobs total). 17 tasks,
# batch_size=512, attention pooling, encoder fixed to RoPE+Canon at num_layers=4
# (rank8_no_canon drops Canon only), gradient_clip_val=5.
#
# Each run uses ALL GPUs on the node (configs set devices: auto, resolved by
# train.py's resolve_free_gpus to every free GPU). Since each run claims the whole node,
# jobs run one after another, not in parallel. Skips already-completed runs.
#
# Usage:
#   bash scripts/run_hypermodel_looped_rope_canon_rank_sweep.sh

set -uo pipefail

PROJECT="arc1d_hypermodel_looped_rope_canon_rank_sweep"
LOG_DIR="logs/arc1d_hypermodel_looped_rope_canon_rank_sweep"
CFG_DIR="configs/experiments/arc1d_hypermodel_looped_rope_canon_rank_sweep"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3 4 5)

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
    done < <(find "$CFG_DIR" -maxdepth 1 -name "rank*.yaml" | sort)
done

N_JOBS=${#JOBS[@]}
echo "Running $N_JOBS jobs sequentially (each using all GPUs)"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/"
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
