#!/usr/bin/env bash
# Run arc1d_v2_backbone_capacity (Round 2, Phase 1) across seeds 1-3.
# Max 315 jobs: 7 steps x 15 tasks x 3 seeds (skips already-completed runs).
# Muon (muon_lr=0.005, muon_momentum=0.95), dim=16 only, no L1-L6 sweep -
# see docs/arc1d_story/05_round2_plan.md's Phase 1 for the full rationale.
#
# Usage:
#   bash scripts/run_v2_backbone_capacity.sh       # 8 GPUs (default)
#   bash scripts/run_v2_backbone_capacity.sh 4     # 4 GPUs
#
# Designed to co-locate with other jobs already on the node's GPUs (small
# model, ~11-40k backbone params, batch_size=256) - each GPU slot in this
# script only ever runs one of our jobs at a time, regardless of what else
# is already resident on that physical GPU.

set -uo pipefail

N_GPUS=${1:-8}
PROJECT="arc1d_v2_backbone_capacity"
LOG_DIR="logs/arc1d_v2_backbone_capacity"
CFG_DIR="configs/experiments/arc1d_v2_backbone_capacity"
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
    done < <(find "$CFG_DIR" -mindepth 2 -maxdepth 2 -name "*.yaml" ! -name "base_*" | sort)
done

N_JOBS=${#JOBS[@]}
echo "Launching $N_JOBS jobs across $N_GPUS GPUs (~$((N_JOBS / N_GPUS)) jobs/GPU)"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run - all conditions already at 3 seeds."
    exit 0
fi

for gpu in $(seq 0 $((N_GPUS - 1))); do
    sleep $((gpu * 5))
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        for i in $(seq "$gpu" "$N_GPUS" $((N_JOBS - 1))); do
            IFS='|' read -r cfg seed <<< "${JOBS[$i]}"
            logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
            exp_name="${logging_name}_seed${seed}"
            log="${LOG_DIR}/${exp_name}.log"

            echo "[GPU $gpu] START  ${PROJECT} / ${exp_name}"
            if .venv/bin/python train.py --config "$cfg" \
                seed="$seed" \
                project_name="$PROJECT" \
                experiment_name="${exp_name}" \
                logging_name="${logging_name}" \
                > "$log" 2>&1; then
                echo "[GPU $gpu] DONE   ${PROJECT} / ${exp_name}"
            else
                echo "[GPU $gpu] FAILED ${PROJECT} / ${exp_name}  (see $log)"
            fi
        done
    ) &
done

wait
echo ""
echo "All $N_JOBS jobs finished. Logs in $LOG_DIR/"
