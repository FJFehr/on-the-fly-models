#!/usr/bin/env bash
# Run arc1d_rope_dim_ablation across seeds 1, 2, 3 (324 jobs: 6 conditions × 18 tasks × 3 seeds).
#
# Sweeps inner_dim and skip config to find the smallest architecture solving all tasks.
# Conditions A and F are reused from previous experiments.
#
#   H: inner=16, outer=8,  no skip           →  6,448 params
#   I: inner=16, outer=8,  block+loop skip   →  6,448 params
#   J: inner=64, outer=8,  no skip           → 55,552 params
#   K: inner=64, outer=8,  block+loop skip   → 55,552 params
#   L: inner=32, outer=16, no skip           → 22,624 params
#   M: inner=32, outer=16, block+loop skip   → 22,624 params
#
# Usage:
#   bash scripts/run_ablation_arc1d_rope_dim.sh       # 8 GPUs
#   bash scripts/run_ablation_arc1d_rope_dim.sh 4
#   bash scripts/run_ablation_arc1d_rope_dim.sh 1

set -uo pipefail

N_GPUS=${1:-8}
PROJECT="arc1d_rope_dim_ablation"
LOG_DIR="logs/arc1d_rope_dim_ablation"
CFG_DIR="configs/experiments/arc1d_rope_dim_ablation"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3)

JOBS=()
for SEED in "${SEEDS[@]}"; do
    while IFS= read -r cfg; do
        JOBS+=("${cfg}|${SEED}")
    done < <(find "$CFG_DIR" -mindepth 2 -maxdepth 2 -name "*.yaml" \
                  ! -name "base_*" ! -path "*/overfit/*" | sort)
done

N_JOBS=${#JOBS[@]}
echo "Launching $N_JOBS jobs across $N_GPUS GPUs (~$((N_JOBS / N_GPUS)) jobs/GPU)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/"
echo ""

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
