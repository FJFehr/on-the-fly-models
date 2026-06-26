#!/usr/bin/env bash
# Run arc1d_recursion_ablation_deep across seeds 1, 2, 3 (216 jobs: 3 seeds × 72 configs).
#
# Conditions (all 256h, 8k equivalent steps):
#   A: Transformer, 4 layers, direct, 8000 steps
#   B: RecursiveTransformer, 4 loops, direct, 8000 steps
#   C: Transformer, 4 layers, N_supervision=8, lr=0.000125, 1000 steps
#   D: RecursiveTransformer, 8 loops, N_supervision=8, lr=0.000125, 1000 steps
#
# Usage:
#   bash scripts/run_ablation_arc1d_recursion_ablation_deep.sh       # 8 GPUs
#   bash scripts/run_ablation_arc1d_recursion_ablation_deep.sh 4     # 4 GPUs
#   bash scripts/run_ablation_arc1d_recursion_ablation_deep.sh 1     # single GPU

set -uo pipefail

N_GPUS=${1:-8}
PROJECT="arc1d_recursion_ablation_deep"
LOG_DIR="logs/arc1d_recursion_ablation_deep"
CFG_DIR="configs/experiments/arc1d_recursion_ablation_deep"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3)

JOBS=()
for SEED in "${SEEDS[@]}"; do
    while IFS= read -r cfg; do
        JOBS+=("${cfg}|${SEED}")
    done < <(find "$CFG_DIR" -mindepth 2 -maxdepth 2 -name "*.yaml" \
                  ! -path "*/overfit/*" | sort)
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
