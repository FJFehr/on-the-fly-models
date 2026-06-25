#!/usr/bin/env bash
# Run arc1d_recursion_ablation_large across seeds 1, 2, 3 (216 jobs: 3 seeds × 72 configs).
#
# Usage:
#   bash scripts/run_ablation_arc1d_recursion_ablation_large.sh           # 8 GPUs
#   bash scripts/run_ablation_arc1d_recursion_ablation_large.sh 4         # 4 GPUs
#   bash scripts/run_ablation_arc1d_recursion_ablation_large.sh 1         # sequential, single GPU

set -uo pipefail

N_GPUS=${1:-8}
LOG_DIR="logs/arc1d_recursion_ablation_large"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3)
CFG_DIR="configs/experiments/arc1d_recursion_ablation_large"

JOBS=()
for SEED in "${SEEDS[@]}"; do
    PROJECT="arc1d_recursion_ablation_large_s${SEED}"
    while IFS= read -r cfg; do
        JOBS+=("${cfg}|${SEED}|${PROJECT}")
    done < <(find "$CFG_DIR" -mindepth 2 -maxdepth 2 -name "*.yaml"                   ! -path "*/overfit/*" | sort)
done

N_JOBS=${#JOBS[@]}
echo "Launching $N_JOBS jobs across $N_GPUS GPUs (~$((N_JOBS / N_GPUS)) jobs/GPU)"
echo "Logs: $LOG_DIR/"
echo ""

for gpu in $(seq 0 $((N_GPUS - 1))); do
    sleep $((gpu * 5))
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        for i in $(seq "$gpu" "$N_GPUS" $((N_JOBS - 1))); do
            IFS='|' read -r cfg seed project <<< "${JOBS[$i]}"
            task=$(basename "$(dirname "$cfg")")
            name=$(basename "$cfg" .yaml)
            log="${LOG_DIR}/${project}_${task}_${name}.log"

            echo "[GPU $gpu] START  ${project} / ${task}/${name}"
            if .venv/bin/python train.py --config "$cfg" seed="$seed" project_name="$project"                 > "$log" 2>&1; then
                echo "[GPU $gpu] DONE   ${project} / ${task}/${name}"
            else
                echo "[GPU $gpu] FAILED ${project} / ${task}/${name}  (see $log)"
            fi
        done
    ) &
done

wait
echo ""
echo "All $N_JOBS jobs finished. Logs in $LOG_DIR/"
