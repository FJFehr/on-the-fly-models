#!/usr/bin/env bash
# Run arc1d_rope_story_ablation across seeds 1–5 (255 jobs: 3 conditions × 17 tasks × 5 seeds).
#
# Three new conditions that fill the story between the plain transformer and the full model:
#   S3: Flat 3L RoPE transformer (n_loops=1, dim=16, no Canon)
#   S4: Looped middle (n_loops=4, dim=16, no Canon)
#   S5: Wide middle (outer=8, inner=32, n_loops=4, no Canon)
#
# Note: 1d_padded_fill is excluded from this experiment.
#
# Usage:
#   bash scripts/run_ablation_arc1d_rope_story.sh       # 8 GPUs
#   bash scripts/run_ablation_arc1d_rope_story.sh 4
#   bash scripts/run_ablation_arc1d_rope_story.sh 1

set -uo pipefail

N_GPUS=${1:-8}
PROJECT="arc1d_rope_story_ablation"
LOG_DIR="logs/arc1d_rope_story_ablation"
CFG_DIR="configs/experiments/arc1d_rope_story_ablation"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3 4 5)

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
