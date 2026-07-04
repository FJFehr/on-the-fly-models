#!/usr/bin/env bash
# Run arc1d_uniform_ablation across seeds 1-5.
# Max 1190 jobs: 7 steps x 2 dims x 17 tasks x 5 seeds (skips already-completed runs).
#
# Clean, single-project rebuild of the ARC-1D design-journey story, run at two
# widths (dim=16, dim=32), uniform throughout - replaces the scattered legacy
# projects (arc1d_rope_story_ablation's SC1-SC10, arc1d_32_canon_ablation,
# arc1d_rope_sandwich_ablation, etc.) with one fully self-contained, from-scratch
# reproduction:
#   T1: Vanilla transformer, sin PE, N_sup=1
#   T2: + N_sup=2 loop training
#   T3: + Canon ABCD
#   T4: + RoPE, flat (n_loops=1)
#   T5: + Looped middle (n_loops=4)
#   T6: + Block skip
#   T7: + Per-loop h0 (loop skip)
#
# Note: 1d_padded_fill is excluded from this experiment.
#
# Usage:
#   bash scripts/run_ablation_arc1d_uniform_ablation.sh       # 8 GPUs
#   bash scripts/run_ablation_arc1d_uniform_ablation.sh 4
#   bash scripts/run_ablation_arc1d_uniform_ablation.sh 1

set -uo pipefail

N_GPUS=${1:-8}
PROJECT="arc1d_uniform_ablation"
LOG_DIR="logs/arc1d_uniform_ablation"
CFG_DIR="configs/experiments/arc1d_uniform_ablation"
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
    done < <(find "$CFG_DIR" -mindepth 2 -maxdepth 2 -name "*.yaml" \
                  ! -name "base_*" ! -path "*/overfit/*" | sort)
done

N_JOBS=${#JOBS[@]}
echo "Launching $N_JOBS jobs across $N_GPUS GPUs (~$((N_JOBS / N_GPUS)) jobs/GPU)"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run — all conditions already at 5 seeds."
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
