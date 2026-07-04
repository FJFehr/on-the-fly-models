#!/usr/bin/env bash
# Run arc1d_rope_story_ablation across seeds 1–5.
# Max 1020 jobs: 12 conditions × 17 tasks × 5 seeds (skips already-completed runs).
#
# Twelve conditions. SC1/SC2 feed the 8-step story narrative (steps S3/S4 —
# Canon ABCD, then RoPE); SC3-SC9 are diagnostics (not part of the plotted
# narrative) testing whether the wide-middle "sandwich" actually helps once
# skip connections are added, and whether that's capacity or the outer/inner
# bottleneck structure itself; S3/S4/S5 are the original no-Canon variants,
# kept as untouched reference data and no longer plotted by
# scripts/plot_story_ablation.py:
#   SC1: Canon ABCD added onto the plain N_sup=4 transformer (dim=512, no RoPE)
#   SC2: + RoPE, flat (dim=16, n_loops=1, Canon carries over from SC1)
#   SC3: [diagnostic] SC2 + block skip, still n_loops=1 — isolates block skip
#        from looping/wide-middle, since S5/S6 showed no gain over S4
#   SC4: [diagnostic] uniform dim=16, n_loops=4, + block skip, no wide middle
#        — S7 (skip_abcd_hw) without the wide middle
#   SC5: [diagnostic] uniform dim=16, n_loops=4, + block skip + loop skip,
#        no wide middle — S8 (loop_skip_f4) without the wide middle
#   SC6: [diagnostic] uniform dim=20, n_loops=4, no skip — param-matched
#        control for SC7/S6 (17,660 vs S6's 16,672 backbone params)
#   SC7: [diagnostic] SC6 + block skip — param-matched to S7 (skip_abcd_hw)
#   SC8: [diagnostic] uniform dim=36, n_loops=4, no skip — param-matched
#        control for SC9/cond J of arc1d_rope_dim_ablation (53,100 vs
#        55,552 backbone params)
#   SC9: [diagnostic] SC8 + block skip — param-matched to cond K of
#        arc1d_rope_dim_ablation (note: K also has loop skip)
#   S3:  [reference] Flat 3L RoPE transformer, no Canon (n_loops=1, dim=16)
#   S4:  [reference] Looped middle, no Canon (n_loops=4, dim=16)
#   S5:  [reference] Wide middle, no Canon (outer=8, inner=32, n_loops=4)
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
