#!/usr/bin/env bash
# Full reproduction script: runs ALL rope ablation experiments at 5 seeds each.
# Skips any individual run that already has outputs/<project>/<exp_name>/results.txt,
# so it is safe to re-run after partial completion or interruption.
#
# Experiments (roughly chronological / story order):
#   arc1d_recursion_ablation_large_8k  — plain transformer baseline (S1, S2)
#   arc1d_rope_sandwich_ablation       — RoPE vs sin PE, Canon ABCD, dim ∈ {16,32}
#   arc1d_rope_wide_middle_ablation    — wide looped middle, outer ∈ {8,16} × n_loops ∈ {4,8,16}
#   arc1d_rope_skip_ablation           — block highway skip (Canon subset ablation)
#   arc1d_rope_loop_skip_ablation      — per-iteration loop h0 injection
#   arc1d_rope_dim_ablation            — inner/outer dim sweep
#   arc1d_rope_unet_skip_ablation      — U-Net style single bypass connections
#   arc1d_rope_story_ablation          — story conditions SC1/SC2 (Canon-plain, Canon+RoPE-flat)
#                                         + legacy reference conditions S3/S4/S5 (no longer plotted)
#
# Total from scratch: ~1904 jobs across all experiments × 5 seeds.
#
# Usage:
#   bash scripts/run_all_ablations_5seeds.sh        # 8 GPUs
#   bash scripts/run_all_ablations_5seeds.sh 4
#   bash scripts/run_all_ablations_5seeds.sh 1

set -uo pipefail

N_GPUS=${1:-8}
LOG_DIR="logs/run_all_ablations_5seeds"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Experiment registry: (PROJECT CFG_DIR) — all run seeds 1–5
# ---------------------------------------------------------------------------
declare -a EXPERIMENTS
EXPERIMENTS=(
    "arc1d_recursion_ablation_large_8k  configs/experiments/arc1d_recursion_ablation_large_8k"
    "arc1d_rope_sandwich_ablation       configs/experiments/arc1d_rope_sandwich_ablation"
    "arc1d_rope_wide_middle_ablation    configs/experiments/arc1d_rope_wide_middle_ablation"
    "arc1d_rope_skip_ablation           configs/experiments/arc1d_rope_skip_ablation"
    "arc1d_rope_loop_skip_ablation      configs/experiments/arc1d_rope_loop_skip_ablation"
    "arc1d_rope_dim_ablation            configs/experiments/arc1d_rope_dim_ablation"
    "arc1d_rope_unet_skip_ablation      configs/experiments/arc1d_rope_unet_skip_ablation"
    "arc1d_rope_story_ablation          configs/experiments/arc1d_rope_story_ablation"
)
SEEDS=(1 2 3 4 5)

# ---------------------------------------------------------------------------
# Build global job list, skipping already-completed runs
# ---------------------------------------------------------------------------
JOBS=()
SKIPPED=0

for entry in "${EXPERIMENTS[@]}"; do
    read -r project cfg_dir <<< "$entry"

    for seed in "${SEEDS[@]}"; do
        while IFS= read -r cfg; do
            logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
            exp_name="${logging_name}_seed${seed}"
            results_file="outputs/${project}/${exp_name}/results.txt"

            if [ -f "$results_file" ]; then
                (( SKIPPED++ )) || true
                continue
            fi
            JOBS+=("${project}|${cfg}|${seed}")
        done < <(find "$cfg_dir" -mindepth 2 -maxdepth 2 -name "*.yaml" \
                     ! -name "base_*" ! -path "*/overfit/*" | sort)
    done
done

N_JOBS=${#JOBS[@]}
echo "Total jobs queued : $N_JOBS"
echo "Already complete  : $SKIPPED (skipped)"
echo "GPUs              : $N_GPUS  (~$((N_JOBS / N_GPUS + 1)) jobs/GPU)"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run — all experiments already at 5 seeds."
    exit 0
fi

# ---------------------------------------------------------------------------
# Dispatch across GPUs
# ---------------------------------------------------------------------------
for gpu in $(seq 0 $((N_GPUS - 1))); do
    sleep $((gpu * 5))
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        for i in $(seq "$gpu" "$N_GPUS" $((N_JOBS - 1))); do
            IFS='|' read -r project cfg seed <<< "${JOBS[$i]}"
            logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
            exp_name="${logging_name}_seed${seed}"
            log="${LOG_DIR}/${project}__${exp_name}.log"

            echo "[GPU $gpu] START  ${project} / ${exp_name}"
            if .venv/bin/python train.py --config "$cfg" \
                seed="$seed" \
                project_name="$project" \
                experiment_name="${exp_name}" \
                logging_name="${logging_name}" \
                > "$log" 2>&1; then
                echo "[GPU $gpu] DONE   ${project} / ${exp_name}"
            else
                echo "[GPU $gpu] FAILED ${project} / ${exp_name}  (see $log)"
            fi
        done
    ) &
done

wait
echo ""
echo "All $N_JOBS jobs finished. Logs in $LOG_DIR/"
