#!/usr/bin/env bash
# Run all recursion ablation experiments distributed across N GPUs.
#
# Each GPU runs its assigned experiments sequentially (round-robin assignment).
# Stdout/stderr for each run is saved to logs/arc1d_recursion_ablation/.
#
# Usage:
#   bash scripts/run_ablation.sh           # all 8 GPUs
#   bash scripts/run_ablation.sh 4         # use first 4 GPUs only
#   bash scripts/run_ablation.sh 1         # sequential, single GPU
#
# Monitor a run:
#   tail -f logs/arc1d_recursion_ablation/1d_fill_D_looped_recursive_transformer.log

set -uo pipefail

N_GPUS=${1:-8}
BASE="configs/experiments/arc1d_recursion_ablation"
LOG_DIR="logs/arc1d_recursion_ablation"
mkdir -p "$LOG_DIR"

CONFIGS=(
  # 1d_denoising_1c
  "1d_denoising_1c/A_transformer.yaml"
  "1d_denoising_1c/B_recursive_transformer.yaml"
  "1d_denoising_1c/C_looped_transformer.yaml"
  "1d_denoising_1c/D_looped_recursive_transformer.yaml"
  # 1d_scale_dp
  "1d_scale_dp/A_transformer.yaml"
  "1d_scale_dp/B_recursive_transformer.yaml"
  "1d_scale_dp/C_looped_transformer.yaml"
  "1d_scale_dp/D_looped_recursive_transformer.yaml"
  # 1d_fill
  "1d_fill/A_transformer.yaml"
  "1d_fill/B_recursive_transformer.yaml"
  "1d_fill/C_looped_transformer.yaml"
  "1d_fill/D_looped_recursive_transformer.yaml"
  # 1d_recolor_cmp
  "1d_recolor_cmp/A_transformer.yaml"
  "1d_recolor_cmp/B_recursive_transformer.yaml"
  "1d_recolor_cmp/C_looped_transformer.yaml"
  "1d_recolor_cmp/D_looped_recursive_transformer.yaml"
  # 1d_recolor_cnt
  "1d_recolor_cnt/A_transformer.yaml"
  "1d_recolor_cnt/B_recursive_transformer.yaml"
  "1d_recolor_cnt/C_looped_transformer.yaml"
  "1d_recolor_cnt/D_looped_recursive_transformer.yaml"
  # 1d_recolor_oe
  "1d_recolor_oe/A_transformer.yaml"
  "1d_recolor_oe/B_recursive_transformer.yaml"
  "1d_recolor_oe/C_looped_transformer.yaml"
  "1d_recolor_oe/D_looped_recursive_transformer.yaml"
)

N_JOBS=${#CONFIGS[@]}
echo "Launching $N_JOBS experiments across $N_GPUS GPUs (~$((N_JOBS / N_GPUS)) jobs/GPU)"
echo "Logs: $LOG_DIR/"
echo ""

for gpu in $(seq 0 $((N_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    for i in $(seq "$gpu" "$N_GPUS" $((N_JOBS - 1))); do
      cfg="${CONFIGS[$i]}"
      task=$(dirname "$cfg" | xargs basename)
      name=$(basename "$cfg" .yaml)
      log="$LOG_DIR/${task}_${name}.log"
      echo "[GPU $gpu] START  $task/$name"
      if python train.py --config "$BASE/$cfg" > "$log" 2>&1; then
        echo "[GPU $gpu] DONE   $task/$name"
      else
        echo "[GPU $gpu] FAILED $task/$name  (see $log)"
      fi
    done
  ) &
done

wait
echo ""
echo "All $N_JOBS experiments finished. Logs in $LOG_DIR/"
