#!/usr/bin/env bash
# Run all extended recursion ablation experiments: 4 variants × 3 seeds × 72 configs.
#
# Variants:
#   base      — original model (256h, 2/1 layers), 4k/1k steps
#   8k        — original model, 8k/2k steps
#   large     — 512h, 4/1 layers (A/C: 4 layers; B/D: 1 layer × 4 loops), 4k/1k steps
#   large_8k  — large model + 8k/2k steps
#
# Seeds: 1, 2, 3 for all variants.
# Output root: outputs/{variant}_s{seed}/  (e.g. outputs/arc1d_recursion_ablation_8k_s2/)
#
# Usage:
#   bash scripts/run_ablation_extended.sh           # 8 GPUs
#   bash scripts/run_ablation_extended.sh 4         # 4 GPUs
#   bash scripts/run_ablation_extended.sh 1         # sequential, single GPU
#
# Monitor a run:
#   tail -f logs/arc1d_recursion_ablation_extended/arc1d_recursion_ablation_large_s1_1d_flip_A_transformer.log
#
# Prerequisites: generate configs first with
#   uv run python scripts/gen_ablation_configs.py

set -uo pipefail

N_GPUS=${1:-8}
LOG_DIR="logs/arc1d_recursion_ablation_extended"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3)

# Variant suffix → config directory suffix (same string here)
VARIANT_SUFFIXES=("" "_8k" "_large" "_large_8k")

# ---------------------------------------------------------------------------
# Build flat JOBS array: each entry is "config_path|seed|project_name"
# ---------------------------------------------------------------------------
JOBS=()
for VSUFFIX in "${VARIANT_SUFFIXES[@]}"; do
    CFG_DIR="configs/experiments/arc1d_recursion_ablation${VSUFFIX}"
    BASE_PROJECT="arc1d_recursion_ablation${VSUFFIX}"

    if [[ ! -d "$CFG_DIR" ]]; then
        echo "WARNING: config dir not found: $CFG_DIR — skipping variant '${VSUFFIX}'"
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        PROJECT="${BASE_PROJECT}_s${SEED}"
        # Collect configs in the same order as run_ablation.sh (task/condition)
        while IFS= read -r cfg; do
            JOBS+=("${cfg}|${SEED}|${PROJECT}")
        done < <(find "$CFG_DIR" -mindepth 2 -maxdepth 2 -name "*.yaml" \
                      ! -path "*/overfit/*" | sort)
    done
done

N_JOBS=${#JOBS[@]}
if [[ $N_JOBS -eq 0 ]]; then
    echo "No jobs found. Run scripts/gen_ablation_configs.py first."
    exit 1
fi

echo "Launching $N_JOBS jobs across $N_GPUS GPUs (~$((N_JOBS / N_GPUS)) jobs/GPU)"
echo "Logs: $LOG_DIR/"
echo ""

# ---------------------------------------------------------------------------
# Dispatch: one subshell per GPU, each processes its round-robin slice
# ---------------------------------------------------------------------------
for gpu in $(seq 0 $((N_GPUS - 1))); do
    sleep $((gpu * 5))  # stagger startup to avoid simultaneous cold-starts
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        for i in $(seq "$gpu" "$N_GPUS" $((N_JOBS - 1))); do
            IFS='|' read -r cfg seed project <<< "${JOBS[$i]}"
            task=$(basename "$(dirname "$cfg")")
            name=$(basename "$cfg" .yaml)
            log="${LOG_DIR}/${project}_${task}_${name}.log"

            echo "[GPU $gpu] START  ${project} / ${task}/${name}"
            if .venv/bin/python train.py --config "$cfg" seed="$seed" project_name="$project" \
                > "$log" 2>&1; then
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
