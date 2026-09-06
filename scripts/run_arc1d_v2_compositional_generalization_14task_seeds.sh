#!/usr/bin/env bash
# Run the 5-seed sweep of arc1d_v2_compositional_generalization_14task's 2 arms
# (notd/frozen_td) -- see gen_arc1d_v2_compositional_generalization_14task_seeds.py for how the
# 10 <arm>_seed<N>.yaml leaf configs were generated (seed baked into each file, not passed via
# CLI override -- scripts/eval_compositional_holdout.py has no dotlist-override support, unlike
# train.py, so it needs each seed's own experiment_name/output_path in the config file itself).
#
# Trains all 10 (skipping already-completed ones via results.txt), then zero-shot-evaluates each
# against the shared held-out compositional set (data/arc_1d_compositional_holdout).
#
# Usage (single node, contiguous GPU indices 0..N-1):
#   bash scripts/run_arc1d_v2_compositional_generalization_14task_seeds.sh       # 8 GPUs (default)
#   bash scripts/run_arc1d_v2_compositional_generalization_14task_seeds.sh 4     # 4 GPUs
#
# Set GPU_LIST to explicit physical GPU indices instead (e.g. GPU_LIST="0 1 2" for a 3-GPU node).
# Designed to co-locate with other jobs already on the node's GPUs (tiny model, ~10k total
# hypernetwork params, batch_size=512).

set -uo pipefail

N_GPUS_ARG=${1:-8}
GPU_LIST=${GPU_LIST:-}

if [ -z "$GPU_LIST" ]; then
    GPU_LIST=$(seq 0 $((N_GPUS_ARG - 1)))
fi

PROJECT="arc1d_v2_compositional_generalization_14task"
LOG_DIR="logs/arc1d_v2_compositional_generalization_14task"
CFG_DIR="configs/experiments/arc1d_v2_compositional_generalization_14task"
mkdir -p "$LOG_DIR"

echo "Building the held-out compositional dataset if it doesn't already exist..."
if [ ! -d "data/arc_1d_compositional_holdout" ]; then
    PYTHONPATH=. .venv/bin/python scripts/build_arc1d_compositional.py
fi

# --- Phase 1: train all 10 (arm, seed) cells, round-robin across GPU_LIST ---
JOBS=()
SKIPPED=0
while IFS= read -r cfg; do
    exp_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    results_file="outputs/${PROJECT}/${exp_name}/results.txt"
    if [ -f "$results_file" ]; then
        (( SKIPPED++ )) || true
        continue
    fi
    JOBS+=("$cfg")
done < <(find "$CFG_DIR" -maxdepth 1 -name "*_seed*.yaml" | sort)

N_JOBS=${#JOBS[@]}
N_LOCAL_GPUS=$(echo "$GPU_LIST" | wc -w)
echo "Training: $N_JOBS jobs to run across $N_LOCAL_GPUS GPUs ($SKIPPED already done)"

if [ "$N_JOBS" -gt 0 ]; then
    slot_idx=0
    for gpu in $GPU_LIST; do
        sleep $((slot_idx * 5))
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            for i in $(seq "$slot_idx" "$N_LOCAL_GPUS" $((N_JOBS - 1))); do
                cfg="${JOBS[$i]}"
                exp_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
                log="${LOG_DIR}/${exp_name}.log"
                echo "[GPU $gpu] TRAIN  ${PROJECT} / ${exp_name}"
                if .venv/bin/python train.py --config "$cfg" > "$log" 2>&1; then
                    echo "[GPU $gpu] DONE   ${PROJECT} / ${exp_name}"
                else
                    echo "[GPU $gpu] FAILED ${PROJECT} / ${exp_name}  (see $log)"
                fi
            done
        ) &
        slot_idx=$((slot_idx + 1))
    done
    wait
fi

# --- Phase 2: zero-shot eval each trained arm/seed on the held-out compositional set ---
echo ""
echo "Training phase done. Running held-out compositional eval for all 10 cells..."
while IFS= read -r cfg; do
    exp_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    eval_dir="outputs/${PROJECT}/${exp_name}/compositional_holdout_eval"
    log="${LOG_DIR}/${exp_name}_holdout_eval.log"

    if [ -f "${eval_dir}/results.txt" ]; then
        echo "SKIP   held-out eval / ${exp_name} (already evaluated)"
        continue
    fi

    echo "EVAL   held-out compositional set / ${exp_name}"
    if PYTHONPATH=. .venv/bin/python scripts/eval_compositional_holdout.py --config "$cfg" > "$log" 2>&1; then
        echo "DONE   held-out eval / ${exp_name}"
    else
        echo "FAILED held-out eval / ${exp_name}  (see $log)"
    fi
done < <(find "$CFG_DIR" -maxdepth 1 -name "*_seed*.yaml" | sort)

echo ""
echo "All done. Logs in ${LOG_DIR}/, held-out eval results under each cell's"
echo "outputs/${PROJECT}/<experiment_name>/compositional_holdout_eval/results.txt"
