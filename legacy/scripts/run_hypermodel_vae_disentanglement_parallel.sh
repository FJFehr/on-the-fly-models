#!/usr/bin/env bash
# Parallel launcher for arc1d_hypermodel_looped_rope_canon_vae_disentanglement: one train.py
# process per GPU (single-GPU, no DDP -- these models are small enough that one GPU each is
# plenty), rather than the sequential whole-node-per-job pattern in
# run_hypermodel_vae_disentanglement.sh. Meant for a node with several genuinely free GPUs
# and more jobs queued than the sequential script would clear quickly; queues any leftover
# jobs onto a GPU as soon as it frees up. Skips already-completed jobs (results.txt exists).
#
# Usage:
#   NUM_GPUS=8 CELL_GLOB="beta*_anneal.yaml" bash scripts/run_hypermodel_vae_disentanglement_parallel.sh

set -uo pipefail

PROJECT="arc1d_hypermodel_looped_rope_canon_muon_diag"
LOG_DIR="logs/arc1d_hypermodel_looped_rope_canon_vae_disentanglement"
CFG_DIR="configs/experiments/arc1d_hypermodel_looped_rope_canon_vae_disentanglement"
CELL_GLOB="${CELL_GLOB:-beta*_anneal.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
SEED=1
mkdir -p "$LOG_DIR"

mapfile -t CFGS < <(find "$CFG_DIR" -maxdepth 1 -name "$CELL_GLOB" | sort)

JOBS=()
for cfg in "${CFGS[@]}"; do
    logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    exp_name="${logging_name}_seed${SEED}"
    results_file="outputs/${PROJECT}/${exp_name}/results.txt"
    if [ -f "$results_file" ]; then
        echo "SKIP (already complete): ${exp_name}"
        continue
    fi
    JOBS+=("$cfg")
done

echo "Launching ${#JOBS[@]} jobs across ${NUM_GPUS} GPUs (one job per GPU, queued as slots free)"

declare -A GPU_PID
declare -A GPU_NAME

launch_on_gpu() {
    local gpu="$1" cfg="$2"
    local logging_name exp_name log
    logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    exp_name="${logging_name}_seed${SEED}"
    log="${LOG_DIR}/${exp_name}.log"
    echo "START  gpu=${gpu}  ${exp_name}"
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python train.py --config "$cfg" \
        seed="$SEED" \
        project_name="$PROJECT" \
        experiment_name="${exp_name}" \
        logging_name="${logging_name}" \
        devices=1 \
        accelerator=gpu \
        > "$log" 2>&1 &
    GPU_PID[$gpu]=$!
    GPU_NAME[$gpu]="$exp_name"
}

job_idx=0
for (( gpu=0; gpu<NUM_GPUS && job_idx<${#JOBS[@]}; gpu++ )); do
    launch_on_gpu "$gpu" "${JOBS[$job_idx]}"
    job_idx=$((job_idx+1))
done

while [ "$job_idx" -lt "${#JOBS[@]}" ] || [ "${#GPU_PID[@]}" -gt 0 ]; do
    for gpu in "${!GPU_PID[@]}"; do
        pid="${GPU_PID[$gpu]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            status=$?
            if [ "$status" -eq 0 ]; then
                echo "DONE   gpu=${gpu}  ${GPU_NAME[$gpu]}"
            else
                echo "FAILED gpu=${gpu}  ${GPU_NAME[$gpu]} (exit ${status})"
            fi
            unset 'GPU_PID[$gpu]'
            unset 'GPU_NAME[$gpu]'
            if [ "$job_idx" -lt "${#JOBS[@]}" ]; then
                launch_on_gpu "$gpu" "${JOBS[$job_idx]}"
                job_idx=$((job_idx+1))
            fi
        fi
    done
    sleep 5
done

echo "All jobs finished."
