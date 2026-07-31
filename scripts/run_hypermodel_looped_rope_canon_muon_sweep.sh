#!/usr/bin/env bash
# Run the arc1d_hypermodel_looped_rope_canon_muon_sweep grid: muon_lr x learning_rate (AdamW
# aux) x weight_decay x batch_size x muon_exclude_lora_heads, notd only, 1 seed each (216 jobs
# total by default). See README.md for the grid and how to read results, and
# scripts/gen_hypermodel_muon_sweep_configs.py for how the configs are generated.
#
# Each job uses a single GPU (base.yaml sets devices: 1). Set GPUS to a comma-separated list of
# GPU ids to run that many jobs in parallel, one per GPU, each pinned via
# CUDA_VISIBLE_DEVICES (e.g. GPUS="0,1,2,3" runs 4 at a time) -- with 216 jobs this is not
# optional in practice. Leave GPUS unset to run sequentially instead. Skips already-completed
# runs (idempotent to rerun).
#
# To split this across multiple nodes without clashing, override CELL_GLOB and/or
# SEEDS_OVERRIDE to give each node a disjoint slice, e.g. split by muon_lr:
#   node A: CELL_GLOB="arm_mlr0.008_*.yaml" bash scripts/run_hypermodel_looped_rope_canon_muon_sweep.sh
#   node B: CELL_GLOB="arm_mlr0.01_*.yaml"  bash scripts/run_hypermodel_looped_rope_canon_muon_sweep.sh
#
# Usage:
#   bash scripts/run_hypermodel_looped_rope_canon_muon_sweep.sh                  # sequential, 1 GPU
#   GPUS="0,1,2,3" bash scripts/run_hypermodel_looped_rope_canon_muon_sweep.sh    # 4-way parallel

set -uo pipefail

PROJECT="arc1d_hypermodel_looped_rope_canon_muon_sweep"
LOG_DIR="logs/arc1d_hypermodel_looped_rope_canon_muon_sweep"
CFG_DIR="configs/experiments/arc1d_hypermodel_looped_rope_canon_muon_sweep"
CELL_GLOB="${CELL_GLOB:-arm_*.yaml}"
FREE_GPUS_FLAG="${FREE_GPUS_FLAG:-}"
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1}"
GPUS="${GPUS:-}"
mkdir -p "$LOG_DIR"

read -ra SEEDS <<< "$SEEDS_OVERRIDE"

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
    done < <(find "$CFG_DIR" -maxdepth 1 -name "$CELL_GLOB" | sort)
done

N_JOBS=${#JOBS[@]}

GPU_IDS=()
if [ -n "$GPUS" ]; then
    IFS=',' read -ra GPU_IDS <<< "$GPUS"
fi
N_PARALLEL=${#GPU_IDS[@]}

if [ "$N_PARALLEL" -gt 0 ]; then
    echo "Running $N_JOBS jobs, ${N_PARALLEL}-way parallel across GPUs: ${GPU_IDS[*]}"
else
    echo "Running $N_JOBS jobs sequentially (each using 1 GPU)"
fi
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/  |  CELL_GLOB: $CELL_GLOB  |  SEEDS: ${SEEDS[*]}"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run, everything already complete."
    exit 0
fi

# Guards against launching onto a GPU whose memory hasn't actually been released yet (e.g. right
# after killing a previous session's process) -- CUDA_VISIBLE_DEVICES isolation is airtight once a
# job starts, but two jobs can still momentarily land on the same physical GPU if the prior
# occupant's memory hadn't been freed by the driver yet when this one claims that same index,
# guaranteeing an OOM once both ramp up. Polls up to 60s; proceeds with a warning if a GPU still
# looks occupied after that (it may be legitimately in shared use by another user's job).
wait_for_gpu_free() {
    local gpu_id="$1"
    [ -z "$gpu_id" ] && return 0
    local tries=0 used=""
    while [ "$tries" -lt 30 ]; do
        used=$(nvidia-smi --query-gpu=memory.used --id="$gpu_id" --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ -n "$used" ] && [ "$used" -lt 500 ]; then
            return 0
        fi
        sleep 2
        tries=$((tries + 1))
    done
    echo "WARNING: gpu ${gpu_id} still shows ${used:-unknown} MiB used after 60s -- proceeding anyway" >&2
}

run_job() {
    local cfg="$1" seed="$2" gpu_id="${3:-}"
    local logging_name exp_name log
    logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    exp_name="${logging_name}_seed${seed}"
    log="${LOG_DIR}/${exp_name}.log"
    local gpu_tag=""
    [ -n "$gpu_id" ] && gpu_tag="  (gpu ${gpu_id})"

    wait_for_gpu_free "$gpu_id"
    echo "START  ${PROJECT} / ${exp_name}${gpu_tag}"
    if CUDA_VISIBLE_DEVICES="$gpu_id" .venv/bin/python train.py --config "$cfg" $FREE_GPUS_FLAG \
        seed="$seed" \
        project_name="$PROJECT" \
        experiment_name="${exp_name}" \
        logging_name="${logging_name}" \
        > "$log" 2>&1; then
        echo "DONE   ${PROJECT} / ${exp_name}${gpu_tag}"
    else
        echo "FAILED ${PROJECT} / ${exp_name}${gpu_tag}  (see $log)"
    fi
}

if [ "$N_PARALLEL" -gt 0 ]; then
    declare -a SLOT_PIDS
    for ((s = 0; s < N_PARALLEL; s++)); do SLOT_PIDS[s]=""; done

    job_index=0
    for job in "${JOBS[@]}"; do
        IFS='|' read -r cfg seed <<< "$job"
        slot=$((job_index % N_PARALLEL))
        if [ -n "${SLOT_PIDS[$slot]:-}" ]; then
            wait "${SLOT_PIDS[$slot]}" 2>/dev/null || true
        fi
        gpu_id="${GPU_IDS[$slot]}"
        run_job "$cfg" "$seed" "$gpu_id" &
        SLOT_PIDS[$slot]=$!
        job_index=$((job_index + 1))
    done
    for pid in "${SLOT_PIDS[@]}"; do
        [ -n "$pid" ] && wait "$pid" 2>/dev/null
    done
else
    for job in "${JOBS[@]}"; do
        IFS='|' read -r cfg seed <<< "$job"
        run_job "$cfg" "$seed"
    done
fi

echo ""
echo "All $N_JOBS jobs finished. Logs in $LOG_DIR/"
