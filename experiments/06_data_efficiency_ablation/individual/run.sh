#!/usr/bin/env bash
# Run all arc1d_lowdata_baseline cells: 14 task categories x 4 data levels
# (variants_per_base_task in {1, 2, 3, full}, stratified/nested per base task
# - see README.md) x 5 seeds each (280 jobs total). No hypernetwork -- one
# LoopedSupervisedLightning model per (category, level, seed), trained
# directly on rope_canon_looped_transformer (identical architecture to
# arc1d_lowdata's target_model). Companion sweep to arc1d_lowdata: see its
# README.md for the fairness/asymmetry discussion.
#
# Each job is tiny (<=360 raw training pairs, hidden_dim=16) and uses a single
# GPU (devices: 1 in base.yaml). Set GPUS to a comma-separated list of GPU ids
# to run that many jobs in parallel, one per GPU, each pinned via
# CUDA_VISIBLE_DEVICES (e.g. GPUS="0,1,2" runs 3 at a time). Leave GPUS unset
# to run sequentially instead. Skips already-completed runs (idempotent to
# rerun).
#
# To split this across multiple nodes without clashing, override CATEGORY_GLOB
# and/or SEEDS_OVERRIDE to give each node a disjoint slice, e.g. split by seed:
#   node A: SEEDS_OVERRIDE="1 2 3" bash experiments/06_data_efficiency_ablation/individual/run.sh
#   node B: SEEDS_OVERRIDE="4 5"   bash experiments/06_data_efficiency_ablation/individual/run.sh
# or by category:
#   node A: CATEGORY_GLOB="1d_denoising_1c" bash experiments/06_data_efficiency_ablation/individual/run.sh
#   node B: CATEGORY_GLOB="1d_fill"         bash experiments/06_data_efficiency_ablation/individual/run.sh
#
# Usage:
#   bash experiments/06_data_efficiency_ablation/individual/run.sh                  # sequential, 1 GPU
#   GPUS="0,1,2" bash experiments/06_data_efficiency_ablation/individual/run.sh      # 3-way parallel

set -uo pipefail

PROJECT="arc1d_lowdata_baseline"
LOG_DIR="logs/arc1d_lowdata_baseline"
CFG_DIR="experiments/06_data_efficiency_ablation/individual/configs"
CATEGORY_GLOB="${CATEGORY_GLOB:-*}"
CELL_GLOB="${CELL_GLOB:-*.yaml}"
FREE_GPUS_FLAG="${FREE_GPUS_FLAG:-}"
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3 4 5}"
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
    done < <(find "$CFG_DIR" -mindepth 2 -maxdepth 2 -path "*/${CATEGORY_GLOB}/${CELL_GLOB}" | sort)
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
echo "Project: $PROJECT  |  Logs: $LOG_DIR/  |  CATEGORY_GLOB: $CATEGORY_GLOB  |  CELL_GLOB: $CELL_GLOB  |  SEEDS: ${SEEDS[*]}"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run, everything already complete."
    exit 0
fi

run_job() {
    local cfg="$1" seed="$2" gpu_id="${3:-}"
    local logging_name exp_name log
    logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    exp_name="${logging_name}_seed${seed}"
    log="${LOG_DIR}/${exp_name}.log"
    local gpu_tag=""
    [ -n "$gpu_id" ] && gpu_tag="  (gpu ${gpu_id})"

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
    # Reacts to whichever job finishes first (wait -n), not a fixed slot
    # order -- the previous round-robin version ("wait on slot i%N_PARALLEL's
    # own previous job") blocked the *entire* dispatch loop on one slow job
    # even while every other slot sat idle, once job durations stopped being
    # short/uniform (2026-09-14 unification: max_steps 2000->8000 made every
    # level take similarly long, so this went from a latent inefficiency to
    # actually stalling most of a node's GPUs, most visibly on ../hypernetwork/'s
    # own run). PID_TO_GPU tracks which GPU each in-flight job owns; wait -n
    # unblocks as soon as *any* job exits, so a free GPU gets its next job
    # immediately instead of waiting for its turn in a fixed rotation.
    # n_running, not ${#PID_TO_GPU[@]}, gates both loops below -- bash (confirmed on
    # both 5.1 and this cluster's 4.4.19) throws "unbound variable" on ${#assoc_arr[@]}
    # under set -u when the array is genuinely empty (before the first insert, or
    # after the last delete), even though key-iteration ("${!arr[@]}") over an empty
    # one is fine. Caught by a standalone smoke test before deploying this.
    declare -A PID_TO_GPU
    job_index=0
    n_running=0

    while [ "$job_index" -lt "$N_JOBS" ] && [ "$n_running" -lt "$N_PARALLEL" ]; do
        IFS='|' read -r cfg seed <<< "${JOBS[$job_index]}"
        gpu_id="${GPU_IDS[$n_running]}"
        run_job "$cfg" "$seed" "$gpu_id" &
        PID_TO_GPU[$!]="$gpu_id"
        job_index=$((job_index + 1))
        n_running=$((n_running + 1))
    done

    while [ "$n_running" -gt 0 ]; do
        wait -n 2>/dev/null || true
        for pid in "${!PID_TO_GPU[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                gpu_id="${PID_TO_GPU[$pid]}"
                unset "PID_TO_GPU[$pid]"
                n_running=$((n_running - 1))
                if [ "$job_index" -lt "$N_JOBS" ]; then
                    IFS='|' read -r cfg seed <<< "${JOBS[$job_index]}"
                    run_job "$cfg" "$seed" "$gpu_id" &
                    PID_TO_GPU[$!]="$gpu_id"
                    job_index=$((job_index + 1))
                    n_running=$((n_running + 1))
                fi
            fi
        done
    done
else
    for job in "${JOBS[@]}"; do
        IFS='|' read -r cfg seed <<< "$job"
        run_job "$cfg" "$seed"
    done
fi

echo ""
echo "All $N_JOBS jobs finished. Logs in $LOG_DIR/"
