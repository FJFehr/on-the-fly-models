#!/usr/bin/env bash
# Run arc1d_v2_multitask (Experiment 2: multi-task capacity) across seeds 1-3.
# 6 jobs: 2 configs (notd, td) x 3 seeds. Each trains one RC1-architecture
# model jointly across all 14 task categories - no per-task model. See
# docs/arc1d_story/06_phase1_findings.md.
#
# Usage:
#   bash scripts/run_arc1d_multitask.sh                  # sequential, 1 GPU
#   GPUS="0,1,2" bash scripts/run_arc1d_multitask.sh      # 3-way parallel

set -uo pipefail

PROJECT="arc1d_v2_multitask"
LOG_DIR="logs/arc1d_v2_multitask"
CFG_DIR="configs/experiments/arc1d_v2_multitask"
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3}"
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
    done < <(find "$CFG_DIR" -maxdepth 1 -name "*.yaml" | sort)
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
echo "Project: $PROJECT  |  Logs: $LOG_DIR/  |  SEEDS: ${SEEDS[*]}"
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
    if CUDA_VISIBLE_DEVICES="$gpu_id" .venv/bin/python train.py --config "$cfg" \
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
