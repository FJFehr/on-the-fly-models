#!/usr/bin/env bash
# Generic training launcher shared by the experiments/ trees.
#
# Extracted from what used to be ~35 near-identical, hand-copied run_*.sh
# scripts (see legacy/scripts/ for the originals): glob a config directory,
# skip any (config, seed) pair that already has a results.txt, and launch
# the rest via train.py, either sequentially or round-robined across a list
# of GPUs.
#
# A few experiments have extra behaviour beyond this common core (a
# data-build step, a post-training eval pass, a non-round-robin "claim every
# GPU per job" mode) and keep their own bespoke script, colocated with their
# configs -- see experiments/04_compositional_generalization/
# and experiments/06_data_efficiency_ablation/*/run.sh.
#
# Usage:
#   CFG_DIR=experiments/01_multitask_capacity \
#     bash scripts/run_config.sh                                   # sequential, 1 GPU
#   CFG_DIR=experiments/01_multitask_capacity GPUS="0,1,2" \
#     bash scripts/run_config.sh                                   # 3-way parallel
#
# Env vars:
#   CFG_DIR         (required) config directory to glob (recursively)
#   PROJECT         wandb/output project name (default: basename of CFG_DIR)
#   CELL_GLOB       filename glob for leaf configs (default: "*.yaml")
#   SEEDS_OVERRIDE  space-separated seeds to launch (default: "1")
#   GPUS            comma-separated GPU ids for round-robin parallelism
#                   (default: empty -> sequential, no CUDA_VISIBLE_DEVICES pin)

set -uo pipefail

CFG_DIR="${CFG_DIR:?Set CFG_DIR to the config directory to run, e.g. experiments/01_multitask_capacity}"
PROJECT="${PROJECT:-$(basename "$CFG_DIR")}"
LOG_DIR="${LOG_DIR:-logs/${PROJECT}}"
CELL_GLOB="${CELL_GLOB:-*.yaml}"
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
    done < <(find "$CFG_DIR" -name "$CELL_GLOB" ! -name "base.yaml" | sort)
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
echo "Project: $PROJECT  |  Config dir: $CFG_DIR  |  Logs: $LOG_DIR/  |  CELL_GLOB: $CELL_GLOB  |  SEEDS: ${SEEDS[*]}"
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
