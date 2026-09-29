#!/usr/bin/env bash
# Training launcher shared by every experiment's run.sh.
#
# Finds the leaf configs under CFG_DIR, skips any job that already has a results.txt, and
# runs the rest with train.py: one after another, or in parallel with one job per GPU (a GPU
# takes the next job as soon as its current one finishes). Always safe to rerun: finished
# jobs are skipped, so it resumes after an interruption.
#
# Usage:
#   CFG_DIR=experiments/01_multitask_capacity bash scripts/run_config.sh      # sequential
#   CFG_DIR=experiments/01_multitask_capacity GPUS=0,1,2 bash scripts/run_config.sh
#
# Env vars:
#   CFG_DIR         (required) config directory, searched recursively
#   CELL_GLOB       filename glob for leaf configs (default: "*.yaml"; base.yaml is skipped)
#   SEEDS_OVERRIDE  space-separated seeds, each run as <experiment_name>_seed<N> (default: "1"),
#                   or "config" to run each config once with the seed and experiment_name
#                   written in it
#   PROJECT         output folder and W&B project (default: basename of CFG_DIR); runs land
#                   in outputs/<PROJECT>/<run name>/
#   GPUS            comma-separated GPU ids to run on in parallel (default: sequential)
#   SHARD           "k/N": run only every N-th job, starting at the k-th (0-based), to split
#                   one sweep across nodes (default: all jobs)
#   PYTHON          interpreter (default: .venv/bin/python, created by `uv sync`)

set -uo pipefail

CFG_DIR="${CFG_DIR:?Set CFG_DIR to the config directory to run, e.g. experiments/01_multitask_capacity}"
PROJECT="${PROJECT:-$(basename "$CFG_DIR")}"
LOG_DIR="${LOG_DIR:-logs/${PROJECT}}"
CELL_GLOB="${CELL_GLOB:-*.yaml}"
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1}"
GPUS="${GPUS:-}"
SHARD="${SHARD:-0/1}"
PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p "$LOG_DIR"

IFS='/' read -r SHARD_K SHARD_N <<< "$SHARD"
read -ra SEEDS <<< "$SEEDS_OVERRIDE"

# Each job is "config|seed", with seed "config" meaning: use the config's own seed and name.
run_name() {
    local cfg="$1" seed="$2" name
    name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    if [ "$seed" = "config" ]; then echo "$name"; else echo "${name}_seed${seed}"; fi
}

JOBS=()
SKIPPED=0
JOB_INDEX=-1
for SEED in "${SEEDS[@]}"; do
    while IFS= read -r cfg; do
        # Sharding counts every job, finished or not, so a shard keeps the same jobs when it
        # is relaunched.
        JOB_INDEX=$((JOB_INDEX + 1))
        (( JOB_INDEX % SHARD_N == SHARD_K )) || continue
        if [ -f "outputs/${PROJECT}/$(run_name "$cfg" "$SEED")/results.txt" ]; then
            (( SKIPPED++ )) || true
            continue
        fi
        JOBS+=("${cfg}|${SEED}")
    done < <(find "$CFG_DIR" -name "$CELL_GLOB" ! -name "base.yaml" | sort)
done
N_JOBS=${#JOBS[@]}

GPU_IDS=()
[ -n "$GPUS" ] && IFS=',' read -ra GPU_IDS <<< "$GPUS"
N_PARALLEL=${#GPU_IDS[@]}

echo "Running $N_JOBS jobs on ${N_PARALLEL:-0} GPU(s) ${GPU_IDS[*]:-(sequential)}; $SKIPPED already complete"
echo "Project: $PROJECT | Config dir: $CFG_DIR | Glob: $CELL_GLOB | Seeds: ${SEEDS[*]} | Shard: $SHARD | Logs: $LOG_DIR/"
echo ""
if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run, everything already complete."
    exit 0
fi

run_job() {
    local cfg="$1" seed="$2" gpu_id="${3:-}"
    local name log
    name=$(run_name "$cfg" "$seed")
    log="${LOG_DIR}/${name}.log"
    local overrides=(project_name="$PROJECT")
    if [ "$seed" != "config" ]; then
        overrides+=(seed="$seed" experiment_name="$name"
                    logging_name="$(grep '^experiment_name:' "$cfg" | awk '{print $2}')")
    fi
    echo "START  ${PROJECT} / ${name}${gpu_id:+  (gpu $gpu_id)}"
    if CUDA_VISIBLE_DEVICES="$gpu_id" $PYTHON train.py --config "$cfg" "${overrides[@]}" > "$log" 2>&1; then
        echo "DONE   ${PROJECT} / ${name}${gpu_id:+  (gpu $gpu_id)}"
    else
        echo "FAILED ${PROJECT} / ${name}${gpu_id:+  (gpu $gpu_id)}  (see $log)"
    fi
}

if [ "$N_PARALLEL" -eq 0 ]; then
    for job in "${JOBS[@]}"; do
        IFS='|' read -r cfg seed <<< "$job"
        run_job "$cfg" "$seed"
    done
else
    # Start one job per GPU, then give each GPU the next job as soon as it frees up.
    declare -A GPU_OF_PID
    next=0
    start_next() {
        IFS='|' read -r cfg seed <<< "${JOBS[$next]}"
        run_job "$cfg" "$seed" "$1" &
        GPU_OF_PID[$!]="$1"
        next=$((next + 1))
    }
    for gpu in "${GPU_IDS[@]}"; do
        [ "$next" -lt "$N_JOBS" ] && start_next "$gpu"
    done
    while [ "${#GPU_OF_PID[@]}" -gt 0 ]; do
        wait -n 2>/dev/null || true
        for pid in "${!GPU_OF_PID[@]}"; do
            kill -0 "$pid" 2>/dev/null && continue
            gpu="${GPU_OF_PID[$pid]}"
            unset "GPU_OF_PID[$pid]"
            [ "$next" -lt "$N_JOBS" ] && start_next "$gpu"
        done
    done
fi

echo ""
echo "All $N_JOBS jobs finished. Logs in $LOG_DIR/"
