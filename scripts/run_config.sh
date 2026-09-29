#!/usr/bin/env bash
# Training launcher shared by every experiment's run.sh.
#
# Finds the leaf configs under CFG_DIR, skips any job that already has a results.txt, and runs
# the rest with train.py, keeping a fixed number of jobs running: each job slot takes the next
# job as soon as its current one finishes. Always safe to rerun: finished jobs are skipped, so
# it resumes after an interruption.
#
# Every job uses one GPU but only about 350 MB of its memory; most of a job's time is CPU-bound
# data setup (about 1 core and 3 to 5 GB of RAM per job), so several jobs can share a GPU.
#
# Usage:
#   CFG_DIR=experiments/01_multitask_capacity bash scripts/run_config.sh                  # one at a time
#   CFG_DIR=experiments/01_multitask_capacity JOBS_PER_GPU=4 bash scripts/run_config.sh   # 4 at once, one GPU
#   CFG_DIR=experiments/01_multitask_capacity GPUS=all bash scripts/run_config.sh         # one per GPU
#
# Env vars:
#   CFG_DIR         (required) config directory, searched recursively
#   CELL_GLOB       filename glob for leaf configs (default: "*.yaml"; base.yaml is skipped)
#   SEEDS_OVERRIDE  space-separated seeds, each run as <experiment_name>_seed<N> (default: "1"),
#                   or "config" to run each config once with the seed and experiment_name
#                   written in it
#   PROJECT         output folder and W&B project (default: basename of CFG_DIR); runs land
#                   in outputs/<PROJECT>/<run name>/
#   GPUS            GPU ids to use, comma-separated, or "all" for every visible GPU
#                   (default: the default GPU only)
#   JOBS_PER_GPU    jobs to run at once on each GPU (default: 1)
#   SHARD           "k/N": run only every N-th job, starting at the k-th (0-based), to split
#                   one sweep across machines (default: all jobs)
#   PYTHON          interpreter (default: .venv/bin/python, created by `uv sync`)

set -uo pipefail

CFG_DIR="${CFG_DIR:?Set CFG_DIR to the config directory to run, e.g. experiments/01_multitask_capacity}"
PROJECT="${PROJECT:-$(basename "$CFG_DIR")}"
LOG_DIR="${LOG_DIR:-logs/${PROJECT}}"
CELL_GLOB="${CELL_GLOB:-*.yaml}"
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1}"
GPUS="${GPUS:-}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
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

# One job slot per GPU and repeat. An empty GPU id means the default GPU (CUDA_VISIBLE_DEVICES
# is left as it is).
if [ "$GPUS" = "all" ]; then
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    [ "${#GPU_IDS[@]}" -gt 0 ] || { echo "GPUS=all: nvidia-smi found no GPUs"; exit 1; }
elif [ -n "$GPUS" ]; then
    IFS=',' read -ra GPU_IDS <<< "$GPUS"
else
    GPU_IDS=("")
fi
SLOTS=()
for ((r = 0; r < JOBS_PER_GPU; r++)); do SLOTS+=("${GPU_IDS[@]}"); done

echo "Running $N_JOBS jobs, ${#SLOTS[@]} at a time (GPUs: ${GPUS:-default}, jobs per GPU: $JOBS_PER_GPU); $SKIPPED already complete"
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
    local gpu_env=()
    [ -n "$gpu_id" ] && gpu_env=(CUDA_VISIBLE_DEVICES="$gpu_id")
    echo "START  ${PROJECT} / ${name}${gpu_id:+  (gpu $gpu_id)}"
    if env "${gpu_env[@]}" $PYTHON train.py --config "$cfg" "${overrides[@]}" > "$log" 2>&1; then
        echo "DONE   ${PROJECT} / ${name}${gpu_id:+  (gpu $gpu_id)}"
    else
        echo "FAILED ${PROJECT} / ${name}${gpu_id:+  (gpu $gpu_id)}  (see $log)"
    fi
}

# Fill every slot, then give each slot the next job as soon as its current one finishes.
declare -A GPU_OF_PID
next=0
start_next() {
    IFS='|' read -r cfg seed <<< "${JOBS[$next]}"
    run_job "$cfg" "$seed" "$1" &
    GPU_OF_PID[$!]="$1"
    next=$((next + 1))
}
for gpu in "${SLOTS[@]}"; do
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

echo ""
echo "All $N_JOBS jobs finished. Logs in $LOG_DIR/"
