#!/usr/bin/env bash
# Run experiment 5's full sweep: every one of the 14 base ARC-1D task categories held out in
# turn, x {notd, frozentd}, x 5 seeds -- 140 leaf configs, one job each (the largest single
# sweep in experiments/ so far).
#
# Custom script, not scripts/run_config.sh (the generic launcher) -- gen_configs.py already
# bakes seed and a unique experiment_name into each leaf (<short>_<arm>_seed<N>.yaml, matching
# experiment 4's gen_seeds.py convention), so the generic launcher's own SEEDS_OVERRIDE looping
# would append a second, conflicting _seed<N> suffix and re-override each leaf's baked seed --
# see experiments/04_compositional_generalization/run_seeds.sh, which hits the same issue and
# solves it the same way (this script mirrors its shape).
#
# No separate eval step needed, unlike experiment 4: val_task_categories always includes the
# held-out category, so its zero-shot score already lands in results.txt's automatic
# per-category validation breakdown from this training run alone (no data-build step either --
# every base category already lives in data/arc_1d_looped_augmented's dev/test splits).
#
# Usage (single node, contiguous GPU indices 0..N-1):
#   bash experiments/05_leave_one_out_task_generalization/run.sh       # 8 GPUs (default)
#   bash experiments/05_leave_one_out_task_generalization/run.sh 4     # 4 GPUs
#
# Set GPU_LIST to explicit physical GPU indices instead (e.g. GPU_LIST="0 2 3" to skip a GPU
# already in use by another job on a shared node).
#
# Before launching all 140: run one leaf by hand first to get a real per-job wall-clock time
# (not otherwise documented) and confirm results.txt carries the expected
# val_query_*_by_task_<held_out_category> keys, e.g.:
#   uv run python train.py --config experiments/05_leave_one_out_task_generalization/configs/hollow_notd_seed1.yaml

set -uo pipefail

N_GPUS_ARG=${1:-8}
GPU_LIST=${GPU_LIST:-}

if [ -z "$GPU_LIST" ]; then
    GPU_LIST=$(seq 0 $((N_GPUS_ARG - 1)))
fi

# PROJECT sets the output/W&B project name; SHARD="k/N" runs only every N-th config
# starting at the k-th (0-based), to split the sweep across nodes.
PROJECT="${PROJECT:-05_leave_one_out_task_generalization}"
SHARD="${SHARD:-0/1}"
IFS=/ read -r SHARD_K SHARD_N <<< "$SHARD"
LOG_DIR="logs/${PROJECT}"
CFG_DIR="experiments/05_leave_one_out_task_generalization/configs"
mkdir -p "$LOG_DIR"

JOBS=()
SKIPPED=0
CFG_INDEX=-1
while IFS= read -r cfg; do
    CFG_INDEX=$((CFG_INDEX + 1))
    (( CFG_INDEX % SHARD_N == SHARD_K )) || continue
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
                if .venv/bin/python train.py --config "$cfg" project_name="$PROJECT" > "$log" 2>&1; then
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

echo ""
echo "All done. Logs in ${LOG_DIR}/, each cell's held-out-category zero-shot score in"
echo "outputs/${PROJECT}/<experiment_name>/results.txt (val_query_*_by_task_<held_out_category>)."
