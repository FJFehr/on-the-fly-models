#!/usr/bin/env bash
# Run arc1d_v2_backbone_capacity (Round 2, Phase 1) across seeds 1-3.
# 450 jobs: 5 steps x 15 tasks x 2 widths (dim16, dim10) x 3 seeds (skips
# already-completed runs). Muon (muon_lr=0.005, muon_momentum=0.95) - see
# docs/arc1d_story/05_round2_plan.md's Phase 1 for the full rationale.
#
# Usage (single node, contiguous GPU indices 0..N-1):
#   bash scripts/run_v2_backbone_capacity.sh       # 8 GPUs (default)
#   bash scripts/run_v2_backbone_capacity.sh 4     # 4 GPUs
#
# Usage (split across multiple nodes / non-contiguous GPU indices): set
# GPU_LIST to this invocation's own physical GPU indices, GPU_OFFSET to
# this invocation's position in the *global* round-robin (0 for the first
# node's first GPU, 4 if this node picks up where a 4-GPU first node left
# off, etc.), and TOTAL_GPUS to the grand total across every node running
# concurrently. The shared outputs/results.txt skip-logic (NFS home,
# same for every torrnode) is what actually prevents two nodes duplicating
# a job - GPU_OFFSET/TOTAL_GPUS just needs to keep each node's round-robin
# slice non-overlapping with the others' from the start, since two nodes
# racing to grab the same not-yet-finished job is possible otherwise.
#   # torrnode11, GPUs 2/3/5/6 (this run's global slots 0-3):
#   GPU_LIST="2 3 5 6" GPU_OFFSET=0 TOTAL_GPUS=8 bash scripts/run_v2_backbone_capacity.sh
#   # torrnode12, GPUs 0/1/2/6 (this run's global slots 4-7):
#   GPU_LIST="0 1 2 6" GPU_OFFSET=4 TOTAL_GPUS=8 bash scripts/run_v2_backbone_capacity.sh
#
# Designed to co-locate with other jobs already on the node's GPUs (small
# model, ~5-40k backbone params, batch_size=256) - each GPU slot in this
# script only ever runs one of our jobs at a time, regardless of what else
# is already resident on that physical GPU.

set -uo pipefail

N_GPUS_ARG=${1:-8}
GPU_LIST=${GPU_LIST:-}
GPU_OFFSET=${GPU_OFFSET:-0}
TOTAL_GPUS=${TOTAL_GPUS:-$N_GPUS_ARG}

if [ -z "$GPU_LIST" ]; then
    # Default: contiguous 0..N-1 on a single node, offset 0.
    GPU_LIST=$(seq 0 $((N_GPUS_ARG - 1)))
    TOTAL_GPUS=$N_GPUS_ARG
fi

PROJECT="arc1d_v2_backbone_capacity"
LOG_DIR="logs/arc1d_v2_backbone_capacity"
CFG_DIR="configs/experiments/arc1d_v2_backbone_capacity"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3)

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
    done < <(find "$CFG_DIR" -mindepth 2 -maxdepth 2 -name "*.yaml" ! -name "base_*" | sort)
done

N_JOBS=${#JOBS[@]}
N_LOCAL_GPUS=$(echo "$GPU_LIST" | wc -w)
echo "Launching (up to) $N_JOBS jobs across $N_LOCAL_GPUS local GPUs (global slots ${GPU_OFFSET}-$((GPU_OFFSET + N_LOCAL_GPUS - 1)) of $TOTAL_GPUS)"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run - all conditions already at 3 seeds."
    exit 0
fi

slot_idx=0
for gpu in $GPU_LIST; do
    global_slot=$((GPU_OFFSET + slot_idx))
    slot_idx=$((slot_idx + 1))
    sleep $((slot_idx * 5))
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        for i in $(seq "$global_slot" "$TOTAL_GPUS" $((N_JOBS - 1))); do
            IFS='|' read -r cfg seed <<< "${JOBS[$i]}"
            logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
            exp_name="${logging_name}_seed${seed}"
            log="${LOG_DIR}/${exp_name}.log"

            echo "[GPU $gpu / slot $global_slot] START  ${PROJECT} / ${exp_name}"
            if .venv/bin/python train.py --config "$cfg" \
                seed="$seed" \
                project_name="$PROJECT" \
                experiment_name="${exp_name}" \
                logging_name="${logging_name}" \
                > "$log" 2>&1; then
                echo "[GPU $gpu / slot $global_slot] DONE   ${PROJECT} / ${exp_name}"
            else
                echo "[GPU $gpu / slot $global_slot] FAILED ${PROJECT} / ${exp_name}  (see $log)"
            fi
        done
    ) &
done

wait
echo ""
echo "This invocation's GPUs finished. Logs in $LOG_DIR/"
