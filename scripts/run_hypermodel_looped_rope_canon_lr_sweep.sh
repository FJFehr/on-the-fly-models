#!/usr/bin/env bash
# Run all 7 arc1d_hypermodel_looped_rope_canon_lr_sweep arms (baseline/zhu_all x
# 1e-3/1e-4/5e-5, the 5e-4 cell for each architecture is handled separately below), 5 seeds
# each (35 jobs), then top up the existing
# arc1d_hypermodel_looped_rope_canon_optimizer/arm_adamw.yaml (baseline, lr=5e-4, already has
# 3 seeds) with seeds 4 and 5 so all 8 grid cells end up at 5 seeds. 37 jobs total.
#
# Each run uses ALL GPUs on the node by default (configs set devices: auto), including any
# already in use by other users' jobs, CUDA compute is time-sliced/shared fine as long as
# there's free memory. Set FREE_GPUS_FLAG="--free-gpus" to instead restrict to GPUs with
# <500MB used, if a shared node's other jobs are memory-heavy enough that sharing risks OOM.
# Since each run claims every (free or shared) GPU, jobs run one after another, not in
# parallel. Skips already-completed runs (idempotent to rerun).
#
# To split this across multiple nodes without clashing, override CELL_GLOB to give each node
# a disjoint slice of the main sweep, e.g.:
#   node A: CELL_GLOB="arm_baseline_*.yaml" bash scripts/run_hypermodel_looped_rope_canon_lr_sweep.sh
#   node B: CELL_GLOB="arm_zhuall_*.yaml"   bash scripts/run_hypermodel_looped_rope_canon_lr_sweep.sh
# (run the top-up section, below, on only one of the two nodes to avoid a race on those 2 jobs)
#
# Usage:
#   bash scripts/run_hypermodel_looped_rope_canon_lr_sweep.sh

set -uo pipefail

PROJECT="arc1d_hypermodel_looped_rope_canon_lr_sweep"
LOG_DIR="logs/arc1d_hypermodel_looped_rope_canon_lr_sweep"
CFG_DIR="configs/experiments/arc1d_hypermodel_looped_rope_canon_lr_sweep"
CELL_GLOB="${CELL_GLOB:-arm_*.yaml}"
FREE_GPUS_FLAG="${FREE_GPUS_FLAG:-}"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3 4 5)

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
echo "Main sweep: running $N_JOBS jobs sequentially (each using all GPUs)"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/  |  CELL_GLOB: $CELL_GLOB"
echo ""

for job in "${JOBS[@]}"; do
    IFS='|' read -r cfg seed <<< "$job"
    logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    exp_name="${logging_name}_seed${seed}"
    log="${LOG_DIR}/${exp_name}.log"

    echo "START  ${PROJECT} / ${exp_name}"
    if .venv/bin/python train.py --config "$cfg" $FREE_GPUS_FLAG \
        seed="$seed" \
        project_name="$PROJECT" \
        experiment_name="${exp_name}" \
        logging_name="${logging_name}" \
        > "$log" 2>&1; then
        echo "DONE   ${PROJECT} / ${exp_name}"
    else
        echo "FAILED ${PROJECT} / ${exp_name}  (see $log)"
    fi
done

echo ""
echo "Main sweep finished."
echo ""

# --- Top up the reused baseline/lr=5e-4 cell (arc1d_hypermodel_looped_rope_canon_optimizer)
# with seeds 4 and 5, so it reaches 5 seeds like every other cell in this grid. ---
TOPUP_PROJECT="arc1d_hypermodel_looped_rope_canon_optimizer"
TOPUP_CFG="configs/experiments/arc1d_hypermodel_looped_rope_canon_optimizer/arm_adamw.yaml"
TOPUP_LOG_DIR="logs/arc1d_hypermodel_looped_rope_canon_optimizer"
TOPUP_LOGGING_NAME=$(grep '^experiment_name:' "$TOPUP_CFG" | awk '{print $2}')
mkdir -p "$TOPUP_LOG_DIR"

echo "Top-up: baseline/lr=5e-4 cell (reusing $TOPUP_CFG), seeds 4-5"
for SEED in 4 5; do
    exp_name="${TOPUP_LOGGING_NAME}_seed${SEED}"
    results_file="outputs/${TOPUP_PROJECT}/${exp_name}/results.txt"
    if [ -f "$results_file" ]; then
        echo "SKIP   ${TOPUP_PROJECT} / ${exp_name} (already complete)"
        continue
    fi
    log="${TOPUP_LOG_DIR}/${exp_name}.log"
    echo "START  ${TOPUP_PROJECT} / ${exp_name}"
    if .venv/bin/python train.py --config "$TOPUP_CFG" $FREE_GPUS_FLAG \
        seed="$SEED" \
        project_name="$TOPUP_PROJECT" \
        experiment_name="${exp_name}" \
        logging_name="${TOPUP_LOGGING_NAME}" \
        > "$log" 2>&1; then
        echo "DONE   ${TOPUP_PROJECT} / ${exp_name}"
    else
        echo "FAILED ${TOPUP_PROJECT} / ${exp_name}  (see $log)"
    fi
done

echo ""
echo "All done."
