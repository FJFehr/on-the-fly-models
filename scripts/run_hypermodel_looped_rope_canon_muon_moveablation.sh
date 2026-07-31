#!/usr/bin/env bash
# Train + compositional-holdout-eval the arc1d_hypermodel_looped_rope_canon_muon_moveablation
# grid: weight_decay x adam_lr x move-task-inclusion, muon_lr=0.02, muon_exclude_lora_heads=true,
# and batch_size=2048 all fixed (see base.yaml -- bsz=4096 was tried and genuinely OOMs at
# 44.21/44.40 GiB, dropped from the grid entirely), 3 seeds by default. Follow-up to the
# (cancelled) arc1d_hypermodel_looped_rope_canon_muon_sweep -- see that experiment's README and
# this one's base.yaml for the findings that motivate this grid.
#
# Two phases, matching run_hypermodel_compositional_generalization.sh's train-then-eval pattern
# but GPU-parallel (base.yaml sets devices: 1) for the train phase, since this grid (24 jobs) is
# still bigger than that experiment's 3 arms:
#   1. Train phase: GPU-parallel via GPUS=, idempotent skip via results.txt.
#   2. Eval phase: scripts/eval_compositional_holdout.py against each completed training run's
#      best checkpoint, sequential (matches eval_compositional_holdout.py's own --device cpu
#      default -- eval is cheap: 400 holdout examples on a tiny model), idempotent skip via the
#      eval's own results.txt. Every run also has log_embedding_clusters: true (base.yaml), so
#      both training and this eval phase produce embedding-cluster + linear-probe diagnostics.
#
# Unlike the plain sweep runner, --checkpoint and --output-dir are passed explicitly to
# eval_compositional_holdout.py (rather than relying on its "best" default + the config's own
# output_path) because this runner overrides experiment_name with a per-seed suffix on train.py's
# CLI that the arm yaml's own experiment_name field doesn't know about.
#
# Set GPUS to a comma-separated list of GPU ids to run the train phase that many jobs in
# parallel, one per GPU (e.g. GPUS="0,1,2,3,4,5,6,7"). Leave GPUS unset to run sequentially.
#
# To split across nodes, override CELL_GLOB to give each node a disjoint half of the grid ("moves"
# is the outermost loop in the generator, so splitting by it gives two even 4-cell/12-job halves):
#   node A: CELL_GLOB="arm_bsz2048_*_with.yaml"    bash scripts/run_hypermodel_looped_rope_canon_muon_moveablation.sh
#   node B: CELL_GLOB="arm_bsz2048_*_without.yaml" bash scripts/run_hypermodel_looped_rope_canon_muon_moveablation.sh
#
# Usage:
#   bash scripts/run_hypermodel_looped_rope_canon_muon_moveablation.sh                  # sequential, 1 GPU
#   GPUS="0,1,2,3,4,5,6,7" bash scripts/run_hypermodel_looped_rope_canon_muon_moveablation.sh

set -uo pipefail

PROJECT="arc1d_hypermodel_looped_rope_canon_muon_moveablation"
LOG_DIR="logs/arc1d_hypermodel_looped_rope_canon_muon_moveablation"
CFG_DIR="configs/experiments/arc1d_hypermodel_looped_rope_canon_muon_moveablation"
CELL_GLOB="${CELL_GLOB:-arm_*.yaml}"
FREE_GPUS_FLAG="${FREE_GPUS_FLAG:-}"
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3}"
GPUS="${GPUS:-}"
mkdir -p "$LOG_DIR"

echo "Building the held-out compositional dataset if it doesn't already exist..."
if [ ! -d "data/arc_1d_compositional_holdout" ]; then
    PYTHONPATH=. .venv/bin/python scripts/build_arc1d_compositional.py
fi

read -ra SEEDS <<< "$SEEDS_OVERRIDE"

# --- Phase 1: train ---------------------------------------------------------

TRAIN_JOBS=()
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
        TRAIN_JOBS+=("${cfg}|${SEED}")
    done < <(find "$CFG_DIR" -maxdepth 1 -name "$CELL_GLOB" | sort)
done

N_TRAIN=${#TRAIN_JOBS[@]}

GPU_IDS=()
if [ -n "$GPUS" ]; then
    IFS=',' read -ra GPU_IDS <<< "$GPUS"
fi
N_PARALLEL=${#GPU_IDS[@]}

if [ "$N_PARALLEL" -gt 0 ]; then
    echo "Train phase: $N_TRAIN jobs, ${N_PARALLEL}-way parallel across GPUs: ${GPU_IDS[*]}"
else
    echo "Train phase: $N_TRAIN jobs sequentially (each using 1 GPU)"
fi
echo "Already complete: $SKIPPED (skipped)  |  CELL_GLOB: $CELL_GLOB  |  SEEDS: ${SEEDS[*]}"
echo ""

run_train_job() {
    local cfg="$1" seed="$2" gpu_id="${3:-}"
    local logging_name exp_name log
    logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    exp_name="${logging_name}_seed${seed}"
    log="${LOG_DIR}/${exp_name}.log"
    local gpu_tag=""
    [ -n "$gpu_id" ] && gpu_tag="  (gpu ${gpu_id})"

    echo "TRAIN  ${PROJECT} / ${exp_name}${gpu_tag}"
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

if [ "$N_TRAIN" -gt 0 ]; then
    if [ "$N_PARALLEL" -gt 0 ]; then
        declare -a SLOT_PIDS
        for ((s = 0; s < N_PARALLEL; s++)); do SLOT_PIDS[s]=""; done

        job_index=0
        for job in "${TRAIN_JOBS[@]}"; do
            IFS='|' read -r cfg seed <<< "$job"
            slot=$((job_index % N_PARALLEL))
            if [ -n "${SLOT_PIDS[$slot]:-}" ]; then
                wait "${SLOT_PIDS[$slot]}" 2>/dev/null || true
            fi
            gpu_id="${GPU_IDS[$slot]}"
            run_train_job "$cfg" "$seed" "$gpu_id" &
            SLOT_PIDS[$slot]=$!
            job_index=$((job_index + 1))
        done
        for pid in "${SLOT_PIDS[@]}"; do
            [ -n "$pid" ] && wait "$pid" 2>/dev/null
        done
    else
        for job in "${TRAIN_JOBS[@]}"; do
            IFS='|' read -r cfg seed <<< "$job"
            run_train_job "$cfg" "$seed"
        done
    fi
else
    echo "Nothing to train, everything already complete."
fi

echo ""
echo "Train phase finished."

# --- Phase 2: compositional-holdout eval ------------------------------------

EVAL_JOBS=()
for SEED in "${SEEDS[@]}"; do
    while IFS= read -r cfg; do
        logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
        exp_name="${logging_name}_seed${SEED}"
        checkpoint="outputs/${PROJECT}/${exp_name}/best_model.ckpt"
        eval_dir="outputs/${PROJECT}/${exp_name}/compositional_holdout_eval"

        if [ -f "${eval_dir}/results.txt" ]; then
            continue
        fi
        if [ ! -f "$checkpoint" ]; then
            echo "SKIP   held-out eval / ${exp_name} (no checkpoint -- training incomplete/failed)"
            continue
        fi
        EVAL_JOBS+=("${cfg}|${exp_name}")
    done < <(find "$CFG_DIR" -maxdepth 1 -name "$CELL_GLOB" | sort)
done

echo ""
echo "Eval phase: ${#EVAL_JOBS[@]} jobs (sequential)"

for job in "${EVAL_JOBS[@]}"; do
    IFS='|' read -r cfg exp_name <<< "$job"
    checkpoint="outputs/${PROJECT}/${exp_name}/best_model.ckpt"
    eval_dir="outputs/${PROJECT}/${exp_name}/compositional_holdout_eval"
    log="${LOG_DIR}/${exp_name}_holdout_eval.log"

    echo "EVAL   held-out compositional set / ${exp_name}"
    if PYTHONPATH=. .venv/bin/python scripts/eval_compositional_holdout.py \
        --config "$cfg" \
        --checkpoint "$checkpoint" \
        --output-dir "$eval_dir" \
        > "$log" 2>&1; then
        echo "DONE   held-out eval / ${exp_name}"
    else
        echo "FAILED held-out eval / ${exp_name}  (see $log)"
    fi
done

echo ""
echo "All done. Train logs + eval logs in ${LOG_DIR}/. Per-run results under"
echo "outputs/${PROJECT}/<experiment_name>_seed<N>/results.txt and"
echo "outputs/${PROJECT}/<experiment_name>_seed<N>/compositional_holdout_eval/results.txt"
