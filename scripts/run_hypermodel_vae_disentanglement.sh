#!/usr/bin/env bash
# Run all 6 arc1d_hypermodel_looped_rope_canon_vae_disentanglement configs
# (beta0_01 / beta0_1 / beta0_5 / beta1 / beta2 / beta10), single seed=1 each (6 jobs total),
# reusing the arc1d_hypermodel_looped_rope_canon_muon_diag W&B project so results sit next to
# the notd/td/frozentd comparison this follows on from. See README.md for the grid and how
# to read results.
#
# Single seed per Fabio's direction -- unlike the multi-seed sibling sweeps (e.g.
# rank_sweep's 5 seeds), SEEDS is intentionally a 1-element array here, not swept.
#
# Each run uses ALL GPUs on the node (configs set devices: auto). Since each run claims the
# whole node, jobs run one after another, not in parallel. Skips already-completed runs.
#
# Usage:
#   bash scripts/run_hypermodel_vae_disentanglement.sh

set -uo pipefail

PROJECT="arc1d_hypermodel_looped_rope_canon_muon_diag"
LOG_DIR="logs/arc1d_hypermodel_looped_rope_canon_vae_disentanglement"
CFG_DIR="configs/experiments/arc1d_hypermodel_looped_rope_canon_vae_disentanglement"
mkdir -p "$LOG_DIR"

SEEDS=(1)

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
    done < <(find "$CFG_DIR" -maxdepth 1 -name "beta*.yaml" | sort)
done

N_JOBS=${#JOBS[@]}
echo "Running $N_JOBS jobs sequentially (each using all GPUs)"
echo "Already complete: $SKIPPED (skipped)"
echo "Project: $PROJECT  |  Logs: $LOG_DIR/"
echo ""

if [ "$N_JOBS" -eq 0 ]; then
    echo "Nothing to run, everything already complete."
    exit 0
fi

for job in "${JOBS[@]}"; do
    IFS='|' read -r cfg seed <<< "$job"
    logging_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    exp_name="${logging_name}_seed${seed}"
    log="${LOG_DIR}/${exp_name}.log"

    echo "START  ${PROJECT} / ${exp_name}"
    if .venv/bin/python train.py --config "$cfg" \
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
echo "All $N_JOBS jobs finished. Logs in $LOG_DIR/"
