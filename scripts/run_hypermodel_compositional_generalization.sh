#!/usr/bin/env bash
# Train the 3 arms (notd / td / frozen_td) of
# arc1d_hypermodel_compositional_generalization -- same 15-category recipe as
# arc1d_hypermodel_looped_rope_canon_muon's notd/td/frozentd comparison, single seed (per
# base.yaml's seed: 42, not swept). After training, evaluates notd and frozen_td (not td --
# see scripts/eval_compositional_holdout.py's docstring for why) zero-shot on the held-out
# compositional set. See README.md for the full rationale and how to read results.
#
# Each run uses ALL GPUs on the node (configs set devices: auto), so runs happen one after
# another, not in parallel. Skips already-completed training runs (matched by whether
# outputs/<project>/<experiment_name>/results.txt exists) and already-completed eval runs
# (matched by whether the eval output dir's results.txt exists).
#
# Set FREE_GPUS_FLAG="--free-gpus" to restrict to GPUs with <500MB used (train.py's
# resolve_free_gpus), instead of claiming every GPU on the node -- needed when the node is
# shared with another user's already-running job:
#   FREE_GPUS_FLAG="--free-gpus" bash scripts/run_hypermodel_compositional_generalization.sh
#
# Usage:
#   bash scripts/run_hypermodel_compositional_generalization.sh

set -uo pipefail

PROJECT="arc1d_hypermodel_compositional_generalization"
LOG_DIR="logs/arc1d_hypermodel_compositional_generalization"
CFG_DIR="configs/experiments/arc1d_hypermodel_compositional_generalization"
FREE_GPUS_FLAG="${FREE_GPUS_FLAG:-}"
mkdir -p "$LOG_DIR"

ARMS=(notd td frozen_td)
EVAL_ARMS=(notd frozen_td)  # td excluded -- see eval_compositional_holdout.py's docstring

echo "Building the held-out compositional dataset if it doesn't already exist..."
if [ ! -d "data/arc_1d_compositional_holdout" ]; then
    # PYTHONPATH=. needed here (unlike train.py at repo root): scripts/*.py's own directory
    # is sys.path[0] by default, not the repo root, so `import data_modules` etc. fail without it.
    PYTHONPATH=. .venv/bin/python scripts/build_arc1d_compositional.py
fi

for ARM in "${ARMS[@]}"; do
    cfg="${CFG_DIR}/${ARM}.yaml"
    exp_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    results_file="outputs/${PROJECT}/${exp_name}/results.txt"
    log="${LOG_DIR}/${exp_name}.log"

    if [ -f "$results_file" ]; then
        echo "SKIP   ${PROJECT} / ${exp_name} (already trained)"
        continue
    fi

    echo "TRAIN  ${PROJECT} / ${exp_name}"
    if .venv/bin/python train.py --config "$cfg" $FREE_GPUS_FLAG > "$log" 2>&1; then
        echo "DONE   ${PROJECT} / ${exp_name}"
    else
        echo "FAILED ${PROJECT} / ${exp_name}  (see $log)"
    fi
done

for ARM in "${EVAL_ARMS[@]}"; do
    cfg="${CFG_DIR}/${ARM}.yaml"
    exp_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
    eval_dir="outputs/${PROJECT}/${exp_name}/compositional_holdout_eval"
    log="${LOG_DIR}/${exp_name}_holdout_eval.log"

    if [ -f "${eval_dir}/results.txt" ]; then
        echo "SKIP   held-out eval / ${exp_name} (already evaluated)"
        continue
    fi

    echo "EVAL   held-out compositional set / ${exp_name}"
    if PYTHONPATH=. .venv/bin/python scripts/eval_compositional_holdout.py --config "$cfg" > "$log" 2>&1; then
        echo "DONE   held-out eval / ${exp_name}"
    else
        echo "FAILED held-out eval / ${exp_name}  (see $log)"
    fi
done

echo ""
echo "All arms finished. Logs in ${LOG_DIR}/, held-out eval results under each arm's"
echo "outputs/${PROJECT}/<experiment_name>/compositional_holdout_eval/results.txt"
