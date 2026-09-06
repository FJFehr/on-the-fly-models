#!/usr/bin/env bash
# Train the 2 arms (notd / frozen_td) of arc1d_v2_compositional_generalization_14task -- the
# standard-14-task rerun of arc1d_v2_compositional_generalization (same "matched-scale",
# 10,156-param hypernetwork built around the 1,398-param dim=4 target model), see README.md. No
# plain td arm this time (not built at all -- see notd.yaml/frozen_td.yaml).
# After training, evaluates both arms zero-shot on the held-out compositional set
# (data/arc_1d_compositional_holdout, shared with arc1d_v2_compositional_generalization -- no
# regeneration needed).
#
# Mirrors scripts/run_arc1d_v2_compositional_generalization.sh's structure: single seed (per
# base.yaml's seed: 42, not swept), each run claims the whole node (devices: 1 here, not auto --
# set --free-gpus to restrict to a free GPU index instead of assuming exclusive node access).
# Skips already-completed training runs (results.txt exists) and already-completed eval runs
# (eval output dir's results.txt exists).
#
# Set FREE_GPUS_FLAG="--free-gpus" to restrict to a free GPU instead of relying on `devices: 1`
# picking whichever the accelerator resolves to -- recommended when the node is shared:
#   FREE_GPUS_FLAG="--free-gpus" bash scripts/run_arc1d_v2_compositional_generalization_14task.sh
#
# Usage:
#   bash scripts/run_arc1d_v2_compositional_generalization_14task.sh

set -uo pipefail

PROJECT="arc1d_v2_compositional_generalization_14task"
LOG_DIR="logs/arc1d_v2_compositional_generalization_14task"
CFG_DIR="configs/experiments/arc1d_v2_compositional_generalization_14task"
FREE_GPUS_FLAG="${FREE_GPUS_FLAG:-}"
mkdir -p "$LOG_DIR"

ARMS=(notd frozen_td)
EVAL_ARMS=(notd frozen_td)

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
echo "Both arms finished. Logs in ${LOG_DIR}/, held-out eval results under each arm's"
echo "outputs/${PROJECT}/<experiment_name>/compositional_holdout_eval/results.txt"
