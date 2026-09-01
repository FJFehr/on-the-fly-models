#!/usr/bin/env bash
# Train the 2 arms (notd / frozen_td) of arc1d_v2_compositional_generalization -- the
# "matched-scale" (10,156-param) rerun of the compositional-generalization question, see
# README.md. No plain td arm this time (not built at all -- see notd.yaml/frozen_td.yaml).
# After training, evaluates both arms zero-shot on the held-out compositional set
# (data/arc_1d_compositional_holdout, built once and shared with the original bigger-recipe
# experiment -- no regeneration needed).
#
# 3 seeds by default (SEEDS_OVERRIDE="1 2 3"), same convention as
# scripts/run_arc1d_v2_generalization.sh: each seed's run gets seed=/experiment_name=/
# output_path= overrides appended to both train.py and scripts/eval_compositional_holdout.py
# (the latter needed its own dotlist-override support added for this -- see that script's
# docstring). Each run claims the whole node (devices: 1 here, not auto -- set --free-gpus to
# restrict to a free GPU index instead of assuming exclusive node access).
#
# Skips already-completed training runs (results.txt exists) and already-completed eval runs
# (eval output dir's results.txt exists), per seed.
#
# Set FREE_GPUS_FLAG="--free-gpus" to restrict to a free GPU instead of relying on `devices: 1`
# picking whichever the accelerator resolves to -- recommended when the node is shared:
#   FREE_GPUS_FLAG="--free-gpus" bash scripts/run_arc1d_v2_compositional_generalization.sh
#
# To split across nodes, override SEEDS_OVERRIDE per node, same convention as the other v2
# launcher scripts:
#   node A: SEEDS_OVERRIDE="1 2" bash scripts/run_arc1d_v2_compositional_generalization.sh
#   node B: SEEDS_OVERRIDE="3"   bash scripts/run_arc1d_v2_compositional_generalization.sh
#
# Usage:
#   bash scripts/run_arc1d_v2_compositional_generalization.sh

set -uo pipefail

PROJECT="arc1d_v2_compositional_generalization"
LOG_DIR="logs/arc1d_v2_compositional_generalization"
CFG_DIR="configs/experiments/arc1d_v2_compositional_generalization"
FREE_GPUS_FLAG="${FREE_GPUS_FLAG:-}"
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1 2 3}"
mkdir -p "$LOG_DIR"

read -ra SEEDS <<< "$SEEDS_OVERRIDE"

ARMS=(notd frozen_td)
EVAL_ARMS=(notd frozen_td)

echo "Building the held-out compositional dataset if it doesn't already exist..."
if [ ! -d "data/arc_1d_compositional_holdout" ]; then
    # PYTHONPATH=. needed here (unlike train.py at repo root): scripts/*.py's own directory
    # is sys.path[0] by default, not the repo root, so `import data_modules` etc. fail without it.
    PYTHONPATH=. .venv/bin/python scripts/build_arc1d_compositional.py
fi

for SEED in "${SEEDS[@]}"; do
    for ARM in "${ARMS[@]}"; do
        cfg="${CFG_DIR}/${ARM}.yaml"
        base_exp_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
        exp_name="${base_exp_name}_seed${SEED}"
        results_file="outputs/${PROJECT}/${exp_name}/results.txt"
        log="${LOG_DIR}/${exp_name}.log"

        if [ -f "$results_file" ]; then
            echo "SKIP   ${PROJECT} / ${exp_name} (already trained)"
            continue
        fi

        echo "TRAIN  ${PROJECT} / ${exp_name}"
        if .venv/bin/python train.py --config "$cfg" $FREE_GPUS_FLAG \
            seed="$SEED" \
            experiment_name="$exp_name" \
            output_path="outputs/${PROJECT}/${exp_name}" \
            > "$log" 2>&1; then
            echo "DONE   ${PROJECT} / ${exp_name}"
        else
            echo "FAILED ${PROJECT} / ${exp_name}  (see $log)"
        fi
    done
done

for SEED in "${SEEDS[@]}"; do
    for ARM in "${EVAL_ARMS[@]}"; do
        cfg="${CFG_DIR}/${ARM}.yaml"
        base_exp_name=$(grep '^experiment_name:' "$cfg" | awk '{print $2}')
        exp_name="${base_exp_name}_seed${SEED}"
        eval_dir="outputs/${PROJECT}/${exp_name}/compositional_holdout_eval"
        log="${LOG_DIR}/${exp_name}_holdout_eval.log"

        if [ -f "${eval_dir}/results.txt" ]; then
            echo "SKIP   held-out eval / ${exp_name} (already evaluated)"
            continue
        fi

        echo "EVAL   held-out compositional set / ${exp_name}"
        if PYTHONPATH=. .venv/bin/python scripts/eval_compositional_holdout.py --config "$cfg" \
            seed="$SEED" \
            experiment_name="$exp_name" \
            output_path="outputs/${PROJECT}/${exp_name}" \
            > "$log" 2>&1; then
            echo "DONE   held-out eval / ${exp_name}"
        else
            echo "FAILED held-out eval / ${exp_name}  (see $log)"
        fi
    done
done

echo ""
echo "All seeds/arms finished. Logs in ${LOG_DIR}/, held-out eval results under each arm's"
echo "outputs/${PROJECT}/<experiment_name>_seed<N>/compositional_holdout_eval/results.txt"
