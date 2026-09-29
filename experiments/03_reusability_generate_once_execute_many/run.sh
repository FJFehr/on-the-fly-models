#!/usr/bin/env bash
# Eval-only: no training here. Runs scripts/measure_compute_efficiency.py's Phase E
# (leave-one-out own-vs-cross-instance generalization) against experiment 2's own dim=4
# matched-scale hypernetwork checkpoints (outputs/02_hypernetwork_multitask/
# hyper_multitask_dim4_{notd,frozentd}_seed{1..5}/, 10,156 trainable params each, real
# checkpoints -- see that experiment's README's "Dim=4 rerun"). Each checkpoint's own
# saved config.yaml is self-contained (no _base_, no unresolved ${...}) and points
# --hyper-config directly at it; --hyper-checkpoint best resolves to that config's own
# output_path/best_model.ckpt automatically.
#
# --generalization-split val,test combines both splits of data/arc_1d_looped_augmented
# (100 rows/category each, confirmed by loading the dataset directly) for 200/category --
# richer than either split alone, still fully within data this repo can rebuild from
# scratch (no dependency on the old preliminary pass's separate devtest_shifted dataset).
#
# measure_compute_efficiency.py has no skip/resume logic of its own for --stage
# generalization (each --output-dir is unconditionally overwritten, never merged) -- this
# script's own marker check (generalization_table.csv already present) provides that.
#
# Cheap and CPU-only (~30s/cell) -- no GPU, no cluster needed once experiment 2's
# checkpoints are local. If a checkpoint is missing, fetch it first:
#   REMOTE_HOST=<node> INCLUDE_CHECKPOINTS=1 bash scripts/fetch_experiments.sh 02_hypernetwork_multitask
#
# Usage:
#   bash experiments/03_reusability_generate_once_execute_many/run.sh
#   GENERALIZATION_SPLIT=val bash experiments/03_reusability_generate_once_execute_many/run.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

# Override CKPT_DIR/OUT_ROOT to evaluate another run of experiment 2, and PYTHON to use a
# specific interpreter instead of .venv/bin/python (created by `uv sync`).
PYTHON="${PYTHON:-.venv/bin/python}"
CKPT_DIR="${CKPT_DIR:-outputs/02_hypernetwork_multitask}"
OUT_ROOT="${OUT_ROOT:-outputs/03_reusability_generate_once_execute_many}"
SPLIT="${GENERALIZATION_SPLIT:-val,test}"

for COND in notd frozentd; do
    for SEED in 1 2 3 4 5; do
        CELL="hyper_multitask_dim4_${COND}_seed${SEED}"
        CKPT_CELL_DIR="${CKPT_DIR}/${CELL}"
        OUT_DIR="${OUT_ROOT}/${CELL}"

        if [ -f "${OUT_DIR}/generalization_table.csv" ]; then
            echo "SKIP   ${CELL} (already evaluated)"
            continue
        fi
        if [ ! -f "${CKPT_CELL_DIR}/config.yaml" ] || [ ! -f "${CKPT_CELL_DIR}/best_model.ckpt" ]; then
            echo "SKIP   ${CELL} (checkpoint missing -- see experiment 2's README to rerun/refetch)"
            continue
        fi

        echo "RUN    ${CELL}"
        if PYTHONPATH=. $PYTHON scripts/measure_compute_efficiency.py --stage generalization \
            --hyper-config "${CKPT_CELL_DIR}/config.yaml" --hyper-checkpoint best \
            --generalization-split "${SPLIT}" --output-dir "${OUT_DIR}"; then
            echo "DONE   ${CELL}"
        else
            echo "FAILED ${CELL}"
        fi
    done
done

echo ""
echo "Next: uv run python experiments/03_reusability_generate_once_execute_many/plot_generalization_loo.py --outputs-dir outputs"
