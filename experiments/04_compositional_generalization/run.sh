#!/usr/bin/env bash
# Eval-only: experiment 4 no longer trains its own models -- both notd and frozen_td reuse
# experiment 2's own dim=4 matched-scale checkpoints directly (in-distribution results are
# already experiment 2's own results.txt, since both experiments train on the identical
# 14-category recipe -- see plot_compositional.py). This script adds the one new thing
# experiment 4 needs on top: the zero-shot compositional-holdout eval,
# scripts/eval_compositional_holdout.py, against those checkpoints.
#
# notd: no architecture dependency on num_tasks (no task-identity signal at all) --
# evaluated directly against experiment 2's checkpoint, no changes needed.
#
# frozen_td: experiment 2's checkpoint has num_tasks=18 (no reason for experiment 2 itself
# to reserve the 10 compositional indices); this eval needs num_tasks=28. Since
# freeze_task_indicator=true means that whole projection is random-init and NEVER trained,
# padding it with 10 freshly-random columns (pad_frozentd_checkpoint.py) is statistically
# equivalent to training with num_tasks=28 from the start -- not a retrain, a
# mathematically-justified extension of a layer that was never learned in the first place
# (verified end-to-end: real numbers land in the same ballpark as experiment 4's own,
# separately-trained num_tasks=28 frozen_td checkpoint).
#
# No GPU, no cluster needed once experiment 2's checkpoints are local -- this is a cheap
# CPU eval pass, same story as experiment 3.
#
# Usage:
#   bash experiments/04_compositional_generalization/run.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

# Override CKPT_DIR/OUT_ROOT to evaluate another run of experiment 2, and PYTHON to use a
# specific interpreter instead of `uv run python`.
PYTHON="${PYTHON:-uv run python}"
CKPT_DIR="${CKPT_DIR:-outputs/02_hypernetwork_multitask}"
OUT_ROOT="${OUT_ROOT:-outputs/04_compositional_generalization}"
PADDED_DIR="${PADDED_DIR:-outputs/04_compositional_generalization/padded_checkpoints}"

echo "== notd: direct eval against experiment 2's checkpoints =="
for SEED in 1 2 3 4 5; do
    CELL="notd_seed${SEED}"
    CKPT_CELL_DIR="${CKPT_DIR}/hyper_multitask_dim4_notd_seed${SEED}"
    OUT_DIR="${OUT_ROOT}/${CELL}"

    if [ -f "${OUT_DIR}/results.txt" ]; then
        echo "SKIP   ${CELL} (already evaluated)"
        continue
    fi
    if [ ! -f "${CKPT_CELL_DIR}/config.yaml" ] || [ ! -f "${CKPT_CELL_DIR}/best_model.ckpt" ]; then
        echo "SKIP   ${CELL} (checkpoint missing -- see experiment 2's README to rerun/refetch)"
        continue
    fi

    echo "RUN    ${CELL}"
    if PYTHONPATH=. $PYTHON scripts/eval_compositional_holdout.py \
        --config "${CKPT_CELL_DIR}/config.yaml" --checkpoint best --output-dir "${OUT_DIR}" \
        --num-qualitative 0; then
        echo "DONE   ${CELL}"
    else
        echo "FAILED ${CELL}"
    fi
done

echo ""
echo "== frozen_td: pad experiment 2's checkpoint to num_tasks=28, then eval =="
for SEED in 1 2 3 4 5; do
    CELL="frozentd_seed${SEED}"
    CKPT_CELL_DIR="${CKPT_DIR}/hyper_multitask_dim4_frozentd_seed${SEED}"
    OUT_DIR="${OUT_ROOT}/${CELL}"
    PADDED_CKPT="${PADDED_DIR}/frozentd_seed${SEED}_padded.ckpt"

    if [ -f "${OUT_DIR}/results.txt" ]; then
        echo "SKIP   ${CELL} (already evaluated)"
        continue
    fi
    if [ ! -f "${CKPT_CELL_DIR}/best_model.ckpt" ]; then
        echo "SKIP   ${CELL} (checkpoint missing -- see experiment 2's README to rerun/refetch)"
        continue
    fi

    if [ ! -f "${PADDED_CKPT}" ]; then
        echo "PAD    ${CELL}"
        if ! PYTHONPATH=. $PYTHON experiments/04_compositional_generalization/pad_frozentd_checkpoint.py \
            --checkpoint "${CKPT_CELL_DIR}/best_model.ckpt" --num-tasks 28 --seed "${SEED}" \
            --out "${PADDED_CKPT}"; then
            echo "FAILED ${CELL} (pad)"
            continue
        fi
    fi

    echo "RUN    ${CELL}"
    if PYTHONPATH=. $PYTHON scripts/eval_compositional_holdout.py \
        --config experiments/04_compositional_generalization/configs/frozen_td.yaml \
        --checkpoint "${PADDED_CKPT}" --output-dir "${OUT_DIR}" --num-qualitative 0; then
        echo "DONE   ${CELL}"
    else
        echo "FAILED ${CELL}"
    fi
done

echo ""
echo "Next: uv run python experiments/04_compositional_generalization/plot_compositional.py --outputs-dir outputs"
