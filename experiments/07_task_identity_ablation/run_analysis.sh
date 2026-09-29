#!/usr/bin/env bash
# Experiment 7, evaluation only (after run_indist.sh; needs experiment 2's checkpoints):
#   1. compositional: every arm's in-distribution checkpoint on experiment 4's composite
#      categories, multi-hot then mean task vectors
#   2. representations: dump and score the four representation spaces for the cluster analysis
# Finished steps are skipped, so it is safe to rerun. Set DEVICE=cuda to use a GPU.
#
# Usage:
#   bash experiments/07_task_identity_ablation/run_analysis.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
D=experiments/07_task_identity_ablation

PYTHONPATH=. $PYTHON $D/eval_compositional.py --mode multihot --device "$DEVICE"
PYTHONPATH=. $PYTHON $D/eval_compositional.py --mode mean --arms "frozentd_latent" \
    "learnedtd_latent" "frozentd_input" "learnedtd_input" --device "$DEVICE"
PYTHONPATH=. $PYTHON $D/dump_representations.py --device "$DEVICE"
