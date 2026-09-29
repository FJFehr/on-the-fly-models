#!/usr/bin/env bash
# Copy experiment outputs (results, logs, figures and trained models) from another machine
# into the local outputs/ folder, with rsync over ssh.
#
# Usage:
#   REMOTE=user@host:/path/to/on-the-fly-models bash scripts/fetch_experiments.sh 02_hypernetwork_multitask
#   REMOTE=gpu-box:on-the-fly-models bash scripts/fetch_experiments.sh 01_multitask_capacity 05_leave_one_out_task_generalization
#   EXCLUDE_CHECKPOINTS=1 REMOTE=... bash scripts/fetch_experiments.sh 01_multitask_capacity
#
# Env vars:
#   REMOTE               (required) the repository on the other machine, as an rsync source
#                        (host:path); ssh options such as jump hosts belong in ~/.ssh/config
#   EXCLUDE_CHECKPOINTS  1 to skip *.ckpt files (default: copy them)

set -uo pipefail

REMOTE="${REMOTE:?Set REMOTE to the repository on the other machine, e.g. user@host:/path/to/on-the-fly-models}"
EXCLUDE_CHECKPOINTS="${EXCLUDE_CHECKPOINTS:-}"

if [[ $# -eq 0 ]]; then
    echo "Usage: REMOTE=host:/path/to/repo $0 <experiment> [<experiment> ...]"
    echo "  <experiment>: a folder under outputs/, e.g. 02_hypernetwork_multitask"
    exit 1
fi

EXCLUDES=()
[[ -n "$EXCLUDE_CHECKPOINTS" ]] && EXCLUDES=(--exclude '*.ckpt')

EXIT=0
for ARG in "$@"; do
    NAME="${ARG#outputs/}"
    mkdir -p "outputs/${NAME}"
    echo "==> ${REMOTE}/outputs/${NAME}/ -> outputs/${NAME}/"
    if rsync -az "${EXCLUDES[@]}" "${REMOTE}/outputs/${NAME}/" "outputs/${NAME}/"; then
        echo "    $(find "outputs/${NAME}" -name results.txt | wc -l) runs with results"
    else
        echo "    FAILED" >&2
        EXIT=1
    fi
done
exit $EXIT
