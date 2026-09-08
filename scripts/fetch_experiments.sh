#!/usr/bin/env bash
# Fetch experiment outputs from a torrnode, excluding model checkpoints by default.
#
# Usage:
#   bash scripts/fetch_experiments.sh arc1d_uniform_ablation
#   bash scripts/fetch_experiments.sh outputs/arc1d_uniform_ablation
#   bash scripts/fetch_experiments.sh arc1d_uniform_ablation arc1d_recursion_ablation
#   REMOTE_HOST=torrnode15.priv bash scripts/fetch_experiments.sh 01_multitask_capacity
#   INCLUDE_CHECKPOINTS=1 bash scripts/fetch_experiments.sh 02_hypernetwork_multitask
#
# Env vars:
#   REMOTE_HOST          which torrnode to pull from (default: torrnode11.priv)
#   REMOTE_REPO          repo directory name on that node, under /homes/55/fabiojfehr/
#                        (default: on-the-fly-models)
#   INCLUDE_CHECKPOINTS  1 to also pull *.ckpt / checkpoints/ (default: excluded --
#                        most experiments only need results.txt/logs/figures locally;
#                        set this when you actually need the trained weights, e.g. to
#                        run a downstream eval script against them)

set -uo pipefail

REMOTE_HOST="${REMOTE_HOST:-torrnode11.priv}"
REMOTE_REPO="${REMOTE_REPO:-on-the-fly-models}"
REMOTE_BASE="/homes/55/fabiojfehr/${REMOTE_REPO}/outputs"
LOCAL_BASE="outputs"
JUMP_HOST="robots.ox.ac.uk"
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-}"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <output_dir> [output_dir ...]"
    echo "  output_dir: name under outputs/ (e.g. arc1d_uniform_ablation)"
    echo "              or full path (e.g. outputs/arc1d_uniform_ablation)"
    exit 1
fi

# Detect if direct connection works; fall back to jump host if not.
SSH_OPTS="-o ConnectTimeout=10 -o BatchMode=yes"
if ssh $SSH_OPTS "$REMOTE_HOST" true 2>/dev/null; then
    RSYNC_SSH="ssh"
    echo "Using direct connection to $REMOTE_HOST"
else
    echo "Direct connection failed, trying jump host $JUMP_HOST ..."
    RSYNC_SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -J $JUMP_HOST"
fi

EXIT=0
for ARG in "$@"; do
    # Strip leading "outputs/" if supplied as full path
    DIR_NAME="${ARG#outputs/}"

    REMOTE_PATH="${REMOTE_HOST}:${REMOTE_BASE}/${DIR_NAME}/"
    LOCAL_PATH="${LOCAL_BASE}/${DIR_NAME}/"
    mkdir -p "$LOCAL_PATH"

    echo ""
    echo "==> Fetching: $REMOTE_PATH"
    echo "         to: $LOCAL_PATH"

    RSYNC_EXCLUDES=()
    if [[ -z "$INCLUDE_CHECKPOINTS" ]]; then
        RSYNC_EXCLUDES=(--exclude '*/checkpoints/*' --exclude '*.ckpt')
    fi

    if rsync -avz \
        "${RSYNC_EXCLUDES[@]}" \
        -e "$RSYNC_SSH" \
        "$REMOTE_PATH" "$LOCAL_PATH"; then
        N=$(find "$LOCAL_PATH" -name "results.txt" | wc -l)
        echo "    Done — $N results.txt files in $LOCAL_PATH"
    else
        echo "    FAILED: $REMOTE_PATH" >&2
        EXIT=1
    fi
done

exit $EXIT
