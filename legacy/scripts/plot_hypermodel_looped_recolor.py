"""Heatmap for one arc1d_hypermodel_looped_recolor task: n_loops x (N_supervision, skip).

Single seed per cell (no averaging) -- this sweep was run at 1 seed, and val/test are only
5 examples each, so scores are quantised to 20% steps and noisy. The heatmap shows the raw
per-cell value rather than a mean +/- std.

Usage:
    python scripts/plot_hypermodel_looped_recolor.py 1d_flip
    python scripts/plot_hypermodel_looped_recolor.py 1d_recolor_cmp
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("outputs/arc1d_hypermodel_looped_recolor")
OUT_DIR = Path("outputs/hypermodel_looped_recolor")

LOOPS = [4, 8, 16]
ROWS = [
    ("N_sup=2, no skip", 2, "noskip"),
    ("N_sup=2, skip", 2, "skip"),
    ("N_sup=4, no skip", 4, "noskip"),
    ("N_sup=4, skip", 4, "skip"),
]

VAL_RE = re.compile(r"^val_query_exact_match: ([0-9.]+)", re.MULTILINE)
TEST_RE = re.compile(r"^test_query_exact_match: ([0-9.]+)", re.MULTILINE)


def load_cell(task: str, nsup: int, skip: str, n_loops: int) -> tuple[float, float] | None:
    exp_dir = RESULTS_DIR / f"looped_hyper_{task.removeprefix('1d_')}_n{nsup}_loop{n_loops}_{skip}"
    rf = exp_dir / "results.txt"
    if not rf.exists():
        return None
    text = rf.read_text()
    vm, tm = VAL_RE.search(text), TEST_RE.search(text)
    if not vm or not tm:
        return None
    return float(vm.group(1)), float(tm.group(1))


def make_heatmap(task: str, metric_idx: int, metric_name: str, ax, show_ylabels: bool) -> None:
    data = np.full((len(ROWS), len(LOOPS)), np.nan)
    for r, (_, nsup, skip) in enumerate(ROWS):
        for c, n_loops in enumerate(LOOPS):
            cell = load_cell(task, nsup, skip, n_loops)
            if cell is not None:
                data[r, c] = cell[metric_idx]

    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    for r in range(len(ROWS)):
        for c in range(len(LOOPS)):
            v = data[r, c]
            if np.isnan(v):
                continue
            color = "white" if v < 0.6 else "black"
            ax.text(c, r, f"{v:.0%}", ha="center", va="center", fontsize=11,
                     fontweight="bold", color=color)
    ax.set_xticks(range(len(LOOPS)))
    ax.set_xticklabels([f"n_loops={n}" for n in LOOPS])
    ax.set_yticks(range(len(ROWS)))
    if show_ylabels:
        ax.set_yticklabels([label for label, _, _ in ROWS])
    else:
        ax.set_yticklabels([])
    ax.set_title(f"{metric_name} exact match", fontsize=11)
    return im


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else "1d_flip"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im = None
    for i, (ax, metric_idx, metric_name) in enumerate(zip(axes, (0, 1), ("Validation", "Test"))):
        im = make_heatmap(task, metric_idx, metric_name, ax, show_ylabels=(i == 0))
    fig.subplots_adjust(wspace=0.08)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.suptitle(f"{task} — n_loops x (N_supervision, skip) sweep, 1 seed, 5 val/5 test examples",
                 fontsize=12)

    out_path = OUT_DIR / f"heatmap_{task.removeprefix('1d_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
