"""Heatmaps for arc1d_sandwich_ablation: mean ± std across seeds.

12 conditions (3 n_loops × 2 N_sup × 2 Canon):
  A: sandwich,       n_loops=1 (3L), direct
  B: sandwich,       n_loops=2 (4L), direct
  C: sandwich,       n_loops=4 (6L), direct
  D: sandwich,       n_loops=1 (3L), N_sup=2
  E: sandwich,       n_loops=2 (4L), N_sup=2
  F: sandwich,       n_loops=4 (6L), N_sup=2
  G: canon_sandwich, n_loops=1 (3L), direct
  H: canon_sandwich, n_loops=2 (4L), direct
  I: canon_sandwich, n_loops=4 (6L), direct
  J: canon_sandwich, n_loops=1 (3L), N_sup=2
  K: canon_sandwich, n_loops=2 (4L), N_sup=2
  L: canon_sandwich, n_loops=4 (6L), N_sup=2

Usage:
    python scripts/plot_sandwich_ablation.py
    python scripts/plot_sandwich_ablation.py outputs/arc1d_sandwich_ablation
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Row order: N_sup=1 block (A,B,C,G,H,I) then N_sup=2 block (D,E,F,J,K,L).
# Within each block: plain sandwich first, then canon sandwich.
COND_LABELS = {
    "A": "3L  (baseline)",
    "B": "4L  (n_loops=2)",
    "C": "6L  (n_loops=4)",
    "G": "Canon 3L",
    "H": "Canon 4L",
    "I": "Canon 6L",
    "D": "3L  + N_sup=2",
    "E": "4L  + N_sup=2",
    "F": "6L  + N_sup=2",
    "J": "Canon 3L  + N_sup=2",
    "K": "Canon 4L  + N_sup=2",
    "L": "Canon 6L  + N_sup=2",
}

TASK_DISPLAY = {
    "1d_denoising_1c":  "Denoising 1c",
    "1d_denoising_mc":  "Denoising mc",
    "1d_fill":          "Fill",
    "1d_padded_fill":   "Padded Fill",
    "1d_flip":          "Flip",
    "1d_hollow":        "Hollow",
    "1d_mirror":        "Mirror",
    "1d_move_1p":       "Move 1p",
    "1d_move_2p":       "Move 2p",
    "1d_move_2p_dp":    "Move 2p dp",
    "1d_move_3p":       "Move 3p",
    "1d_move_dp":       "Move dp",
    "1d_pcopy_1c":      "Pattern Copy 1c",
    "1d_pcopy_mc":      "Pattern Copy mc",
    "1d_recolor_cmp":   "Recolor Comparison",
    "1d_recolor_cnt":   "Recolor Count",
    "1d_recolor_oe":    "Recolor Odd-Even",
    "1d_scale_dp":      "Scale dp",
}

# ablation_{COND}_{TASK}_{SUFFIX}[_seed{N}]
_SUFFIXES = (
    "looped_canon_sandwich_4loops|looped_canon_sandwich_2loops|looped_canon_sandwich"
    "|canon_sandwich_4loops|canon_sandwich_2loops|canon_sandwich"
    "|looped_sandwich_4loops|looped_sandwich_2loops|looped_sandwich"
    "|sandwich_4loops|sandwich_2loops|sandwich"
)
EXP_RE = re.compile(
    rf"ablation_([A-L])_(.*?)_({_SUFFIXES})(?:_seed(\d+))?$"
)
VAL_RE  = re.compile(r"val_query_exact_match: ([0-9.]+)")
TEST_RE = re.compile(r"test_query_exact_match: ([0-9.]+)")


def load_results(results_dir: Path):
    """Return {task: {cond: {metric: [seed_values]}}}."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for exp_dir in sorted(results_dir.iterdir()):
        rf = exp_dir / "results.txt"
        if not rf.exists():
            continue
        m = EXP_RE.match(exp_dir.name)
        if not m:
            continue
        cond, task = m.group(1), m.group(2)

        text = rf.read_text()
        vm = VAL_RE.search(text)
        tm = TEST_RE.search(text)
        if not vm or not tm:
            continue

        data[task][cond]["val"].append(float(vm.group(1)))
        data[task][cond]["test"].append(float(tm.group(1)))

    return data


def make_heatmap(data, metric, output_path, title, task_order):
    all_conds = list(COND_LABELS.keys())
    conds = [c for c in all_conds if any(c in data[t] for t in data)]
    tasks = [t for t in task_order if t in data]

    n_tasks = len(tasks)
    n_conds = len(conds)

    means = np.full((n_conds, n_tasks), np.nan)
    stds  = np.full((n_conds, n_tasks), np.nan)
    for r, cond in enumerate(conds):
        for c, task in enumerate(tasks):
            vals = data[task].get(cond, {}).get(metric, [])
            if vals:
                means[r, c] = np.mean(vals)
                stds[r, c]  = np.std(vals)

    row_means = np.nanmean(means, axis=1, keepdims=True)
    row_stds  = np.nanstd(means,  axis=1, keepdims=True)

    data_full = np.hstack([means, row_means])
    n_cols = n_tasks + 1

    fig, ax = plt.subplots(figsize=(max(12, n_cols * 0.85), 5.5))
    im = ax.imshow(data_full, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    for r in range(n_conds):
        for c in range(n_tasks):
            mu = means[r, c]
            sd = stds[r, c]
            if np.isnan(mu):
                continue
            text_color = "white" if mu < 0.6 else "black"
            ax.text(c, r - 0.13, f"{mu:.0%}", ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")
            ax.text(c, r + 0.28, f"±{sd:.0%}", ha="center", va="center",
                    fontsize=5.5, color=text_color, alpha=0.85)

        mu = float(row_means[r, 0])
        sd = float(row_stds[r, 0])
        text_color = "white" if mu < 0.6 else "black"
        ax.text(n_tasks, r - 0.13, f"{mu:.0%}", ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight="bold")
        ax.text(n_tasks, r + 0.28, f"±{sd:.0%}", ha="center", va="center",
                fontsize=5.5, color=text_color, alpha=0.85)

    # Dividers
    ax.axvline(n_tasks - 0.5, color="white", linewidth=2)
    # Major: between N_sup=1 block (A,B,C,G,H,I) and N_sup=2 block (D,E,F,J,K,L)
    if "D" in conds:
        ax.axhline(conds.index("D") - 0.5, color="white", linewidth=2.5)
    # Minor: between plain and canon within N_sup=1 block
    if "G" in conds:
        ax.axhline(conds.index("G") - 0.5, color="cyan", linewidth=1.5)
    # Minor: between plain and canon within N_sup=2 block
    if "J" in conds:
        ax.axhline(conds.index("J") - 0.5, color="cyan", linewidth=1.5)

    ax.set_yticks(range(n_conds))
    ax.set_yticklabels([COND_LABELS[c] for c in conds], fontsize=9)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [TASK_DISPLAY.get(t, t) for t in tasks] + ["Mean ± Std"],
        fontsize=8, rotation=35, ha="right",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    cbar.set_label("Exact Match", fontsize=9)

    ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def val_task_order(data):
    conds = [c for c in COND_LABELS if any(c in data[t] for t in data)]
    return sorted(
        data.keys(),
        key=lambda t: np.mean([
            np.mean(data[t].get(c, {}).get("val", [0.0]))
            for c in conds
        ]),
        reverse=True,
    )


def main():
    dirs = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else [
        Path("outputs/arc1d_sandwich_ablation"),
    ]
    dirs = [d for d in dirs if d.exists()]

    for results_dir in dirs:
        data = load_results(results_dir)
        n_runs = sum(len(v) for t in data.values() for v in t.values())
        print(f"{results_dir.name}: {len(data)} tasks, {n_runs} seed×condition entries")

        out_dir = results_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        order = val_task_order(data)

        split_label = {"val": "Validation", "test": "Test"}
        for metric in ("val", "test"):
            title = (
                f"ARC-1D Sandwich Ablation — {split_label[metric]} Exact Match"
                f"\n{results_dir.name}  (mean ± std across seeds)"
            )
            make_heatmap(data, metric, out_dir / f"heatmap_{metric}.png", title, order)


if __name__ == "__main__":
    main()
