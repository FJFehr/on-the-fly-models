"""Heatmaps for arc1d_rope_sandwich_ablation: mean ± std across seeds.

8 conditions comparing sinusoidal PE vs RoPE × dim ∈ {16,32} × depth ∈ {6L,8L}:
  A: sinusoidal, dim=16, n_loops=4 (6L)
  B: sinusoidal, dim=16, n_loops=6 (8L)
  E: RoPE,       dim=16, n_loops=4 (6L)
  F: RoPE,       dim=16, n_loops=6 (8L)
  C: sinusoidal, dim=32, n_loops=4 (6L)
  D: sinusoidal, dim=32, n_loops=6 (8L)
  G: RoPE,       dim=32, n_loops=4 (6L)
  H: RoPE,       dim=32, n_loops=6 (8L)

Rows are grouped by dim (16 then 32), sin before RoPE within each group.
Cyan dividers separate sin/RoPE within each dim block; white divider between dims.

Usage:
    python scripts/plot_rope_sandwich_ablation.py
    python scripts/plot_rope_sandwich_ablation.py outputs/arc1d_rope_sandwich_ablation
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Row order: dim=16 block (A,B,E,F) then dim=32 block (C,D,G,H).
# Within each dim: sinusoidal first, then RoPE.
COND_LABELS = {
    "A": "Sin  dim=16  6L",
    "B": "Sin  dim=16  8L",
    "E": "RoPE dim=16  6L",
    "F": "RoPE dim=16  8L",
    "C": "Sin  dim=32  6L",
    "D": "Sin  dim=32  8L",
    "G": "RoPE dim=32  6L",
    "H": "RoPE dim=32  8L",
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

_SUFFIXES = (
    "sin_dim16_6L|sin_dim16_8L|sin_dim32_6L|sin_dim32_8L"
    "|rope_dim16_6L|rope_dim16_8L|rope_dim32_6L|rope_dim32_8L"
)
EXP_RE = re.compile(
    rf"ablation_([A-H])_(.*?)_({_SUFFIXES})(?:_seed(\d+))?$"
)
VAL_RE  = re.compile(r"val_query_exact_match: ([0-9.]+)")
TEST_RE = re.compile(r"test_query_exact_match: ([0-9.]+)")


def load_results(results_dir: Path):
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

    # Vertical divider before Mean column
    ax.axvline(n_tasks - 0.5, color="white", linewidth=2)
    # White divider between dim=16 block and dim=32 block (before C)
    if "C" in conds:
        ax.axhline(conds.index("C") - 0.5, color="white", linewidth=2.5)
    # Cyan divider between sin and RoPE within dim=16 block (before E)
    if "E" in conds:
        ax.axhline(conds.index("E") - 0.5, color="cyan", linewidth=1.5)
    # Cyan divider between sin and RoPE within dim=32 block (before G)
    if "G" in conds:
        ax.axhline(conds.index("G") - 0.5, color="cyan", linewidth=1.5)

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
        Path("outputs/arc1d_rope_sandwich_ablation"),
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
                f"ARC-1D RoPE vs Sinusoidal — {split_label[metric]} Exact Match"
                f"\n{results_dir.name}  (mean ± std across seeds)"
            )
            make_heatmap(data, metric, out_dir / f"heatmap_{metric}.png", title, order)


if __name__ == "__main__":
    main()
