"""Heatmaps for arc1d_rope_dim_ablation: mean ± std across seeds.

8 conditions (A and F reused from prior experiments, H–M new):
  A: outer=8,  inner=32, no skip           (from arc1d_rope_wide_middle_ablation)
  F: outer=8,  inner=32, block+loop skip   (from arc1d_rope_loop_skip_ablation)
  H: outer=8,  inner=16, no skip
  I: outer=8,  inner=16, block+loop skip
  J: outer=8,  inner=64, no skip
  K: outer=8,  inner=64, block+loop skip
  L: outer=16, inner=32, no skip
  M: outer=16, inner=32, block+loop skip

White divider separates reference conditions (A, F) from new conditions (H–M).

Usage:
    python scripts/plot_rope_dim_ablation.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COND_LABELS = {
    "A": "outer8  inner32  no skip  (ref)",
    "F": "outer8  inner32  blk+loop skip  (ref)",
    "H": "outer8  inner16  no skip        6.4k",
    "I": "outer8  inner16  blk+loop skip  6.4k",
    "J": "outer8  inner64  no skip       55.6k",
    "K": "outer8  inner64  blk+loop skip 55.6k",
    "L": "outer16 inner32  no skip       22.6k",
    "M": "outer16 inner32  blk+loop skip 22.6k",
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
    "wide8_6L"           # A
    "|loop_skip_f4"      # F
    "|dim_h16_base"      # H
    "|dim_i16_skip"      # I
    "|dim_j64_base"      # J
    "|dim_k64_skip"      # K
    "|dim_l16out_base"   # L
    "|dim_m16out_skip"   # M
)
EXP_RE = re.compile(rf"ablation_([A-M])_(.*?)_({_SUFFIXES})(?:_seed(\d+))?$")
VAL_RE  = re.compile(r"val_query_exact_match: ([0-9.]+)")
TEST_RE = re.compile(r"test_query_exact_match: ([0-9.]+)")


def load_results(*results_dirs: Path):
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for results_dir in results_dirs:
        if not results_dir.exists():
            continue
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

    fig, ax = plt.subplots(figsize=(max(14, n_cols * 0.85), n_conds * 0.9 + 1.5))
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

    ax.axvline(n_tasks - 0.5, color="white", linewidth=2)
    # Divider after reference conditions (A, F)
    if "F" in conds and "H" in conds:
        ax.axhline(conds.index("H") - 0.5, color="white", linewidth=2.5)

    ax.set_yticks(range(n_conds))
    ax.set_yticklabels([COND_LABELS[c] for c in conds], fontsize=8.5)
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
    if len(sys.argv) > 1:
        dirs = [Path(p) for p in sys.argv[1:]]
    else:
        dirs = [
            Path("outputs/arc1d_rope_wide_middle_ablation"),
            Path("outputs/arc1d_rope_loop_skip_ablation"),
            Path("outputs/arc1d_rope_dim_ablation"),
        ]

    data = load_results(*dirs)
    n_runs = sum(len(v) for t in data.values() for v in t.values())
    print(f"Loaded: {len(data)} tasks, {n_runs} seed×condition entries")

    out_dir = Path("outputs/arc1d_rope_dim_ablation/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    order = val_task_order(data)

    split_label = {"val": "Validation", "test": "Test"}
    for metric in ("val", "test"):
        title = (
            f"ARC-1D Dimension Ablation — {split_label[metric]} Exact Match"
            f"\n(mean ± std across seeds, Canon ABCD RoPE n_loops=4)"
        )
        make_heatmap(data, metric, out_dir / f"heatmap_{metric}.png", title, order)


if __name__ == "__main__":
    main()
