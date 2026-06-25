"""Plot the 4-condition recursion ablation results as grouped bar charts."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path("outputs/arc1d_recursion_ablation")

# Condition display names and colours
CONDITIONS = {
    "A": ("Transformer", "#4C72B0"),
    "B": ("Recursive", "#55A868"),
    "C": ("Looped Training", "#C44E52"),
    "D": "Recursive + Looped",
}
COND_LABELS = {
    "A": "Transformer",
    "B": "Recursive",
    "C": "Looped Training",
    "D": "Recursive + Looped",
}
COND_COLOURS = {
    "A": "#4C72B0",
    "B": "#55A868",
    "C": "#C44E52",
    "D": "#DD8452",
}

# Task canonical names: old-dir-task → display name
# Prefer new runs (1d_*); old fill/scale_dp are duplicates → skip
OLD_TO_CANONICAL = {
    "denoising1c": "1d_denoising_1c",
    # fill and scale_dp have new equivalents; handled below
}
SKIP_OLD = {"fill", "scale_dp"}

TASK_DISPLAY = {
    "1d_denoising_1c":   "Denoising 1c",
    "1d_denoising_mc":   "Denoising mc",
    "1d_fill":           "Fill",
    "1d_padded_fill":    "Padded Fill",
    "1d_flip":           "Flip",
    "1d_hollow":         "Hollow",
    "1d_mirror":         "Mirror",
    "1d_move_1p":        "Move 1p",
    "1d_move_2p":        "Move 2p",
    "1d_move_2p_dp":     "Move 2p dp",
    "1d_move_3p":        "Move 3p",
    "1d_move_dp":        "Move dp",
    "1d_pcopy_1c":       "Pattern Copy 1c",
    "1d_pcopy_mc":       "Pattern Copy mc",
    "1d_recolor_cmp":    "Recolor Comparison",
    "1d_recolor_cnt":    "Recolor Count",
    "1d_recolor_oe":     "Recolor Odd-Even",
    "1d_scale_dp":       "Scale dp",
}


def load_results():
    """Return dict: {canonical_task: {cond: {val: float, test: float}}}"""
    results: dict[str, dict[str, dict[str, float]]] = {}

    pattern = re.compile(
        r"ablation_([ABCD])_(.*?)_(transformer|recursive_transformer|looped_transformer|looped_recursive_transformer)$"
    )
    val_re  = re.compile(r"val_query_exact_match: ([0-9.]+)")
    test_re = re.compile(r"test_query_exact_match: ([0-9.]+)")

    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        rf = exp_dir / "results.txt"
        if not rf.exists():
            continue
        m = pattern.match(exp_dir.name)
        if not m:
            continue
        cond, task, _ = m.group(1), m.group(2), m.group(3)

        if task in SKIP_OLD:
            continue
        canonical = OLD_TO_CANONICAL.get(task, task)

        text = rf.read_text()
        vm = val_re.search(text)
        tm = test_re.search(text)
        if not vm or not tm:
            continue

        results.setdefault(canonical, {})[cond] = {
            "val":  float(vm.group(1)),
            "test": float(tm.group(1)),
        }

    return results


def make_figure(results, metric, output_path):
    tasks_all = sorted(results.keys())

    # Sort: average score across all conditions, descending
    def avg_score(task):
        vals = [v[metric] for v in results[task].values()]
        return sum(vals) / len(vals) if vals else 0.0

    tasks = sorted(tasks_all, key=avg_score, reverse=True)

    conds = ["A", "B", "C", "D"]
    n_tasks = len(tasks)
    n_conds = len(conds)

    bar_width = 0.18
    group_gap = 0.1
    group_width = n_conds * bar_width + group_gap
    x = np.arange(n_tasks) * group_width

    fig, ax = plt.subplots(figsize=(max(14, n_tasks * 0.9), 5.5))

    for i, cond in enumerate(conds):
        heights = [results[t].get(cond, {}).get(metric, 0.0) for t in tasks]
        xpos = x + i * bar_width - (n_conds - 1) * bar_width / 2
        ax.bar(xpos, heights, width=bar_width, color=COND_COLOURS[cond],
               label=COND_LABELS[cond], zorder=3, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [TASK_DISPLAY.get(t, t) for t in tasks],
        rotation=35, ha="right", fontsize=9,
    )
    ax.set_ylabel("Exact Match", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--", zorder=1)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    split_label = "Validation" if metric == "val" else "Test"
    ax.set_title(
        f"ARC-1D Recursion Ablation — {split_label} Exact Match\n"
        "(sorted by mean score, highest first)",
        fontsize=12,
    )

    legend = ax.legend(
        loc="upper right", fontsize=9, framealpha=0.9,
        edgecolor="grey", ncol=2,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def make_heatmap(results, metric, output_path, task_order):
    conds = ["A", "B", "C", "D"]
    cond_labels = [COND_LABELS[c] for c in conds]
    task_labels = [TASK_DISPLAY.get(t, t) for t in task_order]

    # Shape: (n_conds, n_tasks) — models on Y, tasks on X
    data = np.array([
        [results[t].get(c, {}).get(metric, float("nan")) for t in task_order]
        for c in conds
    ])

    n_tasks = len(task_order)
    n_conds = len(conds)

    # Summary column: mean per condition
    means = np.nanmean(data, axis=1, keepdims=True)
    stds  = np.nanstd(data,  axis=1, keepdims=True)
    data_full = np.hstack([data, means])

    n_cols = n_tasks + 1
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.75), 3.2))

    im = ax.imshow(data_full, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    # Annotate task cells
    for r in range(n_conds):
        for c in range(n_tasks):
            val = data_full[r, c]
            if not np.isnan(val):
                text_color = "white" if val < 0.6 else "black"
                ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")
        # Annotate summary column: mean ± std
        mu, sd = float(means[r, 0]), float(stds[r, 0])
        text_color = "white" if mu < 0.6 else "black"
        ax.text(n_tasks, r, f"{mu:.0%}\n±{sd:.0%}", ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight="bold", linespacing=1.4)

    # Divider line before summary column
    ax.axvline(n_tasks - 0.5, color="white", linewidth=2)

    ax.set_yticks(range(n_conds))
    ax.set_yticklabels(cond_labels, fontsize=10)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(task_labels + ["Mean ± Std"], fontsize=8.5, rotation=35, ha="right")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    cbar.set_label("Exact Match", fontsize=9)

    split_label = "Validation" if metric == "val" else "Test"
    ax.set_title(f"ARC-1D Ablation — {split_label} Exact Match",
                 fontsize=11, pad=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    results = load_results()
    print(f"Loaded {len(results)} tasks, {sum(len(v) for v in results.values())} condition entries")

    out_dir = Path("outputs/arc1d_recursion_ablation/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Val figures determine sort order; test reuses same order
    def avg_score(task, metric="val"):
        vals = [v[metric] for v in results[task].values()]
        return sum(vals) / len(vals) if vals else 0.0

    task_order_val = sorted(results.keys(), key=lambda t: avg_score(t, "val"), reverse=True)

    make_figure(results, "val",  out_dir / "ablation_val.png")
    make_figure(results, "test", out_dir / "ablation_test.png")
    make_heatmap(results, "val",  out_dir / "ablation_heatmap_val.png",  task_order_val)
    make_heatmap(results, "test", out_dir / "ablation_heatmap_test.png", task_order_val)


if __name__ == "__main__":
    main()
