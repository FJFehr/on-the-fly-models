"""Plot the 8-condition canon ablation results as grouped bar charts and heatmaps.

Conditions A–D are the baselines (plain transformer / recursive / looped variants).
Conditions E–H are their Canon-augmented mirrors (canon_set="ABCD").
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("outputs/arc1d_recursion_ablation_128_canon")

# Paired palette: E/F/G/H are the Canon counterparts of A/B/C/D.
# Canon bars are slightly darker and drawn with a hatch pattern.
COND_LABELS = {
    "A": "Transformer",
    "B": "Recursive",
    "C": "Looped",
    "D": "Recursive + Looped",
    "E": "Canon Transformer",
    "F": "Canon Recursive",
    "G": "Canon Looped",
    "H": "Canon Recursive + Looped",
}

_BASE_COLOURS = {
    "A": "#4C72B0",
    "B": "#55A868",
    "C": "#C44E52",
    "D": "#DD8452",
}
# Canon variants: desaturated + hatched so they visually pair with their baseline
_CANON_COLOURS = {
    "E": "#7FA8D6",  # lighter blue  (mirrors A)
    "F": "#8FCCA0",  # lighter green (mirrors B)
    "G": "#E08090",  # lighter red   (mirrors C)
    "H": "#EDB080",  # lighter orange (mirrors D)
}
COND_COLOURS = {**_BASE_COLOURS, **_CANON_COLOURS}
COND_HATCH = {c: "" for c in "ABCD"} | {c: "//" for c in "EFGH"}

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

_ALL_SUFFIXES = "|".join([
    "transformer", "recursive_transformer",
    "looped_transformer", "looped_recursive_transformer",
    "canon_transformer", "canon_recursive_transformer",
    "looped_canon_transformer", "looped_canon_recursive_transformer",
])


def load_results():
    """Return dict: {task: {cond: {val: float, test: float}}}"""
    results: dict[str, dict[str, dict[str, float]]] = {}

    pattern = re.compile(rf"ablation_([A-H])_(.+?)_({_ALL_SUFFIXES})$")
    val_re = re.compile(r"val_query_exact_match: ([0-9.]+)")
    test_re = re.compile(r"test_query_exact_match: ([0-9.]+)")

    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        rf = exp_dir / "results.txt"
        if not rf.exists():
            continue
        m = pattern.match(exp_dir.name)
        if not m:
            continue
        cond, task = m.group(1), m.group(2)

        text = rf.read_text()
        vm = val_re.search(text)
        tm = test_re.search(text)
        if not vm or not tm:
            continue

        results.setdefault(task, {})[cond] = {
            "val":  float(vm.group(1)),
            "test": float(tm.group(1)),
        }

    return results


def make_figure(results, metric, output_path):
    tasks_all = sorted(results.keys())

    def avg_score(task):
        vals = [v[metric] for v in results[task].values()]
        return sum(vals) / len(vals) if vals else 0.0

    tasks = sorted(tasks_all, key=avg_score, reverse=True)

    # Group bars as paired clusters: (A,E), (B,F), (C,G), (D,H) per task
    pair_order = [("A", "E"), ("B", "F"), ("C", "G"), ("D", "H")]
    conds_flat = [c for pair in pair_order for c in pair]
    n_tasks = len(tasks)
    n_conds = len(conds_flat)

    bar_width = 0.10
    pair_gap = 0.04   # gap between pairs
    group_gap = 0.14  # extra gap between tasks

    # Build x positions: 4 pairs per task, small gap between pairs
    group_width = 4 * (2 * bar_width + pair_gap) + group_gap
    x = np.arange(n_tasks) * group_width

    fig, ax = plt.subplots(figsize=(max(16, n_tasks * 1.1), 5.5))

    pair_positions = []
    for pi, (base, canon) in enumerate(pair_order):
        pair_left = x + pi * (2 * bar_width + pair_gap) - (4 * (2 * bar_width + pair_gap) - pair_gap) / 2
        for j, cond in enumerate([base, canon]):
            xpos = pair_left + j * bar_width
            heights = [results[t].get(cond, {}).get(metric, 0.0) for t in tasks]
            ax.bar(
                xpos, heights, width=bar_width,
                color=COND_COLOURS[cond], hatch=COND_HATCH[cond],
                label=COND_LABELS[cond], zorder=3,
                edgecolor="white", linewidth=0.4,
            )
        pair_positions.append(pair_left + bar_width / 2)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY.get(t, t) for t in tasks], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Exact Match", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--", zorder=1)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    split_label = "Validation" if metric == "val" else "Test"
    ax.set_title(
        f"ARC-1D Canon Ablation — {split_label} Exact Match\n"
        "Hatched = Canon (ABCD); solid = baseline | sorted by mean score",
        fontsize=12,
    )

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8, framealpha=0.9,
              edgecolor="grey", ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def make_heatmap(results, metric, output_path, task_order):
    conds = ["A", "E", "B", "F", "C", "G", "D", "H"]
    cond_labels = [COND_LABELS[c] for c in conds]
    task_labels = [TASK_DISPLAY.get(t, t) for t in task_order]

    data = np.array([
        [results[t].get(c, {}).get(metric, float("nan")) for t in task_order]
        for c in conds
    ])

    n_tasks = len(task_order)
    n_conds = len(conds)

    means = np.nanmean(data, axis=1, keepdims=True)
    stds = np.nanstd(data, axis=1, keepdims=True)
    data_full = np.hstack([data, means])
    n_cols = n_tasks + 1

    fig, ax = plt.subplots(figsize=(max(12, n_cols * 0.75), 0.7 * n_conds + 1.5))
    im = ax.imshow(data_full, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    for r in range(n_conds):
        for c in range(n_tasks):
            val = data_full[r, c]
            if not np.isnan(val):
                text_color = "white" if val < 0.6 else "black"
                ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                        fontsize=7.5, color=text_color, fontweight="bold")
        mu, sd = float(means[r, 0]), float(stds[r, 0])
        text_color = "white" if mu < 0.6 else "black"
        ax.text(n_tasks, r, f"{mu:.0%}\n±{sd:.0%}", ha="center", va="center",
                fontsize=7, color=text_color, fontweight="bold", linespacing=1.4)

    # Horizontal dividers between pairs (after every 2 rows)
    for div in [1.5, 3.5, 5.5]:
        ax.axhline(div, color="white", linewidth=2)
    ax.axvline(n_tasks - 0.5, color="white", linewidth=2)

    ax.set_yticks(range(n_conds))
    ax.set_yticklabels(cond_labels, fontsize=9)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(task_labels + ["Mean ± Std"], fontsize=8.5, rotation=35, ha="right")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    cbar.set_label("Exact Match", fontsize=9)

    split_label = "Validation" if metric == "val" else "Test"
    ax.set_title(f"ARC-1D Canon Ablation — {split_label} Exact Match (A/E, B/F, C/G, D/H pairs)",
                 fontsize=11, pad=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    results = load_results()
    print(f"Loaded {len(results)} tasks, {sum(len(v) for v in results.values())} condition entries")

    out_dir = Path("outputs/arc1d_recursion_ablation_128_canon/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    def avg_score(task, metric="val"):
        vals = [v[metric] for v in results[task].values()]
        return sum(vals) / len(vals) if vals else 0.0

    task_order_val = sorted(results.keys(), key=lambda t: avg_score(t, "val"), reverse=True)

    make_figure(results, "val",  out_dir / "canon_ablation_val.png")
    make_figure(results, "test", out_dir / "canon_ablation_test.png")
    make_heatmap(results, "val",  out_dir / "canon_ablation_heatmap_val.png",  task_order_val)
    make_heatmap(results, "test", out_dir / "canon_ablation_heatmap_test.png", task_order_val)


if __name__ == "__main__":
    main()
