"""Heatmaps for multi-seed ablation results: mean ± std across seeds.

Usage:
    # one variant
    python scripts/plot_ablation_seeds.py outputs/arc1d_recursion_ablation_large_8k

    # all four variants (produces one plot directory each)
    python scripts/plot_ablation_seeds.py \
        outputs/arc1d_recursion_ablation \
        outputs/arc1d_recursion_ablation_8k \
        outputs/arc1d_recursion_ablation_large \
        outputs/arc1d_recursion_ablation_large_8k
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COND_LABELS = {
    "A": "Transformer",
    "B": "Recursive",
    "C": "Looped Training",
    "D": "Recursive + Looped",
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

# ablation_{COND}_{TASK}_{MODEL}_seed{N}  OR  ablation_{COND}_{TASK}_{MODEL} (no seed)
EXP_RE = re.compile(
    r"ablation_([ABCD])_(.*?)_(transformer|recursive_transformer|looped_transformer|looped_recursive_transformer)"
    r"(?:_seed(\d+))?$"
)
VAL_RE  = re.compile(r"val_query_exact_match: ([0-9.]+)")
TEST_RE = re.compile(r"test_query_exact_match: ([0-9.]+)")

OLD_TO_CANONICAL = {
    "denoising1c": "1d_denoising_1c",
    "fill":        "1d_fill",
    "scale_dp":    "1d_scale_dp",
}
SKIP_OLD: set[str] = set()


def load_results(results_dir: Path):
    """Return {task: {cond: {metric: [seed_values]}}}."""
    data: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for exp_dir in sorted(results_dir.iterdir()):
        rf = exp_dir / "results.txt"
        if not rf.exists():
            continue
        m = EXP_RE.match(exp_dir.name)
        if not m:
            continue
        cond, task = m.group(1), m.group(2)

        if task in SKIP_OLD:
            continue
        task = OLD_TO_CANONICAL.get(task, task)

        text = rf.read_text()
        vm = VAL_RE.search(text)
        tm = TEST_RE.search(text)
        if not vm or not tm:
            continue

        data[task][cond]["val"].append(float(vm.group(1)))
        data[task][cond]["test"].append(float(tm.group(1)))

    return data


def make_heatmap(data, metric, output_path, title, task_order):
    conds = ["A", "B", "C", "D"]
    tasks = [t for t in task_order if t in data]

    n_tasks = len(tasks)
    n_conds = len(conds)

    # mean and std matrices: (n_conds, n_tasks)
    means = np.full((n_conds, n_tasks), np.nan)
    stds  = np.full((n_conds, n_tasks), np.nan)
    for r, cond in enumerate(conds):
        for c, task in enumerate(tasks):
            vals = data[task].get(cond, {}).get(metric, [])
            if vals:
                means[r, c] = np.mean(vals)
                stds[r, c]  = np.std(vals)

    # Summary column: mean across tasks per condition
    row_means = np.nanmean(means, axis=1, keepdims=True)
    row_stds  = np.nanstd(means,  axis=1, keepdims=True)

    data_full = np.hstack([means, row_means])
    n_cols = n_tasks + 1

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.85), 3.4))
    im = ax.imshow(data_full, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    for r in range(n_conds):
        for c in range(n_tasks):
            mu = means[r, c]
            sd = stds[r, c]
            if np.isnan(mu):
                continue
            text_color = "white" if mu < 0.6 else "black"
            # Mean in normal size, ±std smaller beneath it
            ax.text(c, r - 0.13, f"{mu:.0%}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")
            ax.text(c, r + 0.28, f"±{sd:.0%}", ha="center", va="center",
                    fontsize=6, color=text_color, alpha=0.85)

        # Summary column
        mu = float(row_means[r, 0])
        sd = float(row_stds[r, 0])
        text_color = "white" if mu < 0.6 else "black"
        ax.text(n_tasks, r - 0.13, f"{mu:.0%}", ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold")
        ax.text(n_tasks, r + 0.28, f"±{sd:.0%}", ha="center", va="center",
                fontsize=6, color=text_color, alpha=0.85)

    ax.axvline(n_tasks - 0.5, color="white", linewidth=2)

    ax.set_yticks(range(n_conds))
    ax.set_yticklabels([COND_LABELS[c] for c in conds], fontsize=10)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [TASK_DISPLAY.get(t, t) for t in tasks] + ["Mean ± Std"],
        fontsize=8.5, rotation=35, ha="right",
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
    """Return tasks sorted by mean val exact match descending."""
    conds = ["A", "B", "C", "D"]
    return sorted(
        data.keys(),
        key=lambda t: np.mean([
            np.mean(data[t].get(c, {}).get("val", [0.0]))
            for c in conds
        ]),
        reverse=True,
    )


def process(results_dir: Path, task_order=None):
    data = load_results(results_dir)
    n_runs = sum(len(v) for t in data.values() for v in t.values())
    print(f"{results_dir.name}: {len(data)} tasks, {n_runs} seed×condition entries")

    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    order = task_order if task_order is not None else val_task_order(data)

    split_label = {"val": "Validation", "test": "Test"}
    for metric in ("val", "test"):
        title = (
            f"ARC-1D Ablation — {split_label[metric]} Exact Match"
            f"\n{results_dir.name}  (mean ± std across seeds)"
        )
        make_heatmap(data, metric, out_dir / f"heatmap_{metric}.png", title, order)


def main():
    dirs = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else [
        Path("outputs/arc1d_recursion_ablation_8k"),
        Path("outputs/arc1d_recursion_ablation_large"),
        Path("outputs/arc1d_recursion_ablation_large_8k"),
    ]
    dirs = [d for d in dirs if d.exists()]

    # Derive fixed task order from 8k val results
    ref_dir = Path("outputs/arc1d_recursion_ablation_8k")
    ref_data = load_results(ref_dir)
    order = val_task_order(ref_data)
    print(f"Task order (from {ref_dir.name} val): {order}\n")

    for d in dirs:
        process(d, task_order=order)


if __name__ == "__main__":
    main()
