"""Heatmaps for arc1d_uniform_ablation: 7-step design journey at dim=16 and dim=32.

Clean, single-project rebuild of the ARC-1D story (replaces the scattered
SC1-SC10 diagnostics and legacy projects) - uniform width throughout, no
outer/inner "sandwich" split, run at two capacities:

  T1: Vanilla transformer, sin PE, N_sup=1
  T2: + N_sup=2 loop training
  T3: + Canon ABCD
  T4: + RoPE, flat (n_loops=1)
  T5: + Looped middle (n_loops=4)
  T6: + Block skip
  T7: + Per-loop h0 (loop skip)

Produces two independent 7-row heatmaps (dim=16, dim=32) per metric, since each
width is its own self-contained story.

Usage:
    python scripts/plot_uniform_ablation.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STEP_LABELS = {
    "T1": "Vanilla transformer  (sin PE, N_sup=1)",
    "T2": "+ N_sup=2 loop training",
    "T3": "+ Canon ABCD",
    "T4": "+ RoPE  (flat, n_loops=1)",
    "T5": "+ Looped middle  (n_loops=4)",
    "T6": "+ Block skip",
    "T7": "+ Per-loop h0  (loop skip)",
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

_SUFFIXES = "vanilla|nsup|canon|rope_flat|looped|block_skip|loop_skip"
EXP_RE = re.compile(rf"ablation_(T[1-7])_dim(16|32)_(.*?)_({_SUFFIXES})(?:_seed(\d+))?$")

VAL_RE  = re.compile(r"val_query_exact_match: ([0-9.]+)")
TEST_RE = re.compile(r"test_query_exact_match: ([0-9.]+)")
SKIP_TASKS = {"1d_padded_fill"}
_RECOLOR_TASKS = {"1d_recolor_cmp", "1d_recolor_cnt", "1d_recolor_oe"}


def load_results(results_dir: Path):
    # data[dim][task][step][metric] -> list of seed values
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    if not results_dir.exists():
        return data
    for exp_dir in sorted(results_dir.iterdir()):
        rf = exp_dir / "results.txt"
        if not rf.exists():
            continue
        m = EXP_RE.match(exp_dir.name)
        if not m:
            continue
        step, dim, task, _suffix, _seed = m.groups()
        if task in SKIP_TASKS:
            continue

        text = rf.read_text()
        vm = VAL_RE.search(text)
        tm = TEST_RE.search(text)
        if not vm or not tm:
            continue

        data[int(dim)][task][step]["val"].append(float(vm.group(1)))
        data[int(dim)][task][step]["test"].append(float(tm.group(1)))
    return data


def make_heatmap(data, metric, output_path, title, task_order):
    steps = list(STEP_LABELS.keys())
    present = [s for s in steps if any(s in data[t] for t in data)]
    tasks = [t for t in task_order if t in data and t not in SKIP_TASKS]

    n_steps = len(present)
    n_tasks = len(tasks)

    means = np.full((n_steps, n_tasks), np.nan)
    stds  = np.full((n_steps, n_tasks), np.nan)
    for r, step in enumerate(present):
        for c, task in enumerate(tasks):
            vals = data[task].get(step, {}).get(metric, [])
            if vals:
                means[r, c] = np.mean(vals)
                stds[r, c]  = np.std(vals)

    row_means = np.nanmean(means, axis=1, keepdims=True)
    row_stds  = np.nanstd(means,  axis=1, keepdims=True)

    data_full = np.hstack([means, row_means])
    n_cols = n_tasks + 1

    fig, ax = plt.subplots(figsize=(max(14, n_cols * 0.85), n_steps * 0.9 + 1.8))
    im = ax.imshow(data_full, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    for r in range(n_steps):
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

    # Chapter dividers: plain-transformer family (T1-T3) vs RoPE family (T4-T7)
    chapter_ends = {"T3", "T5"}
    for i, step in enumerate(present):
        if step in chapter_ends and i < n_steps - 1:
            ax.axhline(i + 0.5, color="white", linewidth=2.5)

    ax.set_yticks(range(n_steps))
    ax.set_yticklabels([STEP_LABELS[s] for s in present], fontsize=9)
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
    # Sort by T7 val performance; recolor tasks pinned to the right end
    all_steps = list(STEP_LABELS.keys())
    ref_step = "T7" if any("T7" in data[t] for t in data) else all_steps[0]

    def sort_key(t):
        return np.mean(data[t].get(ref_step, {}).get("val", [0.0]))

    non_recolor = sorted(
        (t for t in data if t not in _RECOLOR_TASKS), key=sort_key, reverse=True
    )
    recolor = sorted(
        (t for t in data if t in _RECOLOR_TASKS), key=sort_key, reverse=True
    )
    return non_recolor + recolor


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/arc1d_uniform_ablation")

    data_by_dim = load_results(results_dir)

    out_dir = Path("outputs/uniform_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_label = {"val": "Validation", "test": "Test"}
    for dim in sorted(data_by_dim.keys()):
        data = data_by_dim[dim]
        n_runs = sum(len(v) for t in data.values() for v in t.values())
        print(f"dim={dim}: {len(data)} tasks, {n_runs} seed×step entries")

        steps_found = sorted({s for t in data.values() for s in t})
        print(f"  Steps found: {steps_found}")

        order = val_task_order(data)

        for metric in ("val", "test"):
            title = (
                f"ARC-1D Uniform Ablation — dim={dim} — {split_label[metric]} Exact Match\n"
                f"(T1–T3: plain transformer, num_layers=3  |  T4–T7: RoPE+Canon, n_loops 1→4)"
            )
            make_heatmap(data, metric,
                         out_dir / f"heatmap_dim{dim}_{metric}.png", title, order)


if __name__ == "__main__":
    main()
