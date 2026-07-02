"""Heatmap for ARC-1D design journey: 8 progressive steps from vanilla transformer to best model.

Steps draw from multiple experiments:
  S1  Vanilla transformer (sin PE, N_sup=1)      — arc1d_recursion_ablation_large_8k
  S2  + N_sup=2 loop training (MAML-style)        — arc1d_recursion_ablation_large_8k
  S3  + RoPE (flat 3-layer, no Canon)             — arc1d_rope_story_ablation
  S4  + Looped middle (n_loops=4, no Canon)       — arc1d_rope_story_ablation
  S5  + Wide middle (outer=8, inner=32, no Canon) — arc1d_rope_story_ablation
  S6  + Canon ABCD                                — arc1d_rope_wide_middle_ablation
  S7  + Block skip                                — arc1d_rope_skip_ablation
  S8  + Block + per-iter loop h0                  — arc1d_rope_loop_skip_ablation

Usage:
    python scripts/plot_story_ablation.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STEP_LABELS = {
    "S1": "Vanilla transformer  (sin PE, N_sup=1)",
    "S2": "+ N_sup=2 loop training  (MAML-style)",
    "S3": "+ RoPE  (flat 3-layer, no Canon)",
    "S4": "+ Looped middle  (n_loops=4, weight-shared, no Canon)",
    "S5": "+ Wide middle  (outer=8 → inner=32, fewer params, more capacity)",
    "S6": "+ Canon ABCD  (neighbourhood reasoning)",
    "S7": "+ Block skip  (x += x_in per block)",
    "S8": "+ Per-iter loop h0  (context anchor each iteration)",
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

# Recursion ablation: suffixes identify the story step
_RECUR_SUFFIXES = (
    "looped_canon_recursive_transformer|looped_canon_transformer"
    "|canon_recursive_transformer|canon_transformer"
    "|looped_recursive_transformer|looped_transformer"
    "|recursive_transformer|transformer"
)
RECUR_RE = re.compile(rf"ablation_([A-H])_(.*?)_({_RECUR_SUFFIXES})(?:_seed(\d+))?$")
RECUR_SUFFIX_TO_STEP = {
    "transformer":        "S1",
    "looped_transformer": "S2",
}

# All rope experiments: suffixes identify story steps S3–S8
_ROPE_SUFFIXES = (
    "story_rope_flat|story_rope_loop|story_wide_nc"
    "|wide8_6L|skip_abcd_hw|loop_skip_f4"
)
ROPE_RE = re.compile(rf"ablation_([A-Z0-9])_(.*?)_({_ROPE_SUFFIXES})(?:_seed(\d+))?$")
ROPE_SUFFIX_TO_STEP = {
    "story_rope_flat": "S3",
    "story_rope_loop": "S4",
    "story_wide_nc":   "S5",
    "wide8_6L":        "S6",
    "skip_abcd_hw":    "S7",
    "loop_skip_f4":    "S8",
}

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

            step = None
            # Try recursion ablation pattern first
            m = RECUR_RE.match(exp_dir.name)
            if m:
                suffix = m.group(3)
                step = RECUR_SUFFIX_TO_STEP.get(suffix)
                task = m.group(2)
            else:
                m = ROPE_RE.match(exp_dir.name)
                if m:
                    suffix = m.group(3)
                    step = ROPE_SUFFIX_TO_STEP.get(suffix)
                    task = m.group(2)

            if step is None:
                continue

            text = rf.read_text()
            vm = VAL_RE.search(text)
            tm = TEST_RE.search(text)
            if not vm or not tm:
                continue

            data[task][step]["val"].append(float(vm.group(1)))
            data[task][step]["test"].append(float(tm.group(1)))
    return data


def make_heatmap(data, metric, output_path, title, task_order):
    steps = list(STEP_LABELS.keys())
    present = [s for s in steps if any(s in data[t] for t in data)]
    tasks = [t for t in task_order if t in data]

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

    # Chapter dividers (white lines) separating the 4 story beats
    chapter_ends = {"S2", "S5", "S6"}
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
    # Sort by S8 val performance (the best model ranks tasks by final difficulty)
    s8_tasks = {t for t in data if "S8" in data[t]}
    if s8_tasks:
        return sorted(
            data.keys(),
            key=lambda t: np.mean(data[t].get("S8", {}).get("val", [0.0])),
            reverse=True,
        )
    # Fallback: mean across all available steps
    all_steps = list(STEP_LABELS.keys())
    return sorted(
        data.keys(),
        key=lambda t: np.mean([
            np.mean(data[t].get(s, {}).get("val", [0.0]))
            for s in all_steps
        ]),
        reverse=True,
    )


def main():
    if len(sys.argv) > 1:
        dirs = [Path(p) for p in sys.argv[1:]]
    else:
        dirs = [
            Path("outputs/arc1d_recursion_ablation_large_8k"),
            Path("outputs/arc1d_rope_story_ablation"),
            Path("outputs/arc1d_rope_wide_middle_ablation"),
            Path("outputs/arc1d_rope_skip_ablation"),
            Path("outputs/arc1d_rope_loop_skip_ablation"),
        ]

    data = load_results(*dirs)
    n_runs = sum(len(v) for t in data.values() for v in t.values())
    print(f"Loaded: {len(data)} tasks, {n_runs} seed×step entries")

    steps_found = sorted({s for t in data.values() for s in t})
    print(f"Steps found: {steps_found}")

    out_dir = Path("outputs/story_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    order = val_task_order(data)

    split_label = {"val": "Validation", "test": "Test"}
    for metric in ("val", "test"):
        title = (
            f"ARC-1D Design Journey — {split_label[metric]} Exact Match\n"
            f"(S1–S2: dim=64 plain transformer  |  S3–S8: outer=8–16 inner=32 RoPE sandwich)"
        )
        make_heatmap(data, metric,
                     out_dir / f"heatmap_{metric}.png", title, order)


if __name__ == "__main__":
    main()
