"""Heatmap for ARC-1D design journey: 8 progressive steps from vanilla transformer to best model.

Steps draw from multiple experiments:
  S1  Vanilla transformer (sin PE, N_sup=1)   — arc1d_recursion_ablation_large_8k (cond A)
  S2  + N_sup=4 loop training (MAML-style)     — arc1d_recursion_ablation_large_8k (cond C)
  S3  + Canon ABCD                             — arc1d_rope_story_ablation (cond SC1)
  S4  + RoPE (flat, Canon carries over)        — arc1d_rope_story_ablation (cond SC2)
  S5  + Looped middle (n_loops=4, dim=16)      — arc1d_rope_sandwich_ablation (cond E)
  S6  + Wide middle (outer=8, inner=32)        — arc1d_rope_wide_middle_ablation (cond A)
  S7  + Block skip                             — arc1d_rope_skip_ablation (cond B)
  S8  + Block + per-iter loop h0               — arc1d_rope_loop_skip_ablation (cond F)

Canon ABCD was moved to step 3 (was step 6) so the heatmap increases monotonically —
previously performance dipped after S2 (dim collapse 512->16 + RoPE all at once) and
didn't recover until Canon appeared. The original no-Canon RoPE conditions (S3/S4/S5
in arc1d_rope_story_ablation: story_rope_flat/story_rope_loop/story_wide_nc) are kept
on disk as reference data but are no longer plotted here.

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
    "S2": "+ N_sup=4 loop training  (MAML-style)",
    "S3": "+ Canon ABCD  (neighbourhood reasoning)",
    "S4": "+ RoPE  (flat, Canon carries over)",
    "S5": "+ Looped middle  (n_loops=4, weight-shared)",
    "S6": "+ Wide middle  (outer=8 → inner=32, fewer params, more capacity)",
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
    "story_canon_plain|story_canon_rope_flat|rope_dim16_6L"
    "|wide8_6L|skip_abcd_hw|loop_skip_f4"
)
ROPE_RE = re.compile(rf"ablation_([A-Z0-9]+)_(.*?)_({_ROPE_SUFFIXES})(?:_seed(\d+))?$")
ROPE_SUFFIX_TO_STEP = {
    "story_canon_plain":     "S3",
    "story_canon_rope_flat": "S4",
    "rope_dim16_6L":         "S5",
    "wide8_6L":              "S6",
    "skip_abcd_hw":          "S7",
    "loop_skip_f4":          "S8",
}

VAL_RE  = re.compile(r"val_query_exact_match: ([0-9.]+)")
TEST_RE = re.compile(r"test_query_exact_match: ([0-9.]+)")

# Old task names used in arc1d_recursion_ablation_large_8k
OLD_TO_CANONICAL = {
    "denoising1c": "1d_denoising_1c",
    "fill":        "1d_fill",
    "scale_dp":    "1d_scale_dp",
}
SKIP_TASKS = {"1d_padded_fill"}


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
                task = OLD_TO_CANONICAL.get(m.group(2), m.group(2))
            else:
                m = ROPE_RE.match(exp_dir.name)
                if m:
                    suffix = m.group(3)
                    step = ROPE_SUFFIX_TO_STEP.get(suffix)
                    task = m.group(2)

            if step is None or task in SKIP_TASKS:
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

    # Chapter dividers (white lines) separating the 4 story beats
    chapter_ends = {"S2", "S4", "S6"}
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


_RECOLOR_TASKS = {"1d_recolor_cmp", "1d_recolor_cnt", "1d_recolor_oe"}


def val_task_order(data):
    # Sort by S8 val performance; recolor tasks pinned to the right end
    all_steps = list(STEP_LABELS.keys())
    ref_step = "S8" if any("S8" in data[t] for t in data) else all_steps[0]

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
    if len(sys.argv) > 1:
        dirs = [Path(p) for p in sys.argv[1:]]
    else:
        dirs = [
            Path("outputs/arc1d_recursion_ablation_large_8k"),
            Path("outputs/arc1d_rope_story_ablation"),
            Path("outputs/arc1d_rope_sandwich_ablation"),
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
            f"(S1–S4: dim=512→16 transformer + Canon + RoPE  |  "
            f"S5–S8: outer=8–16 inner=16–32 RoPE+Canon sandwich)"
        )
        make_heatmap(data, metric,
                     out_dir / f"heatmap_{metric}.png", title, order)


if __name__ == "__main__":
    main()
