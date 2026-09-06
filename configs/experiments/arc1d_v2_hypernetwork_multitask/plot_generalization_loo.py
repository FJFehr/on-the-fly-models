"""Macro-average generalization bar chart: task ID vs. no task ID, own vs. cross-instance.

Companion to plot_capacity_cliff.py/plot_per_task.py, same palette/templating, but for a
different question: scripts/measure_compute_efficiency.py's Phase E leave-one-out test (does
ONE generated weight set, from one instance, solve the OTHER instances of its own task
category?) run on the dim=4 arc1d_v2_hypernetwork_multitask checkpoints -- notd (seed2, the
best of 3 freshly-checkpointed retrains) vs. frozen_td (seed1) -- both evaluated on the same
richer eval pool (data/arc_1d_looped_augmented_devtest_shifted, n=100 instances/category, not
the base dataset's fixed 5+5) for a fair, low-noise comparison.

Deliberately simplistic (Fabio's own framing): two x-axis groups (Task ID / No Task ID), each
with two bars -- Own (per-instance generation, each instance scored on its own query) and
Cross-instance (leave-one-out: every instance in turn as the reference, scored on the other
99). Bar height = macro-average across the 14 task categories; whiskers = population std
(statistics.pstdev, matching plot_capacity_cliff.py/plot_per_task.py's own convention) of
those 14 per-category means -- how consistent the condition is ACROSS task types, not the
within-category instance-level spread (that lives in results_generalization_loo.csv's
own_std/loo_std columns instead, and in the earlier per-task horizontal chart).

Color encodes condition (task ID vs. not), matching the legend labels already established in
plot_per_task.py/plot_capacity_cliff.py: plum for task ID (Fabio's explicit call, replacing
the purple used there), teal for no task ID. Fill darkness (full color vs. lighten()'d)
encodes the metric (Own vs. Cross-instance) within each condition -- reusing the same
PAPER_COLORS/PAPER_FILL_COLORS outline-in-full-color/lighter-fill-inside treatment as every
other figure in this repo, just applied along a different axis than usual.

All raw per-task numbers (this script's own macro-average is computed from them, not
hardcoded) live in results_generalization_loo.csv next to this script (committed -- outputs/
itself is gitignored and only exists on the cluster node that ran the eval).

Usage
-----
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_generalization_loo.py
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch
from matplotlib import pyplot as plt

from visualisation.arc_paper import PAPER_COLORS, PAPER_FILL_COLORS
from visualisation.style import apply_latex_style

HERE = Path(__file__).parent
CSV_PATH = HERE / "results_generalization_loo.csv"
PLOT_PATH = HERE / "generalization_loo.png"

# Fabio's explicit color call: plum (not purple) for task ID, teal for no task ID -- same
# PAPER_COLORS palette every other figure in this repo draws from.
COLORS = {
    "notd": PAPER_COLORS[5],  # teal -- joint, no task ID
    "td": PAPER_COLORS[9],  # plum -- joint, with task ID
}
FILL_COLORS = {
    "notd": PAPER_FILL_COLORS[5],
    "td": PAPER_FILL_COLORS[9],
}
# Matches the legend wording already established in plot_per_task.py/plot_capacity_cliff.py.
CONDITION_LABELS = {"td": "Task ID", "notd": "w/o Task ID"}
# "Same-instance" (not "Own"): weights generated from THIS instance's own support set, scored
# on this instance's own query -- the hypernetwork's normal, no-reuse behavior. Named to read
# as the direct opposite of "Cross-instance" (weights from a DIFFERENT instance, reused here).
METRIC_LABELS = {"own": "Same-instance", "loo": "Cross-instance"}


def read_macro_series(path: Path) -> dict[str, dict[str, dict]]:
    """condition -> metric ('own'/'loo') -> {mean, std, n}, macro-averaged across the 14 task
    categories in the CSV (mean/pstdev of the per-category means, not the within-category
    instance-level own_std/loo_std columns)."""
    per_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            per_task[row["condition"]]["own"].append(float(row["own_mean"]))
            per_task[row["condition"]]["loo"].append(float(row["loo_mean"]))

    series: dict[str, dict[str, dict]] = {}
    for condition, metrics in per_task.items():
        series[condition] = {
            metric: {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "n": len(values),
            }
            for metric, values in metrics.items()
        }
    return series


def plot(series: dict[str, dict[str, dict]], out_path: Path) -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
        }
    )

    conditions = ("td", "notd")
    metrics = ("own", "loo")  # Same-instance first, Cross-instance second
    x = np.arange(len(conditions))
    bar_w = 0.32

    fig, ax = plt.subplots(figsize=(4.8, 4.3))

    # Cross-instance (the headline result -- what we want emphasized) gets the darker, full
    # -saturation fill; Own gets the lighter fill. Same outline-in-full-colour/lighter-fill
    # -inside treatment every other figure in this repo uses, just distinguishing metric here
    # (own vs. cross-instance) instead of category, and inverted from the usual "primary
    # thing = full color" default because the emphasis here is on Cross-instance, not Own.
    metric_face = {"own": FILL_COLORS, "loo": COLORS}

    for j, metric in enumerate(metrics):
        offset = (j - (len(metrics) - 1) / 2) * bar_w
        means = np.array([series[cond][metric]["mean"] for cond in conditions])
        stds = np.array([series[cond][metric]["std"] for cond in conditions])
        lower_err = means - np.clip(means - stds, 0, 1)
        upper_err = np.clip(means + stds, 0, 1) - means

        for i, cond in enumerate(conditions):
            ax.bar(
                x[i] + offset, means[i], width=bar_w, color=metric_face[metric][cond],
                edgecolor=COLORS[cond], linewidth=1.4, zorder=3,
            )
        ax.errorbar(
            x + offset, means, yerr=[lower_err, upper_err], fmt="o",
            color="black", markersize=4, ecolor="0.25",
            elinewidth=1.2, capsize=3.5, capthick=1.2, zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions])
    # Kept short -- the legend below already spells out own vs. cross-instance per condition,
    # so the axis label doesn't need to carry that too.
    ax.set_ylabel("Average exact match accuracy")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Condition (Task ID / w/o Task ID) is already labeled on the x-axis and encoded in bar
    # color -- the legend only needs to disambiguate the metric (fill darkness), so it's
    # neutral grey rather than repeating the condition colors: solid grey for Cross-instance
    # (the darker, emphasized fill), lighter grey outline for Same-instance.
    legend_handles = [
        Patch(facecolor="0.85", edgecolor="0.35", linewidth=1.2, label=METRIC_LABELS["own"]),
        Patch(facecolor="0.35", edgecolor="0.15", linewidth=1.2, label=METRIC_LABELS["loo"]),
    ]
    ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=2, frameon=False,
    )

    fig.tight_layout()
    # PDF (vector, for the paper) alongside PNG (for quick preview / the README).
    fig.savefig(out_path)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"Saved {out_path} and {pdf_path}")


def main() -> None:
    series = read_macro_series(CSV_PATH)
    for cond in ("td", "notd"):
        for metric in ("own", "loo"):
            s = series[cond][metric]
            print(
                f"{cond:6s} {METRIC_LABELS[metric]:28s} "
                f"mean={s['mean']:.3f} std={s['std']:.3f} n={s['n']}"
            )
    plot(series, PLOT_PATH)


if __name__ == "__main__":
    main()
