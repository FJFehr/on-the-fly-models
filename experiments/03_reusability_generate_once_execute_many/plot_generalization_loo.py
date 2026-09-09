"""Macro-average generalization bar chart: task ID vs. no task ID, own vs. cross-instance.

Companion to plot_capacity_cliff.py/plot_per_task.py, same palette/templating, but for a
different question: scripts/measure_compute_efficiency.py's Phase E leave-one-out test (does
ONE generated weight set, from one instance, solve the OTHER instances of its own task
category?) run on all 5 seeds of experiment 2's own dim=4 matched-scale hypernetwork
checkpoints (outputs/02_hypernetwork_multitask/hyper_multitask_dim4_{notd,frozentd}_
seed{1..5}/, 10,156 trainable params each, real checkpoints -- see that experiment's README's
"Dim=4 rerun"), both arms, `--generalization-split val,test` (200 instances/category, both
splits of data/arc_1d_looped_augmented -- fully reproducible from this repo's own data-build
recipe, not a separately hand-built eval pool).

Deliberately simplistic (Fabio's own framing): two x-axis groups (Task ID / No Task ID), each
with two bars -- Own (per-instance generation, each instance scored on its own query) and
Cross-instance (leave-one-out: every instance in turn as the reference, scored on the
others). Bar height = macro-average across the 14 task categories; whiskers = population std
(statistics.pstdev, matching plot_capacity_cliff.py/plot_per_task.py's own convention) of
those 14 per-category means, where each per-category mean is itself already averaged across
the 5 seeds -- how consistent the condition is ACROSS task types, not the within-category
instance-level spread (that lives in results_generalization_loo.csv's own_std/loo_std columns
instead) or the across-seed spread (that lives in results_generalization_loo_raw.csv).

Color encodes condition (task ID vs. not), matching the legend labels already established in
plot_per_task.py/plot_capacity_cliff.py: plum for task ID (Fabio's explicit call, replacing
the purple used there), teal for no task ID. Fill darkness (full color vs. lighten()'d)
encodes the metric (Own vs. Cross-instance) within each condition -- reusing the same
PAPER_COLORS/PAPER_FILL_COLORS outline-in-full-color/lighter-fill-inside treatment as every
other figure in this repo, just applied along a different axis than usual.

Raw per-(condition, seed, category) numbers live in results_generalization_loo_raw.csv;
this script's own per-(condition, category) macro-average is computed from
results_generalization_loo.csv (itself averaged across seeds from the raw file) -- neither is
committed, see "Figures and results" in experiments/README.md. --outputs-dir rescans
outputs/compute_efficiency/03_reusability_generate_once_execute_many/ (written by run.sh) and
refreshes both CSVs; without it, replots from results_generalization_loo.csv as committed.

Usage
-----
    uv run python experiments/03_reusability_generate_once_execute_many/plot_generalization_loo.py
    # replots from results_generalization_loo.csv as committed; add --outputs-dir outputs
    # to rescan run.sh's output and refresh both CSVs first
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from visualisation.core.style import apply_latex_style
from visualisation.paper.arc_paper import PAPER_COLORS, PAPER_FILL_COLORS

OUTPUTS_SUBDIR = "compute_efficiency/03_reusability_generate_once_execute_many"
RAW_CSV_PATH = Path(
    "outputs/results/03_reusability_generate_once_execute_many/results_generalization_loo_raw.csv"
)
CSV_PATH = Path(
    "outputs/results/03_reusability_generate_once_execute_many/results_generalization_loo.csv"
)
PLOT_PATH = Path(
    "outputs/figures/03_reusability_generate_once_execute_many/generalization_loo.png"
)

# Same condition labels every other run-dir/CSV in this repo uses -- report_linear_probe.py's
# own "frozen_td" if cond == "frozentd" else cond convention, not the bare "td" this script
# (and the old preliminary CSV it read from) used to use, which collides with a different,
# unrelated condition (the deprecated *learned* task-embedding arm) elsewhere in this repo.
CONDITION_LABEL = {"notd": "notd", "frozentd": "frozen_td"}

# Fabio's explicit color call: plum (not purple) for task ID, teal for no task ID -- same
# PAPER_COLORS palette every other figure in this repo draws from.
COLORS = {
    "notd": PAPER_COLORS[5],  # teal -- joint, no task ID
    "frozen_td": PAPER_COLORS[9],  # plum -- joint, with task ID
}
FILL_COLORS = {
    "notd": PAPER_FILL_COLORS[5],
    "frozen_td": PAPER_FILL_COLORS[9],
}
# Matches the legend wording already established in plot_per_task.py/plot_capacity_cliff.py.
CONDITION_LABELS = {"frozen_td": "Task ID", "notd": "w/o Task ID"}
# "Same-instance" (not "Own"): weights generated from THIS instance's own support set, scored
# on this instance's own query -- the hypernetwork's normal, no-reuse behavior. Named to read
# as the direct opposite of "Cross-instance" (weights from a DIFFERENT instance, reused here).
METRIC_LABELS = {"own": "Same-instance", "loo": "Cross-instance"}

RAW_FIELDS = [
    "condition", "seed", "category", "n_instances",
    "own_exact_match_mean", "own_exact_match_std",
    "n_loo_references", "loo_mean", "loo_std",
    "n_loo_pairs", "loo_pooled_mean", "loo_pooled_std",
]
SUMMARY_FIELDS = ["condition", "category", "own_mean", "own_std", "loo_mean", "loo_std", "n_seeds"]


def extract_records(outputs_dir: Path) -> list[dict]:
    """Scan run.sh's 10 per-(condition, seed) generalization_table.csv files (written by
    scripts/measure_compute_efficiency.py --stage generalization) and return one dict per
    (condition, seed, category) row -- 140 rows (2 conditions x 5 seeds x 14 categories)."""
    root = outputs_dir / OUTPUTS_SUBDIR
    cell_re = re.compile(r"^hyper_multitask_dim4_(notd|frozentd)_seed(\d+)$")
    records = []
    for cell_dir in sorted(root.glob("*")):
        m = cell_re.match(cell_dir.name)
        if not m:
            continue
        condition, seed = CONDITION_LABEL[m.group(1)], int(m.group(2))
        csv_path = cell_dir / "generalization_table.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                records.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "category": row["category"],
                        "n_instances": row["n_instances"],
                        "own_exact_match_mean": row["own_exact_match_mean"],
                        "own_exact_match_std": row["own_exact_match_std"],
                        "n_loo_references": row["n_loo_references"],
                        "loo_mean": row["loo_mean"],
                        "loo_std": row["loo_std"],
                        "n_loo_pairs": row["n_loo_pairs"],
                        "loo_pooled_mean": row["loo_pooled_mean"],
                        "loo_pooled_std": row["loo_pooled_std"],
                    }
                )
    records.sort(key=lambda r: (r["condition"], r["seed"], r["category"]))
    return records


def aggregate_across_seeds(records: list[dict]) -> list[dict]:
    """One row per (condition, category), own_mean/own_std and loo_mean/loo_std computed as
    mean/population-stdev ACROSS THE 5 SEEDS of each seed's own per-category mean -- 28 rows
    (2 conditions x 14 categories). This is the file read_macro_series consumes; each of its
    28 numbers is now a real 5-seed mean, not a single seed's own_exact_match_mean/loo_mean."""
    by_key: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"own": [], "loo": []}
    )
    for r in records:
        key = (r["condition"], r["category"])
        by_key[key]["own"].append(float(r["own_exact_match_mean"]))
        by_key[key]["loo"].append(float(r["loo_mean"]))

    summary = []
    for (condition, category), metrics in sorted(by_key.items()):
        summary.append(
            {
                "condition": condition,
                "category": category,
                "own_mean": statistics.mean(metrics["own"]),
                "own_std": statistics.pstdev(metrics["own"]),
                "loo_mean": statistics.mean(metrics["loo"]),
                "loo_std": statistics.pstdev(metrics["loo"]),
                "n_seeds": len(metrics["own"]),
            }
        )
    return summary


def write_csv(records: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {path}")


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

    conditions = ("frozen_td", "notd")
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"Saved {out_path} and {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="If given, rescan run.sh's raw generalization_table.csv files under this outputs/ "
        "dir and refresh both CSVs before plotting. Without it, replots from "
        "results_generalization_loo.csv as committed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.outputs_dir is not None:
        raw_records = extract_records(args.outputs_dir)
        write_csv(raw_records, RAW_CSV_PATH, RAW_FIELDS)
        summary_records = aggregate_across_seeds(raw_records)
        write_csv(summary_records, CSV_PATH, SUMMARY_FIELDS)

    series = read_macro_series(CSV_PATH)
    for cond in ("frozen_td", "notd"):
        for metric in ("own", "loo"):
            s = series[cond][metric]
            print(
                f"{cond:9s} {METRIC_LABELS[metric]:28s} "
                f"mean={s['mean']:.3f} std={s['std']:.3f} n={s['n']}"
            )
    plot(series, PLOT_PATH)


if __name__ == "__main__":
    main()
