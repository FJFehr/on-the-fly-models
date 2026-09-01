"""Per-task breakdown, dim=4: individual, joint-DIRECT (Exp 2), and joint-
HYPERNETWORK (Exp 4), all five conditions on one chart.

The full story in one figure: individual training (the ceiling) vs. the two
ways of sharing one model across tasks -- direct joint training (which hits
the capacity cliff) and the hypernetwork (which closes it, with a task-ID
signal). Colours pulled from visualisation.arc_paper's PAPER_COLORS (the
same muted-rainbow palette as outputs/visualisations/arc_1d_task_1d_flip_46.pdf),
not the purple/green scheme used elsewhere in this repo's per-task plots --
this figure needs 5 distinguishable conditions, not 2-3.

Move family (1p/2p/3p/2p_dp/dp) pinned as a contiguous block on the left,
ascending by hyper_notd within each block -- same convention as
plot_per_task.py, now with joint-direct's much larger failures visible too.

Raw per-seed data in results_per_task_dim4_combined.csv (individual/joint_td/
joint_notd pulled from arc1d_v2_multitask/results_per_task.csv, hyper_td/
hyper_notd from this folder's own results_per_task_dim4.csv).

Usage
-----
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_per_task_combined.py
"""

import csv
import statistics
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from visualisation.arc_paper import PAPER_COLORS, lighten
from visualisation.style import apply_latex_style, format_task_category


def darken(hex_color: str, amount: float = 0.2) -> str:
    """Blend a hex color toward black -- the inverse of arc_paper.lighten,
    for when a palette entry needs to read as more saturated/burnt, not
    lighter."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c * (1 - amount)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"

HERE = Path(__file__).parent
CSV_PATH = HERE / "results_per_task_dim4_combined.csv"

# PAPER_COLORS keys: 0 khaki/tan, 6 slate blue, 8 purple, 3 gold, 4 olive --
# picked to match Fabio's naming (beige/blue/purple/yellow/green) as closely
# as this palette allows (it has no pure yellow or green -- gold and olive
# are the closest muted equivalents).
COLORS = {
    "individual": PAPER_COLORS[0],  # beige / light khaki-tan
    "joint_td": PAPER_COLORS[6],  # blue (slate blue) -- dark = with task ID
    "joint_notd": lighten(PAPER_COLORS[6], amount=0.35),  # same blue, lighter = no task ID
    "hyper_td": PAPER_COLORS[9],  # plum/pink -- task ID
    "hyper_notd": PAPER_COLORS[5],  # teal -- no task ID
}
LABELS = {
    "individual": "Individual model per task",
    "joint_td": "Joint direct, task ID",
    "joint_notd": "Joint direct, w/o task ID",
    "hyper_td": "Hypernetwork, task ID",
    "hyper_notd": "Hypernetwork, w/o task ID",
}
# Task-ID always to the left of its no-task-ID counterpart within each pair.
CONDITIONS = ("individual", "joint_td", "joint_notd", "hyper_td", "hyper_notd")

MOVE_TASKS = {"1d_move_1p", "1d_move_2p", "1d_move_3p", "1d_move_2p_dp", "1d_move_dp"}

FILL_COLORS = {cond: lighten(hex_) for cond, hex_ in COLORS.items()}
# "individual" = PAPER_COLORS[0] (khaki/tan) direct, no override -- matches
# arc1d_v2_multitask/plot_per_task.py's current scheme exactly (per_task_dim6.pdf),
# so this experiment's "individual" bars read as the same colour everywhere.


def shorten_label(label: str) -> str:
    label = label.replace(" Pixels", "").replace(" Pixel", "")
    return label.replace("Multicolor", "MC")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [
            {
                "condition": row["condition"],
                "task": row["task"],
                "seed": row["seed"],
                "val_exact_match": float(row["val_exact_match"]),
            }
            for row in csv.DictReader(f)
        ]


def build_series(records: list[dict]) -> dict[str, dict[str, dict]]:
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in records:
        by_key.setdefault((r["condition"], r["task"]), []).append(r["val_exact_match"])

    series: dict[str, dict[str, dict]] = {c: {} for c in CONDITIONS}
    for (cond, task), values in by_key.items():
        series[cond][task] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "n": len(values),
        }
    return series


def ordered_tasks(series: dict[str, dict[str, dict]]) -> list[str]:
    """Move family first (ascending hyper_notd score), then everything
    else (also ascending hyper_notd score)."""
    all_tasks = sorted(series["hyper_notd"], key=lambda t: series["hyper_notd"][t]["mean"])
    move = [t for t in all_tasks if t in MOVE_TASKS]
    other = [t for t in all_tasks if t not in MOVE_TASKS]
    return move + other


def plot(series: dict[str, dict[str, dict]], out_path: Path) -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 17,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
        }
    )

    tasks = ordered_tasks(series)
    n_tasks = len(tasks)
    n_cond = len(CONDITIONS)
    bar_w = 0.85 / n_cond
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(16.0, 5.5))

    for i, cond in enumerate(CONDITIONS):
        offset = (i - (n_cond - 1) / 2) * bar_w
        means = np.array([series[cond].get(t, {"mean": 0})["mean"] for t in tasks])
        stds = np.array([series[cond].get(t, {"std": 0})["std"] for t in tasks])
        lower_err = means - np.clip(means - stds, 0, 1)
        upper_err = np.clip(means + stds, 0, 1) - means
        ax.bar(
            x + offset, means, width=bar_w, color=FILL_COLORS[cond],
            edgecolor=COLORS[cond], linewidth=1.2, label=LABELS[cond], zorder=3,
        )
        ax.errorbar(
            x + offset, means, yerr=[lower_err, upper_err], fmt="o",
            color="black", markersize=2.8, ecolor="0.25",
            elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=4,
        )

    n_move = sum(1 for t in tasks if t in MOVE_TASKS)
    ax.axvline(n_move - 0.5, color="0.6", linewidth=1.0, linestyle="--", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [shorten_label(format_task_category(t)) for t in tasks], rotation=40, ha="right",
    )
    ax.set_ylabel("Validation exact match accuracy")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"Saved {out_path} and {pdf_path}")


def main() -> None:
    records = read_csv(CSV_PATH)
    series = build_series(records)
    plot(series, HERE / "per_task_dim4_combined.png")


if __name__ == "__main__":
    main()
