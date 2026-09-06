"""Radar/spider version of plot_per_task_combined.py, dim=4, all 5
conditions. Experimental alternative to the grouped bar chart -- same data
(results_per_task_dim4_combined.csv), same colour scheme, testing whether
14 categories around one circle reads more clearly than 14 groups of 5
bars each. Low scores sit close to the center by construction (radius =
accuracy), so a condition that's failing broadly should visually collapse
toward the middle rather than needing to be read bar-by-bar.

Usage
-----
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_per_task_radar.py
"""

import csv
import statistics
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from plot_per_task_combined import COLORS, LABELS, MOVE_TASKS, ordered_tasks
from visualisation.style import apply_latex_style, format_task_category

HERE = Path(__file__).parent
CSV_PATH = HERE / "results_per_task_dim4_combined.csv"
CONDITIONS = ("individual", "joint_td", "joint_notd", "hyper_td", "hyper_notd")

# No fill (a wash of translucent colour under every line was too distracting
# with 5 overlapping series) -- distinguish lines by colour + linestyle +
# alpha=0.7 (so overlaps stay visible) instead. "individual" is the ceiling
# reference, dotted; the two headline hypernetwork conditions get solid
# (they're the point of the figure); the two joint-direct conditions get
# dashed/dash-dot so they read as "the comparison," not the headline.
LINESTYLES = {
    "individual": ":",
    "joint_notd": "--",
    "joint_td": "-.",
    "hyper_notd": "-",
    "hyper_td": "-",
}
LINE_ALPHA = 0.7
LINEWIDTH = 3.0


def shorten_label(label: str) -> str:
    label = label.replace(" Pixels", "").replace(" Pixel", "")
    return label.replace("Multicolor", "MC")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [
            {
                "condition": row["condition"],
                "task": row["task"],
                "val_exact_match": float(row["val_exact_match"]),
            }
            for row in csv.DictReader(f)
        ]


def build_series(records: list[dict]) -> dict[str, dict[str, float]]:
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in records:
        by_key.setdefault((r["condition"], r["task"]), []).append(r["val_exact_match"])
    series: dict[str, dict[str, float]] = {c: {} for c in CONDITIONS}
    for (cond, task), values in by_key.items():
        series[cond][task] = statistics.mean(values)
    return series


def plot(series: dict[str, dict[str, float]], tasks: list[str], out_path: Path) -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "xtick.labelsize": 13,
            "ytick.labelsize": 11,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
        }
    )

    n = len(tasks)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for cond in CONDITIONS:
        values = [series[cond].get(t, 0.0) for t in tasks]
        values += values[:1]
        ax.plot(
            angles, values, color=COLORS[cond], linewidth=LINEWIDTH,
            linestyle=LINESTYLES[cond], alpha=LINE_ALPHA, label=LABELS[cond],
            solid_capstyle="round", dash_capstyle="round", zorder=3,
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([shorten_label(format_task_category(t)) for t in tasks])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
    # Radial (percentage) axis labels along the horizontal, pointing right --
    # rlabel_position is in the same (offset+direction-transformed) data-theta
    # space as the plotted angles, so 90 degrees here lands at screen-angle 0
    # given set_theta_offset(pi/2)/set_theta_direction(-1) above.
    ax.set_rlabel_position(90)
    ax.grid(alpha=0.35, linewidth=0.6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"Saved {out_path} and {pdf_path}")


def main() -> None:
    records = read_csv(CSV_PATH)
    series = build_series(records)
    # Reuse the same move-family-first-then-ascending ordering as the bar
    # chart for a fair visual comparison, keyed the same way (hyper_notd).
    series_for_order = {"hyper_notd": series["hyper_notd"]}
    series_for_order["hyper_notd"] = {
        t: {"mean": v} for t, v in series["hyper_notd"].items()
    }
    tasks = ordered_tasks(series_for_order)
    plot(series, tasks, HERE / "per_task_dim4_radar.png")


if __name__ == "__main__":
    main()
