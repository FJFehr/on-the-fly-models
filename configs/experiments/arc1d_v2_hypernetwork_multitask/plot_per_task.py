"""Per-task breakdown for Experiment 4 (joint hypernetwork), dim=4: the
headline pair only -- individual vs. hypernetwork with/without a task-ID
signal. Drops the joint-direct-training conditions (see
plot_per_task_combined.py for the full 5-condition version with those
included) to isolate the actual point: task ID essentially solves every
task; without it, the damage is narrow and specific to the move family.

Reuses the finalized colour/label scheme from plot_per_task_combined.py
(beige/teal/gold, "no task ID" left of "task ID") rather than a separate
palette, so every per-task figure in this experiment reads consistently.

Move family (1p/2p/3p/2p_dp/dp) pinned as a contiguous block on the left,
ascending by hyper_notd within each block -- that's where the notd failures
concentrate, the reason this plot exists.

Data: results_per_task_dim4_combined.csv (same source as the 5-condition
and radar plots), individual/hyper_notd/hyper_td rows only.

Usage
-----
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_per_task.py
"""

import csv
import statistics
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from plot_per_task_combined import COLORS, LABELS, MOVE_TASKS
from visualisation.style import apply_latex_style, format_task_category

HERE = Path(__file__).parent
CSV_PATH = HERE / "results_per_task_dim4_combined.csv"
CONDITIONS = ("individual", "hyper_td", "hyper_notd")


def _lighten(hex_color: str, amount: float = 0.45) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


FILL_COLORS = {cond: _lighten(COLORS[cond]) for cond in CONDITIONS}


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
            if row["condition"] in CONDITIONS
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
            "xtick.labelsize": 18,
            "ytick.labelsize": 17,
            "legend.fontsize": 19,
            "pdf.fonttype": 42,
        }
    )

    tasks = ordered_tasks(series)
    n_tasks = len(tasks)
    n_cond = len(CONDITIONS)
    bar_w = 0.8 / n_cond
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(14.0, 5.5))

    for i, cond in enumerate(CONDITIONS):
        offset = (i - (n_cond - 1) / 2) * bar_w
        means = np.array([series[cond].get(t, {"mean": 0})["mean"] for t in tasks])
        stds = np.array([series[cond].get(t, {"std": 0})["std"] for t in tasks])
        lower_err = means - np.clip(means - stds, 0, 1)
        upper_err = np.clip(means + stds, 0, 1) - means
        ax.bar(
            x + offset, means, width=bar_w, color=FILL_COLORS[cond],
            edgecolor=COLORS[cond], linewidth=1.4, label=LABELS[cond], zorder=3,
        )
        ax.errorbar(
            x + offset, means, yerr=[lower_err, upper_err], fmt="o",
            color="black", markersize=3.5, ecolor="0.25",
            elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=4,
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
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"Saved {out_path} and {pdf_path}")


def main() -> None:
    records = read_csv(CSV_PATH)
    series = build_series(records)
    plot(series, HERE / "per_task_dim4.png")


if __name__ == "__main__":
    main()
