"""Per-task breakdown for Experiment 4 (joint hypernetwork), dim=4: the
headline pair only -- individual vs. hypernetwork with/without a task-ID
signal. Drops the joint-direct-training conditions (see
plot_per_task_combined.py for the full 5-condition version with those
included) to isolate the actual point: task ID essentially solves every
task; without it, the damage is narrow and specific to the move family.

Reuses the finalized colour/label scheme from plot_per_task_combined.py
(beige/teal/gold, "no task ID" left of "task ID") rather than a separate
palette, so every per-task figure in this experiment reads consistently.

Tasks ordered by TASK_ORDER (Moves -> Transformations -> Denoise/Copy),
matching the appendix's task-example figure groupings -- not sorted by
score. The move family (1p/2p/3p/2p_dp/dp) is where the notd failures
concentrate, the reason this plot exists.

Data: results_per_task_dim4_combined.csv (same source as the 5-condition
and radar plots), individual/hyper_notd/hyper_td rows only.

Usage
-----
    uv run python experiments/02_hypernetwork_multitask/plot_per_task.py
"""

import csv
import statistics
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from plot_per_task_combined import COLORS, LABELS
from visualisation.core.style import apply_latex_style, format_task_category

HERE = Path(__file__).parent
CSV_PATH = Path("outputs/results/02_hypernetwork_multitask/results_per_task_dim4_combined.csv")
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


# Fixed family order (Moves -> Transformations -> Denoise/Copy), matching
# the task-example figures in the appendix (arc1d-movement-scale /
# arc1d-object-transformations / arc1d-denoising-pcopy) so the same task
# always sits in the same relative position across every figure in the
# paper. Kept in sync by hand with the identically-named constant in
# arc1d_v2_multitask/plot_per_task.py. Replaces the old
# increasing-by-hyper_notd-score sort, which put the two figures and the
# appendix in three different orders.
TASK_ORDER = [
    "1d_move_1p", "1d_move_2p", "1d_move_2p_dp", "1d_move_3p", "1d_move_dp", "1d_scale_dp",
    "1d_fill", "1d_hollow", "1d_flip", "1d_mirror",
    "1d_denoising_1c", "1d_denoising_mc", "1d_pcopy_1c", "1d_pcopy_mc",
]


def ordered_tasks(series: dict[str, dict[str, dict]]) -> list[str]:
    return [t for t in TASK_ORDER if t in series["hyper_notd"]]


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"Saved {out_path} and {pdf_path}")


def main() -> None:
    records = read_csv(CSV_PATH)
    series = build_series(records)
    plot(series, Path("outputs/figures/02_hypernetwork_multitask/per_task_dim4.png"))


if __name__ == "__main__":
    main()
