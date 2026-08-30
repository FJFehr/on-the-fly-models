"""Per-task breakdown for Experiment 2 (multi-task capacity).

Companion to plot_capacity_cliff.py -- that plot shows the aggregate story
across model size; this one asks "which tasks actually fail," per model
size. Starting point is dim=6 (2,444 params), the pivot size where
individual training is already saturated (98.9%) but joint+task-ID training
is not (74.0%); dim=4 and dim=10 follow the same shape once available.

Uses val_query_exact_match_by_task_<category> throughout, for ALL THREE
conditions -- not test_query_exact_match like plot_capacity_cliff.py. Only
the validation split has a per-task breakdown (no equivalent exists for
test, see models/direct_supervised_lightning.py's on_validation_epoch_end
vs test_step), so this chart is internally consistent (same split across
all bars) but not directly comparable, split-wise, to the capacity-cliff
plot's numbers.

All raw per-seed-per-task results for every size live in
results_per_task.csv next to this script (same reasoning as
plot_capacity_cliff.py's results.csv: outputs/ is gitignored and normally
only exists on the cluster node that ran the jobs). --dim selects which
size to render; --outputs-dir rescans every size found under
arc1d_v2_minimal_size / arc1d_v2_multitask and refreshes the CSV first.

Usage
-----
    uv run python configs/experiments/arc1d_v2_multitask/plot_per_task.py
    uv run python configs/experiments/arc1d_v2_multitask/plot_per_task.py --dim 10
    uv run python configs/experiments/arc1d_v2_multitask/plot_per_task.py --outputs-dir outputs
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from visualisation.style import apply_latex_style, format_task_category

HERE = Path(__file__).parent
CSV_PATH = HERE / "results_per_task.csv"

DIMS = ("4", "6", "10")
COLORS = {
    "individual": "#1E8449",  # dark green
    "td": "#5B2C82",  # dark purple  -- joint, with task ID
    "notd": "#B276B2",  # light purple -- joint, no task ID
}
LABELS = {
    "individual": "Individual models per task",
    "td": "Joint model with task ID",
    "notd": "Joint model without task ID",
}
CSV_FIELDS = ["condition", "dim", "task", "seed", "val_exact_match"]

# Manual nudges on top of the increasing-by-td sort: keep related tasks
# together even when their scores don't happen to land next to each other.
# Each entry moves `task` to sit immediately after `anchor`.
ORDER_OVERRIDES = [
    ("1d_move_1p", "1d_move_2p"),  # Move 1 -> next to Move 2 (moves together)
    ("1d_pcopy_mc", "1d_pcopy_1c"),  # Pattern Copy Multicolor -> next to Pattern Copy
]


def apply_order_overrides(tasks: list[str], overrides: list[tuple[str, str]]) -> list[str]:
    tasks = list(tasks)
    for task, anchor in overrides:
        if task not in tasks or anchor not in tasks:
            continue
        tasks.remove(task)
        tasks.insert(tasks.index(anchor) + 1, task)
    return tasks


def lighten(hex_color: str, amount: float = 0.45) -> str:
    """Blend a hex color toward white -- used for bar fills, with the
    original color kept as the outline (matplotlib legend patches pick up
    both automatically)."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


FILL_COLORS = {cond: lighten(hex_) for cond, hex_ in COLORS.items()}


def shorten_label(label: str) -> str:
    """Drop the word 'Pixel(s)' and abbreviate 'Multicolor' to 'MC' in a
    task display label -- redundant/verbose once it's clear from context."""
    label = label.replace(" Pixels", "").replace(" Pixel", "")
    return label.replace("Multicolor", "MC")


# ---------------------------------------------------------------------------
# Extraction (outputs/ -> results_per_task.csv)
# ---------------------------------------------------------------------------


def parse_val_by_task(results_path: Path) -> dict[str, float]:
    """Return {task: val_query_exact_match_by_task_<task>} from results.txt."""
    scores = {}
    prefix = "val_query_exact_match_by_task_"
    for line in results_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            key, _, value = stripped.partition(":")
            task = key[len(prefix) :]
            scores[task] = float(value.strip())
    return scores


def extract_records(outputs_dir: Path) -> list[dict]:
    """Scan every available dim under both experiment output dirs."""
    records = []

    minsize_re = re.compile(r"^v2_minsize_dim(\d+)_(.+)_seed(\d+)$")
    for path in sorted((outputs_dir / "arc1d_v2_minimal_size").glob("*/results.txt")):
        m = minsize_re.match(path.parent.name)
        if not m:
            continue
        dim, task, seed = m.group(1), m.group(2), int(m.group(3))
        scores = parse_val_by_task(path)
        score = scores.get(task)
        if score is not None:
            records.append(
                {"condition": "individual", "dim": dim, "task": task, "seed": seed,
                 "val_exact_match": score}
            )

    # dim=10 individual lives under a different project (the original Phase 1
    # RC1 run), not arc1d_v2_minimal_size (which only ever swept dim 4/6).
    rc1_re = re.compile(r"^v2_RC1_dim10_(.+)_rope_canon_n1_seed(\d+)$")
    for path in sorted((outputs_dir / "arc1d_v2_backbone_capacity").glob("*/results.txt")):
        m = rc1_re.match(path.parent.name)
        if not m:
            continue
        task, seed = m.group(1), int(m.group(2))
        scores = parse_val_by_task(path)
        score = scores.get(task)
        if score is not None:
            records.append(
                {"condition": "individual", "dim": "10", "task": task, "seed": seed,
                 "val_exact_match": score}
            )

    pattern = re.compile(r"^v2_multitask_(notd|td)(?:_dim(\d+))?_seed(\d+)$")
    for path in sorted((outputs_dir / "arc1d_v2_multitask").glob("*/results.txt")):
        m = pattern.match(path.parent.name)
        if not m:
            continue
        cond, dim, seed = m.group(1), m.group(2) or "10", int(m.group(3))
        for task, score in parse_val_by_task(path).items():
            records.append(
                {"condition": cond, "dim": dim, "task": task, "seed": seed,
                 "val_exact_match": score}
            )

    records.sort(key=lambda r: (r["condition"], int(r["dim"]), r["task"], r["seed"]))
    return records


def write_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {path}")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [
            {
                "condition": row["condition"],
                "dim": row["dim"],
                "task": row["task"],
                "seed": int(row["seed"]),
                "val_exact_match": float(row["val_exact_match"]),
            }
            for row in csv.DictReader(f)
        ]


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------


def build_series(records: list[dict], dim: str) -> dict[str, dict[str, dict]]:
    """condition -> task -> summary stats, for one model size."""
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in records:
        if r["dim"] != dim:
            continue
        by_key.setdefault((r["condition"], r["task"]), []).append(r["val_exact_match"])

    series: dict[str, dict[str, dict]] = {"individual": {}, "td": {}, "notd": {}}
    for (cond, task), values in by_key.items():
        series[cond][task] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "n": len(values),
        }
    return series


def plot(series: dict[str, dict[str, dict]], dim: str, out_path: Path) -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
        }
    )

    # Increasing order on "Joint model with task ID" -- surfaces the
    # failures on the left, the point of this chart.
    tasks = sorted(series["td"], key=lambda t: series["td"][t]["mean"])
    tasks = apply_order_overrides(tasks, ORDER_OVERRIDES)

    n_tasks = len(tasks)
    conditions = ("individual", "td", "notd")
    n_cond = len(conditions)
    bar_w = 0.8 / n_cond
    x = np.arange(n_tasks)

    # Wider than tall -- single-column paper figure, but 14 task groups need
    # the horizontal room.
    fig, ax = plt.subplots(figsize=(14.0, 5.5))

    for i, cond in enumerate(conditions):
        offset = (i - (n_cond - 1) / 2) * bar_w
        means = np.array([series[cond].get(t, {"mean": 0})["mean"] for t in tasks])
        stds = np.array([series[cond].get(t, {"std": 0})["std"] for t in tasks])
        # Clip whiskers to [0, 1] rather than letting them overshoot the
        # axis bounds (asymmetric once a bar sits close to 0% or 100%).
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
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="If given, rescan this outputs/ dir (every size found) and refresh the CSV first.",
    )
    parser.add_argument(
        "--dim", choices=DIMS, default="6", help="Which model size to plot (default: 6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.outputs_dir is not None:
        records = extract_records(args.outputs_dir)
        write_csv(records, CSV_PATH)
    else:
        records = read_csv(CSV_PATH)

    available = {r["dim"] for r in records}
    if args.dim not in available:
        msg = f"No data for dim={args.dim} in {CSV_PATH} (have: {sorted(available)})"
        raise SystemExit(msg)

    series = build_series(records, args.dim)
    plot(series, args.dim, HERE / f"per_task_dim{args.dim}.png")


if __name__ == "__main__":
    main()
