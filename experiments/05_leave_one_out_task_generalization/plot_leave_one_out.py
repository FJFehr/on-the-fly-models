"""Per-category chart for experiment 5: zero-shot leave-one-out held-out score, notd vs.
frozen_td, across all 14 base categories.

Each leaf config (<short>_<arm>_seed<N>) trains on 13 of the 14 base categories and reports
the held-out 14th category's own validation breakdown (val_query_accuracy_by_task_<category>,
val_query_exact_match_by_task_<category>) straight from that run's own results.txt -- unlike
experiment 4, everything needed lives in this experiment's own outputs/ tree, no cross-
experiment reuse.

Two CSVs:
- results_holdout.csv: the held-out category's own score, per (category, arm, seed) -- the
  headline number.
- results_indist.csv: the mean over the *other* 13 (in-distribution) categories in that same
  leaf, per (held-out category, arm, seed) -- context for the generalisation gap, not the
  headline.

One figure (per_category_holdout.png): held-out token accuracy (val_query_accuracy), notd vs.
frozen_td, all 14 categories -- same style as experiments 02/04's per-task bar charts.
Exact match is reported in the console summary only, not as a bar chart, if (as expected from
the legacy 3-seed precedent) it's at-or-near 0 for most categories -- a flat bar chart would add
nothing a table doesn't already say better.

Usage
-----
    uv run python experiments/05_leave_one_out_task_generalization/plot_leave_one_out.py
    # replots from the CSVs as committed; add --outputs-dir outputs to rescan + refresh both
"""

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from visualisation.core.style import apply_latex_style, format_task_category
from visualisation.paper.arc_paper import PAPER_COLORS, lighten

RESULTS_DIR = Path("outputs/results/05_leave_one_out_task_generalization")
FIGURES_DIR = Path("outputs/figures/05_leave_one_out_task_generalization")
HOLDOUT_CSV_PATH = RESULTS_DIR / "results_holdout.csv"
INDIST_CSV_PATH = RESULTS_DIR / "results_indist.csv"

CONDITIONS = ("notd", "frozen_td")
SEEDS = (1, 2, 3, 4, 5)
# Same palette/labels convention as experiments 02/04's own notd/frozen_td charts.
COLORS = {"notd": PAPER_COLORS[5], "frozen_td": PAPER_COLORS[9]}
FILL_COLORS = {cond: lighten(hex_) for cond, hex_ in COLORS.items()}
LABELS = {"frozen_td": "Task ID", "notd": "w/o Task ID"}

ALL_CATEGORIES = [
    "1d_denoising_1c", "1d_denoising_mc", "1d_fill", "1d_flip", "1d_hollow", "1d_mirror",
    "1d_move_1p", "1d_move_2p", "1d_move_2p_dp", "1d_move_3p", "1d_move_dp",
    "1d_pcopy_1c", "1d_pcopy_mc", "1d_scale_dp",
]


def short_name(category: str) -> str:
    """Mirrors gen_configs.py's own short_name -- must stay identical to correctly map each
    leaf's directory name back to its held-out category."""
    return category.removeprefix("1d_").replace("_", "")


SHORT_TO_CATEGORY = {short_name(c): c for c in ALL_CATEGORIES}
DIRNAME_COND = {"notd": "notd", "frozen_td": "frozentd"}


def parse_by_task(results_path: Path, prefix: str) -> dict[str, float]:
    """Return {task: value} for lines matching '<prefix><task>: <value>' in results.txt.
    Duplicated from experiments 02/04's own identical helper -- keeps each experiment's plot
    script self-contained rather than importing across experiment folders."""
    scores = {}
    for line in results_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            key, _, value = stripped.partition(":")
            scores[key[len(prefix) :]] = float(value.strip())
    return scores


def extract_records(outputs_dir: Path) -> tuple[list[dict], list[dict]]:
    root = outputs_dir / "05_leave_one_out_task_generalization"
    holdout_records = []
    indist_records = []
    for held_out in ALL_CATEGORIES:
        short = short_name(held_out)
        train_categories = [c for c in ALL_CATEGORIES if c != held_out]
        for cond, dirname_cond in DIRNAME_COND.items():
            for seed in SEEDS:
                path = root / f"{short}_{dirname_cond}_seed{seed}" / "results.txt"
                if not path.exists():
                    continue
                acc_by_task = parse_by_task(path, "val_query_accuracy_by_task_")
                em_by_task = parse_by_task(path, "val_query_exact_match_by_task_")

                holdout_records.append(
                    {
                        "condition": cond, "category": held_out, "seed": seed,
                        "val_accuracy": acc_by_task[held_out],
                        "val_exact_match": em_by_task[held_out],
                    }
                )
                indist_acc = statistics.mean(acc_by_task[c] for c in train_categories)
                indist_em = statistics.mean(em_by_task[c] for c in train_categories)
                indist_records.append(
                    {
                        "condition": cond, "held_out_category": held_out, "seed": seed,
                        "indist_accuracy": indist_acc, "indist_exact_match": indist_em,
                    }
                )
    holdout_records.sort(key=lambda r: (r["condition"], r["category"], r["seed"]))
    indist_records.sort(key=lambda r: (r["condition"], r["held_out_category"], r["seed"]))
    return holdout_records, indist_records


def write_csv(records: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {path}")


def read_csv(path: Path, float_fields: list[str]) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in float_fields:
            row[key] = float(row[key])
        if "seed" in row:
            row["seed"] = int(row["seed"])
    return rows


def build_series(
    records: list[dict], group_key: str, value_key: str
) -> dict[str, dict[str, dict]]:
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in records:
        by_key.setdefault((r["condition"], r[group_key]), []).append(float(r[value_key]))

    series: dict[str, dict[str, dict]] = {c: {} for c in CONDITIONS}
    for (cond, group), values in by_key.items():
        series[cond][group] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "n": len(values),
        }
    return series


def plot_holdout(series: dict[str, dict[str, dict]], out_path: Path) -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 15,
            "pdf.fonttype": 42,
        }
    )

    categories = [c for c in ALL_CATEGORIES if c in series["frozen_td"] or c in series["notd"]]
    labels = [format_task_category(c) for c in categories]

    n_groups = len(categories)
    n_cond = len(CONDITIONS)
    bar_w = 0.8 / n_cond
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    for i, cond in enumerate(CONDITIONS):
        offset = (i - (n_cond - 1) / 2) * bar_w
        means = np.array([series[cond].get(c, {"mean": 0})["mean"] for c in categories])
        stds = np.array([series[cond].get(c, {"std": 0})["std"] for c in categories])
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
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Held-out zero-shot token accuracy")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

    fig.tight_layout()
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
        help="If given, rescan outputs/05_leave_one_out_task_generalization and refresh both "
        "CSVs before plotting. Without it, replots from the CSVs as committed.",
    )
    return parser.parse_args()


HOLDOUT_FIELDS = ["condition", "category", "seed", "val_accuracy", "val_exact_match"]
INDIST_FIELDS = ["condition", "held_out_category", "seed", "indist_accuracy", "indist_exact_match"]


def main() -> None:
    args = parse_args()

    if args.outputs_dir is not None:
        holdout_records, indist_records = extract_records(args.outputs_dir)
        write_csv(holdout_records, HOLDOUT_CSV_PATH, HOLDOUT_FIELDS)
        write_csv(indist_records, INDIST_CSV_PATH, INDIST_FIELDS)
    else:
        holdout_records = read_csv(HOLDOUT_CSV_PATH, ["val_accuracy", "val_exact_match"])
        indist_records = read_csv(INDIST_CSV_PATH, ["indist_accuracy", "indist_exact_match"])

    print("\n== held-out zero-shot, macro over 14 categories, mean +- 1 s.d. across 5 seeds ==")
    acc_series = build_series(holdout_records, "category", "val_accuracy")
    em_series = build_series(holdout_records, "category", "val_exact_match")
    notd_wins = 0
    for cat in ALL_CATEGORIES:
        n = acc_series["notd"].get(cat, {"mean": None})["mean"]
        f = acc_series["frozen_td"].get(cat, {"mean": None})["mean"]
        if n is not None and f is not None and n > f:
            notd_wins += 1
    for cond in CONDITIONS:
        acc_vals = [v["mean"] for v in acc_series[cond].values()]
        em_vals = [v["mean"] for v in em_series[cond].values()]
        print(
            f"  {LABELS[cond]:12s} token_accuracy macro-mean={statistics.mean(acc_vals):.3f}  "
            f"exact_match macro-mean={statistics.mean(em_vals):.4f}"
        )
    print(f"  notd wins {notd_wins}/{len(ALL_CATEGORIES)} categories on token accuracy")
    plot_holdout(acc_series, FIGURES_DIR / "per_category_holdout.png")

    print("\n== in-distribution (mean over the other 13 categories per leaf), for context ==")
    indist_series = build_series(indist_records, "held_out_category", "indist_accuracy")
    indist_em_series = build_series(indist_records, "held_out_category", "indist_exact_match")
    for cond in CONDITIONS:
        acc_vals = [v["mean"] for v in indist_series[cond].values()]
        em_vals = [v["mean"] for v in indist_em_series[cond].values()]
        print(
            f"  {LABELS[cond]:12s} token_accuracy macro-mean={statistics.mean(acc_vals):.3f}  "
            f"exact_match macro-mean={statistics.mean(em_vals):.4f}"
        )


if __name__ == "__main__":
    main()
