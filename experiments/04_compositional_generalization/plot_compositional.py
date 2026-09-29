"""Per-category charts for experiment 4: in-distribution and zero-shot compositional holdout.

Both arms now reuse experiment 2's own dim=4 matched-scale checkpoints (see run.sh and this
experiment's README for why) -- in-distribution numbers are literally experiment 2's own
results.txt (both experiments train on the identical 14-category recipe, confirmed by
diffing their configs), so this script's own --outputs-dir rescan reads across TWO
experiments' outputs/ trees: experiment 2's for in-distribution, this experiment's own
outputs/04_compositional_generalization/ (written by run.sh) for the
zero-shot compositional eval.

Two figures:
- per_task_indist.png: in-distribution validation exact match, 14 base categories, notd vs.
  frozen_td -- same style as experiments/02_hypernetwork_multitask/plot_per_task.py (same
  palette, same error-bar convention), since this is the same underlying models/data.
- per_task_holdout.png: zero-shot seq_accuracy on the 10 held-out composite categories,
  notd vs. frozen_td. seq_accuracy (token-level), not exact_match, is the informative metric
  here -- exact_match is at or near 0 for every category/seed/condition (see the README's
  own "Findings"), so a bar chart of it would show nothing but a flat line; the console
  summary below still reports the real exact_match totals, just not as a figure.

Usage
-----
    uv run python experiments/04_compositional_generalization/plot_compositional.py
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

RESULTS_DIR = Path("outputs/results/04_compositional_generalization")
FIGURES_DIR = Path("outputs/figures/04_compositional_generalization")
INDIST_CSV_PATH = RESULTS_DIR / "results_indist.csv"
HOLDOUT_CSV_PATH = RESULTS_DIR / "results_holdout.csv"

CONDITIONS = ("notd", "frozen_td")
SEEDS = (1, 2, 3, 4, 5)
# Same palette/labels as plot_generalization_loo.py's own notd/frozen_td convention.
COLORS = {"notd": PAPER_COLORS[5], "frozen_td": PAPER_COLORS[9]}
FILL_COLORS = {cond: lighten(hex_) for cond, hex_ in COLORS.items()}
LABELS = {"frozen_td": "Task ID", "notd": "w/o Task ID"}

TASK_ORDER = [
    "1d_move_1p", "1d_move_2p", "1d_move_2p_dp", "1d_move_3p", "1d_move_dp", "1d_scale_dp",
    "1d_fill", "1d_hollow", "1d_flip", "1d_mirror",
    "1d_denoising_1c", "1d_denoising_mc", "1d_pcopy_1c", "1d_pcopy_mc",
]


def parse_val_by_task(results_path: Path) -> dict[str, float]:
    """Return {task: val_query_exact_match_by_task_<task>} from results.txt. Duplicated in
    full from experiments/02_hypernetwork_multitask/plot_per_task_combined.py's own helper
    (same field/format every experiment logs this under) rather than importing across
    experiment folders -- keeps each experiment's plot script self-contained."""
    scores = {}
    prefix = "val_query_exact_match_by_task_"
    for line in results_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            key, _, value = stripped.partition(":")
            scores[key[len(prefix) :]] = float(value.strip())
    return scores


def extract_indist_records(outputs_dir: Path) -> list[dict]:
    """notd/frozen_td in-distribution val exact match, straight from experiment 2's own
    outputs/02_hypernetwork_multitask/hyper_multitask_dim4_{notd,frozentd}_seed{N}/results.txt
    -- not duplicated anywhere in this experiment's own tree, that IS the source."""
    exp2_dir = outputs_dir / "02_hypernetwork_multitask"
    records = []
    for cond, dirname_cond in (("notd", "notd"), ("frozen_td", "frozentd")):
        for seed in SEEDS:
            path = exp2_dir / f"hyper_multitask_dim4_{dirname_cond}_seed{seed}" / "results.txt"
            if not path.exists():
                continue
            for task, score in parse_val_by_task(path).items():
                records.append(
                    {"condition": cond, "task": task, "seed": seed, "val_exact_match": score}
                )
    records.sort(key=lambda r: (r["condition"], r["task"], r["seed"]))
    return records


def parse_holdout_table(results_path: Path) -> dict[str, dict[str, float]]:
    """Parse eval_compositional_holdout.py's plain-text results.txt (not a CSV) --
    {category: {n, exact_match, seq_accuracy}}, skipping the header/blank lines and the
    trailing 'overall' row (which has no seq_accuracy column)."""
    rows = {}
    for line in results_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 4 or not parts[0].startswith("1d_comp_"):
            continue
        category, n, exact_match, seq_accuracy = parts
        rows[category] = {
            "n": int(n),
            "exact_match": float(exact_match),
            "seq_accuracy": float(seq_accuracy),
        }
    return rows


def extract_holdout_records(outputs_dir: Path) -> list[dict]:
    """notd/frozen_td zero-shot compositional-holdout numbers, from run.sh's own output --
    outputs/04_compositional_generalization/{notd,frozentd}_seed{N}/results.txt."""
    root = outputs_dir / "04_compositional_generalization"
    records = []
    for cond, dirname_cond in (("notd", "notd"), ("frozen_td", "frozentd")):
        for seed in SEEDS:
            path = root / f"{dirname_cond}_seed{seed}" / "results.txt"
            if not path.exists():
                continue
            for category, scores in parse_holdout_table(path).items():
                records.append(
                    {
                        "condition": cond, "category": category, "seed": seed,
                        "n": scores["n"], "exact_match": scores["exact_match"],
                        "seq_accuracy": scores["seq_accuracy"],
                    }
                )
    records.sort(key=lambda r: (r["condition"], r["category"], r["seed"]))
    return records


def write_csv(records: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {path}")


def read_csv(path: Path, fieldnames: list[str]) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in fieldnames:
            if key not in ("condition", "task", "category"):
                row[key] = float(row[key]) if "." in row[key] or key != "n" else int(row[key])
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


def _bar_plot(
    series: dict[str, dict[str, dict]],
    groups: list[str],
    group_labels: list[str],
    ylabel: str,
    out_path: Path,
) -> None:
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

    n_groups = len(groups)
    n_cond = len(CONDITIONS)
    bar_w = 0.8 / n_cond
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    for i, cond in enumerate(CONDITIONS):
        offset = (i - (n_cond - 1) / 2) * bar_w
        means = np.array([series[cond].get(g, {"mean": 0})["mean"] for g in groups])
        stds = np.array([series[cond].get(g, {"std": 0})["std"] for g in groups])
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
    ax.set_xticklabels(group_labels, rotation=40, ha="right")
    ax.set_ylabel(ylabel)
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


def plot_indist(series: dict[str, dict[str, dict]], out_path: Path) -> None:
    tasks = [t for t in TASK_ORDER if t in series["frozen_td"] or t in series["notd"]]
    labels = [format_task_category(t) for t in tasks]
    _bar_plot(series, tasks, labels, "Validation exact match accuracy", out_path)


def plot_holdout(series: dict[str, dict[str, dict]], out_path: Path) -> None:
    categories = sorted(set(series["frozen_td"]) | set(series["notd"]))
    labels = [c.replace("1d_comp_", "").replace("_", " + ") for c in categories]
    _bar_plot(series, categories, labels, "Zero-shot sequence accuracy", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="If given, rescan outputs/ (both experiment 2's and this experiment's own) and "
        "refresh both CSVs before plotting. Without it, replots from the CSVs as committed.",
    )
    return parser.parse_args()


INDIST_FIELDS = ["condition", "task", "seed", "val_exact_match"]
HOLDOUT_FIELDS = ["condition", "category", "seed", "n", "exact_match", "seq_accuracy"]


def main() -> None:
    args = parse_args()

    if args.outputs_dir is not None:
        indist_records = extract_indist_records(args.outputs_dir)
        write_csv(indist_records, INDIST_CSV_PATH, INDIST_FIELDS)
        holdout_records = extract_holdout_records(args.outputs_dir)
        write_csv(holdout_records, HOLDOUT_CSV_PATH, HOLDOUT_FIELDS)
    else:
        indist_records = read_csv(INDIST_CSV_PATH, INDIST_FIELDS)
        holdout_records = read_csv(HOLDOUT_CSV_PATH, HOLDOUT_FIELDS)

    print("\n== in-distribution, mean +- 1 s.d. across 5 seeds, macro over 14 tasks ==")
    indist_series = build_series(indist_records, "task", "val_exact_match")
    for cond in CONDITIONS:
        vals = [v["mean"] for v in indist_series[cond].values()]
        print(f"  {LABELS[cond]:12s} macro-mean={statistics.mean(vals):.3f}  n_tasks={len(vals)}")
    plot_indist(indist_series, FIGURES_DIR / "per_task_indist.png")

    print("\n== zero-shot compositional holdout, macro over 10 held-out categories ==")
    holdout_seq_series = build_series(holdout_records, "category", "seq_accuracy")
    holdout_em_series = build_series(holdout_records, "category", "exact_match")
    for cond in CONDITIONS:
        seq_vals = [v["mean"] for v in holdout_seq_series[cond].values()]
        em_vals = [v["mean"] for v in holdout_em_series[cond].values()]
        print(
            f"  {LABELS[cond]:12s} seq_accuracy macro-mean={statistics.mean(seq_vals):.3f}  "
            f"exact_match macro-mean={statistics.mean(em_vals):.4f}"
        )
    plot_holdout(holdout_seq_series, FIGURES_DIR / "per_task_holdout.png")


if __name__ == "__main__":
    main()
