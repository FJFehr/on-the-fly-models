"""Optimizer-ablation capacity-cliff for experiment 1: Muon vs. AdamW.

Companion to plot_capacity_cliff.py and plot_canon_ablation.py -- same
three conditions (Individual, Joint notd, Joint td) across the same four
sizes, split by optimizer instead of by Canon: Muon (configs/{individual,
notd,td}*, the default everywhere else in this experiment) vs. AdamW
(configs/adamw/..., optimizer: AdamW). Canon stays enabled in both arms, so
this isolates the optimizer choice alone. Colour still separates condition
(khaki/blue/purple, same palette as the other capacity-cliff plots); line
style separates the ablation: solid = Muon, dashed = AdamW.

The two optimizers' learning rates and shared scheduler are documented in
"Optimizer ablation" in this experiment's README, not on the figure itself.

Usage
-----
    uv run python experiments/01_multitask_capacity/plot_optimizer_ablation.py \\
        --outputs-dir outputs

    uv run python experiments/01_multitask_capacity/plot_optimizer_ablation.py
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from plot_capacity_cliff import COLORS, DIM_PARAMS, LABELS, parse_test_exact_match
from scipy.interpolate import PchipInterpolator

from visualisation.core.style import apply_latex_style

HERE = Path(__file__).parent
CSV_PATH = Path("outputs/results/01_multitask_capacity/results_optimizer_ablation.csv")
PLOT_PATH = Path("outputs/figures/01_multitask_capacity/capacity_cliff_optimizer_ablation.png")
CSV_FIELDS = ["condition", "muon", "dim", "params", "seed", "test_exact_match"]


def extract_records(outputs_dir: Path) -> list[dict]:
    """Scan outputs/ and return one row per (condition, muon, dim, seed)."""
    project_dir = outputs_dir / "01_multitask_capacity"
    records = []

    individual_raw: dict[bool, dict[str, dict[int, list[float]]]] = {
        True: {d: {} for d in DIM_PARAMS},
        False: {d: {} for d in DIM_PARAMS},
    }
    for muon, pat in (
        (True, re.compile(r"^individual_dim(\d+)_.+_seed(\d+)$")),
        (False, re.compile(r"^individual_adamw_dim(\d+)_.+_seed(\d+)$")),
    ):
        for path in sorted(project_dir.glob("*/results.txt")):
            m = pat.match(path.parent.name)
            if not m:
                continue
            dim, seed = m.group(1), int(m.group(2))
            score = parse_test_exact_match(path)
            if score is not None:
                individual_raw[muon][dim].setdefault(seed, []).append(score)
    for muon, by_dim in individual_raw.items():
        for dim, per_seed in by_dim.items():
            for seed, scores in per_seed.items():
                records.append(
                    {
                        "condition": "individual",
                        "muon": muon,
                        "dim": dim,
                        "params": DIM_PARAMS[dim],
                        "seed": seed,
                        "test_exact_match": sum(scores) / len(scores),
                    }
                )

    for muon, pat in (
        (True, re.compile(r"^joint_(notd|td)_dim(\d+)_seed(\d+)$")),
        (False, re.compile(r"^joint_adamw_(notd|td)_dim(\d+)_seed(\d+)$")),
    ):
        for path in sorted(project_dir.glob("*/results.txt")):
            m = pat.match(path.parent.name)
            if not m:
                continue
            cond, dim, seed = m.group(1), m.group(2), int(m.group(3))
            score = parse_test_exact_match(path)
            if score is not None:
                records.append(
                    {
                        "condition": cond,
                        "muon": muon,
                        "dim": dim,
                        "params": DIM_PARAMS[dim],
                        "seed": seed,
                        "test_exact_match": score,
                    }
                )

    records.sort(key=lambda r: (r["condition"], not r["muon"], int(r["dim"]), r["seed"]))
    return records


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                "muon": row["muon"] == "True",
                "dim": row["dim"],
                "params": int(row["params"]),
                "seed": int(row["seed"]),
                "test_exact_match": float(row["test_exact_match"]),
            }
            for row in csv.DictReader(f)
        ]


def build_series(records: list[dict]) -> dict[tuple[str, bool], dict[str, dict]]:
    """(condition, muon) -> dim -> summary stats."""
    by_key: dict[tuple[str, bool, str], list[float]] = {}
    for r in records:
        by_key.setdefault((r["condition"], r["muon"], r["dim"]), []).append(
            r["test_exact_match"]
        )
    series: dict[tuple[str, bool], dict[str, dict]] = {}
    for (cond, muon, dim), values in by_key.items():
        mean = sum(values) / len(values)
        series.setdefault((cond, muon), {})[dim] = {
            "mean": mean,
            "std": (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5,
            "n": len(values),
        }
    return series


def plot(series: dict[tuple[str, bool], dict[str, dict]], out_path: Path) -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 23,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    dims_sorted = sorted(DIM_PARAMS, key=lambda d: DIM_PARAMS[d])

    for cond in ("individual", "td", "notd"):
        for muon, linestyle, alpha in ((True, "-", 1.0), (False, "--", 0.75)):
            s = series.get((cond, muon))
            if not s:
                continue
            dims_present = [d for d in dims_sorted if d in s]
            if len(dims_present) < 2:
                continue
            x = np.array([DIM_PARAMS[d] for d in dims_present], dtype=float)
            means = np.array([s[d]["mean"] for d in dims_present])
            stds = np.array([s[d]["std"] for d in dims_present])

            log_x = np.log10(x)
            log_x_fine = np.linspace(log_x.min(), log_x.max(), 200)
            x_fine = 10**log_x_fine
            mean_fine = PchipInterpolator(log_x, means)(log_x_fine)
            lo_fine = np.clip(PchipInterpolator(log_x, means - stds)(log_x_fine), 0, 1)
            hi_fine = np.clip(PchipInterpolator(log_x, means + stds)(log_x_fine), 0, 1)

            opt_tag = "Muon" if muon else "AdamW"
            ax.fill_between(
                x_fine, lo_fine, hi_fine, color=COLORS[cond], alpha=0.10, linewidth=0,
                zorder=1,
            )
            ax.plot(
                x_fine, mean_fine, color=COLORS[cond], linestyle=linestyle, alpha=alpha,
                linewidth=2.2, label=f"{LABELS[cond]} ({opt_tag})", zorder=3,
            )
            ax.plot(
                x, means, "o" if muon else "s", color=COLORS[cond], alpha=alpha,
                markersize=8, markeredgecolor="white", markeredgewidth=1.0, zorder=4,
            )

    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    tick_values = [DIM_PARAMS[d] for d in dims_sorted]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([f"{v / 1000:.1f}K" for v in tick_values])
    ax.set_xlabel("Model parameters (log scale)")
    ax.set_ylabel("Test exact match accuracy")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="lower right", frameon=False, ncol=1,
        handlelength=1.6, labelspacing=0.3, borderaxespad=0.3,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"Saved {out_path} and {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="If given, rescan this outputs/ dir and refresh results_optimizer_ablation.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.outputs_dir is not None:
        records = extract_records(args.outputs_dir)
        write_csv(records, CSV_PATH)
    else:
        records = read_csv(CSV_PATH)

    series = build_series(records)
    dims_sorted = sorted(DIM_PARAMS, key=lambda d: DIM_PARAMS[d])
    for cond in ("individual", "notd", "td"):
        for muon in (True, False):
            s = series.get((cond, muon), {})
            for dim in dims_sorted:
                stats = s.get(dim)
                if stats:
                    tag = "muon " if muon else "adamw"
                    print(
                        f"{cond:10s} {tag} dim={dim:>2s} ({DIM_PARAMS[dim]:>5,} params): "
                        f"mean={stats['mean']:.3f} std={stats['std']:.3f} n={stats['n']}"
                    )

    plot(series, PLOT_PATH)


if __name__ == "__main__":
    main()
