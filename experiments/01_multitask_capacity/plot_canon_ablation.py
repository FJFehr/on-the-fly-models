"""Canon-ablation capacity-cliff for experiment 1: with vs. without Canon.

Companion to plot_capacity_cliff.py -- same three conditions (Individual,
Joint notd, Joint td) across the same four sizes, but split by whether the
RoPE+Canon backbone's Canon layer is enabled (configs/{individual,notd,td}*)
or disabled (configs/nocanon/..., canon_set: '' instead of 'ABCD'). Colour
still separates condition (khaki/blue/purple, same palette as the main
capacity-cliff plot); line style now separates the ablation instead of
condition: solid = canon, dashed = no canon. See "Canon ablation" in this
experiment's README for the numbers and the finding.

Usage
-----
    uv run python experiments/01_multitask_capacity/plot_canon_ablation.py \\
        --outputs-dir outputs

    uv run python experiments/01_multitask_capacity/plot_canon_ablation.py
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
CSV_PATH = Path("outputs/results/01_multitask_capacity/results_canon_ablation.csv")
PLOT_PATH = Path("outputs/figures/01_multitask_capacity/capacity_cliff_canon_ablation.png")
CSV_FIELDS = ["condition", "canon", "dim", "params", "seed", "test_exact_match"]


def extract_records(outputs_dir: Path) -> list[dict]:
    """Scan outputs/ and return one row per (condition, canon, dim, seed)."""
    project_dir = outputs_dir / "01_multitask_capacity"
    records = []

    individual_raw: dict[bool, dict[str, dict[int, list[float]]]] = {
        True: {d: {} for d in DIM_PARAMS},
        False: {d: {} for d in DIM_PARAMS},
    }
    for canon, pat in (
        (True, re.compile(r"^individual_dim(\d+)_.+_seed(\d+)$")),
        (False, re.compile(r"^individual_nocanon_dim(\d+)_.+_seed(\d+)$")),
    ):
        for path in sorted(project_dir.glob("*/results.txt")):
            m = pat.match(path.parent.name)
            if not m:
                continue
            dim, seed = m.group(1), int(m.group(2))
            score = parse_test_exact_match(path)
            if score is not None:
                individual_raw[canon][dim].setdefault(seed, []).append(score)
    for canon, by_dim in individual_raw.items():
        for dim, per_seed in by_dim.items():
            for seed, scores in per_seed.items():
                records.append(
                    {
                        "condition": "individual",
                        "canon": canon,
                        "dim": dim,
                        "params": DIM_PARAMS[dim],
                        "seed": seed,
                        "test_exact_match": sum(scores) / len(scores),
                    }
                )

    for canon, pat in (
        (True, re.compile(r"^joint_(notd|td)_dim(\d+)_seed(\d+)$")),
        (False, re.compile(r"^joint_nocanon_(notd|td)_dim(\d+)_seed(\d+)$")),
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
                        "canon": canon,
                        "dim": dim,
                        "params": DIM_PARAMS[dim],
                        "seed": seed,
                        "test_exact_match": score,
                    }
                )

    records.sort(key=lambda r: (r["condition"], not r["canon"], int(r["dim"]), r["seed"]))
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
                "canon": row["canon"] == "True",
                "dim": row["dim"],
                "params": int(row["params"]),
                "seed": int(row["seed"]),
                "test_exact_match": float(row["test_exact_match"]),
            }
            for row in csv.DictReader(f)
        ]


def build_series(records: list[dict]) -> dict[tuple[str, bool], dict[str, dict]]:
    """(condition, canon) -> dim -> summary stats."""
    by_key: dict[tuple[str, bool, str], list[float]] = {}
    for r in records:
        by_key.setdefault((r["condition"], r["canon"], r["dim"]), []).append(
            r["test_exact_match"]
        )
    series: dict[tuple[str, bool], dict[str, dict]] = {}
    for (cond, canon, dim), values in by_key.items():
        series.setdefault((cond, canon), {})[dim] = {
            "mean": sum(values) / len(values),
            "std": (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values))
            ** 0.5,
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
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    dims_sorted = sorted(DIM_PARAMS, key=lambda d: DIM_PARAMS[d])

    for cond in ("individual", "td", "notd"):
        for canon, linestyle, alpha in ((True, "-", 1.0), (False, "--", 0.75)):
            s = series.get((cond, canon))
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

            canon_tag = "canon" if canon else "no canon"
            ax.fill_between(
                x_fine, lo_fine, hi_fine, color=COLORS[cond], alpha=0.10, linewidth=0,
                zorder=1,
            )
            ax.plot(
                x_fine, mean_fine, color=COLORS[cond], linestyle=linestyle, alpha=alpha,
                linewidth=2.2, label=f"{LABELS[cond]} ({canon_tag})", zorder=3,
            )
            ax.plot(
                x, means, "o" if canon else "s", color=COLORS[cond], alpha=alpha,
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
    ax.legend(loc="lower right", frameon=False, ncol=1)

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
        help="If given, rescan this outputs/ dir and refresh results_canon_ablation.csv.",
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
        for canon in (True, False):
            s = series.get((cond, canon), {})
            for dim in dims_sorted:
                stats = s.get(dim)
                if stats:
                    tag = "canon   " if canon else "nocanon "
                    print(
                        f"{cond:10s} {tag} dim={dim:>2s} ({DIM_PARAMS[dim]:>5,} params): "
                        f"mean={stats['mean']:.3f} std={stats['std']:.3f} n={stats['n']}"
                    )

    plot(series, PLOT_PATH)


if __name__ == "__main__":
    main()
