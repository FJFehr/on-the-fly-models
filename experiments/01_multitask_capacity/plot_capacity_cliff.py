"""Capacity-cliff figure for Experiment 2 (multi-task capacity).

Compares three training regimes across model size (parameter count):

  1. Individual   -- one model per task, no cross-task sharing
                     (arc1d_v2_minimal_size for dim 4/6, arc1d_v2_backbone_capacity
                     RC1 for dim 10 -- the original Phase 1 run)
  2. Joint, no ID -- one model trained jointly across all 14 task categories,
                     no task-identity signal (arc1d_v2_multitask notd*)
  3. Joint + ID   -- same joint setup, plus a per-task identity embedding
                     (arc1d_v2_multitask td*)

All raw per-seed results live in results.csv next to this script (one row =
one seed's test_query_exact_match; "individual" rows are already averaged
across that seed's 14 separate per-task models). That CSV is what gets
committed to git -- outputs/ itself is gitignored and typically only exists
on whichever cluster node ran the jobs, so plotting from the CSV works
anywhere without needing that node.

Usage
-----
Plot from the committed CSV (works anywhere, no outputs/ needed):

    uv run python experiments/01_multitask_capacity/plot_capacity_cliff.py

Refresh results.csv from a live outputs/ directory (run on the node that
has it), then plot:

    uv run python experiments/01_multitask_capacity/plot_capacity_cliff.py \\
        --outputs-dir outputs
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator

from visualisation.paper.arc_paper import PAPER_COLORS
from visualisation.core.style import apply_latex_style

HERE = Path(__file__).parent
CSV_PATH = Path("outputs/results/01_multitask_capacity/results.csv")
# Rendered PNG/PDF are regenerated from CSV_PATH on demand, not committed --
# see "Figures" in experiments/README.md. Run from the repo
# root (same assumption as --outputs-dir's "outputs" default above).
PLOT_PATH = Path("outputs/figures/01_multitask_capacity/capacity_cliff.png")

# Same architecture at three widths; embedding_dim fixed at 10 throughout.
# Total parameter counts measured directly (embedder + backbone + head; +180
# for the task-embedding table in the "Joint + ID" arms, not shown on the
# x-axis since the swept quantity is backbone width, not that fixed addition).
DIM_PARAMS = {"4": 1398, "6": 2444, "10": 5400, "14": 9508}

# Same palette as the hypernetwork multitask comparison
# (experiments/02_hypernetwork_multitask/per_task_dim4_combined.png)
# so a reader sees one consistent colour language across both experiments:
# baseline/individual = khaki-brown, joint+task-ID = slate blue, joint-no-ID
# = purple. From visualisation.paper.arc_paper's PAPER_COLORS (the muted rainbow
# also used for ARC cell values 0-9), not picked ad hoc.
COLORS = {
    "individual": PAPER_COLORS[0],  # khaki/tan -- baseline
    "notd": PAPER_COLORS[6],  # slate blue -- joint, no task ID
    "td": PAPER_COLORS[8],  # purple -- joint, with task ID
}
LINESTYLES = {"individual": ":", "notd": "-", "td": "-"}
LABELS = {
    "individual": "Individual models",
    "td": "Joint model, Task ID",
    "notd": "Joint model, w/o Task ID",
}
CSV_FIELDS = ["condition", "dim", "params", "seed", "test_exact_match"]


# ---------------------------------------------------------------------------
# Extraction (outputs/ -> results.csv)
# ---------------------------------------------------------------------------


def parse_test_exact_match(results_path: Path) -> float | None:
    """Return test_query_exact_match from a results.txt file, or None."""
    in_test = False
    for line in results_path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "test_metrics:":
            in_test = True
            continue
        if in_test and stripped.startswith("test_query_exact_match:"):
            return float(stripped.split(":", 1)[1].strip())
    return None


def extract_records(outputs_dir: Path) -> list[dict]:
    """Scan outputs/ and return one row per (condition, dim, seed)."""
    individual_raw: dict[str, dict[int, list[float]]] = {"4": {}, "6": {}, "10": {}}

    minsize_re = re.compile(r"^v2_minsize_dim(\d+)_.+_seed(\d+)$")
    for path in sorted((outputs_dir / "arc1d_v2_minimal_size").glob("*/results.txt")):
        m = minsize_re.match(path.parent.name)
        if not m:
            continue
        dim, seed = m.group(1), int(m.group(2))
        score = parse_test_exact_match(path)
        if score is not None:
            individual_raw[dim].setdefault(seed, []).append(score)

    rc1_re = re.compile(r"^v2_RC1_dim10_.+_rope_canon_n1_seed(\d+)$")
    for path in sorted((outputs_dir / "arc1d_v2_backbone_capacity").glob("*/results.txt")):
        m = rc1_re.match(path.parent.name)
        if not m:
            continue
        seed = int(m.group(1))
        score = parse_test_exact_match(path)
        if score is not None:
            individual_raw["10"].setdefault(seed, []).append(score)

    records = []
    for dim, per_seed in individual_raw.items():
        for seed, scores in per_seed.items():
            records.append(
                {
                    "condition": "individual",
                    "dim": dim,
                    "params": DIM_PARAMS[dim],
                    "seed": seed,
                    "test_exact_match": sum(scores) / len(scores),
                }
            )

    pattern = re.compile(r"^v2_multitask_(notd|td)(?:_dim(\d+))?_seed(\d+)$")
    for path in sorted((outputs_dir / "arc1d_v2_multitask").glob("*/results.txt")):
        m = pattern.match(path.parent.name)
        if not m:
            continue
        cond, dim, seed = m.group(1), m.group(2) or "10", int(m.group(3))
        score = parse_test_exact_match(path)
        if score is not None:
            records.append(
                {
                    "condition": cond,
                    "dim": dim,
                    "params": DIM_PARAMS[dim],
                    "seed": seed,
                    "test_exact_match": score,
                }
            )

    records.sort(key=lambda r: (r["condition"], int(r["dim"]), r["seed"]))
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
                "dim": row["dim"],
                "params": int(row["params"]),
                "seed": int(row["seed"]),
                "test_exact_match": float(row["test_exact_match"]),
            }
            for row in csv.DictReader(f)
        ]


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------


def build_series(records: list[dict]) -> dict[str, dict[str, dict]]:
    """condition -> dim -> summary stats, ready for plotting."""
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in records:
        by_key.setdefault((r["condition"], r["dim"]), []).append(r["test_exact_match"])

    series: dict[str, dict[str, dict]] = {"individual": {}, "notd": {}, "td": {}}
    for (cond, dim), values in by_key.items():
        series[cond][dim] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "n": len(values),
            "min": min(values),
            "max": max(values),
        }
    return series


def plot(series: dict[str, dict[str, dict]], out_path: Path) -> None:
    apply_latex_style()
    # Bump past the shared style's default sizes (tuned for dense multi-panel
    # LaTeX figures) -- this is a single standalone chart, larger text reads
    # better at its actual display size. Font swapped from Computer Modern to
    # Nimbus Roman (metric-compatible with Times New Roman -- what LaTeX's
    # `times` package actually renders as) to match the NeurIPS template.
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 23,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 16,
            "pdf.fonttype": 42,  # embed as TrueType, not the default Type 3
        }
    )
    # Square, sized for a single-column paper figure.
    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    dims_sorted = sorted(DIM_PARAMS, key=lambda d: DIM_PARAMS[d])

    for cond in ("individual", "td", "notd"):
        dims_present = [d for d in dims_sorted if d in series[cond]]
        if len(dims_present) < 2:
            continue
        x = np.array([DIM_PARAMS[d] for d in dims_present], dtype=float)
        means = np.array([series[cond][d]["mean"] for d in dims_present])
        stds = np.array([series[cond][d]["std"] for d in dims_present])

        log_x = np.log10(x)
        log_x_fine = np.linspace(log_x.min(), log_x.max(), 200)
        x_fine = 10**log_x_fine

        mean_fine = PchipInterpolator(log_x, means)(log_x_fine)
        lo_fine = np.clip(PchipInterpolator(log_x, means - stds)(log_x_fine), 0, 1)
        hi_fine = np.clip(PchipInterpolator(log_x, means + stds)(log_x_fine), 0, 1)

        ax.fill_between(
            x_fine, lo_fine, hi_fine, color=COLORS[cond], alpha=0.18, linewidth=0,
            zorder=1,
        )
        ax.plot(
            x_fine,
            mean_fine,
            color=COLORS[cond],
            linestyle=LINESTYLES[cond],
            linewidth=2.2,
            label=LABELS[cond],
            zorder=3,
        )
        ax.plot(
            x, means, "o", color=COLORS[cond], markersize=10,
            markeredgecolor="white", markeredgewidth=1.2, zorder=4,
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
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    # PDF (vector, for the paper) alongside PNG (for quick preview / the README).
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
        help="If given, rescan this outputs/ dir and refresh results.csv before plotting.",
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
    for cond in ("individual", "notd", "td"):
        for dim in ("4", "6", "10"):
            s = series[cond].get(dim)
            if s:
                print(
                    f"{cond:10s} dim={dim:>2s} ({DIM_PARAMS[dim]:>5,} params): "
                    f"mean={s['mean']:.3f} std={s['std']:.3f} n={s['n']}"
                )

    plot(series, PLOT_PATH)


if __name__ == "__main__":
    main()
