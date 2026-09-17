"""Data-efficiency figure for Experiment 6 (data efficiency ablation).

Same style/design as Experiment 1's capacity-cliff figure
(01_multitask_capacity/plot_capacity_cliff.py) -- log-x, PCHIP-smoothed mean
line with a shaded +-1 std band, markers at the actual measured levels -- but
the swept quantity on the x-axis is training-data amount (rows/category),
not model parameter count, and the three conditions compared are:

  1. Individual   -- one model per task category, no cross-task sharing at
                     all (arc1d_lowdata_baseline, dim=4 matched-scale target)
  2. Hyper, Task ID    -- the hypernetwork, frozen_td (arc1d_lowdata)
  3. Hyper, w/o Task ID -- the hypernetwork, notd (arc1d_lowdata)

The joint (direct, no-hypernetwork) arm is deliberately left out of this
figure -- it's a different comparison (weight generation vs. plain task-id
conditioning, at a different model scale entirely, see
06_data_efficiency_ablation/joint/README.md) from the one this figure is
making (does cross-task weight generation buy data efficiency over having no
sharing at all).

Individual only has 4 v/full levels (v1/v2/v3/full -- see
06_data_efficiency_ablation/individual/README.md for why 4/5/20 are out of
scope there); the hypernetwork conditions have all 7. Each series is plotted
over whichever levels it actually has data for, same as capacity_cliff.py's
per-condition `dims_present` handling.

2026-09-15: added the sub-40-base-tasks-per-category axis (t1/t3/t5/t10/t20,
always at variants_per_base_task=1 -- see hypernetwork/README.md's "Data
levels" for the full story) to all three series, extending the primary axis
below 40 rows/category down to 1.

All raw per-seed records live in results.csv next to this script (one row =
one seed's test_query_exact_match; "individual" rows are already averaged
across that seed's 14 per-category models, exactly mirroring
01_multitask_capacity's own individual-row convention). That CSV is what
gets committed to git -- outputs/ itself is gitignored.

Usage
-----
Plot from the committed CSV (works anywhere, no outputs/ needed):

    uv run python experiments/06_data_efficiency_ablation/plot_data_efficiency.py

Refresh results.csv from a live outputs/ directory, then plot:

    uv run python experiments/06_data_efficiency_ablation/plot_data_efficiency.py \\
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
CSV_PATH = Path("outputs/results/06_data_efficiency_ablation/results_data_efficiency_cliff.csv")
PLOT_PATH = Path("outputs/figures/06_data_efficiency_ablation/data_efficiency_cliff.png")

CATEGORIES = [
    "1d_denoising_1c", "1d_denoising_mc", "1d_fill", "1d_flip", "1d_hollow",
    "1d_mirror", "1d_move_1p", "1d_move_2p", "1d_move_2p_dp", "1d_move_3p",
    "1d_move_dp", "1d_pcopy_1c", "1d_pcopy_mc", "1d_scale_dp",
]

# variants_per_base_task -> rows/category (40 base tasks/category, per the
# nested/cumulative design -- see hypernetwork/README.md's "Data levels").
# "full" is ~1000 variants/base task -- 40,000 rows/category. t{N} (2026-09-15)
# is the sub-40-base-tasks-per-category axis -- N base tasks x 1 variant
# (the original) each = N rows/category exactly, no approximation needed.
LEVEL_ROWS = {
    "t1": 1, "t3": 3, "t5": 5, "t10": 10, "t20": 20,
    "v1": 40, "v2": 80, "v3": 120, "v4": 160, "v5": 200, "v20": 800, "full": 40_000,
}

# individual = khaki-brown baseline (same as 01/02's own "individual" role).
# td/notd match 02_hypernetwork_multitask/plot_per_task_combined.py's own
# hyper_td/hyper_notd colours exactly (plum / teal) -- not 01's joint-model
# purple/slate-blue, since this figure's td/notd lines are the hypernetwork's,
# the same condition 02 already colours this way.
COLORS = {
    "individual": PAPER_COLORS[0],
    "notd": PAPER_COLORS[5],  # teal -- matches 02's hyper_notd
    "td": PAPER_COLORS[9],  # plum/pink -- matches 02's hyper_td
}
LINESTYLES = {"individual": ":", "notd": "-", "td": "-"}
LABELS = {
    "individual": "Individual models",
    "td": "Hypernetwork, Task ID",
    "notd": "Hypernetwork, w/o Task ID",
}
CSV_FIELDS = ["condition", "level", "data_amount", "seed", "test_exact_match"]


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
    """Scan outputs/ and return one row per (condition, level, seed)."""
    records = []

    # --- hypernetwork: lowdata_{frozentd,notd}_{level}_seed{n} ---
    cond_map = {"frozentd": "td", "notd": "notd"}
    hyper_re = re.compile(r"^lowdata_(frozentd|notd)_(v\d+|t\d+|full)_seed(\d+)$")
    hyper_dir = outputs_dir / "arc1d_lowdata"
    for path in sorted(hyper_dir.glob("*/results.txt")):
        m = hyper_re.match(path.parent.name)
        if not m:
            continue  # old pre-resize orphan (different naming) -- skip
        raw_cond, level, seed = m.group(1), m.group(2), int(m.group(3))
        score = parse_test_exact_match(path)
        if score is not None:
            records.append({
                "condition": cond_map[raw_cond], "level": level,
                "data_amount": LEVEL_ROWS[level], "seed": seed,
                "test_exact_match": score,
            })

    # --- individual: lowdata_baseline_{category}_{level}_seed{n} ---
    # Average across the 14 categories per (level, seed) first, exactly
    # mirroring 01_multitask_capacity's own "individual" rows (already
    # averaged across that seed's 14 separate per-task models).
    cat_alt = "|".join(re.escape(c) for c in CATEGORIES)
    indiv_re = re.compile(rf"^lowdata_baseline_({cat_alt})_(v\d+|t\d+|full)_seed(\d+)$")
    indiv_dir = outputs_dir / "arc1d_lowdata_baseline"
    indiv_raw: dict[tuple[str, int], list[float]] = {}
    for path in sorted(indiv_dir.glob("*/results.txt")):
        m = indiv_re.match(path.parent.name)
        if not m:
            continue
        _category, level, seed = m.group(1), m.group(2), int(m.group(3))
        score = parse_test_exact_match(path)
        if score is not None:
            indiv_raw.setdefault((level, seed), []).append(score)

    for (level, seed), scores in indiv_raw.items():
        records.append({
            "condition": "individual", "level": level,
            "data_amount": LEVEL_ROWS[level], "seed": seed,
            "test_exact_match": sum(scores) / len(scores),
        })

    records.sort(key=lambda r: (r["condition"], r["data_amount"], r["seed"]))
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
                "level": row["level"],
                "data_amount": int(row["data_amount"]),
                "seed": int(row["seed"]),
                "test_exact_match": float(row["test_exact_match"]),
            }
            for row in csv.DictReader(f)
        ]


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------


def build_series(records: list[dict]) -> dict[str, dict[int, dict]]:
    """condition -> data_amount -> summary stats, ready for plotting."""
    by_key: dict[tuple[str, int], list[float]] = {}
    for r in records:
        by_key.setdefault((r["condition"], r["data_amount"]), []).append(r["test_exact_match"])

    series: dict[str, dict[int, dict]] = {"individual": {}, "notd": {}, "td": {}}
    for (cond, amount), values in by_key.items():
        series[cond][amount] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "n": len(values),
        }
    return series


def plot(series: dict[str, dict[int, dict]], out_path: Path) -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 23,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 16,
            "pdf.fonttype": 42,
        }
    )
    # Width matches 01_multitask_capacity/plot_capacity_cliff.py's own 5.8x5.8 square
    # exactly; height is taller by the extra vertical space the second x-axis row
    # (augmentations/example) below needs, so the *axes* (the actual plot rectangle,
    # not the figure) end up the same physical size in both -- verified empirically:
    # capacity_cliff's rendered axes are 4.34x4.81in; at 5.8x5.8 this file's own axes
    # were only 4.34x3.56in (tight_layout gave the whole deficit to the bottom margin
    # for the second axis row), so 5.8 + (4.81-3.56) = 7.05 recovers the same 4.81in
    # axes height while keeping that same bottom margin for the second row.
    fig, ax = plt.subplots(figsize=(5.8, 6.74))

    all_amounts = sorted({a for cond in series.values() for a in cond})

    for cond in ("individual", "td", "notd"):
        amounts_present = sorted(a for a in all_amounts if a in series[cond])
        if len(amounts_present) < 2:
            continue
        x = np.array(amounts_present, dtype=float)
        means = np.array([series[cond][a]["mean"] for a in amounts_present])
        stds = np.array([series[cond][a]["std"] for a in amounts_present])

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
            x_fine, mean_fine, color=COLORS[cond], linestyle=LINESTYLES[cond],
            linewidth=2.2, label=LABELS[cond], zorder=3,
        )
        ax.plot(
            x, means, "o", color=COLORS[cond], markersize=10,
            markeredgecolor="white", markeredgewidth=1.2, zorder=4,
        )

    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    # Labelling every measured level crowds several values into one log-decade
    # and the tick labels collide. Label a well-spaced subset instead -- the
    # markers themselves still show every measured point, labelled or not.
    tick_values = [1, 10, 40, 200, 800, 40_000]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(
        [f"{v // 1000}K" if v >= 1000 else str(v) for v in tick_values]
    )
    ax.set_xlabel("Training rows / category (log scale)")

    # Second x-axis row: augmentations/example. For the v{N}/full family this
    # is a genuine linear function of rows/category (rows = 40 base
    # tasks/category x variants_per_base_task). But the 2026-09-15 t{N} family
    # varies base-task *count* at a fixed variants_per_base_task=1 -- multiple
    # distinct primary-axis positions (1,3,5,10,20,40 rows -- t1..t20 and v1)
    # all genuinely mean "0 additional augmentation", which no single
    # computed function can express (it would need to be non-injective, and
    # 0 augmentations has no defined position on a log axis via rows/40
    # anyway). So this axis is now an *identity* secondary_xaxis (shares the
    # primary axis's own rows/category coordinates exactly, keeping automatic
    # pixel-alignment under resize/pan) with a hand-picked (position, label)
    # table instead of a formula -- v1's own label deliberately changes from
    # the old "1" to "0" here, so the whole zero-additional-augmentation
    # family (t20/t10/t5/t3/t1/v1) reads consistently as "0", while
    # v3/v5/v20/full keep their previously-correct labels. This is an
    # intentional redefinition, not a bug.
    secax = ax.secondary_xaxis(-0.18, functions=(lambda x: x, lambda x: x))
    # Only the first (1) and last (40) position of the zero-augmentation run
    # (t1/t3/t5/t10/t20/v1) get a "0" tick -- labelling all six crowded the
    # axis with repeats of the same value. The intermediate t-levels are
    # still plotted (markers at 3/5/10/20) -- just not tick-labelled here.
    # "3" (v3, row 120) is dropped too -- too close to "5" (v5, row 200) on
    # the log scale to both need a label; the v3 marker itself still shows.
    sec_tick_positions = [1, 40, 200, 800, 40_000]
    sec_tick_labels = ["0", "0", "5", "20", "1000"]
    secax.set_xticks(sec_tick_positions)
    secax.set_xticklabels(sec_tick_labels)
    secax.set_xlabel("Augmentations / example")
    ax.set_ylabel("Test exact match accuracy")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

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
        for amount in sorted(series[cond]):
            s = series[cond][amount]
            print(
                f"{cond:10s} rows/cat={amount:>6,}: "
                f"mean={s['mean']:.3f} std={s['std']:.3f} n={s['n']}"
            )

    plot(series, PLOT_PATH)


if __name__ == "__main__":
    main()
