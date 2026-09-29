"""Aggregate and plot experiment 8: the Muon grid (arm A) and the plain-AdamW grid (arm B).

Reads every outputs/08_optimiser_tuning/<config>_seed<N>/results.txt and writes:
- outputs/results/08_optimiser_tuning/results.csv: one row per run.
- outputs/figures/08_optimiser_tuning/muon_val_loss.{png,pdf}: mean val_loss over seeds, Muon
  learning rate x Adam-group learning rate, one panel per weight decay; the best cell is boxed.
- outputs/figures/08_optimiser_tuning/marginals.{png,pdf}: for each tuned setting, the best
  mean val_loss reachable at each of its values (Muon: 3 panels), and AdamW's val_loss against
  its learning rate, one line per weight decay.

The console summary lists each arm's top cells (mean and s.d. over seeds), flags a best cell
on a grid edge (stage 2 then extends that grid), compares the sanity cell with experiment 2,
and compares the best Muon cell with the best AdamW cell. Partial sweeps are fine: missing
runs are skipped and each cell reports how many seeds it has.

Usage:
    uv run python experiments/08_optimiser_tuning/plot_all.py
    uv run python experiments/08_optimiser_tuning/plot_all.py --outputs-dir /path/to/outputs
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

import numpy as np
from gen_configs import ADAMW_LRS, MUON_ADAM_LRS, MUON_LRS, WEIGHT_DECAYS
from matplotlib import pyplot as plt

from visualisation.core.style import apply_latex_style
from visualisation.paper.arc_paper import PAPER_COLORS

PROJECT = "08_optimiser_tuning"
RESULTS_DIR = Path("outputs/results") / PROJECT
FIGURES_DIR = Path("outputs/figures") / PROJECT
SANITY_CELL = ("muon", "0.005", "1e-3", "0.01")
EXP02_SANITY_RUNS = "02_hypernetwork_multitask/hyper_multitask_dim4_notd_seed{seed}"

METRICS = [
    "val_loss",
    "val_query_accuracy",
    "val_query_exact_match",
    "test_loss",
    "test_query_accuracy",
    "test_query_exact_match",
]
CSV_FIELDS = ["arm", "muon_lr", "learning_rate", "weight_decay", "seed", *METRICS]

MUON_RUN = re.compile(r"^muon_m(?P<m>[^_]+)_a(?P<a>[^_]+)_wd(?P<wd>[^_]+)_seed(?P<seed>\d+)$")
ADAMW_RUN = re.compile(r"^adamw_lr(?P<a>[^_]+)_wd(?P<wd>[^_]+)_seed(?P<seed>\d+)$")
# Fixed order: one colour per weight decay, the same in every panel.
WD_COLORS = dict(
    zip(WEIGHT_DECAYS, (PAPER_COLORS[5], PAPER_COLORS[2], PAPER_COLORS[8]), strict=True)
)


def parse_results(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition(":")
        if sep and key in METRICS:
            values[key] = float(value)
    return values


def collect(outputs_dir: Path) -> list[dict]:
    rows = []
    for results_path in sorted((outputs_dir / PROJECT).glob("*/results.txt")):
        name = results_path.parent.name
        if match := MUON_RUN.match(name):
            row = {"arm": "muon", "muon_lr": match["m"]}
        elif match := ADAMW_RUN.match(name):
            row = {"arm": "adamw", "muon_lr": ""}
        else:
            continue
        row |= {"learning_rate": match["a"], "weight_decay": match["wd"], "seed": match["seed"]}
        rows.append(row | parse_results(results_path))
    return rows


def write_csv(rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "results.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def cell_key(row: dict) -> tuple[str, str, str, str]:
    return (row["arm"], row["muon_lr"], row["learning_rate"], row["weight_decay"])


def summarise(rows: list[dict]) -> dict[tuple, dict]:
    """{cell: {metric: (mean, s.d.), 'n': seeds}} over the seeds each cell has."""
    by_cell: dict[tuple, list[dict]] = {}
    for row in rows:
        by_cell.setdefault(cell_key(row), []).append(row)
    cells = {}
    for key, runs in by_cell.items():
        stats: dict = {"n": len(runs)}
        for metric in METRICS:
            values = [run[metric] for run in runs if metric in run]
            if values:
                sd = statistics.stdev(values) if len(values) > 1 else 0.0
                stats[metric] = (statistics.mean(values), sd)
        cells[key] = stats
    return cells


def cell_label(key: tuple) -> str:
    arm, muon_lr, lr, wd = key
    if arm == "muon":
        return f"Muon {muon_lr}, Adam group {lr}, wd {wd}"
    return f"AdamW {lr}, wd {wd}"


def edge_settings(key: tuple) -> list[str]:
    """The tuned settings of `key` that sit on the edge of their grid."""
    arm, muon_lr, lr, wd = key
    axes = [("weight decay", wd, WEIGHT_DECAYS)]
    if arm == "muon":
        axes = [("Muon lr", muon_lr, MUON_LRS), ("Adam-group lr", lr, MUON_ADAM_LRS), *axes]
    else:
        axes = [("AdamW lr", lr, ADAMW_LRS), *axes]
    # Weight decay 0 cannot be extended downwards, so only its upper edge counts.
    return [
        name
        for name, value, grid in axes
        if value == grid[-1] or (value == grid[0] and name != "weight decay")
    ]


def fmt(stats: dict, metric: str) -> str:
    if metric not in stats:
        return "n/a"
    mean, sd = stats[metric]
    return f"{mean:.4f} ± {sd:.4f}"


def print_summary(cells: dict[tuple, dict], outputs_dir: Path, top: int = 10) -> None:
    best = {}
    for arm in ("muon", "adamw"):
        ranked = sorted(
            (key for key in cells if key[0] == arm and "val_loss" in cells[key]),
            key=lambda key: cells[key]["val_loss"][0],
        )
        if not ranked:
            print(f"\n{arm}: no finished runs yet")
            continue
        best[arm] = ranked[0]
        n_expected = 60 if arm == "muon" else 15
        print(f"\n{arm}: {len(ranked)}/{n_expected} cells with results; top {top} by val_loss")
        for key in ranked[:top]:
            stats = cells[key]
            print(
                f"  {cell_label(key):40s} val_loss {fmt(stats, 'val_loss')}  "
                f"val EM {fmt(stats, 'val_query_exact_match')}  (n={stats['n']})"
            )
        if edges := edge_settings(ranked[0]):
            print(f"  best cell is on the grid edge for: {', '.join(edges)} -> extend in stage 2")

    if SANITY_CELL in cells:
        exp02 = [
            parse_results(p)["val_loss"]
            for seed in (1, 2, 3)
            if (p := outputs_dir / EXP02_SANITY_RUNS.format(seed=seed) / "results.txt").exists()
        ]
        print(f"\nSanity cell ({cell_label(SANITY_CELL)}): {fmt(cells[SANITY_CELL], 'val_loss')}")
        if exp02:
            sd = statistics.stdev(exp02) if len(exp02) > 1 else 0.0
            print(
                f"  experiment 2 dim-4 notd, seeds 1-{len(exp02)}: "
                f"{statistics.mean(exp02):.4f} ± {sd:.4f}"
            )

    if len(best) == 2:
        print("\nBest Muon vs best AdamW")
        for arm in ("muon", "adamw"):
            stats = cells[best[arm]]
            print(f"  {cell_label(best[arm])}")
            for metric in (
                "val_loss",
                "val_query_exact_match",
                "test_query_accuracy",
                "test_query_exact_match",
            ):
                print(f"    {metric:24s} {fmt(stats, metric)}")


def mean_val_loss(cells: dict, key: tuple) -> float:
    return cells[key]["val_loss"][0] if key in cells and "val_loss" in cells[key] else np.nan


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{name}.{suffix}")
    print(f"Wrote {FIGURES_DIR / name}.png")


def plot_muon_heatmaps(cells: dict) -> None:
    grids = {
        wd: np.array(
            [[mean_val_loss(cells, ("muon", m, a, wd)) for a in MUON_ADAM_LRS] for m in MUON_LRS]
        )
        for wd in WEIGHT_DECAYS
    }
    if all(np.isnan(grid).all() for grid in grids.values()):
        return
    vmin = np.nanmin([np.nanmin(g) for g in grids.values() if not np.isnan(g).all()])
    vmax = np.nanmax([np.nanmax(g) for g in grids.values() if not np.isnan(g).all()])
    best_key = min(
        (key for key in cells if key[0] == "muon" and "val_loss" in cells[key]),
        key=lambda key: cells[key]["val_loss"][0],
    )

    fig, axes = plt.subplots(
        1, len(WEIGHT_DECAYS), figsize=(4.2 * len(WEIGHT_DECAYS), 4.2), sharey=True
    )
    for ax, wd in zip(axes, WEIGHT_DECAYS, strict=True):
        # Single-hue sequential scale, darker = lower (better) val_loss, shared across panels.
        image = ax.imshow(grids[wd], cmap="Blues_r", vmin=vmin, vmax=vmax, aspect="auto")
        for i, m in enumerate(MUON_LRS):
            for j, a in enumerate(MUON_ADAM_LRS):
                key = ("muon", m, a, wd)
                if key not in cells or "val_loss" not in cells[key]:
                    continue
                mean, sd = cells[key]["val_loss"]
                dark = (mean - vmin) < 0.5 * (vmax - vmin)
                ax.text(
                    j,
                    i,
                    f"{mean:.3f}\n±{sd:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if dark else "black",
                )
                if key == best_key:
                    ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2, ec="black")
                    )
        ax.set_xticks(range(len(MUON_ADAM_LRS)), MUON_ADAM_LRS)
        ax.set_yticks(range(len(MUON_LRS)), MUON_LRS)
        ax.set_xlabel("Adam-group learning rate")
        ax.set_title(f"weight decay {wd}")
    axes[0].set_ylabel("Muon learning rate")
    fig.colorbar(image, ax=axes, shrink=0.85, label="val loss (mean over seeds)")
    save(fig, "muon_val_loss")
    plt.close(fig)


def best_over_rest(cells: dict, match) -> tuple[float, float]:
    """Best mean val_loss (and its s.d.) among the Muon cells that satisfy `match`."""
    candidates = [
        cells[k]["val_loss"]
        for k in cells
        if k[0] == "muon" and match(k) and "val_loss" in cells[k]
    ]
    return min(candidates) if candidates else (np.nan, np.nan)


def plot_marginals(cells: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    muon_axes = [
        ("Muon learning rate", MUON_LRS, 1),
        ("Adam-group learning rate", MUON_ADAM_LRS, 2),
        ("weight decay", WEIGHT_DECAYS, 3),
    ]
    for ax, (label, values, index) in zip(axes[:3], muon_axes, strict=True):
        points = [best_over_rest(cells, lambda k, v=v, i=index: k[i] == v) for v in values]
        ax.errorbar(
            range(len(values)),
            [p[0] for p in points],
            yerr=[p[1] for p in points],
            color=PAPER_COLORS[6],
            marker="o",
            markersize=6,
            lw=2,
            capsize=3,
        )
        ax.set_xticks(range(len(values)), values)
        ax.set_xlabel(label)
        ax.set_title("Muon: best over the other settings", fontsize=9)
    axes[0].set_ylabel("val loss (mean over seeds)")

    ax = axes[3]
    for wd in WEIGHT_DECAYS:
        points = [
            cells.get(("adamw", "", lr, wd), {}).get("val_loss", (np.nan, np.nan))
            for lr in ADAMW_LRS
        ]
        ax.errorbar(
            range(len(ADAMW_LRS)),
            [p[0] for p in points],
            yerr=[p[1] for p in points],
            color=WD_COLORS[wd],
            marker="o",
            markersize=6,
            lw=2,
            capsize=3,
            label=f"wd {wd}",
        )
    ax.set_xticks(range(len(ADAMW_LRS)), ADAMW_LRS)
    ax.set_xlabel("AdamW learning rate")
    ax.set_title("plain AdamW", fontsize=9)
    ax.legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", color="0.9", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "marginals")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    apply_latex_style()

    rows = collect(args.outputs_dir)
    print(f"{len(rows)} runs found; wrote {write_csv(rows)}")
    cells = summarise(rows)
    print_summary(cells, args.outputs_dir)
    plot_muon_heatmaps(cells)
    plot_marginals(cells)


if __name__ == "__main__":
    main()
