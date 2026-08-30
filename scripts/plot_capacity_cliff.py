"""Plot the capacity-cliff figure for Experiment 2 (multi-task capacity).

Compares three training regimes across model size (parameter count):

  1. Individual   -- one model per task, no cross-task sharing
                     (arc1d_v2_minimal_size for dim 4/6, arc1d_v2_backbone_capacity
                     RC1 for dim 10 -- the original Phase 1 run)
  2. Joint, no ID -- one model trained jointly across all 14 task categories,
                     no task-identity signal (arc1d_v2_multitask notd*)
  3. Joint + ID   -- same joint setup, plus a per-task identity embedding
                     (arc1d_v2_multitask td*)

Reads test-split exact match from results.txt files under --outputs-dir.
See docs/arc1d_story/06_phase1_findings.md for the underlying experiments.

Usage
-----
    uv run python scripts/plot_capacity_cliff.py \\
        --outputs-dir outputs \\
        --out-dir configs/experiments/arc1d_v2_multitask
"""

import argparse
import re
import statistics
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator

from visualisation.style import FONT_SIZES, apply_latex_style

# Same architecture at three widths; embedding_dim fixed at 10 throughout.
# Total parameter counts measured directly (embedder + backbone + head; +180
# for the task-embedding table in the "Joint + ID" arms, not shown on the
# x-axis since the swept quantity is backbone width, not that fixed addition).
DIM_PARAMS = {"4": 1398, "6": 2444, "10": 5400}

COLORS = {
    "individual": "#2ECC71",  # green
    "notd": "#B276B2",  # light purple -- joint, no task ID
    "td": "#5B2C82",  # dark purple  -- joint, with task ID
}
LINESTYLES = {"individual": ":", "notd": "-", "td": "-"}
LABELS = {
    "individual": "Individual (one model per task)",
    "notd": "Joint, no task ID",
    "td": "Joint + task-ID embedding",
}


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


def load_individual(outputs_dir: Path) -> dict[str, dict[int, list[float]]]:
    """dim -> seed -> [test_query_exact_match per task]."""
    by_dim: dict[str, dict[int, list[float]]] = {"4": {}, "6": {}, "10": {}}

    minsize_re = re.compile(r"^v2_minsize_dim(\d+)_.+_seed(\d+)$")
    for path in sorted((outputs_dir / "arc1d_v2_minimal_size").glob("*/results.txt")):
        m = minsize_re.match(path.parent.name)
        if not m:
            continue
        dim, seed = m.group(1), int(m.group(2))
        score = parse_test_exact_match(path)
        if score is not None:
            by_dim[dim].setdefault(seed, []).append(score)

    rc1_re = re.compile(r"^v2_RC1_dim10_.+_rope_canon_n1_seed(\d+)$")
    for path in sorted((outputs_dir / "arc1d_v2_backbone_capacity").glob("*/results.txt")):
        m = rc1_re.match(path.parent.name)
        if not m:
            continue
        seed = int(m.group(1))
        score = parse_test_exact_match(path)
        if score is not None:
            by_dim["10"].setdefault(seed, []).append(score)

    return by_dim


def load_joint(outputs_dir: Path) -> dict[str, dict[str, dict[int, float]]]:
    """condition (notd/td) -> dim -> seed -> test_query_exact_match."""
    by_cond: dict[str, dict[str, dict[int, float]]] = {"notd": {}, "td": {}, }
    for cond in by_cond:
        by_cond[cond] = {"4": {}, "6": {}, "10": {}}

    pattern = re.compile(r"^v2_multitask_(notd|td)(?:_dim(\d+))?_seed(\d+)$")
    for path in sorted(outputs_dir.glob("arc1d_v2_multitask/*/results.txt")):
        m = pattern.match(path.parent.name)
        if not m:
            continue
        cond, dim, seed = m.group(1), m.group(2) or "10", int(m.group(3))
        score = parse_test_exact_match(path)
        if score is not None:
            by_cond[cond][dim][seed] = score

    return by_cond


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "n": len(values),
        "min": min(values),
        "max": max(values),
    }


def build_series(
    individual_raw: dict[str, dict[int, list[float]]],
    joint_raw: dict[str, dict[str, dict[int, float]]],
) -> dict[str, dict[str, dict]]:
    """condition -> dim -> summary stats, ready for plotting."""
    series: dict[str, dict[str, dict]] = {"individual": {}, "notd": {}, "td": {}}

    for dim, per_seed in individual_raw.items():
        per_seed_means = [sum(v) / len(v) for v in per_seed.values()]
        if per_seed_means:
            series["individual"][dim] = summarize(per_seed_means)

    for cond in ("notd", "td"):
        for dim, per_seed in joint_raw[cond].items():
            values = list(per_seed.values())
            if values:
                series[cond][dim] = summarize(values)

    return series


def plot(series: dict[str, dict[str, dict]], out_path: Path) -> None:
    apply_latex_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    dims_sorted = sorted(DIM_PARAMS, key=lambda d: DIM_PARAMS[d])
    x_all = [DIM_PARAMS[d] for d in dims_sorted]

    for cond in ("individual", "notd", "td"):
        dims_present = [d for d in dims_sorted if d in series[cond]]
        if len(dims_present) < 2:
            continue
        x = np.array([DIM_PARAMS[d] for d in dims_present], dtype=float)
        means = np.array([series[cond][d]["mean"] for d in dims_present])
        stds = np.array([series[cond][d]["std"] for d in dims_present])

        # Smooth monotone-preserving spline through the mean points (log-x).
        log_x = np.log10(x)
        spline = PchipInterpolator(log_x, means)
        log_x_fine = np.linspace(log_x.min(), log_x.max(), 200)
        x_fine = 10**log_x_fine

        ax.plot(
            x_fine,
            spline(log_x_fine),
            color=COLORS[cond],
            linestyle=LINESTYLES[cond],
            linewidth=2.2,
            zorder=3 if cond != "notd" else 2,
        )
        ax.errorbar(
            x,
            means,
            yerr=stds,
            fmt="o",
            color=COLORS[cond],
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=4,
            elinewidth=1.4,
            zorder=4,
            label=LABELS[cond],
        )

    # Annotate the gap at dim=6 -- the clearest single-size demonstration.
    if "6" in series["individual"] and "6" in series["td"]:
        x6 = DIM_PARAMS["6"]
        y_top = series["individual"]["6"]["mean"]
        y_bot = series["td"]["6"]["mean"]
        ax.annotate(
            "",
            xy=(x6 * 1.05, y_bot),
            xytext=(x6 * 1.05, y_top),
            arrowprops={"arrowstyle": "<->", "color": "0.35", "linewidth": 1.0},
        )
        ax.text(
            x6 * 1.1,
            (y_top + y_bot) / 2,
            f"{(y_top - y_bot) * 100:.0f} pt gap\nat {x6:,} params",
            fontsize=FONT_SIZES["annotation"],
            va="center",
            color="0.25",
        )

    ax.set_xscale("log")
    ax.set_xticks(x_all)
    ax.set_xticklabels([f"{DIM_PARAMS[d]:,}" for d in dims_sorted])
    ax.set_xlabel("Model size (parameters, log scale)")
    ax.set_ylabel("Test exact match")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=FONT_SIZES["legend"])
    ax.set_title(
        "Individually solvable, but not jointly -- even with task identity",
        fontsize=FONT_SIZES["title"],
    )

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("configs/experiments/arc1d_v2_multitask")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    individual_raw = load_individual(args.outputs_dir)
    joint_raw = load_joint(args.outputs_dir)
    series = build_series(individual_raw, joint_raw)

    for cond in ("individual", "notd", "td"):
        for dim in ("4", "6", "10"):
            s = series[cond].get(dim)
            if s:
                print(
                    f"{cond:10s} dim={dim:>2s} ({DIM_PARAMS[dim]:>5,} params): "
                    f"mean={s['mean']:.3f} std={s['std']:.3f} n={s['n']}"
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot(series, args.out_dir / "capacity_cliff.png")


if __name__ == "__main__":
    main()
