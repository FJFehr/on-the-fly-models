"""Plot analysis for arc1d_capacity experiments.

Reads results.txt files from an experiment output directory and produces two plots:

  1. Bar chart  — how many tasks each model type solved (val_query_exact_match == 1.0)
  2. Heatmap    — val_query_exact_match per (task, model) cell

Usage
-----
    uv run python visualisation/plot_capacity.py \\
        --output-dir outputs/arc1d_capacity_binary_baseline_experiments
"""

import argparse
import math
import re
from contextlib import suppress
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from visualisation.core.style import (
    FONT_SIZES,
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    apply_latex_style,
    format_task_category,
    normalize_task_category,
)

KNOWN_MODELS = ["rnn", "cnn", "transformer", "mlp"]
SOLVED_THRESHOLD = 1.0
SEED_SUFFIX_RE = re.compile(r"^(?P<base>.+)_seed_(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory containing per-experiment subdirectories with results.txt files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_results_file(path: Path) -> dict[str, float]:
    """Return the validation metrics dict from a results.txt file."""
    metrics: dict[str, float] = {}
    in_val = False
    for line in path.read_text().splitlines():
        if line.strip() == "validation_metrics:":
            in_val = True
            continue
        if line.strip() == "test_metrics:":
            break
        if in_val and ":" in line:
            key, _, value = line.partition(":")
            with suppress(ValueError):
                metrics[key.strip()] = float(value.strip())
    return metrics


def infer_model(experiment_name: str) -> str | None:
    """Extract model name from experiment folder name (suffix match)."""
    for model in KNOWN_MODELS:
        if experiment_name.endswith(f"_{model}"):
            return model
    return None


def split_seed_suffix(experiment_name: str) -> tuple[str, int | None]:
    """Strip a trailing `_seed_<n>` suffix when present."""
    match = SEED_SUFFIX_RE.match(experiment_name)
    if match is None:
        return experiment_name, None
    return match.group("base"), int(match.group("seed"))


def infer_task(experiment_name: str, model: str) -> str:
    """Strip the model suffix and normalize the task category key."""
    task = experiment_name[: -(len(model) + 1)]
    return normalize_task_category(task)


def load_results(output_dir: Path) -> list[dict]:
    """Walk output_dir for results.txt files and return parsed records."""
    records = []
    for results_path in sorted(output_dir.glob("*/results.txt")):
        base_name, seed = split_seed_suffix(results_path.parent.name)
        model = infer_model(base_name)
        if model is None:
            continue
        task = infer_task(base_name, model)
        metrics = parse_results_file(results_path)
        val_score = metrics.get("val_query_exact_match")
        if val_score is None:
            continue
        records.append(
            {
                "task": task,
                "model": model,
                "seed": seed,
                "val_query_exact_match": val_score,
            }
        )
    return records


def aggregate_seed_metrics(records: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    """Aggregate seeded records by canonical `(task, model)` pair."""
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        key = (record["task"], record["model"])
        grouped.setdefault(key, []).append(record["val_query_exact_match"])

    aggregated = {}
    for key, values in grouped.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        aggregated[key] = {
            "max": max(values),
            "mean": mean,
            "std": variance**0.5,
            "count": float(len(values)),
        }
    return aggregated


def warn_on_incomplete_seed_coverage(aggregated: dict[tuple[str, str], dict[str, float]]) -> None:
    """Print a warning when `(task, model)` cells have different seed counts."""
    if not aggregated:
        return

    counts = {int(stats["count"]) for stats in aggregated.values()}
    if len(counts) == 1:
        print(f"Detected {counts.pop()} seed(s) for every task-model cell.")
        return

    print(
        "Warning: inconsistent seed coverage across task-model cells "
        f"(min={min(counts)}, max={max(counts)})."
    )


# ---------------------------------------------------------------------------
# Plot 1: bar chart of solved tasks per model
# ---------------------------------------------------------------------------


def plot_solved_bar(records: list[dict], output_path: Path) -> None:
    aggregated = aggregate_seed_metrics(records)
    models_present = sorted(
        {model for _, model in aggregated},
        key=lambda m: KNOWN_MODELS.index(m) if m in KNOWN_MODELS else len(KNOWN_MODELS),
    )
    solved_counts = {
        model: sum(
            1
            for (task, record_model), stats in aggregated.items()
            if record_model == model and stats["max"] >= SOLVED_THRESHOLD
        )
        for model in models_present
    }
    total_tasks = len({task for task, _ in aggregated})

    fig, ax = plt.subplots(figsize=(max(5, len(models_present) * 1.5), 5))

    x = range(len(models_present))
    bars = ax.bar(
        x,
        [solved_counts[m] for m in models_present],
        width=0.5,
        color=[MODEL_COLORS.get(m, "#4C72B0") for m in models_present],
        edgecolor="#333333",
    )

    for bar, model in zip(bars, models_present, strict=True):
        count = solved_counts[model]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(count),
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["annotation"],
            fontweight="bold",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models_present], fontsize=FONT_SIZES["tick"]
    )
    ax.set_yticks(range(total_tasks + 1))
    ax.set_ylim(0, total_tasks + 0.8)
    ax.set_ylabel("Tasks solved (val exact match = 1.0)", fontsize=FONT_SIZES["label"])
    ax.set_title("Tasks solved per model (best seed)", fontsize=FONT_SIZES["title"], pad=12)
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved bar chart to {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: heatmap of val_query_exact_match per (task, model)
# ---------------------------------------------------------------------------


def plot_heatmap(records: list[dict], output_path: Path) -> None:
    aggregated = aggregate_seed_metrics(records)
    tasks = sorted({task for task, _ in aggregated})
    models = sorted(
        {model for _, model in aggregated},
        key=lambda m: KNOWN_MODELS.index(m) if m in KNOWN_MODELS else len(KNOWN_MODELS),
    )

    matrix = [  # noqa: F841
        [aggregated.get((task, model), {}).get("mean", math.nan) for model in models]
        for task in tasks
    ]

    fig, ax = plt.subplots(figsize=(max(4, len(tasks) * 0.9), max(3, len(models) * 1.4)))

    cmap = LinearSegmentedColormap.from_list(
        "brown_to_blue", [MODEL_COLORS["cnn"], MODEL_COLORS["rnn"]]
    )
    norm = Normalize(vmin=0.0, vmax=1.0)

    for row_idx, model in enumerate(models):
        for col_idx, task in enumerate(tasks):
            stats = aggregated.get((task, model))
            value = math.nan if stats is None else stats["mean"]
            if math.isnan(value):
                color = "#cccccc"
                label = "n/a"
            else:
                color = cmap(norm(value))
                label = f"{stats['mean']:.2f}±{stats['std']:.2f}"
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, color=color))
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=FONT_SIZES["annotation"],
                color="black",
                fontweight="bold",
            )

    ax.set_xlim(0, len(tasks))
    ax.set_ylim(0, len(models))
    ax.set_xticks([i + 0.5 for i in range(len(tasks))])
    ax.set_xticklabels(
        [format_task_category(t) for t in tasks],
        rotation=55,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"])
    ax.set_yticks([i + 0.5 for i in range(len(models))])
    ax.set_yticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models], fontsize=FONT_SIZES["tick"]
    )
    ax.set_title(
        "Validation exact match by task and model (mean across seeds)",
        fontsize=FONT_SIZES["title"],
        pad=12,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("mean val exact match", fontsize=FONT_SIZES["label"])
    cbar.ax.tick_params(labelsize=FONT_SIZES["tick"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    apply_latex_style()
    args = parse_args()

    if not args.output_dir.is_dir():
        raise SystemExit(f"Output directory not found: {args.output_dir}")

    records = load_results(args.output_dir)
    if not records:
        raise SystemExit(f"No results.txt files found under {args.output_dir}")

    print(f"Loaded {len(records)} experiment results from {args.output_dir}")
    warn_on_incomplete_seed_coverage(aggregate_seed_metrics(records))

    plots_dir = args.output_dir / "plots"
    plot_solved_bar(records, plots_dir / "solved_per_model.png")
    plot_heatmap(records, plots_dir / "heatmap_task_model.png")


if __name__ == "__main__":
    main()
