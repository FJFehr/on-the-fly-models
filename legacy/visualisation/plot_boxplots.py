"""Plot grouped 1D-ARC sequence-length boxplots by category and split."""

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from visualisation.style import FONT_SIZES, apply_latex_style, format_task_category

DEFAULT_SPLITS = ("train", "dev", "test")
SPLIT_SHADES = {
    "train": 0.55,
    "dev": 0.25,
    "test": -0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/arc_1d"),
        help="Directory containing the saved ARC task dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/task_visualisations/arc_1d_length_boxplots_by_split.png"),
        help="Path where the grouped boxplot figure will be saved.",
    )
    return parser.parse_args()


def load_task_data(data_dir: Path) -> Dataset | DatasetDict:
    if (data_dir / "dataset_dict.json").exists():
        return DatasetDict.load_from_disk(str(data_dir))
    return load_from_disk(str(data_dir))


def validate_task_dataset(data: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(data, Dataset):
        msg = "Expected a DatasetDict with train/dev/test splits."
        raise ValueError(msg)

    required_columns = {"task_category", "sequence_length"}
    for split_name in DEFAULT_SPLITS:
        if split_name not in data:
            available = ", ".join(data.keys())
            msg = f"Missing required split {split_name!r}. Available: {available}"
            raise ValueError(msg)
        if not required_columns.issubset(data[split_name].column_names):
            msg = f"Split {split_name!r} is missing required task columns: {required_columns}"
            raise ValueError(msg)

    return data


def blend_color(color: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    target = np.array([1.0, 1.0, 1.0]) if amount >= 0 else np.array([0.0, 0.0, 0.0])
    base = np.array(color)
    return tuple(base + (target - base) * abs(amount))


def category_base_colors(categories: list[str]) -> dict[str, tuple[float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return {category: cmap(index % cmap.N)[:3] for index, category in enumerate(categories)}


def build_length_lookup(data: DatasetDict) -> dict[str, dict[str, np.ndarray]]:
    categories = sorted(
        {category for split in DEFAULT_SPLITS for category in data[split]["task_category"]}
    )
    lengths_by_split = {
        split_name: {
            category: np.array(
                [
                    example["sequence_length"]
                    for example in data[split_name]
                    if example["task_category"] == category
                ]
            )
            for category in categories
        }
        for split_name in DEFAULT_SPLITS
    }
    return lengths_by_split


def plot_grouped_boxplots(data: DatasetDict, output_path: Path) -> None:
    lengths_by_split = build_length_lookup(data)
    categories = sorted(lengths_by_split[DEFAULT_SPLITS[0]])
    base_colors = category_base_colors(categories)

    fig_width = max(14, len(categories) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 8.0))

    group_centers = np.arange(len(categories)) * 1.35
    offsets = {"train": -0.28, "dev": 0.0, "test": 0.28}
    width = 0.22
    grand_total_tokens = 0

    for split_name in DEFAULT_SPLITS:
        data_series = []
        positions = []
        face_colors = []
        for index, category in enumerate(categories):
            lengths = lengths_by_split[split_name][category]
            if len(lengths) == 0:
                continue
            data_series.append(lengths)
            positions.append(group_centers[index] + offsets[split_name])
            face_colors.append(blend_color(base_colors[category], SPLIT_SHADES[split_name]))
            grand_total_tokens += lengths.sum() * 2

        boxplot = ax.boxplot(
            data_series,
            positions=positions,
            widths=width,
            patch_artist=True,
            manage_ticks=False,
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#444444", "linewidth": 1.2},
            capprops={"color": "#444444", "linewidth": 1.2},
            boxprops={"edgecolor": "#333333", "linewidth": 1.1},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#222222",
                "markeredgecolor": "#222222",
                "markersize": 3,
                "alpha": 0.6,
            },
        )

        for patch, color in zip(boxplot["boxes"], face_colors, strict=True):
            patch.set_facecolor(color)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [format_task_category(category) for category in categories], rotation=55, ha="right"
    )
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"])
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])
    ax.set_ylabel("Sequence length", fontsize=FONT_SIZES["label"])
    ax.set_title(
        "1D-ARC sequence-length distributions by category and split",
        fontsize=FONT_SIZES["title"],
        pad=34,
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.40)
    for split_name in DEFAULT_SPLITS:
        for index, category in enumerate(categories):
            lengths = lengths_by_split[split_name][category]
            if len(lengths) == 0:
                continue
            ax.text(
                group_centers[index] + offsets[split_name],
                ymax * 1.02,
                f"{lengths.sum() * 2 / 1000:.1f}K",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZES["annotation"],
                rotation=90,
            )

    ax.text(
        0.99,
        0.97,
        f"Total: {grand_total_tokens / 1000:.1f}K tokens",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SIZES["label"],
    )

    legend_handles = [
        Patch(facecolor="#E0E0E0", edgecolor="#333333", label="train"),
        Patch(facecolor="#9C9C9C", edgecolor="#333333", label="dev"),
        Patch(facecolor="#555555", edgecolor="#333333", label="test"),
    ]
    ax.legend(
        handles=legend_handles,
        title="Split shade",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZES["legend"],
        title_fontsize=FONT_SIZES["legend"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_latex_style()
    args = parse_args()
    data = validate_task_dataset(load_task_data(args.data_dir))
    plot_grouped_boxplots(data, args.output_path)
    print(f"Saved grouped boxplot to {args.output_path}")


if __name__ == "__main__":
    main()
