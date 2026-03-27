"""Visualise task-level 1D-ARC examples from a saved Hugging Face dataset."""

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# Standard ARC color palette (integers 0-9)
ARC_COLORS = {
    0: "#000000",  # Black
    1: "#0074D9",  # Blue
    2: "#FF4136",  # Red
    3: "#2ECC40",  # Green
    4: "#FFDC00",  # Yellow
    5: "#AAAAAA",  # Grey
    6: "#F012BE",  # Magenta
    7: "#FF851B",  # Orange
    8: "#7FDBFF",  # Cyan
    9: "#870C25",  # Maroon
}

ARC_CMAP = mcolors.ListedColormap([ARC_COLORS[i] for i in range(10)])
ARC_NORM = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, 10.5, 1), ncolors=10)
MASK_CELL_COLOR = "#F4F1EA"
MASK_EDGE_COLOR = "#B7B0A4"
PANEL_LABEL_PAD = 8
ARROW_COLUMN_WIDTH = 0.45
TASK_CATEGORY_DISPLAY_NAMES = {
    "1d_move_1p": "Move 1 Pixel",
    "1d_move_2p": "Move 2 Pixels",
    "1d_move_3p": "Move 3 Pixels",
    "1d_move_dp": "Move Dynamic",
    "1d_move_2p_dp": "Move 2 Pixels Towards",
    "1d_fill": "Fill",
    "1d_padded_fill": "Padded Fill",
    "1d_hollow": "Hollow",
    "1d_flip": "Flip",
    "1d_mirror": "Mirror",
    "1d_denoising_1c": "Denoise",
    "1d_denoising_mc": "Denoise Multicolor",
    "1d_pcopy_1c": "Pattern Copy",
    "1d_pcopy_mc": "Pattern Copy Multicolor",
    "1d_recolor_oe": "Recolor by Odd Even",
    "1d_recolor_cnt": "Recolor by Size",
    "1d_recolor_cmp": "Recolor by Size Comparison",
    "1d_scale_dp": "Scaling",
}


def format_task_category(category: str) -> str:
    normalized_category = category.removeprefix("dataset/")
    return TASK_CATEGORY_DISPLAY_NAMES.get(normalized_category, normalized_category)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/arc_1d",
        help="Directory containing the saved ARC task dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to inspect. Use 'all' to concatenate all available splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/task_visualisations"),
        help="Directory where rendered task images will be written.",
    )
    parser.add_argument(
        "--task-category",
        default=None,
        help="Optional task category filter.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Optional task id filter. If provided, a single matching task is rendered.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to render after filtering.",
    )
    parser.add_argument(
        "--first-per-category",
        action="store_true",
        help="Render only the first task encountered for each category.",
    )
    return parser.parse_args()


def select_dataset(data: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(data, Dataset):
        if split != "all":
            print("Loaded a single dataset; ignoring --split.")
        return data

    if split == "all":
        ordered_splits = [
            split_name
            for split_name in ("train", "dev", "test", "holdout_test")
            if split_name in data
        ]
        return concatenate_datasets([data[split_name] for split_name in ordered_splits])

    if split not in data:
        available_splits = ", ".join(data.keys())
        msg = f"Unknown split {split!r}. Available splits: {available_splits}"
        raise ValueError(msg)

    return data[split]


def load_task_data(data_dir: str | Path) -> Dataset | DatasetDict:
    data_path = Path(data_dir)

    # Prefer the explicit DatasetDict loader because the directory can also contain
    # legacy root-level dataset files that confuse the generic loader.
    if (data_path / "dataset_dict.json").exists():
        return DatasetDict.load_from_disk(str(data_path))

    return load_from_disk(str(data_path))


def is_task_dataset(ds: Dataset) -> bool:
    required_columns = {
        "task_category",
        "task_id",
        "support_inputs",
        "support_outputs",
        "query_input",
        "query_output",
    }
    return required_columns.issubset(ds.column_names)


def filter_tasks(
    ds: Dataset,
    task_category: str | None,
    task_id: int | None,
    first_per_category: bool,
    max_tasks: int | None,
) -> list[dict]:
    selected_tasks = []
    seen_categories = set()

    for task in ds:
        if task_category is not None and task["task_category"] != task_category:
            continue
        if task_id is not None and task["task_id"] != task_id:
            continue
        if first_per_category and task["task_category"] in seen_categories:
            continue

        selected_tasks.append(task)
        seen_categories.add(task["task_category"])

        if max_tasks is not None and len(selected_tasks) >= max_tasks:
            break

    return selected_tasks


def draw_sequence(ax: plt.Axes, sequence: list[int], label: str) -> None:
    grid = np.array(sequence).reshape(1, -1)
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, aspect="equal")
    ax.set_xticks(np.arange(-0.5, len(sequence), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(
        label,
        fontsize=14,
        fontweight="bold",
        pad=PANEL_LABEL_PAD,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_masked_sequence(ax: plt.Axes, sequence_length: int, label: str) -> None:
    ax.set_xlim(-0.5, sequence_length - 0.5)
    ax.set_ylim(0.5, -0.5)
    ax.set_aspect("equal")

    for index in range(sequence_length):
        ax.add_patch(
            Rectangle(
                (index - 0.5, -0.5),
                1,
                1,
                facecolor=MASK_CELL_COLOR,
                edgecolor=MASK_EDGE_COLOR,
                linewidth=1.5,
            )
        )
        ax.text(index, 0, "?", ha="center", va="center", fontsize=16, fontweight="bold")

    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(
        label,
        fontsize=14,
        fontweight="bold",
        pad=PANEL_LABEL_PAD,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_io_arrow(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.annotate(
        "",
        xy=(0.88, 0.5),
        xytext=(0.12, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "mutation_scale": 18,
            "linewidth": 2.0,
            "color": "#333333",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def visualise_task(task: dict, save_path: Path) -> None:
    support_inputs = task["support_inputs"]
    support_outputs = task["support_outputs"]
    query_input = task["query_input"]
    sequence_length = len(query_input)

    num_support_rows = len(support_inputs)
    grid_rows = num_support_rows + 2
    panel_width = max(sequence_length * 0.6, 4.5)
    fig = plt.figure(
        figsize=(
            panel_width * 2 + ARROW_COLUMN_WIDTH,
            max(num_support_rows * 1.2 + 2.0, 5.1),
        )
    )
    grid = fig.add_gridspec(
        grid_rows,
        3,
        width_ratios=[panel_width, ARROW_COLUMN_WIDTH, panel_width],
        height_ratios=[1] * num_support_rows + [0.16, 1],
        hspace=0.0,
        wspace=0.02,
    )

    axes = np.empty((num_support_rows + 1, 2), dtype=object)

    for row_index, (support_input, support_output) in enumerate(
        zip(support_inputs, support_outputs, strict=True)
    ):
        axes[row_index, 0] = fig.add_subplot(grid[row_index, 0])
        arrow_ax = fig.add_subplot(grid[row_index, 1])
        axes[row_index, 1] = fig.add_subplot(grid[row_index, 2])
        draw_sequence(axes[row_index, 0], support_input, f"S{row_index + 1} In")
        draw_sequence(axes[row_index, 1], support_output, f"S{row_index + 1} Out")
        draw_io_arrow(arrow_ax)

    axes[-1, 0] = fig.add_subplot(grid[-1, 0])
    arrow_ax = fig.add_subplot(grid[-1, 1])
    axes[-1, 1] = fig.add_subplot(grid[-1, 2])
    draw_sequence(axes[-1, 0], query_input, "Query In")
    draw_masked_sequence(axes[-1, 1], sequence_length, "Query Out")
    draw_io_arrow(arrow_ax)

    task_title = format_task_category(task["task_category"])
    fig.suptitle(task_title, fontsize=22, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, top=0.88)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_output_path(output_dir: Path, task: dict) -> Path:
    category = task["task_category"]
    task_id = task["task_id"]
    return output_dir / f"arc_1d_task_{category}_{task_id}.png"


def main() -> None:
    args = parse_args()
    data = load_task_data(args.data_dir)
    ds = select_dataset(data, args.split)

    if not is_task_dataset(ds):
        msg = "This script expects the task-level dataset produced by build_arc_1d.py."
        raise ValueError(msg)

    selected_tasks = filter_tasks(
        ds,
        task_category=args.task_category,
        task_id=args.task_id,
        first_per_category=args.first_per_category,
        max_tasks=args.max_tasks,
    )
    if not selected_tasks:
        msg = "No tasks matched the requested filters."
        raise ValueError(msg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in selected_tasks:
        output_path = build_output_path(args.output_dir, task)
        visualise_task(task, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
