"""Shared ARC visualisation helpers for scripts and W&B logging."""

import matplotlib
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

import wandb

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from visualisation.style import FONT_SIZES, format_task_category

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

def draw_sequence(ax: plt.Axes, sequence: list[int], label: str) -> None:
    grid = np.array(sequence).reshape(1, -1)
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, aspect="equal")
    ax.set_xticks(np.arange(-0.5, len(sequence), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(label, fontsize=FONT_SIZES["panel_label"], fontweight="bold", pad=PANEL_LABEL_PAD)
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
        ax.text(
            index, 0, "?",
            ha="center", va="center", fontsize=FONT_SIZES["title"], fontweight="bold",
        )

    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(label, fontsize=FONT_SIZES["panel_label"], fontweight="bold", pad=PANEL_LABEL_PAD)
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


def render_task_figure(task: dict) -> plt.Figure:
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
    fig.suptitle(task_title, fontsize=FONT_SIZES["title"], fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, top=0.88)
    return fig


def render_val_example_figure(
    input_sequence: list[int],
    target_sequence: list[int],
    prediction_sequence: list[int],
    title: str | None = None,
) -> plt.Figure:
    sequence_length = max(
        len(input_sequence),
        len(target_sequence),
        len(prediction_sequence),
    )
    panel_width = max(sequence_length * 0.6, 4.5)
    fig, axes = plt.subplots(3, 1, figsize=(panel_width, 5.8))

    draw_sequence(axes[0], input_sequence, "Input")
    draw_sequence(axes[1], target_sequence, "Target")
    draw_sequence(axes[2], prediction_sequence, "Prediction")

    if title:
        fig.suptitle(title, fontsize=FONT_SIZES["title"], fontweight="bold", y=0.98)
        fig.subplots_adjust(top=0.86, hspace=0.65)
    else:
        fig.subplots_adjust(top=0.92, hspace=0.65)
    return fig


def render_task_prediction_figure(
    support_inputs: list[list[int]],
    support_targets: list[list[int]],
    support_predictions: list[list[int]],
    query_input: list[int],
    query_target: list[int],
    query_prediction: list[int],
    task_category: str,
    task_id: int,
    query_exact_match: bool | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Render all task examples as 4 columns with input/target/prediction stacked."""
    example_columns = [
        ("Support 1", support_inputs[0], support_targets[0], support_predictions[0]),
        ("Support 2", support_inputs[1], support_targets[1], support_predictions[1]),
        ("Support 3", support_inputs[2], support_targets[2], support_predictions[2]),
        ("Query", query_input, query_target, query_prediction),
    ]
    sequence_length = max(
        max(len(input_sequence), len(target_sequence), len(prediction_sequence))
        for _, input_sequence, target_sequence, prediction_sequence in example_columns
    )
    panel_width = max(sequence_length * 0.55, 4.0)
    fig, axes = plt.subplots(3, 4, figsize=(panel_width * 4.2, 6.4))

    for column_index, (
        column_label,
        input_sequence,
        target_sequence,
        prediction_sequence,
    ) in enumerate(
        example_columns
    ):
        draw_sequence(axes[0, column_index], input_sequence, f"{column_label} Input")
        draw_sequence(axes[1, column_index], target_sequence, f"{column_label} Target")
        draw_sequence(axes[2, column_index], prediction_sequence, f"{column_label} Prediction")

    if title is None:
        title = f"{format_task_category(task_category)} | task_id={task_id}"
        if query_exact_match is not None:
            title += f" | query_exact_match={query_exact_match}"

    fig.suptitle(title, fontsize=FONT_SIZES["title"], fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.88, hspace=0.65, wspace=0.15)
    return fig


def figure_to_wandb_image(fig: plt.Figure, caption: str | None = None) -> wandb.Image:
    image = wandb.Image(fig, caption=caption)
    plt.close(fig)
    return image
