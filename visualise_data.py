import numpy as np
from datasets import load_from_disk
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

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


def visualise_arc_1d_example(
    input_seq: list[int],
    output_seq: list[int],
    title: str = "",
    save_path: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Visualise a single ARC 1D example as colored cells (input above, output below)."""
    max_len = max(len(input_seq), len(output_seq))

    fig, axes = plt.subplots(2, 1, figsize=(max(max_len * 0.6, 3), 2.4))

    for ax, seq, label in zip(axes, [input_seq, output_seq], ["Input", "Output"], strict=True):
        grid = np.array(seq).reshape(1, -1)
        ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, aspect="equal")

        # Grid lines
        ax.set_xticks(np.arange(-0.5, len(seq), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 1, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_ylabel(label, fontsize=16, rotation=0, labelpad=55, va="center")

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig, axes


def main():
    ds = load_from_disk("data/arc_1d")
    example = ds[100]
    title = f"{example['task_category']} (task {example['task_id']}, {example['split']})"
    visualise_arc_1d_example(
        example["input"],
        example["output"],
        title=title,
        save_path="data/arc_1d_example.png",
    )
    print("Saved visualisation to data/arc_1d_example.png")


if __name__ == "__main__":
    main()
