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


def analyse_lengths(ds):
    """Analyse and report the distribution of sequence lengths in the dataset."""
    input_lens = np.array([len(ex["input"]) for ex in ds])
    output_lens = np.array([len(ex["output"]) for ex in ds])

    # Check input vs output length agreement
    mismatches = np.sum(input_lens != output_lens)
    if mismatches == 0:
        print("All input and output sequences have the same length within each example.")
    else:
        print(f"WARNING: {mismatches} examples have different input/output lengths.")

    # Check if all problems are the same size
    unique_lens = np.unique(input_lens)
    if len(unique_lens) == 1:
        print(f"\nAll {len(ds)} examples have the same length: {unique_lens[0]}")
        return

    print(f"\nSequence lengths are NOT uniform — {len(unique_lens)} distinct lengths found.")
    print(f"  Min: {input_lens.min()}")
    print(f"  Max: {input_lens.max()}")
    print(f"  Mean: {input_lens.mean():.1f}")
    print(f"  Median: {np.median(input_lens):.1f}")
    print(f"  Std: {input_lens.std():.1f}")

    # Per-category breakdown
    categories = sorted(set(ds["task_category"]))
    print(f"\nPer-category breakdown ({len(categories)} categories):")
    hdr = f"  {'Category':<25s} {'Min':>5s} {'Max':>5s} {'Mean':>6s} {'N':>5s}"
    print(hdr)
    print(f"  {'-' * 25} {'-' * 5} {'-' * 5} {'-' * 6} {'-' * 5}")
    for cat in categories:
        mask = np.array([c == cat for c in ds["task_category"]])
        cat_lens = input_lens[mask]
        print(
            f"  {cat:<25s} {cat_lens.min():>5d} {cat_lens.max():>5d}"
            f" {cat_lens.mean():>6.1f} {len(cat_lens):>5d}"
        )

    # Quartiles
    q25, q50, q75 = np.percentile(input_lens, [25, 50, 75])
    print(f"\n  Quartiles: Q1={q25:.0f}, Median={q50:.0f}, Q3={q75:.0f}")


def plot_category_length_stats(ds, save_path: str = "data/arc_1d_category_lengths.png"):
    """Box plots of per-category sequence length distributions."""
    input_lens = np.array([len(ex["input"]) for ex in ds])
    categories = sorted(set(ds["task_category"]))

    data = []
    labels = []
    for cat in categories:
        mask = np.array([c == cat for c in ds["task_category"]])
        data.append(input_lens[mask])
        labels.append(cat.replace("1d_", ""))

    cat_colors = list(ARC_COLORS.values()) + list(ARC_COLORS.values())

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], cat_colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Sequence length", fontsize=14)
    ax.set_title("Per-category sequence length distributions", fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Add headroom for labels and place per-category totals above each box
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.25)
    for i, cat_lens in enumerate(data):
        cat_total = cat_lens.sum() * 2  # input + output
        ax.text(
            i + 1, ymax * 1.02, f"{cat_total / 1000:.1f}K",
            ha="center", va="bottom", fontsize=10, rotation=90,
        )

    # Grand total in the top right
    grand_total = input_lens.sum() * 2
    ax.text(
        0.99,
        0.97,
        f"Total: {grand_total / 1000:.1f}K tokens",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_length_distribution(ds, save_path: str = "data/arc_1d_length_distribution.png"):
    """Plot a histogram of sequence lengths with quartile lines."""
    input_lens = np.array([len(ex["input"]) for ex in ds])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        input_lens,
        bins=range(input_lens.min(), input_lens.max() + 2),
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of 1D-ARC sequence lengths")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Distribution saved to {save_path}")


# One representative example index per task category, each with a distinct
# dominant colour where possible (short examples preferred).
CATEGORY_EXAMPLES = {
    "1d_denoising_1c": 0,  # yellow
    "1d_denoising_mc": 201,  # red
    "1d_fill": 472,  # orange
    "1d_flip": 792,  # green
    "1d_hollow": 873,  # magenta
    "1d_mirror": 1113,  # grey
    "1d_move_1p": 1358,  # cyan
    "1d_move_2p": 1480,  # blue
    "1d_move_2p_dp": 1756,  # orange
    "1d_move_3p": 1956,  # orange
    "1d_move_dp": 2156,  # orange
    "1d_padded_fill": 2272,  # orange
    "1d_pcopy_1c": 2404,  # maroon
    "1d_pcopy_mc": 2600,  # blue
    "1d_recolor_cmp": 2816,  # blue
    "1d_recolor_cnt": 3192,  # grey
    "1d_recolor_oe": 3268,  # blue
    "1d_scale_dp": 3556,  # orange
}


def main():
    ds = load_from_disk("data/arc_1d")

    analyse_lengths(ds)
    plot_length_distribution(ds)
    plot_category_length_stats(ds)

    # Visualise one example per category (no titles)
    for cat, idx in CATEGORY_EXAMPLES.items():
        example = ds[idx]
        visualise_arc_1d_example(
            example["input"],
            example["output"],
            title=cat,
            save_path=f"data/arc_1d_{cat}.png",
        )
        print(f"  {cat} (index {idx}) -> data/arc_1d_{cat}.png")


if __name__ == "__main__":
    main()
