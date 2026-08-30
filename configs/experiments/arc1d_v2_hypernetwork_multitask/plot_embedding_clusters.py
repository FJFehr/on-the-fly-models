"""Themed, individual cluster-map plots for the joint hypernetwork's pooled
task latent (Experiment 4).

Reads the raw pooled-embedding vectors dumped by scripts/dump_embedding_clusters.py
(one .npz per condition, e.g. embeddings_dim6_notd.npz) and renders each
available projection (PCA / t-SNE / UMAP) as its own standalone figure,
styled uniform with the rest of the paper's figures:

- Nimbus Roman, larger labels, no title, top/right spines stripped
  (matching plot_capacity_cliff.py / plot_per_task.py)
- category colours drawn from visualisation.arc_paper's PAPER_COLORS /
  PAPER_FILL_COLORS -- the same muted rainbow used in the task-example
  figures (e.g. outputs/visualisations/arc_1d_task_1d_flip_46.pdf), not a
  generic matplotlib colormap
- category labels run through arc_paper.shorten_task_label (drop
  "Pixel(s)", "Multicolor" -> "MC"), same as plot_per_task.py's x-axis
- legend as a large, wrapped block along the bottom

Reuses visualisation.embedding_clusters's projection-computation and
scatter-drawing logic (_compute_projection_panels / _draw_scatter_panel)
rather than reimplementing it -- this script only supplies the colours,
labels, and chrome that need to differ from that module's own
render_single_projection_figures/render_embedding_cluster_figure, which are
still what training's own end-of-run logging uses unchanged.

Why a checkpoint dump instead of just re-plotting from the original PNG:
save_checkpoints was false for the original run, so the raw vectors that
produced cluster_dim6_notd.png/cluster_dim6_frozentd.png were never saved --
only rendered once into that PNG. dim6_notd and dim6_frozentd were rerun
once with save_checkpoints=true specifically so this script (and any future
restyle) never needs to retrain again -- rerun it any time with no cluster
access needed, ~15s locally.

Usage
-----
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_embedding_clusters.py
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_embedding_clusters.py \\
        --projections t-SNE   # only render the ones actually used in the paper
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from visualisation.arc_paper import PAPER_COLORS, PAPER_FILL_COLORS, shorten_task_label
from visualisation.embedding_clusters import _compute_projection_panels, _draw_scatter_panel
from visualisation.style import apply_latex_style, format_task_category

HERE = Path(__file__).parent
CONDITIONS = {
    "notd": "embeddings_dim6_notd.npz",
    "frozentd": "embeddings_dim6_frozentd.npz",
}

# PAPER_COLORS/PAPER_FILL_COLORS are keyed 0-9 for ARC cell values (0 = null/
# background, not a real category colour) -- reused here as an 18-slot
# categorical cycle (9 solid + 9 light) for task categories instead, since
# that's the same "muted rainbow" palette every other paper figure draws
# from. 14 task categories comfortably fit within the 18 slots with no repeats.
PALETTE_CYCLE = [PAPER_COLORS[v] for v in range(1, 10)] + [
    PAPER_FILL_COLORS[v] for v in range(1, 10)
]


def apply_theme() -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            "pdf.fonttype": 42,
        }
    )


def build_display_categories(
    task_categories: list[str],
) -> tuple[list[str], list[str], dict[str, str]]:
    """Map raw category keys to short display labels and assign each a
    colour from the shared paper palette (visualisation.arc_paper)."""
    display_categories = [shorten_task_label(format_task_category(c)) for c in task_categories]
    unique_categories = sorted(set(display_categories))
    if len(unique_categories) > len(PALETTE_CYCLE):
        msg = (
            f"{len(unique_categories)} categories exceed the "
            f"{len(PALETTE_CYCLE)}-colour palette cycle."
        )
        raise ValueError(msg)
    color_by_category = dict(zip(unique_categories, PALETTE_CYCLE, strict=False))
    return display_categories, unique_categories, color_by_category


def render_themed_panel(
    coords: np.ndarray, display_categories: list[str], unique_categories: list[str],
    color_by_category: dict[str, str],
) -> plt.Figure:
    """One projection's scatter, styled uniform with the other plots in this
    project: square, no title, top/right spines stripped, large legend
    wrapped across the bottom."""
    group_values = ["__all__"] * len(display_categories)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    handles_by_category: dict[str, object] = {}
    _draw_scatter_panel(
        ax, coords, display_categories, group_values, unique_categories, ["__all__"],
        color_by_category, {}, False, handles_by_category, {},
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ncol = min(len(unique_categories), 4)
    fig.legend(
        handles_by_category.values(), handles_by_category.keys(),
        loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=ncol, frameon=False,
        markerscale=1.8,
    )
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projections",
        nargs="+",
        default=None,
        choices=["PCA", "t-SNE", "UMAP"],
        help="Subset to render (default: all three available).",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None, choices=list(CONDITIONS),
        help="Subset to render (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_theme()

    conditions = args.conditions or list(CONDITIONS)
    for condition in conditions:
        npz_path = HERE / CONDITIONS[condition]
        if not npz_path.exists():
            print(f"Skipping {condition}: {npz_path} not found (run dump_embedding_clusters.py).")
            continue

        data = np.load(npz_path, allow_pickle=True)
        vectors, task_categories = data["vectors"], list(data["task_categories"])
        display_categories, unique_categories, color_by_category = build_display_categories(
            task_categories
        )

        panels = _compute_projection_panels(vectors)

        for name, coords in panels:
            if args.projections and name not in args.projections:
                continue
            figure = render_themed_panel(
                coords, display_categories, unique_categories, color_by_category
            )
            projection_key = name.lower().replace("-", "")  # "t-SNE" -> "tsne"
            out_stem = HERE / f"cluster_dim6_{condition}_{projection_key}"
            figure.savefig(out_stem.with_suffix(".png"))
            figure.savefig(out_stem.with_suffix(".pdf"))
            print(f"Saved {out_stem}.png / .pdf")


if __name__ == "__main__":
    main()
