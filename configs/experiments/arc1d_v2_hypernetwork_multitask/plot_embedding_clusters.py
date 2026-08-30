"""Themed, individual cluster-map plots for the joint hypernetwork's pooled
task latent (Experiment 4).

Reads the raw pooled-embedding vectors dumped by scripts/dump_embedding_clusters.py
(one .npz per condition, e.g. embeddings_dim6_notd.npz) and renders each
available projection (PCA / t-SNE / UMAP) as its own standalone figure,
styled uniform with plot_capacity_cliff.py / plot_per_task.py: Nimbus Roman,
larger labels, no title, top/right spines stripped, legend as a wrapped row
across the top rather than matplotlib's default bottom block. Reuses
visualisation.embedding_clusters's projection-computation and scatter-colour
logic (_compute_projection_panels / _resolve_common_styling /
_draw_scatter_panel) rather than reimplementing it -- this script only
supplies the chrome (title/legend/spines) that needs to differ from that
module's own render_single_projection_figures/render_embedding_cluster_figure,
which are still what training's own end-of-run logging uses unchanged.

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

from visualisation.embedding_clusters import (
    _compute_projection_panels,
    _draw_scatter_panel,
    _resolve_common_styling,
)
from visualisation.style import apply_latex_style

HERE = Path(__file__).parent
CONDITIONS = {
    "notd": "embeddings_dim6_notd.npz",
    "frozentd": "embeddings_dim6_frozentd.npz",
}


def apply_theme() -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
        }
    )


def render_themed_panel(coords: np.ndarray, styling: tuple) -> plt.Figure:
    """One projection's scatter, styled uniform with the other plots in this
    project: square, no title, top/right spines stripped, legend wrapped
    across the top instead of matplotlib's default bottom block."""
    (
        display_categories, unique_categories, color_by_category,
        group_styles, group_values, unique_groups,
    ) = styling

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    handles_by_category: dict[str, object] = {}
    _draw_scatter_panel(
        ax, coords, display_categories, group_values, unique_categories, unique_groups,
        color_by_category, group_styles, False, handles_by_category, {},
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ncol = min(len(unique_categories), 5)
    fig.legend(
        handles_by_category.values(), handles_by_category.keys(),
        loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=ncol, frameon=False,
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

        panels = _compute_projection_panels(vectors)
        styling = _resolve_common_styling(task_categories, None, None)

        for name, coords in panels:
            if args.projections and name not in args.projections:
                continue
            figure = render_themed_panel(coords, styling)
            projection_key = name.lower().replace("-", "")  # "t-SNE" -> "tsne"
            out_stem = HERE / f"cluster_dim6_{condition}_{projection_key}"
            figure.savefig(out_stem.with_suffix(".png"))
            figure.savefig(out_stem.with_suffix(".pdf"))
            print(f"Saved {out_stem}.png / .pdf")


if __name__ == "__main__":
    main()
