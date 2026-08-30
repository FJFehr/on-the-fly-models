"""Themed, paired cluster-map plots for the joint hypernetwork's pooled
task latent (Experiment 4).

Reads the raw pooled-embedding vectors dumped by scripts/dump_embedding_clusters.py
(one .npz per condition, e.g. embeddings_dim6_notd.npz) and renders each
available projection (PCA / t-SNE / UMAP) as one figure with both
conditions side by side -- notd (no task ID) vs frozentd (frozen task-ID
embedding) -- sharing one legend, so the two are directly comparable rather
than living in six separate single-condition files. Styled uniform with the
rest of the paper's figures:

- Nimbus Roman, larger labels, no chart title (each panel keeps a small
  condition label -- "No task ID" / "Frozen task ID" -- since two panels
  side by side need that to stay legible, unlike the single-panel plots
  elsewhere), top/right spines stripped
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
still what training's own end-of-run logging uses unchanged. Each
condition's projection is still fit independently (t-SNE/UMAP can't be
fit jointly across two separately-trained models' embedding spaces) --
pairing only happens at the figure/layout level.

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
        --projections t-SNE   # only the pairing actually used in the paper
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
    "frozentd": "embeddings_dim6_frozentd.npz",
    "notd": "embeddings_dim6_notd.npz",
}
CONDITION_LABELS = {"notd": "No task ID", "frozentd": "Task ID"}

# PAPER_COLORS/PAPER_FILL_COLORS are keyed 0-9 for ARC cell values (0 = null/
# background, not a real category colour) -- reused here as an 18-slot
# categorical cycle (9 solid + 9 light) for task categories instead, since
# that's the same "muted rainbow" palette every other paper figure draws
# from. 14 task categories comfortably fit within the 18 slots with no repeats.
PALETTE_CYCLE = [PAPER_COLORS[v] for v in range(1, 10)] + [
    PAPER_FILL_COLORS[v] for v in range(1, 10)
]

MARKER_STYLE = {"s": 18 * 9, "alpha": 0.7}  # 3x visual diameter (area scales by 3**2); see below


def apply_theme() -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 21,
            "axes.titlesize": 21,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 19,
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


def render_paired_panel(
    coords_by_condition: dict[str, np.ndarray],
    display_categories: list[str],
    unique_categories: list[str],
    color_by_category: dict[str, str],
) -> plt.Figure:
    """One projection, both conditions side by side, sharing one legend
    along the bottom. Square-ish per panel, no chart title (each panel's
    own small label distinguishes it instead), top/right spines stripped."""
    group_values = ["__all__"] * len(display_categories)
    # _draw_scatter_panel's default marker style ({"s": 18, "alpha": 0.75}) is
    # only reachable through its group-style override path -- groups_enabled
    # is set True here purely to reuse that path for a single pseudo-group,
    # not because these points actually belong to different groups (the
    # legend still only ever uses handles_by_category, never handles_by_group).
    marker_style = {"__all__": MARKER_STYLE}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.5))
    handles_by_category: dict[str, object] = {}
    for ax, condition in zip(axes, CONDITIONS, strict=True):
        _draw_scatter_panel(
            ax, coords_by_condition[condition], display_categories, group_values,
            unique_categories, ["__all__"], color_by_category, marker_style, True,
            handles_by_category, {},
        )
        ax.set_title(CONDITION_LABELS[condition], pad=10)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    ncol = min(len(unique_categories), 4)
    fig.legend(
        handles_by_category.values(), handles_by_category.keys(),
        loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=ncol, frameon=False,
        markerscale=0.8,
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
    return parser.parse_args()


def load_condition(condition: str) -> tuple[np.ndarray, list[str]] | None:
    npz_path = HERE / CONDITIONS[condition]
    if not npz_path.exists():
        print(f"Skipping {condition}: {npz_path} not found (run dump_embedding_clusters.py).")
        return None
    data = np.load(npz_path, allow_pickle=True)
    return data["vectors"], list(data["task_categories"])


def main() -> None:
    args = parse_args()
    apply_theme()

    loaded = {c: load_condition(c) for c in CONDITIONS}
    if any(v is None for v in loaded.values()):
        return

    # Both conditions share the same underlying 14-task/70-record structure,
    # so category display names/colours only need computing once.
    _, task_categories = next(iter(loaded.values()))
    display_categories, unique_categories, color_by_category = build_display_categories(
        task_categories
    )

    panels_by_condition = {c: dict(_compute_projection_panels(v)) for c, (v, _) in loaded.items()}
    available_projections = set.intersection(*(set(p) for p in panels_by_condition.values()))

    for name in ("PCA", "t-SNE", "UMAP"):
        if name not in available_projections:
            continue
        if args.projections and name not in args.projections:
            continue
        coords_by_condition = {c: panels_by_condition[c][name] for c in CONDITIONS}
        figure = render_paired_panel(
            coords_by_condition, display_categories, unique_categories, color_by_category
        )
        projection_key = name.lower().replace("-", "")  # "t-SNE" -> "tsne"
        out_stem = HERE / f"cluster_dim6_paired_{projection_key}"
        figure.savefig(out_stem.with_suffix(".png"))
        figure.savefig(out_stem.with_suffix(".pdf"))
        print(f"Saved {out_stem}.png / .pdf")


if __name__ == "__main__":
    main()
