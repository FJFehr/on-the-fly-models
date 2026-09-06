"""Paired cluster-map plots for a hypernetwork's pooled task latent.

The default look-and-feel for comparing two trained runs' embedding
disentanglement: reads two .npz dumps -- saved automatically by training
itself (`log_embedding_clusters: true`, see training.logging's
log_embedding_cluster_plots) as `outputs/<project>/<run>/embedding_clusters/embeddings.npz`,
or by legacy/scripts/dump_embedding_clusters.py for a run that predates
that -- and renders each available projection (PCA / t-SNE / UMAP) as one figure
with both runs side by side, sharing one legend -- so any two conditions
(task-ID vs no task-ID, two model sizes, two seeds, ...) are directly
comparable rather than living in separate single-condition files. Styled
uniform with the rest of the paper's figures:

- Nimbus Roman, larger labels, no chart title (each panel keeps a small
  label naming that run, since two panels side by side need that to stay
  legible, unlike single-panel plots), top/right spines stripped
  (matching experiments/01_multitask_capacity/plot_capacity_cliff.py
  and plot_per_task.py)
- category colours drawn from visualisation.arc_paper's PAPER_COLORS /
  PAPER_FILL_COLORS -- the same muted rainbow used in the task-example
  figures (e.g. outputs/figures/00_task_examples/arc_1d_task_1d_flip_46.pdf),
  not a generic matplotlib colormap
- category labels run through arc_paper.shorten_task_label (drop
  "Pixel(s)", "Multicolor" -> "MC"), same as plot_per_task.py's x-axis
- legend as a large, wrapped block along the bottom
- markers at 3x the base visual diameter, alpha 0.7

Reuses visualisation.embedding_clusters's projection-computation and
scatter-drawing logic (_compute_projection_panels / _draw_scatter_panel)
rather than reimplementing it -- this module only supplies the colours,
labels, and chrome that need to differ from that module's own
render_single_projection_figures/render_embedding_cluster_figure, which are
still what training's own end-of-run logging uses unchanged (that hook only
ever sees one run's embeddings at a time, so it can't produce a *paired*
figure by itself -- pairing is necessarily a post-hoc, two-checkpoint step).
Each run's projection is still fit independently (t-SNE/UMAP can't be fit
jointly across two different embedding spaces); only the figure layout
pairs them.

Usage
-----
Train two runs with `log_embedding_clusters: true` first (each dumps its own
embeddings.npz automatically), then:

    uv run python -m visualisation.plot_embedding_clusters \\
        --left-npz path/to/run_a.npz --left-label "Task ID" \\
        --right-npz path/to/run_b.npz --right-label "No task ID" \\
        --out-dir path/to/output/dir --out-prefix cluster_paired

For a specific experiment's usual pair, prefer that experiment's own thin
preset script if one exists, e.g.
experiments/02_hypernetwork_multitask/plot_embedding_clusters.py
-- same underlying code, just with that experiment's paths/labels filled in
so it can be rerun with no arguments.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from visualisation.arc_paper import PAPER_COLORS, PAPER_FILL_COLORS, shorten_task_label
from visualisation.embedding_clusters import _compute_projection_panels, _draw_scatter_panel
from visualisation.style import apply_latex_style, format_task_category

# PAPER_COLORS/PAPER_FILL_COLORS are keyed 0-9 for ARC cell values (0 = null/
# background, not a real category colour) -- reused here as an 18-slot
# categorical cycle (9 solid + 9 light) for task categories instead, since
# that's the same "muted rainbow" palette every other paper figure draws
# from. Raise past 18 distinct categories and this cycle repeats.
PALETTE_CYCLE = [PAPER_COLORS[v] for v in range(1, 10)] + [
    PAPER_FILL_COLORS[v] for v in range(1, 10)
]

MARKER_STYLE = {"s": 18 * 9, "alpha": 0.7}  # 3x visual diameter (area scales by 3**2)


def apply_theme() -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 21,
            "axes.titlesize": 25,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 20,
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
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    left_label: str,
    right_label: str,
    display_categories: list[str],
    unique_categories: list[str],
    color_by_category: dict[str, str],
) -> plt.Figure:
    """One projection, two runs side by side (left, right), sharing one
    legend along the bottom. No chart title (each panel's own small label
    distinguishes it instead), top/right spines stripped."""
    group_values = ["__all__"] * len(display_categories)
    # _draw_scatter_panel's default marker style ({"s": 18, "alpha": 0.75}) is
    # only reachable through its group-style override path -- groups_enabled
    # is set True here purely to reuse that path for a single pseudo-group,
    # not because these points actually belong to different groups (the
    # legend still only ever uses handles_by_category, never handles_by_group).
    marker_style = {"__all__": MARKER_STYLE}

    # Shorter than wide -- fits a single-column paper figure better than the
    # squarer per-panel proportions used elsewhere; margins trimmed below
    # (tight_layout's own pad, plus the legend sitting close under the axes)
    # keep it from reading as mostly whitespace once compressed this much.
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    handles_by_category: dict[str, object] = {}
    panels = ((axes[0], left_coords, left_label), (axes[1], right_coords, right_label))
    for ax, coords, label in panels:
        _draw_scatter_panel(
            ax, coords, display_categories, group_values,
            unique_categories, ["__all__"], color_by_category, marker_style, True,
            handles_by_category, {},
        )
        ax.set_title(label, pad=6)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    ncol = min(len(unique_categories), 7)
    fig.legend(
        handles_by_category.values(), handles_by_category.keys(),
        loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False,
        markerscale=1.0, handletextpad=0.4, columnspacing=1.0,
    )
    fig.tight_layout(pad=0.6)
    return fig


def _load_npz(npz_path: Path) -> tuple[np.ndarray, list[str]] | None:
    if not npz_path.exists():
        print(f"Skipping: {npz_path} not found (train with log_embedding_clusters: true first).")
        return None
    data = np.load(npz_path, allow_pickle=True)
    return data["vectors"], list(data["task_categories"])


def render_paired_cluster_maps(
    left_npz: Path,
    left_label: str,
    right_npz: Path,
    right_label: str,
    out_dir: Path,
    out_prefix: str = "cluster_paired",
    projections: list[str] | None = None,
) -> None:
    """Load both .npz dumps and render every shared projection as one
    left/right paired figure (PNG + PDF) in out_dir."""
    apply_theme()

    left = _load_npz(left_npz)
    right = _load_npz(right_npz)
    if left is None or right is None:
        return
    left_vectors, left_categories = left
    right_vectors, right_categories = right

    if sorted(left_categories) != sorted(right_categories):
        msg = (
            "left/right task categories differ -- these two runs weren't "
            "evaluated on the same category set, so they can't share one "
            "legend/colour mapping."
        )
        raise ValueError(msg)

    display_categories, unique_categories, color_by_category = build_display_categories(
        left_categories
    )

    left_panels = dict(_compute_projection_panels(left_vectors))
    right_panels = dict(_compute_projection_panels(right_vectors))
    available = set(left_panels) & set(right_panels)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("PCA", "t-SNE", "UMAP"):
        if name not in available:
            continue
        if projections and name not in projections:
            continue
        figure = render_paired_panel(
            left_panels[name], right_panels[name], left_label, right_label,
            display_categories, unique_categories, color_by_category,
        )
        projection_key = name.lower().replace("-", "")  # "t-SNE" -> "tsne"
        out_stem = out_dir / f"{out_prefix}_{projection_key}"
        figure.savefig(out_stem.with_suffix(".png"))
        figure.savefig(out_stem.with_suffix(".pdf"))
        print(f"Saved {out_stem}.png / .pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-npz", required=True, type=Path)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-npz", required=True, type=Path)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-prefix", default="cluster_paired")
    parser.add_argument(
        "--projections", nargs="+", default=None, choices=["PCA", "t-SNE", "UMAP"],
        help="Subset to render (default: all three available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_paired_cluster_maps(
        args.left_npz, args.left_label, args.right_npz, args.right_label,
        args.out_dir, args.out_prefix, args.projections,
    )


if __name__ == "__main__":
    main()
